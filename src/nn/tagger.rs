use spacy_core::{Doc, StringStore};
use spacy_model::Bundle;
use thiserror::Error;

use crate::{Matrix, ModelOpError, SoftmaxLayer};

#[derive(Debug, Error)]
pub enum TaggerError {
    #[error(transparent)]
    Model(#[from] ModelOpError),
    #[error("tagger component {0:?} is missing")]
    MissingComponent(String),
    #[error("tagger graph is invalid: {0}")]
    InvalidGraph(String),
    #[error("tagger output has {actual} rows for {expected} tokens")]
    RowCount { expected: usize, actual: usize },
}

pub struct Tagger {
    output: SoftmaxLayer,
    labels: Vec<String>,
}

impl Tagger {
    /// Load a tagger output layer and its ordered labels.
    ///
    /// # Errors
    ///
    /// Returns an error if the component, softmax node, or labels are missing.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, TaggerError> {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| TaggerError::MissingComponent(component_name.to_owned()))?;
        let softmax = component
            .nodes
            .iter()
            .find(|node| node.name == "softmax")
            .ok_or_else(|| TaggerError::InvalidGraph("missing softmax node".to_owned()))?;
        let outputs = softmax.dims.get("nO").copied().flatten().ok_or_else(|| {
            TaggerError::InvalidGraph("softmax output dimension is missing".to_owned())
        })?;
        if component.labels.len() != outputs {
            return Err(TaggerError::InvalidGraph(format!(
                "softmax has {outputs} outputs but {} labels",
                component.labels.len()
            )));
        }
        Ok(Self {
            output: SoftmaxLayer::load(bundle, softmax)?,
            labels: component.labels.clone(),
        })
    }

    /// Compute raw tagger scores from `tok2vec` rows.
    ///
    /// # Errors
    ///
    /// Returns an error when the input width differs from the model.
    pub fn scores(&self, tok2vec: &Matrix) -> Result<Matrix, TaggerError> {
        Ok(self.output.forward(tok2vec)?)
    }

    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Select the first maximum-scoring label for each token.
    #[must_use]
    pub fn predict(&self, scores: &Matrix) -> Vec<usize> {
        (0..scores.rows())
            .map(|row| {
                scores
                    .row(row)
                    .iter()
                    .copied()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |best, candidate| {
                        if candidate.1 > best.1 {
                            candidate
                        } else {
                            best
                        }
                    })
                    .0
            })
            .collect()
    }

    /// Write predicted fine-grained tags to a document.
    ///
    /// # Errors
    ///
    /// Returns an error if score rows do not match tokens or an index is
    /// outside the exported label list.
    pub fn annotate(&self, doc: &mut Doc, scores: &Matrix) -> Result<(), TaggerError> {
        if scores.rows() != doc.len() {
            return Err(TaggerError::RowCount {
                expected: doc.len(),
                actual: scores.rows(),
            });
        }
        for (token, label) in doc.tokens_mut().iter_mut().zip(self.predict(scores)) {
            let label = self.labels.get(label).ok_or_else(|| {
                TaggerError::InvalidGraph(format!("predicted label index {label} is out of bounds"))
            })?;
            token.tag = StringStore::id(label);
        }
        Ok(())
    }
}
