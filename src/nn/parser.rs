use spacy_model::{Bundle, ComponentManifest, NodeManifest};
use thiserror::Error;

use crate::{LinearLayer, Matrix, ModelOpError, PrecomputableAffineLayer, PrecomputedAffine};

#[derive(Debug, Error)]
pub enum TransitionScorerError {
    #[error(transparent)]
    Model(#[from] ModelOpError),
    #[error("transition component {0:?} is missing")]
    MissingComponent(String),
    #[error("transition model graph is invalid: {0}")]
    InvalidGraph(String),
}

/// Neural scoring portion shared by spaCy's dependency parser and NER.
pub struct TransitionScorer {
    projection: LinearLayer,
    lower: PrecomputableAffineLayer,
    upper: Option<LinearLayer>,
}

impl TransitionScorer {
    /// Load a parser-style transition scorer.
    ///
    /// # Errors
    ///
    /// Returns an error if graph references or required layer parameters are
    /// missing or incompatible.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, TransitionScorerError> {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| TransitionScorerError::MissingComponent(component_name.to_owned()))?;
        let root = root_node(component)?;
        let lower = reference_node(component, root, "lower")?;
        let upper = reference_node(component, root, "upper")?;
        let tok2vec = reference_node(component, root, "tok2vec")?;
        let projection = tok2vec
            .children
            .iter()
            .rev()
            .filter_map(|index| node(component, *index))
            .find(|node| node.name == "linear")
            .ok_or_else(|| {
                TransitionScorerError::InvalidGraph(
                    "tok2vec branch has no linear projection".to_owned(),
                )
            })?;
        let upper = if upper.name == "noop" {
            None
        } else {
            Some(LinearLayer::load(bundle, upper)?)
        };
        Ok(Self {
            projection: LinearLayer::load(bundle, projection)?,
            lower: PrecomputableAffineLayer::load(bundle, lower)?,
            upper,
        })
    }

    /// Project shared token vectors and precompute state feature values.
    ///
    /// # Errors
    ///
    /// Returns an error if the shared token vector width is incompatible.
    pub fn precompute(&self, tok2vec: &Matrix) -> Result<PrecomputedAffine, TransitionScorerError> {
        let projected = self.projection.forward(tok2vec)?;
        Ok(self.lower.precompute(&projected)?)
    }

    /// Score one state represented by its ordered context token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid feature IDs or incompatible model layers.
    pub fn score(
        &self,
        cache: &PrecomputedAffine,
        token_ids: &[i32],
    ) -> Result<Matrix, TransitionScorerError> {
        let hidden = cache.hidden(token_ids)?;
        match &self.upper {
            Some(upper) => Ok(upper.forward(&hidden)?),
            None => Ok(hidden),
        }
    }

    #[must_use]
    pub fn class_count(&self) -> usize {
        self.upper
            .as_ref()
            .map_or_else(|| self.lower.outputs(), LinearLayer::outputs)
    }
}

fn root_node(component: &ComponentManifest) -> Result<&NodeManifest, TransitionScorerError> {
    let index = component.root_node.ok_or_else(|| {
        TransitionScorerError::InvalidGraph("component has no root node".to_owned())
    })?;
    node(component, index)
        .ok_or_else(|| TransitionScorerError::InvalidGraph(format!("root node {index} is missing")))
}

fn reference_node<'a>(
    component: &'a ComponentManifest,
    root: &NodeManifest,
    name: &str,
) -> Result<&'a NodeManifest, TransitionScorerError> {
    let index = root.refs.get(name).copied().flatten().ok_or_else(|| {
        TransitionScorerError::InvalidGraph(format!("root has no {name:?} reference"))
    })?;
    node(component, index).ok_or_else(|| {
        TransitionScorerError::InvalidGraph(format!(
            "{name:?} reference points to missing node {index}"
        ))
    })
}

fn node(component: &ComponentManifest, index: usize) -> Option<&NodeManifest> {
    component.nodes.iter().find(|node| node.index == index)
}
