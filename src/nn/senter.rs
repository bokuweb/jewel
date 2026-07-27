use spacy_core::Doc;
use spacy_model::{Bundle, ComponentManifest};
use thiserror::Error;

use crate::{Matrix, Tagger, TaggerError, Tok2Vec, Tok2VecError};

#[derive(Debug, Error)]
pub enum SentenceRecognizerError {
    #[error(transparent)]
    Tok2Vec(#[from] Tok2VecError),
    #[error(transparent)]
    Classifier(#[from] TaggerError),
    #[error("sentence recognizer setting {name:?} is missing or invalid")]
    InvalidSetting { name: &'static str },
    #[error("sentence recognizer labels are invalid: expected [\"I\", \"S\"], got {0:?}")]
    InvalidLabels(Vec<String>),
    #[error("sentence recognizer requires vectors from an upstream tok2vec component")]
    ExternalTok2VecRequired,
}

/// Trainable sentence boundary detection compatible with spaCy's
/// `SentenceRecognizer`.
pub struct SentenceRecognizer {
    encoder: Option<Tok2Vec>,
    classifier: Tagger,
    overwrite: bool,
}

impl SentenceRecognizer {
    /// Load a sentence recognizer when the exported pipeline contains one.
    ///
    /// # Errors
    ///
    /// Returns an error when the graph, labels, or settings are incompatible.
    pub fn load_optional(bundle: &Bundle) -> Result<Option<Self>, SentenceRecognizerError> {
        bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.factory == "senter")
            .map(|component| Self::from_component(bundle, component))
            .transpose()
    }

    fn from_component(
        bundle: &Bundle,
        component: &ComponentManifest,
    ) -> Result<Self, SentenceRecognizerError> {
        if component.labels != ["I", "S"] {
            return Err(SentenceRecognizerError::InvalidLabels(
                component.labels.clone(),
            ));
        }
        let overwrite = component
            .settings
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .ok_or(SentenceRecognizerError::InvalidSetting { name: "overwrite" })?;
        let external = component
            .nodes
            .iter()
            .any(|node| node.name == "tok2vec-listener");
        Ok(Self {
            encoder: if external {
                None
            } else {
                Some(Tok2Vec::load(bundle, &component.name)?)
            },
            classifier: Tagger::load(bundle, &component.name)?,
            overwrite,
        })
    }

    /// Return whether this component consumes vectors from an upstream
    /// `tok2vec` component.
    #[must_use]
    pub const fn requires_external_tok2vec(&self) -> bool {
        self.encoder.is_none()
    }

    /// Predict and attach sentence starts.
    ///
    /// # Errors
    ///
    /// Returns an error when an external encoder is required or inference
    /// fails.
    pub fn annotate(&self, doc: &mut Doc) -> Result<(), SentenceRecognizerError> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or(SentenceRecognizerError::ExternalTok2VecRequired)?;
        let vectors = encoder.forward(doc)?;
        self.annotate_with_tok2vec(doc, &vectors)
    }

    /// Predict and attach sentence starts from shared upstream token vectors.
    ///
    /// # Errors
    ///
    /// Returns an error when classifier inference fails.
    pub fn annotate_with_tok2vec(
        &self,
        doc: &mut Doc,
        vectors: &Matrix,
    ) -> Result<(), SentenceRecognizerError> {
        let scores = self.classifier.scores(vectors)?;
        if scores.rows() != doc.len() {
            return Err(TaggerError::RowCount {
                expected: doc.len(),
                actual: scores.rows(),
            }
            .into());
        }
        set_annotations(doc, &self.classifier.predict(&scores), self.overwrite);
        Ok(())
    }
}

fn set_annotations(doc: &mut Doc, classes: &[usize], overwrite: bool) {
    for (token, class) in doc.tokens_mut().iter_mut().zip(classes) {
        if token.sent_start == 0 || overwrite {
            token.sent_start = if *class == 1 { 1 } else { -1 };
        }
    }
}

#[cfg(test)]
mod tests {
    use spacy_core::Doc;

    use super::set_annotations;

    #[test]
    fn class_one_is_a_sentence_start() {
        let mut doc =
            Doc::from_words(&["Alice", "works", ".", "Bob"], &[true, false, true, false]).unwrap();
        set_annotations(&mut doc, &[1, 0, 0, 1], false);
        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| token.sent_start)
                .collect::<Vec<_>>(),
            [1, -1, -1, 1]
        );
    }

    #[test]
    fn overwrite_controls_existing_sentence_annotations() {
        let mut preserved =
            Doc::from_words(&["Alice", "works", "today"], &[true, true, false]).unwrap();
        preserved.tokens_mut()[1].sent_start = 1;
        set_annotations(&mut preserved, &[0, 0, 1], false);
        assert_eq!(
            preserved
                .tokens()
                .iter()
                .map(|token| token.sent_start)
                .collect::<Vec<_>>(),
            [-1, 1, 1]
        );

        let mut replaced = preserved.clone();
        set_annotations(&mut replaced, &[1, 0, 0], true);
        assert_eq!(
            replaced
                .tokens()
                .iter()
                .map(|token| token.sent_start)
                .collect::<Vec<_>>(),
            [1, -1, -1]
        );
    }
}
