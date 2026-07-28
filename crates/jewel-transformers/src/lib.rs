//! Transformer encoder contracts for Jewel model adapters.
//!
//! This crate intentionally does not select an inference engine. CPU, GPU,
//! ONNX, and native Rust implementations can implement [`TransformerEncoder`]
//! without adding their dependencies to `jewel-core`.

use jewel_core::{Doc, Matrix};
use thiserror::Error;

/// Static transformer configuration exported with a model bundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransformerSpec {
    pub model: String,
    pub hidden_width: usize,
    pub max_tokens: usize,
    pub stride: usize,
}

impl TransformerSpec {
    /// Validate dimensions needed by the shared transformer pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error for empty model names, zero dimensions, or a stride
    /// that cannot advance the configured token window.
    pub fn validate(&self) -> Result<(), TransformerError> {
        if self.model.is_empty() {
            return Err(TransformerError::InvalidSpec(
                "model name must not be empty".to_owned(),
            ));
        }
        if self.hidden_width == 0 {
            return Err(TransformerError::InvalidSpec(
                "hidden width must be greater than zero".to_owned(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(TransformerError::InvalidSpec(
                "max tokens must be greater than zero".to_owned(),
            ));
        }
        if self.stride >= self.max_tokens {
            return Err(TransformerError::InvalidSpec(
                "stride must be smaller than max tokens".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Failure produced by a transformer adapter or its inference backend.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TransformerError {
    #[error("invalid transformer specification: {0}")]
    InvalidSpec(String),
    #[error("transformer backend failed: {0}")]
    Backend(String),
    #[error(
        "transformer returned {actual_rows} token rows with width {actual_width}; \
         expected {expected_rows} rows with width {expected_width}"
    )]
    InvalidOutput {
        expected_rows: usize,
        actual_rows: usize,
        expected_width: usize,
        actual_width: usize,
    },
}

/// Backend-neutral encoder producing one contextual vector per Jewel token.
pub trait TransformerEncoder: Send + Sync {
    /// Return the immutable model configuration used by this encoder.
    fn spec(&self) -> &TransformerSpec;

    /// Encode a tokenized document into one row per Jewel token.
    ///
    /// # Errors
    ///
    /// Returns an error when token alignment or backend inference fails.
    fn encode(&self, doc: &Doc) -> Result<Matrix, TransformerError>;
}

/// Validate an encoder result before passing it to a spaCy listener.
///
/// # Errors
///
/// Returns an error when the matrix is not aligned to the document or model
/// width.
pub fn validate_token_vectors(
    doc: &Doc,
    spec: &TransformerSpec,
    vectors: &Matrix,
) -> Result<(), TransformerError> {
    if vectors.rows() != doc.len() || vectors.cols() != spec.hidden_width {
        return Err(TransformerError::InvalidOutput {
            expected_rows: doc.len(),
            actual_rows: vectors.rows(),
            expected_width: spec.hidden_width,
            actual_width: vectors.cols(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use jewel_core::{Doc, Matrix};

    use super::{validate_token_vectors, TransformerError, TransformerSpec};

    fn spec() -> TransformerSpec {
        TransformerSpec {
            model: "example/electra".to_owned(),
            hidden_width: 4,
            max_tokens: 128,
            stride: 96,
        }
    }

    #[test]
    fn validates_transformer_windows_and_output_alignment() {
        spec().validate().unwrap();
        let doc = Doc::from_words(&["契約", "締結"], &[false, false]).unwrap();
        let vectors = Matrix::zeros(2, 4);
        validate_token_vectors(&doc, &spec(), &vectors).unwrap();

        let error = validate_token_vectors(&doc, &spec(), &Matrix::zeros(3, 4)).unwrap_err();
        assert_eq!(
            error,
            TransformerError::InvalidOutput {
                expected_rows: 2,
                actual_rows: 3,
                expected_width: 4,
                actual_width: 4,
            }
        );
    }

    #[test]
    fn rejects_non_advancing_windows() {
        let mut invalid = spec();
        invalid.stride = invalid.max_tokens;
        assert!(matches!(
            invalid.validate(),
            Err(TransformerError::InvalidSpec(_))
        ));
    }
}
