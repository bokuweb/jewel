//! Tokenization runtimes for exported spaCy language rules.

use std::error::Error as StdError;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use spacy_core::Doc;
use thiserror::Error;

#[cfg(feature = "delarocha-tokenizer")]
mod delarocha;
#[cfg(feature = "sudachi-tokenizer")]
mod japanese;
mod regex;

#[cfg(feature = "delarocha-tokenizer")]
pub use delarocha::{
    DelarochaCompatibilityRule, DelarochaFeatureSchema, DelarochaRuleToken, DelarochaTokenizer,
    DelarochaTokenizerConfig, DelarochaTokenizerError,
};
#[cfg(feature = "sudachi-tokenizer")]
pub use japanese::{JapaneseTokenizer, JapaneseTokenizerConfig, JapaneseTokenizerError, SplitMode};
pub use regex::{ExceptionToken, RegexTokenizer, RegexTokenizerConfig, TokenizerError};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TagBigramRule {
    pub tag: String,
    pub next_tag: String,
    pub pos: Option<u64>,
    pub next_pos: Option<u64>,
}

/// Thread-safe tokenization boundary used by inference pipelines.
///
/// Implementations must return spaCy-compatible token attributes and Unicode
/// code-point offsets expected by the exported model.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into a spaCy-compatible document.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizeError`] when tokenization or document construction
    /// fails.
    fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError>;

    /// Create request-local state for repeated tokenization.
    ///
    /// The default session delegates to [`Tokenizer::tokenize`]. Backends with
    /// reusable scratch buffers can override this method without changing the
    /// application-owned tokenizer injection contract.
    fn session(&self) -> Box<dyn TokenizerSession + '_> {
        Box::new(BorrowedTokenizerSession { tokenizer: self })
    }
}

/// Request-local tokenizer state used by sequential and batch inference.
pub trait TokenizerSession {
    /// Tokenize one document while retaining backend scratch capacity for the
    /// next call.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizeError`] when tokenization or document construction
    /// fails.
    fn tokenize(&mut self, text: &str) -> Result<Doc, TokenizeError>;
}

struct BorrowedTokenizerSession<'a, T: Tokenizer + ?Sized> {
    tokenizer: &'a T,
}

impl<T: Tokenizer + ?Sized> TokenizerSession for BorrowedTokenizerSession<'_, T> {
    fn tokenize(&mut self, text: &str) -> Result<Doc, TokenizeError> {
        self.tokenizer.tokenize(text)
    }
}

/// Shareable tokenizer handle accepted by pipeline injection constructors.
pub type SharedTokenizer = Arc<dyn Tokenizer>;

#[derive(Debug, Error)]
#[error("tokenization failed: {source}")]
pub struct TokenizeError {
    #[source]
    source: Box<dyn StdError + Send + Sync>,
}

impl TokenizeError {
    /// Erase a concrete tokenizer error while preserving its source chain.
    pub fn new(error: impl StdError + Send + Sync + 'static) -> Self {
        Self {
            source: Box::new(error),
        }
    }
}

impl Tokenizer for RegexTokenizer {
    fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError> {
        RegexTokenizer::tokenize(self, text).map_err(TokenizeError::new)
    }
}

#[cfg(feature = "sudachi-tokenizer")]
impl Tokenizer for JapaneseTokenizer {
    fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError> {
        JapaneseTokenizer::tokenize(self, text).map_err(TokenizeError::new)
    }
}

#[cfg(feature = "delarocha-tokenizer")]
impl Tokenizer for DelarochaTokenizer {
    fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError> {
        DelarochaTokenizer::tokenize(self, text).map_err(TokenizeError::new)
    }

    fn session(&self) -> Box<dyn TokenizerSession + '_> {
        self.reusable_session()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use spacy_core::Doc;

    use super::{SharedTokenizer, TokenizeError, Tokenizer};

    struct UpstreamTokenizer;

    impl Tokenizer for UpstreamTokenizer {
        fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError> {
            Doc::from_words(&[text], &[false]).map_err(TokenizeError::new)
        }
    }

    #[test]
    fn upper_layer_tokenizer_can_be_type_erased_and_shared() {
        let tokenizer: SharedTokenizer = Arc::new(UpstreamTokenizer);
        let doc = tokenizer.tokenize("山田太郎").unwrap();

        assert_eq!(doc.text(), "山田太郎");
    }

    #[test]
    fn upper_layer_tokenizer_gets_a_default_reusable_session() {
        let tokenizer: SharedTokenizer = Arc::new(UpstreamTokenizer);
        let mut session = tokenizer.session();

        assert_eq!(session.tokenize("山田太郎").unwrap().text(), "山田太郎");
        assert_eq!(session.tokenize("佐藤花子").unwrap().text(), "佐藤花子");
    }
}
