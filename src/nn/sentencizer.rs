use std::collections::BTreeSet;

use spacy_core::Doc;
use spacy_model::{Bundle, ComponentManifest};
use thiserror::Error;
use unicode_categories::UnicodeCategories;

#[derive(Debug, Error)]
pub enum SentencizerError {
    #[error("sentencizer setting {name:?} is missing or invalid")]
    InvalidSetting { name: &'static str },
}

/// Rule-based sentence boundary detection compatible with spaCy's
/// `Sentencizer`.
pub struct Sentencizer {
    punct_chars: BTreeSet<Box<str>>,
    overwrite: bool,
}

impl Sentencizer {
    /// Load a sentencizer when the exported pipeline contains one.
    ///
    /// # Errors
    ///
    /// Returns an error when the exported component settings are malformed.
    pub fn load_optional(bundle: &Bundle) -> Result<Option<Self>, SentencizerError> {
        bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.factory == "sentencizer")
            .map(Self::from_component)
            .transpose()
    }

    fn from_component(component: &ComponentManifest) -> Result<Self, SentencizerError> {
        let punct_chars = component
            .settings
            .get("punct_chars")
            .and_then(serde_json::Value::as_array)
            .ok_or(SentencizerError::InvalidSetting {
                name: "punct_chars",
            })?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(Into::into)
                    .ok_or(SentencizerError::InvalidSetting {
                        name: "punct_chars",
                    })
            })
            .collect::<Result<_, _>>()?;
        let overwrite = component
            .settings
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .ok_or(SentencizerError::InvalidSetting { name: "overwrite" })?;
        Ok(Self {
            punct_chars,
            overwrite,
        })
    }

    /// Attach sentence starts using the configured terminal token strings.
    pub fn annotate(&self, doc: &mut Doc) {
        let mut guesses = vec![false; doc.len()];
        if !doc.is_empty() {
            guesses[0] = true;
            let mut start = 0;
            let mut seen_period = false;
            for (index, token) in doc.tokens().iter().enumerate() {
                let is_terminal = self.punct_chars.contains(token.text.as_ref());
                let is_punctuation = !token.text.is_empty()
                    && token
                        .text
                        .chars()
                        .all(|character| character.is_punctuation());
                if seen_period && !is_punctuation && !is_terminal {
                    guesses[start] = true;
                    start = index;
                    seen_period = false;
                } else if is_terminal {
                    seen_period = true;
                }
            }
            guesses[start] = true;
        }

        for (token, is_start) in doc.tokens_mut().iter_mut().zip(guesses) {
            if token.sent_start == 0 || self.overwrite {
                token.sent_start = if is_start { 1 } else { -1 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use spacy_core::Doc;
    use spacy_model::ComponentManifest;

    use super::Sentencizer;

    #[derive(Deserialize)]
    struct Fixture {
        spacy_version: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        words: Vec<String>,
        spaces: Vec<bool>,
        punct_chars: Vec<String>,
        overwrite: bool,
        initial: Vec<i8>,
        sent_starts: Vec<i8>,
    }

    fn sentencizer(punct_chars: &[&str], overwrite: bool) -> Sentencizer {
        let component: ComponentManifest = serde_json::from_value(serde_json::json!({
            "name": "sentencizer",
            "factory": "sentencizer",
            "kind": "rule_based",
            "root_node": null,
            "settings": {
                "punct_chars": punct_chars,
                "overwrite": overwrite
            }
        }))
        .unwrap();
        Sentencizer::from_component(&component).unwrap()
    }

    #[test]
    fn starts_after_terminal_and_trailing_punctuation_tokens() {
        let mut doc = Doc::from_words(
            &["Alice", ".", ")", "Bob", "?", "Carol"],
            &[false, false, true, false, true, false],
        )
        .unwrap();
        sentencizer(&[".", "?"], false).annotate(&mut doc);
        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| token.sent_start)
                .collect::<Vec<_>>(),
            [1, -1, -1, 1, -1, 1]
        );
    }

    #[test]
    fn supports_japanese_and_custom_terminal_tokens() {
        let mut doc = Doc::from_words(&["甲", "。", "乙", "END", "丙"], &[false; 5]).unwrap();
        sentencizer(&["。", "END"], false).annotate(&mut doc);
        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| token.sent_start)
                .collect::<Vec<_>>(),
            [1, -1, 1, -1, 1]
        );
    }

    #[test]
    fn overwrite_controls_existing_sentence_annotations() {
        let mut preserved = Doc::from_words(&["Alice", ".", "Bob"], &[false, true, false]).unwrap();
        preserved.tokens_mut()[1].sent_start = 1;
        sentencizer(&["."], false).annotate(&mut preserved);
        assert_eq!(preserved.tokens()[1].sent_start, 1);

        let mut replaced = preserved.clone();
        sentencizer(&["."], true).annotate(&mut replaced);
        assert_eq!(
            replaced
                .tokens()
                .iter()
                .map(|token| token.sent_start)
                .collect::<Vec<_>>(),
            [1, -1, 1]
        );
    }

    #[test]
    fn matches_spacy_3_8_golden_sentence_boundaries() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/sentencizer_spacy_3_8.json"
        ))
        .unwrap();
        assert_eq!(fixture.spacy_version, "3.8.13");
        for case in fixture.cases {
            let mut doc = Doc::from_words(&case.words, &case.spaces).unwrap();
            for (token, sent_start) in doc.tokens_mut().iter_mut().zip(case.initial) {
                token.sent_start = sent_start;
            }
            let punct_chars = case
                .punct_chars
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            sentencizer(&punct_chars, case.overwrite).annotate(&mut doc);
            assert_eq!(
                doc.tokens()
                    .iter()
                    .map(|token| token.sent_start)
                    .collect::<Vec<_>>(),
                case.sent_starts
            );
        }
    }
}
