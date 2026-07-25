use std::collections::BTreeMap;

use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use spacy_core::{Doc, StringStore, TokenData};
use thiserror::Error;

const CURRENT_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RegexTokenizerConfig {
    pub format_version: u32,
    pub language: String,
    pub prefix: Option<String>,
    pub suffix: Option<String>,
    pub infix: Option<String>,
    pub token_match: Option<String>,
    pub url_match: Option<String>,
    pub exceptions: BTreeMap<String, Vec<ExceptionToken>>,
    #[serde(default)]
    pub norm_overrides: BTreeMap<u64, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExceptionToken {
    pub orth: String,
    pub norm: Option<String>,
}

#[derive(Debug)]
pub struct RegexTokenizer {
    language: String,
    prefix: Option<Regex>,
    suffix: Option<Regex>,
    infix: Option<Regex>,
    token_match: Option<Regex>,
    url_match: Option<Regex>,
    exceptions: BTreeMap<String, Vec<ExceptionToken>>,
    norm_overrides: BTreeMap<u64, String>,
    max_exception_chars: usize,
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer configuration is invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported tokenizer format {actual}; this runtime supports {supported}")]
    UnsupportedFormat { actual: u32, supported: u32 },
    #[error("tokenizer regex {name} is invalid: {source}")]
    RegexCompile {
        name: &'static str,
        source: Box<fancy_regex::Error>,
    },
    #[error("tokenizer regex execution failed: {0}")]
    Regex(Box<fancy_regex::Error>),
    #[error("tokenizer exception {text:?} does not reconstruct its source text")]
    InvalidException { text: String },
}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(error: fancy_regex::Error) -> Self {
        Self::Regex(Box::new(error))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Piece {
    text: String,
    norm: Option<String>,
    has_space: bool,
}

impl Piece {
    fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            norm: None,
            has_space: false,
        }
    }
}

impl RegexTokenizer {
    /// Build a tokenizer from an exported JSON configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError`] if JSON decoding, format validation, regex
    /// compilation, or exception validation fails.
    pub fn from_json(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let config: RegexTokenizerConfig = serde_json::from_slice(bytes)?;
        Self::from_config(config)
    }

    /// Build a tokenizer from a decoded configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError`] for unsupported formats, invalid regular
    /// expressions, or malformed special cases.
    pub fn from_config(config: RegexTokenizerConfig) -> Result<Self, TokenizerError> {
        if config.format_version != CURRENT_FORMAT_VERSION {
            return Err(TokenizerError::UnsupportedFormat {
                actual: config.format_version,
                supported: CURRENT_FORMAT_VERSION,
            });
        }
        for (text, exception) in &config.exceptions {
            let reconstructed: String = exception.iter().map(|token| token.orth.as_str()).collect();
            if reconstructed != *text {
                return Err(TokenizerError::InvalidException { text: text.clone() });
            }
        }

        let max_exception_chars = config
            .exceptions
            .keys()
            .map(|text| text.chars().count())
            .max()
            .unwrap_or(0);
        Ok(Self {
            language: config.language,
            prefix: compile("prefix", config.prefix)?,
            suffix: compile("suffix", config.suffix)?,
            infix: compile("infix", config.infix)?,
            token_match: compile("token_match", config.token_match)?,
            url_match: compile("url_match", config.url_match)?,
            exceptions: config.exceptions,
            norm_overrides: config.norm_overrides,
            max_exception_chars,
        })
    }

    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Tokenize text into the core owned document representation.
    ///
    /// # Errors
    ///
    /// Returns [`TokenizerError::Regex`] if a backtracking regex fails during
    /// execution.
    pub fn tokenize(&self, text: &str) -> Result<Doc, TokenizerError> {
        if text.is_empty() {
            return Ok(Doc::default());
        }

        let chars: Vec<char> = text.chars().collect();
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut in_whitespace = chars[0].is_whitespace();

        for (index, character) in chars.iter().copied().enumerate() {
            if character.is_whitespace() != in_whitespace {
                if start < index {
                    let span: String = chars[start..index].iter().collect();
                    self.tokenize_span(&span, &mut pieces)?;
                }
                if character == ' ' {
                    if let Some(last) = pieces.last_mut() {
                        last.has_space = true;
                    }
                    start = index + 1;
                } else {
                    start = index;
                }
                in_whitespace = !in_whitespace;
            }
        }

        if start < chars.len() {
            let span: String = chars[start..].iter().collect();
            self.tokenize_span(&span, &mut pieces)?;
            if chars.last() == Some(&' ') && !in_whitespace {
                if let Some(last) = pieces.last_mut() {
                    last.has_space = true;
                }
            }
        }

        self.apply_phrase_exceptions(&mut pieces);
        Ok(pieces_to_doc(pieces, &self.norm_overrides))
    }

    fn tokenize_span(&self, span: &str, output: &mut Vec<Piece>) -> Result<(), TokenizerError> {
        if let Some(exception) = self.exceptions.get(span) {
            append_exception(output, exception);
            return Ok(());
        }

        let (core, prefixes, suffixes) = self.split_affixes(span)?;
        output.extend(prefixes);
        self.attach_core(&core, output)?;
        output.extend(suffixes.into_iter().rev());
        Ok(())
    }

    fn split_affixes(
        &self,
        input: &str,
    ) -> Result<(String, Vec<Piece>, Vec<Piece>), TokenizerError> {
        let mut current = input.to_owned();
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        let mut last_size = usize::MAX;

        while !current.is_empty() && current.len() != last_size {
            if self.matches_token(&current)? || self.exceptions.contains_key(&current) {
                break;
            }
            last_size = current.len();

            let prefix_len = match_len(self.prefix.as_ref(), &current)?;
            let prefix = current[..prefix_len].to_owned();
            let minus_prefix = current[prefix_len..].to_owned();
            if prefix_len > 0
                && !minus_prefix.is_empty()
                && self.exceptions.contains_key(&minus_prefix)
            {
                current = minus_prefix;
                prefixes.push(Piece::plain(prefix));
                break;
            }

            let suffix_len = match_len(self.suffix.as_ref(), &current[prefix_len..])?;
            let suffix_start = current.len().saturating_sub(suffix_len);
            let suffix = current[suffix_start..].to_owned();
            let minus_suffix = current[..suffix_start].to_owned();
            if suffix_len > 0
                && !minus_suffix.is_empty()
                && self.exceptions.contains_key(&minus_suffix)
            {
                current = minus_suffix;
                suffixes.push(Piece::plain(suffix));
                break;
            }

            if prefix_len > 0 && suffix_len > 0 && prefix_len + suffix_len <= current.len() {
                current = current[prefix_len..suffix_start].to_owned();
                prefixes.push(Piece::plain(prefix));
                suffixes.push(Piece::plain(suffix));
            } else if prefix_len > 0 {
                current = minus_prefix;
                prefixes.push(Piece::plain(prefix));
            } else if suffix_len > 0 {
                current = minus_suffix;
                suffixes.push(Piece::plain(suffix));
            }
        }
        Ok((current, prefixes, suffixes))
    }

    fn attach_core(&self, core: &str, output: &mut Vec<Piece>) -> Result<(), TokenizerError> {
        if core.is_empty() {
            return Ok(());
        }
        if let Some(exception) = self.exceptions.get(core) {
            append_exception(output, exception);
            return Ok(());
        }
        if self.matches_token(core)? || is_match(self.url_match.as_ref(), core)? {
            output.push(Piece::plain(core));
            return Ok(());
        }

        let Some(infix) = &self.infix else {
            output.push(Piece::plain(core));
            return Ok(());
        };
        let matches = infix.find_iter(core).collect::<Result<Vec<_>, _>>()?;
        if matches.is_empty() {
            output.push(Piece::plain(core));
            return Ok(());
        }

        let mut start = 0;
        for matched in matches {
            if matched.start() == 0 {
                continue;
            }
            if matched.start() != start {
                output.push(Piece::plain(&core[start..matched.start()]));
            }
            if matched.start() != matched.end() {
                output.push(Piece::plain(&core[matched.start()..matched.end()]));
            }
            start = matched.end();
        }
        if start < core.len() {
            output.push(Piece::plain(&core[start..]));
        }
        Ok(())
    }

    fn matches_token(&self, text: &str) -> Result<bool, TokenizerError> {
        is_match(self.token_match.as_ref(), text)
    }

    fn apply_phrase_exceptions(&self, pieces: &mut Vec<Piece>) {
        if self.max_exception_chars == 0 || pieces.is_empty() {
            return;
        }

        let original = std::mem::take(pieces);
        let mut result = Vec::with_capacity(original.len());
        let mut start = 0;
        while start < original.len() {
            let mut surface = String::new();
            let mut best: Option<(usize, &Vec<ExceptionToken>)> = None;
            for end in start..original.len() {
                if end > start && original[end - 1].has_space {
                    surface.push(' ');
                }
                surface.push_str(&original[end].text);
                if surface.chars().count() > self.max_exception_chars {
                    break;
                }
                if let Some(exception) = self.exceptions.get(&surface) {
                    best = Some((end + 1, exception));
                }
            }

            if let Some((end, exception)) = best {
                let final_space = original[end - 1].has_space;
                let result_start = result.len();
                append_exception(&mut result, exception);
                if let Some(last) = result.last_mut() {
                    last.has_space = final_space;
                }
                debug_assert!(result.len() > result_start);
                start = end;
            } else {
                result.push(original[start].clone());
                start += 1;
            }
        }
        *pieces = result;
    }
}

fn compile(name: &'static str, pattern: Option<String>) -> Result<Option<Regex>, TokenizerError> {
    pattern
        .map(|pattern| {
            Regex::new(&pattern).map_err(|source| TokenizerError::RegexCompile {
                name,
                source: Box::new(source),
            })
        })
        .transpose()
}

fn match_len(regex: Option<&Regex>, text: &str) -> Result<usize, TokenizerError> {
    let Some(regex) = regex else {
        return Ok(0);
    };
    let matched = regex.find(text).map_err(TokenizerError::from)?;
    Ok(matched.map_or(0, |matched| matched.end() - matched.start()))
}

fn is_match(regex: Option<&Regex>, text: &str) -> Result<bool, TokenizerError> {
    let Some(regex) = regex else {
        return Ok(false);
    };
    regex.is_match(text).map_err(TokenizerError::from)
}

fn append_exception(output: &mut Vec<Piece>, exception: &[ExceptionToken]) {
    output.extend(exception.iter().map(|token| Piece {
        text: token.orth.clone(),
        norm: token.norm.clone(),
        has_space: false,
    }));
}

fn pieces_to_doc(pieces: Vec<Piece>, norm_overrides: &BTreeMap<u64, String>) -> Doc {
    let mut offset = 0;
    let mut tokens = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let mut token = TokenData::new(&piece.text, piece.has_space, offset);
        let norm = piece.norm.unwrap_or_else(|| {
            norm_overrides
                .get(&token.orth)
                .cloned()
                .unwrap_or_else(|| piece.text.to_lowercase())
        });
        token.norm = StringStore::id(&norm);
        offset += piece.text.chars().count() + usize::from(piece.has_space);
        tokens.push(token);
    }
    Doc::new(tokens)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{ExceptionToken, RegexTokenizer, RegexTokenizerConfig};

    fn tokenizer() -> RegexTokenizer {
        let mut exceptions = BTreeMap::new();
        exceptions.insert(
            "can't".to_owned(),
            vec![
                ExceptionToken {
                    orth: "ca".to_owned(),
                    norm: Some("can".to_owned()),
                },
                ExceptionToken {
                    orth: "n't".to_owned(),
                    norm: Some("not".to_owned()),
                },
            ],
        );
        RegexTokenizer::from_config(RegexTokenizerConfig {
            format_version: 1,
            language: "en".to_owned(),
            prefix: Some(r#"^[\(\"]"#.to_owned()),
            suffix: Some(r#"[\.\!\?\)\"]$"#.to_owned()),
            infix: Some(r"(?<=[A-Za-z])-(?=[A-Za-z])".to_owned()),
            token_match: None,
            url_match: Some(r"^https?://[^ ]+$".to_owned()),
            exceptions,
            norm_overrides: BTreeMap::new(),
        })
        .unwrap()
    }

    #[test]
    fn handles_affixes_infixes_exceptions_and_spaces() {
        let doc = tokenizer()
            .tokenize(r#""I can't use state-of-the-art.""#)
            .unwrap();
        let tokens: Vec<_> = doc
            .tokens()
            .iter()
            .map(|token| token.text.as_ref())
            .collect();
        assert_eq!(
            tokens,
            [
                "\"", "I", "ca", "n't", "use", "state", "-", "of", "-", "the", "-", "art", ".",
                "\""
            ]
        );
        assert_eq!(doc.text(), r#""I can't use state-of-the-art.""#);
    }

    #[test]
    fn preserves_nonstandard_whitespace_as_tokens() {
        let doc = tokenizer().tokenize("a  b\nc").unwrap();
        let tokens: Vec<_> = doc
            .tokens()
            .iter()
            .map(|token| token.text.as_ref())
            .collect();
        assert_eq!(tokens, ["a", " ", "b", "\n", "c"]);
        assert_eq!(doc.text(), "a  b\nc");
    }
}
