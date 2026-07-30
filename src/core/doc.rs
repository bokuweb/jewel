use std::ops::Range;

use thiserror::Error;

use crate::{StringId, StringStore};

/// Token-boundary alignment used by spaCy's `Doc.char_span`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CharSpanAlignment {
    /// Require both character offsets to match token boundaries exactly.
    #[default]
    Strict,
    /// Keep only tokens completely contained by the character range.
    Contract,
    /// Include every token touched by the character range.
    Expand,
}

/// Owned per-token state. String-valued annotations use spaCy-compatible IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenData {
    pub orth: StringId,
    pub text: Box<str>,
    pub has_space: bool,
    pub idx: usize,
    pub lemma: StringId,
    pub norm: StringId,
    pub pos: u64,
    pub tag: StringId,
    pub dep: StringId,
    pub head: i32,
    pub sent_start: i8,
    pub ent_iob: u8,
    pub ent_type: StringId,
    pub ent_id: StringId,
    pub ent_kb_id: StringId,
    pub morph: StringId,
}

impl TokenData {
    #[must_use]
    pub fn new(text: &str, has_space: bool, idx: usize) -> Self {
        Self {
            orth: StringStore::id(text),
            text: text.into(),
            has_space,
            idx,
            lemma: 0,
            norm: 0,
            pos: 0,
            tag: 0,
            dep: 0,
            head: 0,
            sent_start: 0,
            ent_iob: 0,
            ent_type: 0,
            ent_id: 0,
            ent_kb_id: 0,
            morph: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Doc {
    tokens: Vec<TokenData>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DocError {
    #[error("words and spaces must have equal lengths: words={words}, spaces={spaces}")]
    MismatchedWordAndSpaceCounts { words: usize, spaces: usize },
    #[error("token index {index} is outside document length {len}")]
    TokenOutOfBounds { index: usize, len: usize },
    #[error("invalid span {start}..{end} for document length {len}")]
    InvalidSpan {
        start: usize,
        end: usize,
        len: usize,
    },
}

impl Doc {
    #[must_use]
    pub fn new(tokens: Vec<TokenData>) -> Self {
        Self { tokens }
    }

    /// Construct a document using spaCy's `Doc(vocab, words=..., spaces=...)`
    /// convention. Offsets are Unicode code-point offsets, matching Python
    /// string indexing and `Token.idx`.
    ///
    /// # Errors
    ///
    /// Returns [`DocError::MismatchedWordAndSpaceCounts`] if `words` and
    /// `spaces` have different lengths.
    pub fn from_words<S: AsRef<str>>(words: &[S], spaces: &[bool]) -> Result<Self, DocError> {
        if words.len() != spaces.len() {
            return Err(DocError::MismatchedWordAndSpaceCounts {
                words: words.len(),
                spaces: spaces.len(),
            });
        }

        let mut idx = 0;
        let mut tokens = Vec::with_capacity(words.len());
        for (word, has_space) in words.iter().zip(spaces) {
            let text = word.as_ref();
            tokens.push(TokenData::new(text, *has_space, idx));
            idx += text.chars().count() + usize::from(*has_space);
        }
        Ok(Self { tokens })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    #[must_use]
    pub fn text(&self) -> String {
        let capacity = self
            .tokens
            .iter()
            .map(|token| token.text.len() + usize::from(token.has_space))
            .sum();
        let mut text = String::with_capacity(capacity);
        for token in &self.tokens {
            text.push_str(&token.text);
            if token.has_space {
                text.push(' ');
            }
        }
        text
    }

    /// Borrow a token view.
    ///
    /// # Errors
    ///
    /// Returns [`DocError::TokenOutOfBounds`] if `index` is outside the
    /// document.
    pub fn token(&self, index: usize) -> Result<Token<'_>, DocError> {
        if index >= self.len() {
            return Err(DocError::TokenOutOfBounds {
                index,
                len: self.len(),
            });
        }
        Ok(Token { doc: self, index })
    }

    /// Borrow a token span.
    ///
    /// # Errors
    ///
    /// Returns [`DocError::InvalidSpan`] if the range is reversed or outside
    /// the document.
    pub fn span(&self, range: Range<usize>) -> Result<Span<'_>, DocError> {
        if range.start > range.end || range.end > self.len() {
            return Err(DocError::InvalidSpan {
                start: range.start,
                end: range.end,
                len: self.len(),
            });
        }
        Ok(Span {
            doc: self,
            start: range.start,
            end: range.end,
        })
    }

    /// Return the token span aligned to a Unicode character range.
    ///
    /// Offsets use Python/spaCy Unicode code-point indexing. `Strict` returns
    /// `None` unless both offsets are token boundaries, `Contract` drops
    /// partially covered boundary tokens, and `Expand` includes touched
    /// boundary tokens. Ranges outside the document return `None`.
    #[must_use]
    pub fn char_span(&self, range: Range<usize>, alignment: CharSpanAlignment) -> Option<Span<'_>> {
        let document_end = self.char_len();
        if self.is_empty() || range.start > range.end || range.end > document_end {
            return None;
        }

        let token_end =
            |index: usize| self.tokens[index].idx + self.tokens[index].text.chars().count();
        let token_range = match alignment {
            CharSpanAlignment::Strict => {
                if range.start == range.end {
                    return None;
                }
                let start = self
                    .tokens
                    .iter()
                    .position(|token| token.idx == range.start)?;
                let end = (start..self.len()).find(|index| token_end(*index) == range.end)? + 1;
                (start < end).then_some(start..end)?
            }
            CharSpanAlignment::Contract => {
                if range.start == range.end {
                    return None;
                }
                let start = self.tokens.iter().enumerate().position(|(index, token)| {
                    token.idx >= range.start && token_end(index) <= range.end
                })?;
                let end = self.tokens.iter().enumerate().rposition(|(index, token)| {
                    token.idx >= range.start && token_end(index) <= range.end
                })? + 1;
                (start < end).then_some(start..end)?
            }
            CharSpanAlignment::Expand => {
                let mut first = None;
                let mut last = None;
                for (index, token) in self.tokens.iter().enumerate() {
                    let overlaps = if range.start == range.end {
                        token.idx < range.start && token_end(index) > range.start
                    } else {
                        token_end(index) > range.start && token.idx < range.end
                    };
                    if overlaps {
                        first.get_or_insert(index);
                        last = Some(index);
                    }
                }
                if let (Some(start), Some(end)) = (first, last) {
                    start..end + 1
                } else {
                    let anchor = self
                        .tokens
                        .iter()
                        .position(|token| token.idx >= range.end)
                        .unwrap_or(self.len());
                    let anchor_char = self.span_char(anchor);
                    let previous_end = anchor.checked_sub(1).map_or(0, token_end);
                    let outside_gap = range.start < previous_end || range.end > anchor_char;
                    let empty_at_document_edge =
                        anchor == self.len() && range.start == document_end;
                    if (anchor == 0 && range.end == 0) || outside_gap || empty_at_document_edge {
                        return None;
                    }
                    anchor..anchor
                }
            }
        };
        Some(Span {
            doc: self,
            start: token_range.start,
            end: token_range.end,
        })
    }

    #[must_use]
    pub fn tokens(&self) -> &[TokenData] {
        &self.tokens
    }

    #[must_use]
    pub fn tokens_mut(&mut self) -> &mut [TokenData] {
        &mut self.tokens
    }

    fn char_len(&self) -> usize {
        self.tokens.last().map_or(0, |token| {
            token.idx + token.text.chars().count() + usize::from(token.has_space)
        })
    }

    fn span_char(&self, token: usize) -> usize {
        self.tokens
            .get(token)
            .map_or_else(|| self.char_len(), |token| token.idx)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Token<'doc> {
    doc: &'doc Doc,
    index: usize,
}

impl<'doc> Token<'doc> {
    #[must_use]
    pub fn index(self) -> usize {
        self.index
    }

    #[must_use]
    pub fn data(self) -> &'doc TokenData {
        &self.doc.tokens[self.index]
    }

    #[must_use]
    pub fn text(self) -> &'doc str {
        &self.data().text
    }

    #[must_use]
    pub fn whitespace(self) -> &'static str {
        if self.data().has_space {
            " "
        } else {
            ""
        }
    }

    #[must_use]
    pub fn idx(self) -> usize {
        self.data().idx
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Span<'doc> {
    doc: &'doc Doc,
    start: usize,
    end: usize,
}

impl Span<'_> {
    #[must_use]
    pub fn start(self) -> usize {
        self.start
    }

    #[must_use]
    pub fn end(self) -> usize {
        self.end
    }

    #[must_use]
    pub fn len(self) -> usize {
        self.end - self.start
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    #[must_use]
    pub fn start_char(self) -> usize {
        self.doc.span_char(self.start)
    }

    #[must_use]
    pub fn end_char(self) -> usize {
        if self.is_empty() {
            self.start_char()
        } else {
            let token = &self.doc.tokens[self.end - 1];
            token.idx + token.text.chars().count()
        }
    }

    #[must_use]
    pub fn text(self) -> String {
        let mut text = String::new();
        for (offset, token) in self.doc.tokens[self.start..self.end].iter().enumerate() {
            text.push_str(&token.text);
            if token.has_space && offset + 1 < self.len() {
                text.push(' ');
            }
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::{CharSpanAlignment, Doc};

    #[test]
    fn reconstructs_text_and_unicode_offsets() {
        let doc = Doc::from_words(&["Rust", "で", "解析"], &[true, false, false]).unwrap();
        assert_eq!(doc.text(), "Rust で解析");
        assert_eq!(doc.token(0).unwrap().idx(), 0);
        assert_eq!(doc.token(1).unwrap().idx(), 5);
        assert_eq!(doc.token(2).unwrap().idx(), 6);
        assert_eq!(doc.span(1..3).unwrap().text(), "で解析");
    }

    #[test]
    fn char_spans_match_spacy_3_8_alignment_modes() {
        let doc = Doc::from_words(&["Tokyo", "Shibuya"], &[true, false]).unwrap();

        let exact = doc.char_span(0..5, CharSpanAlignment::Strict).unwrap();
        assert_eq!(
            (exact.start(), exact.end(), exact.text()),
            (0, 1, "Tokyo".to_owned())
        );
        assert_eq!((exact.start_char(), exact.end_char()), (0, 5));

        assert!(doc.char_span(1..5, CharSpanAlignment::Strict).is_none());
        assert!(doc.char_span(1..5, CharSpanAlignment::Contract).is_none());
        let expanded = doc.char_span(1..5, CharSpanAlignment::Expand).unwrap();
        assert_eq!((expanded.start(), expanded.end()), (0, 1));

        let contracted = doc.char_span(0..6, CharSpanAlignment::Contract).unwrap();
        assert_eq!((contracted.start(), contracted.end()), (0, 1));

        let whitespace = doc.char_span(5..6, CharSpanAlignment::Expand).unwrap();
        assert!(whitespace.is_empty());
        assert_eq!((whitespace.start(), whitespace.start_char()), (1, 6));
        assert!(doc.char_span(13..13, CharSpanAlignment::Expand).is_none());
    }

    #[test]
    fn char_spans_use_unicode_code_point_offsets() {
        let doc = Doc::from_words(&["東京", "都"], &[false, false]).unwrap();
        let span = doc.char_span(2..3, CharSpanAlignment::Strict).unwrap();
        assert_eq!(
            (span.start(), span.end(), span.text()),
            (1, 2, "都".to_owned())
        );
        assert_eq!((span.start_char(), span.end_char()), (2, 3));
        assert!(doc.char_span(0..4, CharSpanAlignment::Expand).is_none());
    }

    #[test]
    fn expanded_char_spans_match_spacy_trailing_whitespace_behavior() {
        let doc = Doc::from_words(&["A"], &[true]).unwrap();
        for range in [1..1, 1..2] {
            let span = doc.char_span(range, CharSpanAlignment::Expand).unwrap();
            assert!(span.is_empty());
            assert_eq!((span.start(), span.start_char()), (1, 2));
        }
        assert!(doc.char_span(2..2, CharSpanAlignment::Expand).is_none());
    }
}
