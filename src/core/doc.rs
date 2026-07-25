use std::ops::Range;

use thiserror::Error;

use crate::{StringId, StringStore};

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

    #[must_use]
    pub fn tokens(&self) -> &[TokenData] {
        &self.tokens
    }

    #[must_use]
    pub fn tokens_mut(&mut self) -> &mut [TokenData] {
        &mut self.tokens
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
    use super::Doc;

    #[test]
    fn reconstructs_text_and_unicode_offsets() {
        let doc = Doc::from_words(&["Rust", "で", "解析"], &[true, false, false]).unwrap();
        assert_eq!(doc.text(), "Rust で解析");
        assert_eq!(doc.token(0).unwrap().idx(), 0);
        assert_eq!(doc.token(1).unwrap().idx(), 5);
        assert_eq!(doc.token(2).unwrap().idx(), 6);
        assert_eq!(doc.span(1..3).unwrap().text(), "で解析");
    }
}
