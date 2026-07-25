use std::collections::HashMap;

use thiserror::Error;

use crate::hash_string;

pub type StringId = u64;

/// A bidirectional store for stable spaCy string IDs.
///
/// Text observed while processing a document does not have to be permanently
/// interned here. This keeps long-running services from growing the shared
/// vocabulary for every request.
#[derive(Clone, Debug, Default)]
pub struct StringStore {
    by_id: HashMap<StringId, Box<str>>,
    insertion_order: Vec<StringId>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StringStoreError {
    #[error(
        "spaCy string hash collision for id {id}: existing={existing:?}, incoming={incoming:?}"
    )]
    HashCollision {
        id: StringId,
        existing: String,
        incoming: String,
    },
}

impl StringStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.insertion_order.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.insertion_order.is_empty()
    }

    /// Return spaCy's ID for a string without retaining the string.
    #[must_use]
    pub fn id(text: &str) -> StringId {
        if text.is_empty() {
            0
        } else if let Some(id) = builtin_id(text) {
            id
        } else {
            hash_string(text)
        }
    }

    /// Intern a string and return its stable ID.
    ///
    /// # Errors
    ///
    /// Returns [`StringStoreError::HashCollision`] if the ID is already
    /// associated with different text.
    pub fn add(&mut self, text: &str) -> Result<StringId, StringStoreError> {
        if text.is_empty() {
            return Ok(0);
        }

        let id = Self::id(text);
        if let Some(existing) = self.by_id.get(&id) {
            if existing.as_ref() == text {
                return Ok(id);
            }
            return Err(StringStoreError::HashCollision {
                id,
                existing: existing.to_string(),
                incoming: text.to_owned(),
            });
        }

        self.by_id.insert(id, text.into());
        self.insertion_order.push(id);
        Ok(id)
    }

    #[must_use]
    pub fn get(&self, id: StringId) -> Option<&str> {
        if id == 0 {
            Some("")
        } else {
            self.by_id.get(&id).map(AsRef::as_ref)
        }
    }

    #[must_use]
    pub fn contains_id(&self, id: StringId) -> bool {
        id == 0 || self.by_id.contains_key(&id)
    }

    /// Iterate over interned strings in insertion order.
    ///
    /// # Panics
    ///
    /// Panics only if the store's private map and insertion order are
    /// inconsistent, which indicates an internal implementation defect.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (StringId, &str)> {
        self.insertion_order.iter().map(|id| {
            let text = self
                .by_id
                .get(id)
                .expect("insertion order and string map must stay consistent");
            (*id, text.as_ref())
        })
    }
}

#[allow(clippy::too_many_lines)] // Kept together to make the versioned spaCy symbol table auditable.
fn builtin_id(text: &str) -> Option<u64> {
    const FLAGS: std::ops::RangeInclusive<u64> = 19..=63;
    const DEPRECATED: std::ops::RangeInclusive<u64> = 104..=379;
    const LOW_NAMES: &[&str] = &[
        "IS_ALPHA",
        "IS_ASCII",
        "IS_DIGIT",
        "IS_LOWER",
        "IS_PUNCT",
        "IS_SPACE",
        "IS_TITLE",
        "IS_UPPER",
        "LIKE_URL",
        "LIKE_NUM",
        "LIKE_EMAIL",
        "IS_STOP",
        "IS_OOV_DEPRECATED",
        "IS_BRACKET",
        "IS_QUOTE",
        "IS_LEFT_PUNCT",
        "IS_RIGHT_PUNCT",
        "IS_CURRENCY",
    ];
    const CORE_NAMES: &[&str] = &[
        "ID",
        "ORTH",
        "LOWER",
        "NORM",
        "SHAPE",
        "PREFIX",
        "SUFFIX",
        "LENGTH",
        "CLUSTER",
        "LEMMA",
        "POS",
        "TAG",
        "DEP",
        "ENT_IOB",
        "ENT_TYPE",
        "HEAD",
        "SENT_START",
        "SPACY",
        "PROB",
        "LANG",
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CONJ",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
        "EOL",
        "SPACE",
    ];
    const HIGH_NAMES: &[&str] = &[
        "PERSON",
        "NORP",
        "FACILITY",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LANGUAGE",
        "LAW",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
        "acomp",
        "advcl",
        "advmod",
        "agent",
        "amod",
        "appos",
        "attr",
        "aux",
        "auxpass",
        "cc",
        "ccomp",
        "complm",
        "conj",
        "cop",
        "csubj",
        "csubjpass",
        "dep",
        "det",
        "dobj",
        "expl",
        "hmod",
        "hyph",
        "infmod",
        "intj",
        "iobj",
        "mark",
        "meta",
        "neg",
        "nmod",
        "nn",
        "npadvmod",
        "nsubj",
        "nsubjpass",
        "num",
        "number",
        "oprd",
        "obj",
        "obl",
        "parataxis",
        "partmod",
        "pcomp",
        "pobj",
        "poss",
        "possessive",
        "preconj",
        "prep",
        "prt",
        "punct",
        "quantmod",
        "relcl",
        "rcmod",
        "root",
        "xcomp",
        "acl",
        "ENT_KB_ID",
        "MORPH",
        "ENT_ID",
        "IDX",
        "_",
    ];

    if let Some(index) = LOW_NAMES.iter().position(|name| *name == text) {
        return u64::try_from(index).ok().map(|index| index + 1);
    }
    if let Some(number) = text.strip_prefix("FLAG").and_then(parse_canonical_u64) {
        if FLAGS.contains(&number) {
            return Some(number);
        }
    }
    if let Some(index) = CORE_NAMES.iter().position(|name| *name == text) {
        return u64::try_from(index).ok().map(|index| index + 64);
    }
    if let Some(number) = text
        .strip_prefix("DEPRECATED")
        .filter(|number| number.len() == 3)
        .and_then(|number| number.parse::<u64>().ok())
    {
        let id = number + 103;
        if number > 0 && DEPRECATED.contains(&id) {
            return Some(id);
        }
    }
    HIGH_NAMES
        .iter()
        .position(|name| *name == text)
        .and_then(|index| u64::try_from(index).ok())
        .map(|index| index + 380)
}

fn parse_canonical_u64(text: &str) -> Option<u64> {
    if text.is_empty() || (text.len() > 1 && text.starts_with('0')) {
        None
    } else {
        text.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::StringStore;

    #[test]
    fn empty_string_uses_spacy_reserved_zero_id() {
        let mut store = StringStore::new();
        assert_eq!(store.add("").unwrap(), 0);
        assert_eq!(store.get(0), Some(""));
        assert!(store.is_empty());
    }

    #[test]
    fn adding_a_string_twice_does_not_duplicate_it() {
        let mut store = StringStore::new();
        let first = store.add("pipeline").unwrap();
        let second = store.add("pipeline").unwrap();
        assert_eq!(first, second);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn reserved_symbol_names_use_spacy_ids() {
        assert_eq!(StringStore::id("X"), 101);
        assert_eq!(StringStore::id("dep"), 414);
        assert_eq!(StringStore::id("_"), 456);
        assert_eq!(StringStore::id("FLAG42"), 42);
        assert_eq!(StringStore::id("DEPRECATED276"), 379);
    }
}
