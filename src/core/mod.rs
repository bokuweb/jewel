//! Core data structures shared by the Python-free spaCy runtime.

mod doc;
mod hash;
mod strings;

pub use doc::{Doc, DocError, Span, Token, TokenData};
pub use hash::{hash_bytes, hash_string, SPACY_STRING_SEED};
pub use strings::{StringId, StringStore, StringStoreError};

/// Stable spaCy attribute identifiers used by `Doc.to_array` and `DocBin`.
///
/// These values come from spaCy's `attr_id_t` enum. The remaining identifiers
/// will be generated from the upstream symbol table as format coverage grows.
pub mod attrs {
    pub const ID: u64 = 64;
    pub const ORTH: u64 = 65;
    pub const LOWER: u64 = 66;
    pub const NORM: u64 = 67;
    pub const SHAPE: u64 = 68;
    pub const PREFIX: u64 = 69;
    pub const SUFFIX: u64 = 70;
    pub const LENGTH: u64 = 71;
    pub const CLUSTER: u64 = 72;
    pub const LEMMA: u64 = 73;
    pub const POS: u64 = 74;
    pub const TAG: u64 = 75;
    pub const DEP: u64 = 76;
    pub const ENT_IOB: u64 = 77;
    pub const ENT_TYPE: u64 = 78;
    pub const HEAD: u64 = 79;
    pub const SENT_START: u64 = 80;
    pub const SPACY: u64 = 81;
    pub const PROB: u64 = 82;
    pub const LANG: u64 = 83;
}
