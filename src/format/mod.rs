//! Compatibility readers and writers for serialized spaCy data.

mod docbin;

pub use docbin::{DocBin, DocBinError, DocRecord};
