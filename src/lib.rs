//! A spaCy-compatible inference runtime with no Python runtime dependency.
//!
//! This crate provides Japanese and English tokenization, tagging, dependency
//! parsing, named-entity recognition, and loading for model bundles exported
//! from spaCy.

extern crate self as spacy_core;
extern crate self as spacy_model;
extern crate self as spacy_tokenizer;

mod core;
mod format;
mod model_bundle;
mod nn;
mod tokenizer;

pub use core::*;
pub use format::*;
pub use model_bundle::*;
pub use nn::*;
pub use tokenizer::*;
