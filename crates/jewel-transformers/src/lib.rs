//! Transformer model contracts and exported-asset loading for Jewel.
//!
//! Backends implement [`TransformerEncoder`] and return one pooled contextual
//! vector per Jewel token. `jewel-ginza` owns the downstream parser and NER
//! execution, so encoder implementations do not depend on GiNZA.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use jewel_core::{Bundle, Doc, Matrix};
use serde::{Deserialize, Serialize};
use sudachi::analysis::stateless_tokenizer::StatelessTokenizer;
use sudachi::analysis::{Mode, Tokenize};
use sudachi::config::ConfigBuilder;
use sudachi::dic::dictionary::JapaneseDictionary;
use thiserror::Error;

/// Tokenization settings used by GiNZA's SudachiTra WordPiece tokenizer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransformerTokenizerSpec {
    pub kind: String,
    pub split_mode: String,
    pub word_form_type: String,
    pub do_lower_case: bool,
    pub do_nfkc: bool,
    pub cls_token: String,
    pub sep_token: String,
    pub unk_token: String,
    pub pad_token: String,
}

/// Static transformer configuration exported with a model bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransformerSpec {
    pub architecture: String,
    pub model: String,
    pub hidden_width: usize,
    pub window: usize,
    pub stride: usize,
    pub max_wordpieces: usize,
    pub config_path: String,
    pub weights_path: String,
    pub vocab_path: String,
    pub tokenizer: TransformerTokenizerSpec,
}

impl TransformerSpec {
    /// Load the single transformer component declared by a Jewel bundle.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, ambiguous, malformed, unsafe, or absent
    /// transformer assets.
    pub fn from_bundle(bundle: &Bundle) -> Result<Self, TransformerError> {
        let components = bundle
            .manifest()
            .pipeline
            .iter()
            .filter(|component| component.factory.contains("transformer"))
            .collect::<Vec<_>>();
        let component = match components.as_slice() {
            [] => return Err(TransformerError::MissingComponent),
            [component] => *component,
            _ => {
                return Err(TransformerError::MultipleComponents(
                    components
                        .iter()
                        .map(|component| component.name.clone())
                        .collect(),
                ))
            }
        };
        Self::from_settings(bundle.root(), &component.settings)
    }

    /// Parse transformer settings and resolve their bundle-relative assets.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed settings, invalid dimensions, unsafe
    /// paths, or missing files.
    pub fn from_settings(
        root: &Path,
        settings: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, TransformerError> {
        let value = serde_json::Value::Object(
            settings
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let spec: Self = serde_json::from_value(value)
            .map_err(|error| TransformerError::InvalidSettings(error.to_string()))?;
        spec.validate()?;
        for path in [&spec.config_path, &spec.weights_path, &spec.vocab_path] {
            let resolved = resolve_asset(root, path)?;
            if !resolved.is_file() {
                return Err(TransformerError::MissingAsset(resolved));
            }
        }
        Ok(spec)
    }

    /// Validate dimensions and tokenizer semantics used by the shared runtime.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported architectures or invalid windows.
    pub fn validate(&self) -> Result<(), TransformerError> {
        if self.architecture != "electra" {
            return Err(TransformerError::InvalidSpec(format!(
                "unsupported architecture {:?}",
                self.architecture
            )));
        }
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
        if self.window == 0 {
            return Err(TransformerError::InvalidSpec(
                "window must be greater than zero".to_owned(),
            ));
        }
        if self.stride == 0 || self.stride > self.window {
            return Err(TransformerError::InvalidSpec(
                "stride must be between one and the window size".to_owned(),
            ));
        }
        if self.max_wordpieces < 2 {
            return Err(TransformerError::InvalidSpec(
                "max wordpieces must include CLS and SEP".to_owned(),
            ));
        }
        if self.tokenizer.kind != "sudachitra_wordpiece"
            || self.tokenizer.split_mode != "A"
            || self.tokenizer.word_form_type != "dictionary_and_surface"
        {
            return Err(TransformerError::InvalidSpec(
                "GiNZA Electra requires SudachiTra WordPiece with split mode A \
                 and dictionary_and_surface forms"
                    .to_owned(),
            ));
        }
        for path in [&self.config_path, &self.weights_path, &self.vocab_path] {
            if !is_safe_relative_path(path) {
                return Err(TransformerError::UnsafeAssetPath((*path).to_owned()));
            }
        }
        Ok(())
    }
}

/// Failure produced by transformer asset loading or an inference backend.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TransformerError {
    #[error("bundle has no transformer component")]
    MissingComponent,
    #[error("bundle has multiple transformer components: {0:?}")]
    MultipleComponents(Vec<String>),
    #[error("transformer settings are invalid: {0}")]
    InvalidSettings(String),
    #[error("invalid transformer specification: {0}")]
    InvalidSpec(String),
    #[error("transformer asset path is unsafe: {0:?}")]
    UnsafeAssetPath(String),
    #[error("transformer asset is missing: {0}")]
    MissingAsset(PathBuf),
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
    #[error(
        "transformer returned vectors for {actual} documents; expected vectors for {expected}"
    )]
    InvalidBatchOutput { expected: usize, actual: usize },
}

/// Backend-neutral encoder producing one contextual vector per Jewel token.
pub trait TransformerEncoder: Send + Sync {
    /// Return the immutable model configuration used by this encoder.
    fn spec(&self) -> &TransformerSpec;

    /// Encode a tokenized document into one row per Jewel token.
    ///
    /// Implementations perform source-compatible subword tokenization,
    /// strided transformer inference, and mean pooling over aligned pieces.
    ///
    /// # Errors
    ///
    /// Returns an error when token alignment or backend inference fails.
    fn encode(&self, doc: &Doc) -> Result<Matrix, TransformerError>;

    /// Encode a document batch while preserving input order.
    ///
    /// The default implementation calls [`TransformerEncoder::encode`] for
    /// each document. Backends may override this method to batch execution
    /// across documents.
    ///
    /// # Errors
    ///
    /// Returns the first token alignment or backend inference error.
    fn encode_batch(&self, docs: &[Doc]) -> Result<Vec<Matrix>, TransformerError> {
        docs.iter().map(|doc| self.encode(doc)).collect()
    }
}

/// CPU-native Electra encoder for exported GiNZA model assets.
///
/// GiNZA 5.2's Electra checkpoint uses equal embedding and hidden widths, so
/// its discriminator encoder is structurally equivalent to the BERT encoder
/// implemented by Candle. The tokenizer remains GiNZA-specific.
pub struct CandleElectraEncoder {
    spec: TransformerSpec,
    model: BertModel,
    device: Device,
    wordpieces: WordpieceVocabulary,
    dictionary: Arc<JapaneseDictionary>,
    span_batch_size: usize,
}

impl CandleElectraEncoder {
    /// Default number of overlapping token spans evaluated in one forward pass.
    pub const DEFAULT_SPAN_BATCH_SIZE: usize = 8;

    /// Load exported safetensors, WordPiece vocabulary, and Sudachi assets.
    ///
    /// # Errors
    ///
    /// Returns an error when model configuration, tensors, vocabulary, or the
    /// exported Sudachi dictionary cannot be loaded.
    pub fn load(bundle: &Bundle) -> Result<Self, TransformerError> {
        Self::load_with_span_batch_size(bundle, Self::DEFAULT_SPAN_BATCH_SIZE)
    }

    /// Load an encoder with a caller-selected maximum span batch size.
    ///
    /// Smaller batches reduce peak activation memory for long documents.
    /// Larger batches can improve throughput when the backend has sufficient
    /// memory. The value must be greater than zero.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero batch size or when model configuration,
    /// tensors, vocabulary, or the exported Sudachi dictionary cannot be
    /// loaded.
    pub fn load_with_span_batch_size(
        bundle: &Bundle,
        span_batch_size: usize,
    ) -> Result<Self, TransformerError> {
        if span_batch_size == 0 {
            return Err(TransformerError::InvalidSpec(
                "Electra span batch size must be greater than zero".to_owned(),
            ));
        }
        let spec = TransformerSpec::from_bundle(bundle)?;
        let config_path = resolve_asset(bundle.root(), &spec.config_path)?;
        let config_bytes = std::fs::read(&config_path)
            .map_err(|error| backend_error("read transformer config", error))?;
        let raw_config: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|error| backend_error("parse transformer config", error))?;
        let embedding_size = raw_config
            .get("embedding_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                TransformerError::InvalidSettings(
                    "Electra config is missing embedding_size".to_owned(),
                )
            })?;
        if embedding_size != spec.hidden_width {
            return Err(TransformerError::InvalidSpec(format!(
                "Electra embedding width {embedding_size} differs from hidden width {}",
                spec.hidden_width
            )));
        }
        let config: BertConfig = serde_json::from_slice(&config_bytes)
            .map_err(|error| backend_error("parse Candle Electra config", error))?;
        if config.hidden_size != spec.hidden_width {
            return Err(TransformerError::InvalidSpec(format!(
                "model hidden width {} differs from exported width {}",
                config.hidden_size, spec.hidden_width
            )));
        }

        let device = Device::Cpu;
        let weights_path = resolve_asset(bundle.root(), &spec.weights_path)?;
        let weights = std::fs::read(&weights_path)
            .map_err(|error| backend_error("read transformer weights", error))?;
        let builder = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)
            .map_err(|error| backend_error("load transformer weights", error))?;
        let model = BertModel::load(builder, &config)
            .map_err(|error| backend_error("construct Electra encoder", error))?;
        let wordpieces = WordpieceVocabulary::load(
            &resolve_asset(bundle.root(), &spec.vocab_path)?,
            &spec.tokenizer,
        )?;
        let dictionary = load_sudachi_dictionary(bundle)?;

        Ok(Self {
            spec,
            model,
            device,
            wordpieces,
            dictionary,
            span_batch_size,
        })
    }

    /// Return the maximum number of overlapping spans run in one forward pass.
    #[must_use]
    pub const fn span_batch_size(&self) -> usize {
        self.span_batch_size
    }

    /// Return per-token WordPiece IDs before adding span-level CLS and SEP.
    ///
    /// This is intended for compatibility diagnostics and tokenizer parity
    /// tests; inference callers normally use [`TransformerEncoder::encode`].
    ///
    /// # Errors
    ///
    /// Returns an error when Sudachi analysis fails.
    pub fn token_wordpiece_ids(&self, doc: &Doc) -> Result<Vec<Vec<u32>>, TransformerError> {
        let tokenizer = StatelessTokenizer::new(self.dictionary.clone());
        self.document_pieces(&tokenizer, doc).map(|tokens| {
            tokens
                .into_iter()
                .map(|pieces| pieces.into_iter().map(|piece| piece.id).collect())
                .collect()
        })
    }

    fn prepare_span(
        &self,
        document: usize,
        token_pieces: &[Vec<Wordpiece>],
        start: usize,
        end: usize,
    ) -> Result<PreparedSpan, TransformerError> {
        let mut ids = Vec::new();
        ids.push(self.wordpieces.cls);
        let mut alignments = Vec::with_capacity(end - start);
        for pieces in &token_pieces[start..end] {
            let mut aligned = Vec::new();
            for piece in pieces {
                let index = ids.len();
                ids.push(piece.id);
                if piece.aligned {
                    aligned.push(index);
                }
            }
            alignments.push(aligned);
        }
        ids.push(self.wordpieces.sep);
        if ids.len() > self.spec.max_wordpieces {
            return Err(TransformerError::Backend(format!(
                "token span {start}..{end} produced {} wordpieces, exceeding model limit {}",
                ids.len(),
                self.spec.max_wordpieces
            )));
        }
        Ok(PreparedSpan {
            document,
            start,
            ids,
            alignments,
        })
    }

    fn encode_span_batch(
        &self,
        spans: &[PreparedSpan],
        pools: &mut [PooledDocument],
    ) -> Result<(), TransformerError> {
        let batch_size = spans.len();
        let sequence_length = spans.iter().map(|span| span.ids.len()).max().unwrap_or(0);
        if batch_size == 0 || sequence_length == 0 {
            return Ok(());
        }

        let mut input_ids = vec![self.wordpieces.pad; batch_size.saturating_mul(sequence_length)];
        let mut attention = vec![0_u32; batch_size.saturating_mul(sequence_length)];
        for (batch, span) in spans.iter().enumerate() {
            let offset = batch * sequence_length;
            input_ids[offset..offset + span.ids.len()].copy_from_slice(&span.ids);
            attention[offset..offset + span.ids.len()].fill(1);
        }
        let input_ids = Tensor::from_vec(input_ids, (batch_size, sequence_length), &self.device)
            .map_err(|error| backend_error("build Electra input IDs", error))?;
        let token_types = Tensor::zeros((batch_size, sequence_length), DType::U32, &self.device)
            .map_err(|error| backend_error("build Electra token types", error))?;
        let attention = Tensor::from_vec(attention, (batch_size, sequence_length), &self.device)
            .map_err(|error| backend_error("build Electra attention mask", error))?;
        let output = self
            .model
            .forward(&input_ids, &token_types, Some(&attention))
            .and_then(|tensor| tensor.to_vec3::<f32>())
            .map_err(|error| backend_error("run Electra inference", error))?;
        if output.len() != spans.len() {
            return Err(TransformerError::Backend(format!(
                "Electra returned {} span outputs for a batch of {}",
                output.len(),
                spans.len()
            )));
        }
        for (span, span_output) in spans.iter().zip(&output) {
            let pool = pools.get_mut(span.document).ok_or_else(|| {
                TransformerError::Backend(format!(
                    "prepared span references missing document {}",
                    span.document
                ))
            })?;
            pool_span_output(
                span,
                span_output,
                self.spec.hidden_width,
                &mut pool.sums,
                &mut pool.counts,
            )?;
        }
        Ok(())
    }

    fn document_pieces(
        &self,
        tokenizer: &StatelessTokenizer<Arc<JapaneseDictionary>>,
        doc: &Doc,
    ) -> Result<Vec<Vec<Wordpiece>>, TransformerError> {
        doc.tokens()
            .iter()
            .map(|token| self.token_pieces(tokenizer, &token.text))
            .collect()
    }

    fn token_pieces(
        &self,
        tokenizer: &StatelessTokenizer<Arc<JapaneseDictionary>>,
        text: &str,
    ) -> Result<Vec<Wordpiece>, TransformerError> {
        if text.chars().all(char::is_whitespace) {
            return Ok(Vec::new());
        }
        let morphemes = tokenizer
            .tokenize(text, Mode::A, false)
            .map_err(|error| backend_error("run SudachiTra word tokenization", error))?;
        let mut ids = Vec::new();
        for morpheme in morphemes.iter() {
            if morpheme.surface().chars().all(char::is_whitespace) {
                continue;
            }
            let conjugative = morpheme
                .part_of_speech()
                .first()
                .is_some_and(|part| matches!(part.as_str(), "動詞" | "形容詞" | "助動詞"));
            let pieces = if conjugative {
                self.wordpieces.encode_word(morpheme.surface().as_ref())
            } else {
                self.wordpieces.encode_word(morpheme.dictionary_form())
            };
            ids.extend(pieces);
        }
        Ok(ids)
    }

    fn encode_documents(&self, docs: &[Doc]) -> Result<Vec<Matrix>, TransformerError> {
        let tokenizer = StatelessTokenizer::new(self.dictionary.clone());
        let token_pieces = docs
            .iter()
            .map(|doc| self.document_pieces(&tokenizer, doc))
            .collect::<Result<Vec<_>, _>>()?;
        let lengths = docs.iter().map(Doc::len).collect::<Vec<_>>();
        let ranges = batch_span_ranges(&lengths, self.spec.window, self.spec.stride);
        let mut pools = lengths
            .iter()
            .map(|&length| PooledDocument::new(length, self.spec.hidden_width))
            .collect::<Vec<_>>();
        for range_batch in ranges.chunks(self.span_batch_size) {
            let spans = range_batch
                .iter()
                .map(|&(document, start, end)| {
                    self.prepare_span(document, &token_pieces[document], start, end)
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.encode_span_batch(&spans, &mut pools)?;
        }
        pools
            .into_iter()
            .map(|pool| pool.finish(self.spec.hidden_width))
            .collect()
    }
}

impl TransformerEncoder for CandleElectraEncoder {
    fn spec(&self) -> &TransformerSpec {
        &self.spec
    }

    fn encode(&self, doc: &Doc) -> Result<Matrix, TransformerError> {
        self.encode_documents(std::slice::from_ref(doc))?
            .pop()
            .ok_or_else(|| TransformerError::Backend("missing Electra document output".to_owned()))
    }

    fn encode_batch(&self, docs: &[Doc]) -> Result<Vec<Matrix>, TransformerError> {
        self.encode_documents(docs)
    }
}

struct PreparedSpan {
    document: usize,
    start: usize,
    ids: Vec<u32>,
    alignments: Vec<Vec<usize>>,
}

struct PooledDocument {
    sums: Vec<f32>,
    counts: Vec<usize>,
}

impl PooledDocument {
    fn new(tokens: usize, hidden_width: usize) -> Self {
        Self {
            sums: vec![0.0; tokens.saturating_mul(hidden_width)],
            counts: vec![0; tokens],
        }
    }

    fn finish(mut self, hidden_width: usize) -> Result<Matrix, TransformerError> {
        let rows = self.counts.len();
        for (token, count) in self.counts.into_iter().enumerate() {
            if count == 0 {
                continue;
            }
            let scale = 1.0 / count as f32;
            let offset = token * hidden_width;
            for value in &mut self.sums[offset..offset + hidden_width] {
                *value *= scale;
            }
        }
        Matrix::new(rows, hidden_width, self.sums)
            .map_err(|error| backend_error("construct pooled token vectors", error))
    }
}

fn span_ranges(length: usize, window: usize, stride: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < length {
        let end = start.saturating_add(window).min(length);
        ranges.push((start, end));
        if end == length {
            break;
        }
        start = start.saturating_add(stride);
    }
    ranges
}

fn batch_span_ranges(
    lengths: &[usize],
    window: usize,
    stride: usize,
) -> Vec<(usize, usize, usize)> {
    lengths
        .iter()
        .enumerate()
        .flat_map(|(document, &length)| {
            span_ranges(length, window, stride)
                .into_iter()
                .map(move |(start, end)| (document, start, end))
        })
        .collect()
}

fn pool_span_output(
    span: &PreparedSpan,
    output: &[Vec<f32>],
    hidden_width: usize,
    sums: &mut [f32],
    counts: &mut [usize],
) -> Result<(), TransformerError> {
    for (local_token, pieces) in span.alignments.iter().enumerate() {
        let token = span.start + local_token;
        for &piece in pieces {
            let row = output.get(piece).ok_or_else(|| {
                TransformerError::Backend(format!(
                    "Electra output is missing wordpiece row {piece}"
                ))
            })?;
            if row.len() != hidden_width {
                return Err(TransformerError::Backend(format!(
                    "Electra wordpiece row has width {}, expected {hidden_width}",
                    row.len()
                )));
            }
            let target = token * hidden_width;
            for (sum, value) in sums[target..target + hidden_width].iter_mut().zip(row) {
                *sum += *value;
            }
            counts[token] += 1;
        }
    }
    Ok(())
}

struct WordpieceVocabulary {
    pieces: HashMap<String, u32>,
    cls: u32,
    sep: u32,
    unknown: u32,
    pad: u32,
}

#[derive(Clone, Copy)]
struct Wordpiece {
    id: u32,
    aligned: bool,
}

impl WordpieceVocabulary {
    fn load(path: &Path, tokenizer: &TransformerTokenizerSpec) -> Result<Self, TransformerError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|error| backend_error("read WordPiece vocabulary", error))?;
        let pieces = contents
            .lines()
            .enumerate()
            .map(|(index, piece)| {
                u32::try_from(index)
                    .map(|index| (piece.to_owned(), index))
                    .map_err(|_| {
                        TransformerError::Backend("WordPiece vocabulary exceeds u32 IDs".to_owned())
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let token_id = |token: &str| {
            pieces.get(token).copied().ok_or_else(|| {
                TransformerError::InvalidSettings(format!(
                    "WordPiece vocabulary is missing special token {token:?}"
                ))
            })
        };
        Ok(Self {
            cls: token_id(&tokenizer.cls_token)?,
            sep: token_id(&tokenizer.sep_token)?,
            unknown: token_id(&tokenizer.unk_token)?,
            pad: token_id(&tokenizer.pad_token)?,
            pieces,
        })
    }

    fn encode_word(&self, word: &str) -> Vec<Wordpiece> {
        let chars = word.chars().collect::<Vec<_>>();
        if chars.is_empty() {
            return Vec::new();
        }
        if chars.len() > 100 {
            return vec![Wordpiece {
                id: self.unknown,
                aligned: false,
            }];
        }
        let mut result = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched = None;
            while start < end {
                let body = chars[start..end].iter().collect::<String>();
                let piece = if start == 0 {
                    body
                } else {
                    format!("##{body}")
                };
                if let Some(id) = self.pieces.get(&piece).copied() {
                    matched = Some((end, id));
                    break;
                }
                end -= 1;
            }
            let Some((next, id)) = matched else {
                return vec![Wordpiece {
                    id: self.unknown,
                    aligned: false,
                }];
            };
            result.push(Wordpiece { id, aligned: true });
            start = next;
        }
        result
    }
}

fn load_sudachi_dictionary(bundle: &Bundle) -> Result<Arc<JapaneseDictionary>, TransformerError> {
    let tokenizer_path = bundle.root().join(&bundle.manifest().tokenizer.path);
    let bytes = std::fs::read(&tokenizer_path)
        .map_err(|error| backend_error("read Jewel Sudachi configuration", error))?;
    let config: jewel_core::JapaneseTokenizerConfig = serde_json::from_slice(&bytes)
        .map_err(|error| backend_error("parse Jewel Sudachi configuration", error))?;
    let config_path = resolve_asset(bundle.root(), &config.config_path)?;
    let dictionary_path = resolve_asset(bundle.root(), &config.dictionary_path)?;
    let builder = ConfigBuilder::from_file(&config_path)
        .map_err(|error| backend_error("read Sudachi configuration", error))?;
    let sudachi_config = builder.system_dict(dictionary_path).build();
    JapaneseDictionary::from_cfg(&sudachi_config)
        .map(Arc::new)
        .map_err(|error| backend_error("load Sudachi dictionary", error))
}

fn backend_error(context: &str, error: impl std::fmt::Display) -> TransformerError {
    TransformerError::Backend(format!("{context}: {error}"))
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

fn resolve_asset(root: &Path, path: &str) -> Result<PathBuf, TransformerError> {
    if !is_safe_relative_path(path) {
        return Err(TransformerError::UnsafeAssetPath(path.to_owned()));
    }
    Ok(root.join(path))
}

fn is_safe_relative_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use jewel_core::{Doc, Matrix};

    use super::{
        batch_span_ranges, is_safe_relative_path, pool_span_output, span_ranges,
        validate_token_vectors, PooledDocument, PreparedSpan, TransformerEncoder, TransformerError,
        TransformerSpec, TransformerTokenizerSpec, WordpieceVocabulary,
    };

    struct DefaultBatchEncoder {
        spec: TransformerSpec,
    }

    impl TransformerEncoder for DefaultBatchEncoder {
        fn spec(&self) -> &TransformerSpec {
            &self.spec
        }

        fn encode(&self, doc: &Doc) -> Result<Matrix, TransformerError> {
            Ok(Matrix::zeros(doc.len(), self.spec.hidden_width))
        }
    }

    fn spec() -> TransformerSpec {
        TransformerSpec {
            architecture: "electra".to_owned(),
            model: "example/electra".to_owned(),
            hidden_width: 4,
            window: 128,
            stride: 96,
            max_wordpieces: 512,
            config_path: "transformer/config.json".to_owned(),
            weights_path: "transformer/model.safetensors".to_owned(),
            vocab_path: "transformer/vocab.txt".to_owned(),
            tokenizer: TransformerTokenizerSpec {
                kind: "sudachitra_wordpiece".to_owned(),
                split_mode: "A".to_owned(),
                word_form_type: "dictionary_and_surface".to_owned(),
                do_lower_case: false,
                do_nfkc: false,
                cls_token: "[CLS]".to_owned(),
                sep_token: "[SEP]".to_owned(),
                unk_token: "[UNK]".to_owned(),
                pad_token: "[PAD]".to_owned(),
            },
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
    fn default_batch_encoder_preserves_document_order_and_shapes() {
        let encoder = DefaultBatchEncoder { spec: spec() };
        let docs = [
            Doc::from_words(&["東京"], &[false]).unwrap(),
            Doc::from_words(&["株式会社", "青空"], &[false; 2]).unwrap(),
        ];
        let vectors = encoder.encode_batch(&docs).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!((vectors[0].rows(), vectors[0].cols()), (1, 4));
        assert_eq!((vectors[1].rows(), vectors[1].cols()), (2, 4));
    }

    #[test]
    fn rejects_invalid_windows_and_asset_paths() {
        let mut invalid = spec();
        invalid.stride = invalid.window + 1;
        assert!(matches!(
            invalid.validate(),
            Err(TransformerError::InvalidSpec(_))
        ));
        assert!(is_safe_relative_path("transformer/model.safetensors"));
        assert!(!is_safe_relative_path("../model.safetensors"));
        assert!(!is_safe_relative_path("/tmp/model.safetensors"));
    }

    #[test]
    fn wordpiece_unknowns_are_sent_to_electra_but_not_pooled() {
        let vocabulary = WordpieceVocabulary {
            pieces: HashMap::from([
                ("[UNK]".to_owned(), 1),
                ("契".to_owned(), 2),
                ("##約".to_owned(), 3),
            ]),
            cls: 4,
            sep: 5,
            unknown: 1,
            pad: 0,
        };
        let known = vocabulary.encode_word("契約");
        assert_eq!(
            known
                .iter()
                .map(|piece| (piece.id, piece.aligned))
                .collect::<Vec<_>>(),
            vec![(2, true), (3, true)]
        );
        let unknown = vocabulary.encode_word("〒");
        assert_eq!(
            unknown
                .iter()
                .map(|piece| (piece.id, piece.aligned))
                .collect::<Vec<_>>(),
            vec![(1, false)]
        );
    }

    #[test]
    fn span_ranges_preserve_the_existing_overlap_schedule() {
        assert_eq!(span_ranges(0, 128, 96), Vec::new());
        assert_eq!(span_ranges(12, 128, 96), vec![(0, 12)]);
        assert_eq!(
            span_ranges(300, 128, 96),
            vec![(0, 128), (96, 224), (192, 300)]
        );
    }

    #[test]
    fn batch_span_ranges_preserve_documents_and_skip_empty_inputs() {
        assert_eq!(
            batch_span_ranges(&[0, 12, 300], 128, 96),
            vec![(1, 0, 12), (2, 0, 128), (2, 96, 224), (2, 192, 300)]
        );
    }

    #[test]
    fn pooled_batch_outputs_accumulate_overlapping_wordpieces() {
        let first = PreparedSpan {
            document: 0,
            start: 0,
            ids: vec![4, 2, 3, 5],
            alignments: vec![vec![1], vec![2]],
        };
        let second = PreparedSpan {
            document: 0,
            start: 1,
            ids: vec![4, 3, 5],
            alignments: vec![vec![1]],
        };
        let mut sums = vec![0.0; 6];
        let mut counts = vec![0; 3];
        pool_span_output(
            &first,
            &[
                vec![0.0, 0.0],
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![0.0, 0.0],
            ],
            2,
            &mut sums,
            &mut counts,
        )
        .unwrap();
        pool_span_output(
            &second,
            &[vec![0.0, 0.0], vec![5.0, 6.0], vec![0.0, 0.0]],
            2,
            &mut sums,
            &mut counts,
        )
        .unwrap();
        assert_eq!(sums, vec![1.0, 2.0, 8.0, 10.0, 0.0, 0.0]);
        assert_eq!(counts, vec![1, 2, 0]);

        let pooled = PooledDocument { sums, counts }.finish(2).unwrap();
        assert_eq!(pooled.as_slice(), &[1.0, 2.0, 4.0, 5.0, 0.0, 0.0]);
    }
}
