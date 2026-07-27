//! Native inference-only numerical kernels matching the Thinc operations used
//! by spaCy's small convolutional pipelines.

mod compatibility;
mod dependency_parser;
mod model;
mod ner;
mod parser;
mod pipeline;
mod sentencizer;
mod tagger;
mod tok2vec;

use thiserror::Error;

pub use compatibility::{
    CompatibilityArea, CompatibilityDiagnostic, NerCompatibilityReport,
    COMPATIBILITY_REPORT_VERSION,
};
pub use dependency_parser::{ArcEagerState, DependencyParser, DependencyParserError, ParserAction};
pub use model::{
    HashEmbedLayer, LayerNormLayer, LinearLayer, MaxoutLayer, ModelOpError,
    PrecomputableAffineLayer, PrecomputedAffine, SoftmaxLayer, StaticVectorsLayer,
};
pub use ner::{
    EntityLabelFilter, EntityLabelSelection, EntityRecognizer, EntityRecognizerError, NamedEntity,
    NerAction, NerState,
};
pub use parser::{TransitionScorer, TransitionScorerError};
pub use pipeline::{
    EnglishNerPipeline, EnglishPipeline, EnglishTaggerPipeline, JapaneseNerPipeline, NerLanguage,
    NerPipeline, PipelineError,
};
pub use sentencizer::{Sentencizer, SentencizerError};
pub use tagger::{Tagger, TaggerError};
pub use tok2vec::{extract_features, Tok2Vec, Tok2VecEmbed, Tok2VecError};
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NnError {
    #[error("matrix shape {rows}x{cols} requires {expected} values, got {actual}")]
    InvalidMatrix {
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    #[error("operation {operation} received incompatible shape: {message}")]
    Shape {
        operation: &'static str,
        message: String,
    },
    #[error("sequence lengths total {actual}, expected {expected}")]
    InvalidLengths { expected: usize, actual: usize },
}

impl Matrix {
    /// Construct a row-major matrix.
    ///
    /// # Errors
    ///
    /// Returns [`NnError::InvalidMatrix`] when the data length does not match
    /// `rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, NnError> {
        let expected = rows.saturating_mul(cols);
        if data.len() != expected {
            return Err(NnError::InvalidMatrix {
                rows,
                cols,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows.saturating_mul(cols)],
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }
}

/// Thinc's four-bucket `MurmurHash3` result for one unsigned 64-bit key.
#[must_use]
pub fn hash64_4(value: u64, seed: u32) -> [u32; 4] {
    let mut h1 = value;
    h1 = h1.wrapping_mul(0x87c3_7b91_1142_53d5);
    h1 = h1.rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5_ad43_2745_937f);
    h1 ^= u64::from(seed);
    h1 ^= 8;
    let mut h2 = u64::from(seed) ^ 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    let h1 = h1.to_le_bytes();
    let h2 = h2.to_le_bytes();
    [
        u32::from_le_bytes([h1[0], h1[1], h1[2], h1[3]]),
        u32::from_le_bytes([h1[4], h1[5], h1[6], h1[7]]),
        u32::from_le_bytes([h2[0], h2[1], h2[2], h2[3]]),
        u32::from_le_bytes([h2[4], h2[5], h2[6], h2[7]]),
    ]
}

fn fmix64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    value
}

/// Apply Thinc's four-bucket hash embedding lookup and sum.
///
/// # Errors
///
/// Returns a shape error if the embedding table is empty.
pub fn hash_embed(ids: &[u64], seed: u32, embeddings: &Matrix) -> Result<Matrix, NnError> {
    if embeddings.rows == 0 {
        return Err(shape("hash_embed", "embedding row count must be non-zero"));
    }
    let mut output = Matrix::zeros(ids.len(), embeddings.cols);
    for (row, id) in ids.iter().copied().enumerate() {
        for key in hash64_4(id, seed) {
            let source = embeddings.row(key as usize % embeddings.rows);
            for (target, value) in output.row_mut(row).iter_mut().zip(source) {
                *target += *value;
            }
        }
    }
    Ok(output)
}

/// Concatenate matrices column-wise.
///
/// # Errors
///
/// Returns a shape error unless all matrices have the same row count.
pub fn concatenate(inputs: &[Matrix]) -> Result<Matrix, NnError> {
    let Some(first) = inputs.first() else {
        return Ok(Matrix::zeros(0, 0));
    };
    if inputs.iter().any(|input| input.rows != first.rows) {
        return Err(shape("concatenate", "all row counts must match"));
    }
    let cols = inputs.iter().map(|input| input.cols).sum();
    let mut output = Matrix::zeros(first.rows, cols);
    for row in 0..first.rows {
        let mut offset = 0;
        for input in inputs {
            let end = offset + input.cols;
            output.row_mut(row)[offset..end].copy_from_slice(input.row(row));
            offset = end;
        }
    }
    Ok(output)
}

/// Apply Thinc's affine projection followed by maxout pooling.
///
/// `weights` uses row-major `(nO, nP, nI)` layout and `bias` uses
/// `(nO, nP)` layout.
///
/// # Errors
///
/// Returns a shape error when the parameter sizes are inconsistent.
pub fn maxout(
    input: &Matrix,
    weights: &[f32],
    bias: &[f32],
    outputs: usize,
    pieces: usize,
) -> Result<Matrix, NnError> {
    let expected_weights = outputs.saturating_mul(pieces).saturating_mul(input.cols);
    if pieces == 0
        || weights.len() != expected_weights
        || bias.len() != outputs.saturating_mul(pieces)
    {
        return Err(shape(
            "maxout",
            format!(
                "input={}, outputs={outputs}, pieces={pieces}, weights={}, bias={}",
                input.cols,
                weights.len(),
                bias.len()
            ),
        ));
    }
    let mut result = Matrix::zeros(input.rows, outputs);
    for row in 0..input.rows {
        for output in 0..outputs {
            let mut best = f32::NEG_INFINITY;
            for piece in 0..pieces {
                let parameter = output * pieces + piece;
                let mut value = bias[parameter];
                let weight_start = parameter * input.cols;
                for index in 0..input.cols {
                    value += input.row(row)[index] * weights[weight_start + index];
                }
                best = best.max(value);
            }
            result.row_mut(row)[output] = best;
        }
    }
    Ok(result)
}

/// Apply per-row layer normalization and learned scale/shift.
///
/// # Errors
///
/// Returns a shape error unless `gain` and `bias` match the input width.
pub fn layer_norm(input: &Matrix, gain: &[f32], bias: &[f32]) -> Result<Matrix, NnError> {
    if gain.len() != input.cols || bias.len() != input.cols {
        return Err(shape(
            "layer_norm",
            format!(
                "input={}, gain={}, bias={}",
                input.cols,
                gain.len(),
                bias.len()
            ),
        ));
    }
    let mut output = Matrix::zeros(input.rows, input.cols);
    if input.cols == 0 {
        return Ok(output);
    }
    let width = f32::from(
        u16::try_from(input.cols).map_err(|_| shape("layer_norm", "input width exceeds 65535"))?,
    );
    for row in 0..input.rows {
        let values = input.row(row);
        let mean = values.iter().copied().sum::<f32>() / width;
        let variance = values
            .iter()
            .map(|value| {
                let distance = *value - mean;
                distance * distance
            })
            .sum::<f32>()
            / width
            + 1e-8;
        let inverse_std = variance.sqrt().recip();
        for column in 0..input.cols {
            output.row_mut(row)[column] =
                (values[column] - mean) * inverse_std * gain[column] + bias[column];
        }
    }
    Ok(output)
}

/// Expand each sequence row with left and right context, zero-padding sequence
/// boundaries.
///
/// # Errors
///
/// Returns an error when `lengths` does not partition all input rows.
pub fn expand_window(input: &Matrix, window: usize, lengths: &[usize]) -> Result<Matrix, NnError> {
    let total: usize = lengths.iter().sum();
    if total != input.rows {
        return Err(NnError::InvalidLengths {
            expected: input.rows,
            actual: total,
        });
    }
    let features = window * 2 + 1;
    let mut output = Matrix::zeros(input.rows, input.cols.saturating_mul(features));
    let mut sequence_start = 0;
    for length in lengths {
        let sequence_end = sequence_start + length;
        for row in sequence_start..sequence_end {
            for relative in 0..features {
                let source_row = if relative <= window {
                    row.checked_sub(window - relative)
                } else {
                    row.checked_add(relative - window)
                };
                let Some(source_row) =
                    source_row.filter(|source| *source >= sequence_start && *source < sequence_end)
                else {
                    continue;
                };
                let target_start = relative * input.cols;
                output.row_mut(row)[target_start..target_start + input.cols]
                    .copy_from_slice(input.row(source_row));
            }
        }
        sequence_start = sequence_end;
    }
    Ok(output)
}

/// Element-wise residual addition.
///
/// # Errors
///
/// Returns a shape error unless both matrices have identical shapes.
pub fn residual(left: &Matrix, right: &Matrix) -> Result<Matrix, NnError> {
    if left.rows != right.rows || left.cols != right.cols {
        return Err(shape(
            "residual",
            format!(
                "left={}x{}, right={}x{}",
                left.rows, left.cols, right.rows, right.cols
            ),
        ));
    }
    Matrix::new(
        left.rows,
        left.cols,
        left.data
            .iter()
            .zip(&right.data)
            .map(|(left, right)| left + right)
            .collect(),
    )
}

/// Apply an affine projection and optional row-wise softmax.
///
/// `weights` uses `(nO, nI)` layout.
///
/// # Errors
///
/// Returns a shape error when the parameters do not match the input.
pub fn affine_softmax(
    input: &Matrix,
    weights: &[f32],
    bias: &[f32],
    outputs: usize,
    normalize: bool,
    temperature: f32,
) -> Result<Matrix, NnError> {
    if weights.len() != outputs.saturating_mul(input.cols)
        || bias.len() != outputs
        || temperature <= 0.0
    {
        return Err(shape(
            "affine_softmax",
            format!(
                "input={}, outputs={outputs}, weights={}, bias={}, temperature={temperature}",
                input.cols,
                weights.len(),
                bias.len()
            ),
        ));
    }
    let mut result = Matrix::zeros(input.rows, outputs);
    for row in 0..input.rows {
        for (output, output_bias) in bias.iter().copied().enumerate() {
            let mut value = output_bias;
            let weight_start = output * input.cols;
            for column in 0..input.cols {
                value += input.row(row)[column] * weights[weight_start + column];
            }
            result.row_mut(row)[output] = value;
        }
        if normalize && outputs > 0 {
            let maximum = result
                .row(row)
                .iter()
                .map(|value| *value / temperature)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for value in result.row_mut(row) {
                *value = (*value / temperature - maximum).exp();
                sum += *value;
            }
            for value in result.row_mut(row) {
                *value /= sum;
            }
        }
    }
    Ok(result)
}

/// Apply a row-wise affine projection.
///
/// `weights` uses `(nO, nI)` layout.
///
/// # Errors
///
/// Returns a shape error when the parameters do not match the input.
pub fn linear(
    input: &Matrix,
    weights: &[f32],
    bias: &[f32],
    outputs: usize,
) -> Result<Matrix, NnError> {
    affine_softmax(input, weights, bias, outputs, false, 1.0)
}

fn shape(operation: &'static str, message: impl Into<String>) -> NnError {
    NnError::Shape {
        operation,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_window, hash64_4, residual, Matrix};

    #[test]
    fn thinc_hash_matches_golden_values() {
        assert_eq!(hash64_4(0, 8), [0, 0, 0, 0]);
        assert_eq!(
            hash64_4(18_446_744_073_709_551_615, 13),
            [3_199_409_384, 1_062_937_581, 752_279_700, 3_061_442_444]
        );
    }

    #[test]
    fn window_does_not_cross_sequence_boundaries() {
        let input = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let result = expand_window(&input, 1, &[2, 1]).unwrap();
        assert_eq!(
            result.as_slice(),
            &[0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn residual_adds_equal_shapes() {
        let left = Matrix::new(1, 2, vec![1.0, 2.0]).unwrap();
        let right = Matrix::new(1, 2, vec![3.0, 4.0]).unwrap();
        assert_eq!(residual(&left, &right).unwrap().as_slice(), &[4.0, 6.0]);
    }
}
