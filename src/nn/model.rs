use std::collections::HashMap;

use spacy_model::{Bundle, BundleError, NodeManifest};
use thiserror::Error;

use crate::{affine_softmax, hash_embed, layer_norm, linear, maxout, Matrix, NnError};

#[derive(Debug, Error)]
pub enum ModelOpError {
    #[error(transparent)]
    Bundle(#[from] BundleError),
    #[error(transparent)]
    Kernel(#[from] NnError),
    #[error("node {node} is not a supported {operation} node: {message}")]
    InvalidNode {
        node: usize,
        operation: &'static str,
        message: String,
    },
    #[error("node {node} is missing dimension {name:?}")]
    MissingDimension { node: usize, name: String },
    #[error("node {node} has invalid attribute {name:?}: {message}")]
    InvalidAttribute {
        node: usize,
        name: String,
        message: String,
    },
    #[error("node {node} has invalid parameter {name:?}: {message}")]
    InvalidParameter {
        node: usize,
        name: String,
        message: String,
    },
}

pub struct HashEmbedLayer {
    seed: u32,
    embeddings: Matrix,
}

pub struct MaxoutLayer {
    outputs: usize,
    pieces: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

pub struct LayerNormLayer {
    gain: Vec<f32>,
    bias: Vec<f32>,
}

pub struct LinearLayer {
    outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

pub struct PrecomputableAffineLayer {
    inputs: usize,
    outputs: usize,
    features: usize,
    pieces: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    padding: Vec<f32>,
}

pub struct StaticVectorsLayer {
    outputs: usize,
    projection: Vec<f32>,
    vectors: Matrix,
    rows: HashMap<u64, usize>,
}

/// Cached token projections used by spaCy's transition-based parser models.
pub struct PrecomputedAffine {
    token_count: usize,
    outputs: usize,
    features: usize,
    pieces: usize,
    data: Vec<f32>,
    bias: Vec<f32>,
}

pub struct SoftmaxLayer {
    outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    normalize: bool,
    temperature: f32,
}

impl LinearLayer {
    /// Load a `linear` leaf from an exported model graph.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions or tensors.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "linear")?;
        let outputs = dimension(node, "nO")?;
        let inputs = dimension(node, "nI")?;
        Ok(Self {
            outputs,
            weights: tensor(bundle, node, "W", &[outputs, inputs])?,
            bias: tensor(bundle, node, "b", &[outputs])?,
        })
    }

    /// Run the affine projection.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the input width differs from the weights.
    pub fn forward(&self, input: &Matrix) -> Result<Matrix, ModelOpError> {
        Ok(linear(input, &self.weights, &self.bias, self.outputs)?)
    }

    #[must_use]
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl PrecomputableAffineLayer {
    /// Load a `precomputable_affine` parser layer.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions or tensors.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "precomputable_affine")?;
        let inputs = dimension(node, "nI")?;
        let outputs = dimension(node, "nO")?;
        let features = dimension(node, "nF")?;
        let pieces = dimension(node, "nP")?;
        Ok(Self {
            inputs,
            outputs,
            features,
            pieces,
            weights: tensor(bundle, node, "W", &[features, outputs, pieces, inputs])?,
            bias: tensor(bundle, node, "b", &[outputs, pieces])?,
            padding: tensor(bundle, node, "pad", &[1, features, outputs, pieces])?,
        })
    }

    /// Precompute every token's contribution for each parser state feature.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the input width differs from `nI`.
    pub fn precompute(&self, input: &Matrix) -> Result<PrecomputedAffine, ModelOpError> {
        if input.cols() != self.inputs {
            return Err(NnError::Shape {
                operation: "precomputable_affine",
                message: format!("input={}, expected={}", input.cols(), self.inputs),
            }
            .into());
        }
        let row_width = self
            .features
            .saturating_mul(self.outputs)
            .saturating_mul(self.pieces);
        let mut data = vec![0.0; (input.rows() + 1).saturating_mul(row_width)];
        data[..row_width].copy_from_slice(&self.padding);
        for token in 0..input.rows() {
            for feature in 0..self.features {
                for output in 0..self.outputs {
                    for piece in 0..self.pieces {
                        let parameter =
                            ((feature * self.outputs + output) * self.pieces + piece) * self.inputs;
                        let mut value = 0.0;
                        for column in 0..self.inputs {
                            value += input.row(token)[column] * self.weights[parameter + column];
                        }
                        let target = (token + 1) * row_width
                            + (feature * self.outputs + output) * self.pieces
                            + piece;
                        data[target] = value;
                    }
                }
            }
        }
        Ok(PrecomputedAffine {
            token_count: input.rows(),
            outputs: self.outputs,
            features: self.features,
            pieces: self.pieces,
            data,
            bias: self.bias.clone(),
        })
    }

    #[must_use]
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl StaticVectorsLayer {
    /// Load spaCy's projected vocabulary vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the projection or bundle-level vector table is
    /// missing or incompatible.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "static_vectors")?;
        let outputs = dimension(node, "nO")?;
        let vector_width = dimension(node, "nM")?;
        let manifests =
            bundle.manifest().vectors.as_ref().ok_or_else(|| {
                invalid(node, "static_vectors", "bundle has no vocabulary vectors")
            })?;
        if manifests.data.shape.len() != 2 || manifests.data.shape[1] != vector_width {
            return Err(invalid(
                node,
                "static_vectors",
                format!(
                    "vector data shape is {:?}, expected (*, {vector_width})",
                    manifests.data.shape
                ),
            ));
        }
        let vector_count = manifests.data.shape[0];
        let vectors = bundle.f32_tensor(&manifests.data.key)?;
        let keys = bundle.u64_tensor(&manifests.keys.key)?;
        let row_values = bundle.u64_tensor(&manifests.rows.key)?;
        if keys.shape().len() != 1
            || row_values.shape() != keys.shape()
            || keys.as_slice().len() != row_values.as_slice().len()
        {
            return Err(invalid(
                node,
                "static_vectors",
                "vector keys and rows must be equal-length vectors",
            ));
        }
        let rows = keys
            .as_slice()
            .iter()
            .copied()
            .zip(row_values.as_slice().iter().copied())
            .map(|(key, row)| {
                let row = usize::try_from(row)
                    .map_err(|_| invalid(node, "static_vectors", "vector row exceeds usize"))?;
                if row >= vector_count {
                    return Err(invalid(
                        node,
                        "static_vectors",
                        format!("vector row {row} exceeds table size {vector_count}"),
                    ));
                }
                Ok((key, row))
            })
            .collect::<Result<HashMap<_, _>, ModelOpError>>()?;
        Ok(Self {
            outputs,
            projection: tensor(bundle, node, "W", &[outputs, vector_width])?,
            vectors: Matrix::new(vector_count, vector_width, vectors.into_data())?,
            rows,
        })
    }

    /// Look up `ORTH` IDs and apply the learned vector projection.
    ///
    /// Unknown IDs produce the zero vector, matching spaCy's default vector
    /// table behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if loaded vector dimensions are inconsistent.
    pub fn forward(&self, ids: &[u64]) -> Result<Matrix, ModelOpError> {
        let mut output = Matrix::zeros(ids.len(), self.outputs);
        for (token, id) in ids.iter().enumerate() {
            let Some(row) = self.rows.get(id).copied() else {
                continue;
            };
            for result in 0..self.outputs {
                let mut value = 0.0;
                let weight_start = result * self.vectors.cols();
                for column in 0..self.vectors.cols() {
                    value += self.vectors.row(row)[column] * self.projection[weight_start + column];
                }
                output.row_mut(token)[result] = value;
            }
        }
        Ok(output)
    }

    #[must_use]
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl PrecomputedAffine {
    /// Sum cached state features, add the shared bias once, and max over pieces.
    ///
    /// Negative token IDs select the learned padding vector. Non-negative IDs
    /// are zero-based token positions.
    ///
    /// # Errors
    ///
    /// Returns a shape error for the wrong number of features or an invalid
    /// token position.
    pub fn hidden(&self, token_ids: &[i32]) -> Result<Matrix, ModelOpError> {
        if token_ids.len() != self.features {
            return Err(NnError::Shape {
                operation: "precomputed_affine",
                message: format!(
                    "received {} features, expected {}",
                    token_ids.len(),
                    self.features
                ),
            }
            .into());
        }
        let row_width = self
            .features
            .saturating_mul(self.outputs)
            .saturating_mul(self.pieces);
        let mut values = self.bias.clone();
        for (feature, token_id) in token_ids.iter().copied().enumerate() {
            let row = if token_id < 0 {
                0
            } else {
                let token = usize::try_from(token_id).map_err(|_| NnError::Shape {
                    operation: "precomputed_affine",
                    message: format!("invalid token ID {token_id}"),
                })?;
                if token >= self.token_count {
                    return Err(NnError::Shape {
                        operation: "precomputed_affine",
                        message: format!(
                            "token ID {token} is out of bounds for {} tokens",
                            self.token_count
                        ),
                    }
                    .into());
                }
                token + 1
            };
            let source = row * row_width + feature * self.outputs * self.pieces;
            for (value, cached) in values
                .iter_mut()
                .zip(&self.data[source..source + self.outputs.saturating_mul(self.pieces)])
            {
                *value += *cached;
            }
        }
        let mut hidden = Matrix::zeros(1, self.outputs);
        for output in 0..self.outputs {
            let start = output * self.pieces;
            hidden.row_mut(0)[output] = values[start..start + self.pieces]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
        }
        Ok(hidden)
    }

    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features
    }
}

impl HashEmbedLayer {
    /// Load a `hashembed` leaf from an exported model graph.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions, attributes, or tensors.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "hashembed")?;
        let rows = dimension(node, "nV")?;
        let cols = dimension(node, "nO")?;
        let seed_value = node
            .attrs
            .get("seed")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| invalid_attribute(node, "seed", "expected an unsigned integer"))?;
        let seed = u32::try_from(seed_value)
            .map_err(|_| invalid_attribute(node, "seed", "value exceeds u32"))?;
        let tensor = tensor(bundle, node, "E", &[rows, cols])?;
        Ok(Self {
            seed,
            embeddings: Matrix::new(rows, cols, tensor)?,
        })
    }

    /// Run inference for unsigned feature IDs.
    ///
    /// # Errors
    ///
    /// Returns a numerical shape error for an invalid embedding table.
    pub fn forward(&self, ids: &[u64]) -> Result<Matrix, ModelOpError> {
        Ok(hash_embed(ids, self.seed, &self.embeddings)?)
    }
}

impl MaxoutLayer {
    /// Load a `maxout` leaf from an exported model graph.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions or tensors.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "maxout")?;
        let outputs = dimension(node, "nO")?;
        let pieces = dimension(node, "nP")?;
        let inputs = dimension(node, "nI")?;
        let weights = tensor(bundle, node, "W", &[outputs, pieces, inputs])?;
        let bias = tensor(bundle, node, "b", &[outputs, pieces])?;
        Ok(Self {
            outputs,
            pieces,
            weights,
            bias,
        })
    }

    /// Run the affine maxout operation.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the input width differs from the weights.
    pub fn forward(&self, input: &Matrix) -> Result<Matrix, ModelOpError> {
        Ok(maxout(
            input,
            &self.weights,
            &self.bias,
            self.outputs,
            self.pieces,
        )?)
    }

    #[must_use]
    pub fn outputs(&self) -> usize {
        self.outputs
    }
}

impl LayerNormLayer {
    /// Load a `layernorm` leaf from an exported model graph.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions or tensors.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "layernorm")?;
        let width = dimension(node, "nO")?;
        Ok(Self {
            gain: tensor(bundle, node, "G", &[width])?,
            bias: tensor(bundle, node, "b", &[width])?,
        })
    }

    /// Run layer normalization.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the input width differs from the parameters.
    pub fn forward(&self, input: &Matrix) -> Result<Matrix, ModelOpError> {
        Ok(layer_norm(input, &self.gain, &self.bias)?)
    }
}

impl SoftmaxLayer {
    /// Load a `softmax` leaf from an exported model graph.
    ///
    /// # Errors
    ///
    /// Returns an error for missing dimensions, attributes, or tensors.
    pub fn load(bundle: &Bundle, node: &NodeManifest) -> Result<Self, ModelOpError> {
        require_name(node, "softmax")?;
        let outputs = dimension(node, "nO")?;
        let inputs = dimension(node, "nI")?;
        let normalize = node
            .attrs
            .get("softmax_normalize")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                invalid_attribute(node, "softmax_normalize", "expected a boolean value")
            })?;
        let temperature = node
            .attrs
            .get("softmax_temperature")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                invalid_attribute(node, "softmax_temperature", "expected a numeric value")
            })?
            .to_string()
            .parse::<f32>()
            .map_err(|_| {
                invalid_attribute(
                    node,
                    "softmax_temperature",
                    "value cannot be represented as f32",
                )
            })?;
        Ok(Self {
            outputs,
            weights: tensor(bundle, node, "W", &[outputs, inputs])?,
            bias: tensor(bundle, node, "b", &[outputs])?,
            normalize,
            temperature,
        })
    }

    /// Run the affine projection and configured softmax normalization.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the input width differs from the weights.
    pub fn forward(&self, input: &Matrix) -> Result<Matrix, ModelOpError> {
        Ok(affine_softmax(
            input,
            &self.weights,
            &self.bias,
            self.outputs,
            self.normalize,
            self.temperature,
        )?)
    }
}

fn require_name(node: &NodeManifest, expected: &'static str) -> Result<(), ModelOpError> {
    if node.name == expected {
        Ok(())
    } else {
        Err(invalid(
            node,
            expected,
            format!("node name is {:?}", node.name),
        ))
    }
}

fn dimension(node: &NodeManifest, name: &str) -> Result<usize, ModelOpError> {
    node.dims
        .get(name)
        .copied()
        .flatten()
        .ok_or_else(|| ModelOpError::MissingDimension {
            node: node.index,
            name: name.to_owned(),
        })
}

fn tensor(
    bundle: &Bundle,
    node: &NodeManifest,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, ModelOpError> {
    let tensor = node
        .params
        .get(name)
        .ok_or_else(|| ModelOpError::InvalidParameter {
            node: node.index,
            name: name.to_owned(),
            message: "parameter is missing".to_owned(),
        })?;
    if tensor.shape != shape {
        return Err(ModelOpError::InvalidParameter {
            node: node.index,
            name: name.to_owned(),
            message: format!(
                "parameter {name:?} shape is {:?}, expected {shape:?}",
                tensor.shape
            ),
        });
    }
    let loaded = bundle.f32_tensor(&tensor.key)?;
    if loaded.shape() != shape {
        return Err(ModelOpError::InvalidParameter {
            node: node.index,
            name: name.to_owned(),
            message: format!(
                "stored parameter {name:?} shape is {:?}, expected {shape:?}",
                loaded.shape()
            ),
        });
    }
    Ok(loaded.into_data())
}

fn invalid_attribute(node: &NodeManifest, name: &str, message: impl Into<String>) -> ModelOpError {
    ModelOpError::InvalidAttribute {
        node: node.index,
        name: name.to_owned(),
        message: message.into(),
    }
}

fn invalid(
    node: &NodeManifest,
    operation: &'static str,
    message: impl Into<String>,
) -> ModelOpError {
    ModelOpError::InvalidNode {
        node: node.index,
        operation,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::StaticVectorsLayer;
    use crate::Matrix;

    #[test]
    fn static_vectors_project_rows_and_zero_unknown_ids() {
        let layer = StaticVectorsLayer {
            outputs: 2,
            projection: vec![1.0, 2.0, -1.0, 0.5],
            vectors: Matrix::new(2, 2, vec![3.0, 4.0, 5.0, 6.0]).unwrap(),
            rows: HashMap::from([(10, 0), (20, 1)]),
        };
        let output = layer.forward(&[10, 99, 20]).unwrap();
        assert_eq!(output.as_slice(), &[11.0, -1.0, 0.0, 0.0, 17.0, -2.0]);
    }
}
