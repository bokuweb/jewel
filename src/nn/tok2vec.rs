use spacy_core::{Doc, StringStore};
use spacy_model::{Bundle, ComponentManifest};
use thiserror::Error;

use crate::{
    concatenate, expand_window, residual, HashEmbedLayer, LayerNormLayer, Matrix, MaxoutLayer,
    ModelOpError, NnError, StaticVectorsLayer,
};

const FEATURE_COUNT: usize = 6;

#[derive(Debug, Error)]
pub enum Tok2VecError {
    #[error(transparent)]
    Model(#[from] ModelOpError),
    #[error(transparent)]
    Kernel(#[from] NnError),
    #[error("tok2vec component {0:?} is missing")]
    MissingComponent(String),
    #[error("tok2vec embedding graph is invalid: {0}")]
    InvalidGraph(String),
}

pub struct Tok2VecEmbed {
    embeddings: Vec<HashEmbedLayer>,
    projection: MaxoutLayer,
    normalization: LayerNormLayer,
    feature_columns: Vec<usize>,
    static_vectors: Option<StaticVectorsLayer>,
}

pub struct Tok2Vec {
    embed: Tok2VecEmbed,
    convolution: Vec<(MaxoutLayer, LayerNormLayer)>,
    pad: usize,
}

impl Tok2VecEmbed {
    /// Load the feature embedding and initial projection portion of a spaCy
    /// `tok2vec` component.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph does not contain a supported ordered
    /// feature list, matching hash embeddings, and an initial projection.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, Tok2VecError> {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| Tok2VecError::MissingComponent(component_name.to_owned()))?;
        let feature_columns = feature_columns(component)?;
        let feature_count = feature_columns.len();
        let mut embedding_nodes = component
            .nodes
            .iter()
            .filter(|node| node.name == "hashembed")
            .map(|node| {
                let column = node
                    .attrs
                    .get("column")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|column| usize::try_from(column).ok())
                    .ok_or_else(|| {
                        Tok2VecError::InvalidGraph(format!(
                            "hashembed node {} has no valid column",
                            node.index
                        ))
                    })?;
                Ok((column, node))
            })
            .collect::<Result<Vec<_>, Tok2VecError>>()?;
        embedding_nodes.sort_by_key(|(column, _)| *column);
        if embedding_nodes.len() != feature_count
            || embedding_nodes
                .iter()
                .enumerate()
                .any(|(expected, (actual, _))| expected != *actual)
        {
            return Err(Tok2VecError::InvalidGraph(format!(
                "expected columns 0..{feature_count}, got {:?}",
                embedding_nodes
                    .iter()
                    .map(|(column, _)| *column)
                    .collect::<Vec<_>>()
            )));
        }
        let embedding_width = embedding_nodes[0]
            .1
            .dims
            .get("nO")
            .copied()
            .flatten()
            .ok_or_else(|| Tok2VecError::InvalidGraph("missing embedding width".to_owned()))?;
        let static_vectors = static_vectors(bundle, component)?;
        let projection_width = embedding_width * feature_count
            + static_vectors
                .as_ref()
                .map_or(0, StaticVectorsLayer::outputs);
        let projection_node = component
            .nodes
            .iter()
            .find(|node| {
                node.name == "maxout"
                    && node.dims.get("nI").copied().flatten() == Some(projection_width)
            })
            .ok_or_else(|| {
                Tok2VecError::InvalidGraph(format!(
                    "missing initial maxout with width {projection_width}"
                ))
            })?;
        let parent = component
            .nodes
            .iter()
            .find(|node| {
                node.name == "maxout>>layernorm"
                    && node.children.first() == Some(&projection_node.index)
            })
            .ok_or_else(|| {
                Tok2VecError::InvalidGraph("missing maxout/layernorm chain".to_owned())
            })?;
        let normalization_index = *parent
            .children
            .get(1)
            .ok_or_else(|| Tok2VecError::InvalidGraph("missing layernorm child".to_owned()))?;
        let normalization_node = component
            .nodes
            .iter()
            .find(|node| node.index == normalization_index)
            .ok_or_else(|| {
                Tok2VecError::InvalidGraph(format!("missing layernorm node {normalization_index}"))
            })?;

        let embeddings = embedding_nodes
            .into_iter()
            .map(|(_, node)| HashEmbedLayer::load(bundle, node))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embeddings,
            projection: MaxoutLayer::load(bundle, projection_node)?,
            normalization: LayerNormLayer::load(bundle, normalization_node)?,
            feature_columns,
            static_vectors,
        })
    }

    /// Execute feature extraction, hash embedding, concatenation, maxout, and
    /// layer normalization.
    ///
    /// # Errors
    ///
    /// Returns an error if a loaded parameter shape is inconsistent.
    pub fn forward(&self, doc: &Doc) -> Result<Matrix, Tok2VecError> {
        let features = extract_features(doc);
        let mut embedded = self
            .embeddings
            .iter()
            .zip(&self.feature_columns)
            .map(|(embedding, source_column)| {
                let ids = features
                    .iter()
                    .map(|features| features[*source_column])
                    .collect::<Vec<_>>();
                embedding.forward(&ids)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(static_vectors) = &self.static_vectors {
            let orth = doc
                .tokens()
                .iter()
                .map(|token| token.orth)
                .collect::<Vec<_>>();
            embedded.push(static_vectors.forward(&orth)?);
        }
        let concatenated = concatenate(&embedded)?;
        let projected = self.projection.forward(&concatenated)?;
        Ok(self.normalization.forward(&projected)?)
    }
}

impl Tok2Vec {
    /// Load the complete hash-embedding and residual convolution encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the exported graph does not have the expected
    /// four residual convolution blocks.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, Tok2VecError> {
        let embed = Tok2VecEmbed::load(bundle, component_name)?;
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| Tok2VecError::MissingComponent(component_name.to_owned()))?;
        let encode = component
            .nodes
            .iter()
            .find(|node| {
                node.attrs.get("pad").and_then(serde_json::Value::as_u64) == Some(4)
                    && node.name.contains("residual(expand_window")
            })
            .ok_or_else(|| {
                Tok2VecError::InvalidGraph("missing padded convolution encoder".to_owned())
            })?;
        let pad = encode
            .attrs
            .get("pad")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| Tok2VecError::InvalidGraph("missing encoder padding".to_owned()))?;
        let mut maxout_nodes = component
            .nodes
            .iter()
            .filter(|node| {
                node.name == "maxout"
                    && node.dims.get("nI").copied().flatten() == Some(288)
                    && node.dims.get("nO").copied().flatten() == Some(96)
            })
            .collect::<Vec<_>>();
        maxout_nodes.sort_by_key(|node| node.index);
        if maxout_nodes.len() != 4 {
            return Err(Tok2VecError::InvalidGraph(format!(
                "expected 4 convolution maxout nodes, got {}",
                maxout_nodes.len()
            )));
        }
        let convolution = maxout_nodes
            .into_iter()
            .map(|maxout_node| {
                let parent = component
                    .nodes
                    .iter()
                    .find(|node| {
                        node.name == "maxout>>layernorm"
                            && node.children.first() == Some(&maxout_node.index)
                    })
                    .ok_or_else(|| {
                        Tok2VecError::InvalidGraph(format!(
                            "maxout node {} has no layernorm parent",
                            maxout_node.index
                        ))
                    })?;
                let normalization_index = *parent.children.get(1).ok_or_else(|| {
                    Tok2VecError::InvalidGraph(format!(
                        "maxout node {} has no layernorm sibling",
                        maxout_node.index
                    ))
                })?;
                let normalization_node = component
                    .nodes
                    .iter()
                    .find(|node| node.index == normalization_index)
                    .ok_or_else(|| {
                        Tok2VecError::InvalidGraph(format!(
                            "missing layernorm node {normalization_index}"
                        ))
                    })?;
                Ok((
                    MaxoutLayer::load(bundle, maxout_node)?,
                    LayerNormLayer::load(bundle, normalization_node)?,
                ))
            })
            .collect::<Result<Vec<_>, Tok2VecError>>()?;
        Ok(Self {
            embed,
            convolution,
            pad,
        })
    }

    /// Execute the complete `tok2vec` encoder for one document.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if any exported parameter is inconsistent.
    pub fn forward(&self, doc: &Doc) -> Result<Matrix, Tok2VecError> {
        self.forward_stages(doc)?
            .into_iter()
            .last()
            .ok_or_else(|| Tok2VecError::InvalidGraph("tok2vec emitted no stages".to_owned()))
    }

    /// Return the embedding output and each successive residual block output.
    ///
    /// This is primarily useful for cross-runtime numerical conformance tests.
    ///
    /// # Errors
    ///
    /// Returns a numerical error if any exported parameter is inconsistent.
    pub fn forward_stages(&self, doc: &Doc) -> Result<Vec<Matrix>, Tok2VecError> {
        let embedded = self.embed.forward(doc)?;
        let mut stages = vec![embedded.clone()];
        let mut output = pad_rows(&embedded, self.pad)?;
        for (maxout, normalization) in &self.convolution {
            let expanded = expand_window(&output, 1, &[output.rows()])?;
            let update = maxout.forward(&expanded)?;
            let update = normalization.forward(&update)?;
            output = residual(&output, &update)?;
            stages.push(crop_rows(&output, self.pad, doc.len())?);
        }
        Ok(stages)
    }
}

fn feature_columns(component: &ComponentManifest) -> Result<Vec<usize>, Tok2VecError> {
    let feature_node = component
        .nodes
        .iter()
        .find(|node| node.name == "extract_features")
        .ok_or_else(|| Tok2VecError::InvalidGraph("missing extract_features node".to_owned()))?;
    let columns = feature_node
        .attrs
        .get("columns")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Tok2VecError::InvalidGraph("missing feature column list".to_owned()))?;
    let supported = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"];
    let indices = columns
        .iter()
        .map(|column| {
            let name = column.as_str().ok_or_else(|| {
                Tok2VecError::InvalidGraph(format!("invalid feature column {column:?}"))
            })?;
            supported
                .iter()
                .position(|supported| *supported == name)
                .ok_or_else(|| {
                    Tok2VecError::InvalidGraph(format!("unsupported feature column {name:?}"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if indices.is_empty() || indices.len() > FEATURE_COUNT {
        return Err(Tok2VecError::InvalidGraph(format!(
            "unsupported feature columns {columns:?}"
        )));
    }
    Ok(indices)
}

fn static_vectors(
    bundle: &Bundle,
    component: &ComponentManifest,
) -> Result<Option<StaticVectorsLayer>, Tok2VecError> {
    component
        .nodes
        .iter()
        .find(|node| node.name == "static_vectors")
        .map(|node| StaticVectorsLayer::load(bundle, node))
        .transpose()
        .map_err(Into::into)
}

/// Build the six integer feature columns used by the English small-model
/// embedding: NORM, PREFIX, SUFFIX, SHAPE, SPACY, and `IS_SPACE`.
#[must_use]
pub fn extract_features(doc: &Doc) -> Vec<[u64; FEATURE_COUNT]> {
    doc.tokens()
        .iter()
        .map(|token| {
            let norm = if token.norm == 0 {
                StringStore::id(&token.text.to_lowercase())
            } else {
                token.norm
            };
            [
                norm,
                StringStore::id(&prefix(&token.text)),
                StringStore::id(&suffix(&token.text)),
                StringStore::id(&word_shape(&token.text)),
                u64::from(token.has_space),
                u64::from(token.text.chars().all(char::is_whitespace)),
            ]
        })
        .collect()
}

fn prefix(text: &str) -> String {
    text.chars().next().into_iter().collect()
}

fn suffix(text: &str) -> String {
    let mut characters = text.chars().rev().take(3).collect::<Vec<_>>();
    characters.reverse();
    characters.into_iter().collect()
}

fn word_shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".to_owned();
    }
    let mut shape = String::new();
    let mut last = None;
    let mut sequence = 0;
    for character in text.chars() {
        let shape_character = if character.is_alphabetic() {
            if character.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if is_decimal_digit(character) {
            'd'
        } else {
            character
        };
        if last == Some(shape_character) {
            sequence += 1;
        } else {
            sequence = 0;
            last = Some(shape_character);
        }
        if sequence < 4 {
            shape.push(shape_character);
        }
    }
    shape
}

#[allow(clippy::to_digit_is_some)] // Unicode decimal digits are required, not only ASCII digits.
fn is_decimal_digit(character: char) -> bool {
    character.to_digit(10).is_some()
}

fn pad_rows(input: &Matrix, pad: usize) -> Result<Matrix, NnError> {
    let mut data = vec![0.0; pad.saturating_mul(input.cols())];
    data.extend_from_slice(input.as_slice());
    data.resize(data.len() + pad.saturating_mul(input.cols()), 0.0);
    Matrix::new(input.rows() + pad * 2, input.cols(), data)
}

fn crop_rows(input: &Matrix, start: usize, rows: usize) -> Result<Matrix, NnError> {
    let first = start.saturating_mul(input.cols());
    let last = first + rows.saturating_mul(input.cols());
    Matrix::new(rows, input.cols(), input.as_slice()[first..last].to_vec())
}

#[cfg(test)]
mod tests {
    use spacy_core::{Doc, StringStore};

    use super::{extract_features, word_shape};

    #[test]
    fn word_shape_matches_spacy_rules() {
        assert_eq!(word_shape("Hello12345!"), "Xxxxxdddd!");
        assert_eq!(word_shape(&"a".repeat(100)), "LONG");
    }

    #[test]
    fn extracts_reserved_shape_and_boolean_features() {
        let doc = Doc::from_words(&["I", " "], &[true, false]).unwrap();
        let features = extract_features(&doc);
        assert_eq!(features[0][3], 101);
        assert_eq!(features[0][4], 1);
        assert_eq!(features[1][5], 1);
        assert_eq!(features[0][0], StringStore::id("i"));
    }
}
