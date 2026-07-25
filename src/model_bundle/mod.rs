//! Schema and validation for exported `spacy-rs` model bundles.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use spacy_core::Doc;
use spacy_tokenizer::{JapaneseTokenizer, JapaneseTokenizerError, RegexTokenizer, TokenizerError};
use thiserror::Error;

pub const CURRENT_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BundleManifest {
    pub format_version: u32,
    pub source: SourceManifest,
    pub runtime: RuntimeManifest,
    pub tokenizer: TokenizerManifest,
    #[serde(default)]
    pub vectors: Option<VectorsManifest>,
    pub pipeline: Vec<ComponentManifest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VectorsManifest {
    pub data: TensorManifest,
    pub keys: TensorManifest,
    pub rows: TensorManifest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SourceManifest {
    pub spacy_version: String,
    pub model_name: String,
    pub model_version: String,
    pub lang: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeManifest {
    pub min_runtime_version: String,
    pub requires_python: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizerManifest {
    pub kind: TokenizerKind,
    pub path: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    Regex,
    Sudachi,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ComponentManifest {
    pub name: String,
    pub factory: String,
    pub kind: ComponentKind,
    pub root_node: Option<usize>,
    #[serde(default)]
    pub nodes: Vec<NodeManifest>,
    #[serde(default)]
    pub state_path: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub moves: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Trainable,
    RuleBased,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NodeManifest {
    pub index: usize,
    pub name: String,
    #[serde(default)]
    pub children: Vec<usize>,
    pub dims: BTreeMap<String, Option<usize>>,
    pub refs: BTreeMap<String, Option<usize>>,
    pub params: BTreeMap<String, TensorManifest>,
    #[serde(default)]
    pub attrs: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub omitted_attrs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TensorManifest {
    pub key: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Bundle {
    root: PathBuf,
    manifest: BundleManifest,
    weight_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct F32Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct U64Tensor {
    shape: Vec<usize>,
    data: Vec<u64>,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("model bundle manifest is invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported model bundle format {actual}; this runtime supports {supported}")]
    UnsupportedFormat { actual: u32, supported: u32 },
    #[error("runtime bundle must not require Python")]
    RequiresPython,
    #[error("duplicate pipeline component name {0:?}")]
    DuplicateComponent(String),
    #[error("component name and factory must not be empty")]
    EmptyComponentIdentity,
    #[error("component {component:?} contains duplicate node index {index}")]
    DuplicateNodeIndex { component: String, index: usize },
    #[error("trainable component {0:?} has no valid root node")]
    MissingRootNode(String),
    #[error("component {component:?} node {node} references missing child {child}")]
    MissingChild {
        component: String,
        node: usize,
        child: usize,
    },
    #[error("component {component:?} has unsafe state path {path:?}")]
    UnsafeStatePath { component: String, path: String },
}

#[derive(Debug, Error)]
pub enum BundleError {
    #[error("could not read model bundle file {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error("model bundle is missing required file {0}")]
    MissingFile(PathBuf),
    #[error("model bundle weights are invalid: {0}")]
    Weights(#[from] safetensors::SafeTensorError),
    #[error("manifest references missing tensor {0:?}")]
    MissingTensor(String),
    #[error("manifest references tensor {0:?} more than once")]
    DuplicateTensor(String),
    #[error("tensor {key:?} has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        key: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor {key:?} has dtype {actual}, expected {expected}")]
    TensorDtype {
        key: String,
        expected: String,
        actual: String,
    },
    #[error("tensor {key:?} byte length {actual} is not valid for F32 shape {shape:?}")]
    InvalidF32Data {
        key: String,
        shape: Vec<usize>,
        actual: usize,
    },
    #[error("tensor {key:?} byte length {actual} is not valid for U64 shape {shape:?}")]
    InvalidU64Data {
        key: String,
        shape: Vec<usize>,
        actual: usize,
    },
}

pub enum RuntimeTokenizer {
    Regex(Box<RegexTokenizer>),
    Sudachi(JapaneseTokenizer),
}

#[derive(Debug, Error)]
pub enum RuntimeTokenizerError {
    #[error("could not read tokenizer file {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Regex(#[from] TokenizerError),
    #[error(transparent)]
    Japanese(#[from] JapaneseTokenizerError),
}

impl RuntimeTokenizer {
    /// Tokenize text using the language implementation selected by the bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when regex execution or Sudachi analysis fails.
    pub fn tokenize(&self, text: &str) -> Result<Doc, RuntimeTokenizerError> {
        match self {
            Self::Regex(tokenizer) => Ok(tokenizer.tokenize(text)?),
            Self::Sudachi(tokenizer) => Ok(tokenizer.tokenize(text)?),
        }
    }
}

impl BundleManifest {
    /// Parse and validate a bundle manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] for invalid JSON, an unsupported format, or a
    /// structural contract violation.
    pub fn from_json(bytes: &[u8]) -> Result<Self, ManifestError> {
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate the runtime compatibility contract.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] if this runtime cannot safely execute the
    /// declared bundle structure.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.format_version != CURRENT_FORMAT_VERSION {
            return Err(ManifestError::UnsupportedFormat {
                actual: self.format_version,
                supported: CURRENT_FORMAT_VERSION,
            });
        }
        if self.runtime.requires_python {
            return Err(ManifestError::RequiresPython);
        }
        if !is_safe_relative_path(&self.tokenizer.path) {
            return Err(ManifestError::UnsafeStatePath {
                component: "tokenizer".to_owned(),
                path: self.tokenizer.path.clone(),
            });
        }

        let mut component_names = BTreeSet::new();
        for component in &self.pipeline {
            if component.name.is_empty() || component.factory.is_empty() {
                return Err(ManifestError::EmptyComponentIdentity);
            }
            if !component_names.insert(component.name.as_str()) {
                return Err(ManifestError::DuplicateComponent(component.name.clone()));
            }

            let mut node_indices = BTreeSet::new();
            for node in &component.nodes {
                if !node_indices.insert(node.index) {
                    return Err(ManifestError::DuplicateNodeIndex {
                        component: component.name.clone(),
                        index: node.index,
                    });
                }
            }
            if component.kind == ComponentKind::Trainable {
                let Some(root) = component.root_node else {
                    return Err(ManifestError::MissingRootNode(component.name.clone()));
                };
                if !node_indices.contains(&root) {
                    return Err(ManifestError::MissingRootNode(component.name.clone()));
                }
            }
            for node in &component.nodes {
                for child in &node.children {
                    if !node_indices.contains(child) {
                        return Err(ManifestError::MissingChild {
                            component: component.name.clone(),
                            node: node.index,
                            child: *child,
                        });
                    }
                }
            }

            if let Some(path) = &component.state_path {
                if !is_safe_relative_path(path) {
                    return Err(ManifestError::UnsafeStatePath {
                        component: component.name.clone(),
                        path: path.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

impl Bundle {
    /// Load and validate a Python-free model bundle directory.
    ///
    /// # Errors
    ///
    /// Returns [`BundleError`] if the manifest cannot be read, violates the
    /// compatibility contract, or references a missing file.
    pub fn load(root: impl AsRef<Path>) -> Result<Self, BundleError> {
        let root = root.as_ref();
        let manifest_path = root.join("manifest.json");
        let bytes = std::fs::read(&manifest_path).map_err(|source| BundleError::Io {
            path: manifest_path,
            source,
        })?;
        let manifest = BundleManifest::from_json(&bytes)?;

        let weights_path = root.join("weights.safetensors");
        if !weights_path.is_file() {
            return Err(BundleError::MissingFile(weights_path));
        }
        let weight_bytes = std::fs::read(&weights_path).map_err(|source| BundleError::Io {
            path: weights_path,
            source,
        })?;
        let weights = SafeTensors::deserialize(&weight_bytes)?;
        let tokenizer_path = root.join(&manifest.tokenizer.path);
        if !tokenizer_path.is_file() {
            return Err(BundleError::MissingFile(tokenizer_path));
        }
        let mut tensor_manifests = manifest
            .pipeline
            .iter()
            .flat_map(|component| &component.nodes)
            .flat_map(|node| node.params.values())
            .collect::<Vec<_>>();
        if let Some(vectors) = &manifest.vectors {
            tensor_manifests.extend([&vectors.data, &vectors.keys, &vectors.rows]);
        }
        let mut tensor_keys = BTreeSet::new();
        for tensor_manifest in tensor_manifests {
            if !tensor_keys.insert(tensor_manifest.key.as_str()) {
                return Err(BundleError::DuplicateTensor(tensor_manifest.key.clone()));
            }
            let tensor = weights
                .tensor(&tensor_manifest.key)
                .map_err(|_| BundleError::MissingTensor(tensor_manifest.key.clone()))?;
            if tensor.shape() != tensor_manifest.shape {
                return Err(BundleError::TensorShape {
                    key: tensor_manifest.key.clone(),
                    expected: tensor_manifest.shape.clone(),
                    actual: tensor.shape().to_vec(),
                });
            }
            let actual_dtype = format!("{:?}", tensor.dtype());
            if actual_dtype != tensor_manifest.dtype {
                return Err(BundleError::TensorDtype {
                    key: tensor_manifest.key.clone(),
                    expected: tensor_manifest.dtype.clone(),
                    actual: actual_dtype,
                });
            }
        }
        for component in &manifest.pipeline {
            if let Some(path) = &component.state_path {
                let state_path = root.join(path);
                if !state_path.is_file() {
                    return Err(BundleError::MissingFile(state_path));
                }
            }
        }

        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            weight_bytes,
        })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// Copy one `F32` tensor from the safetensors payload.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is absent, has another dtype, or has
    /// malformed byte storage.
    pub fn f32_tensor(&self, key: &str) -> Result<F32Tensor, BundleError> {
        let weights = SafeTensors::deserialize(&self.weight_bytes)?;
        let tensor = weights
            .tensor(key)
            .map_err(|_| BundleError::MissingTensor(key.to_owned()))?;
        if format!("{:?}", tensor.dtype()) != "F32" {
            return Err(BundleError::TensorDtype {
                key: key.to_owned(),
                expected: "F32".to_owned(),
                actual: format!("{:?}", tensor.dtype()),
            });
        }
        let bytes = tensor.data();
        let chunks = bytes.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            return Err(BundleError::InvalidF32Data {
                key: key.to_owned(),
                shape: tensor.shape().to_vec(),
                actual: bytes.len(),
            });
        }
        let data = chunks
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(F32Tensor {
            shape: tensor.shape().to_vec(),
            data,
        })
    }

    /// Copy one `U64` tensor from the safetensors payload.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is absent, has another dtype, or has
    /// malformed byte storage.
    pub fn u64_tensor(&self, key: &str) -> Result<U64Tensor, BundleError> {
        let weights = SafeTensors::deserialize(&self.weight_bytes)?;
        let tensor = weights
            .tensor(key)
            .map_err(|_| BundleError::MissingTensor(key.to_owned()))?;
        if format!("{:?}", tensor.dtype()) != "U64" {
            return Err(BundleError::TensorDtype {
                key: key.to_owned(),
                expected: "U64".to_owned(),
                actual: format!("{:?}", tensor.dtype()),
            });
        }
        let bytes = tensor.data();
        let chunks = bytes.chunks_exact(8);
        if !chunks.remainder().is_empty() {
            return Err(BundleError::InvalidU64Data {
                key: key.to_owned(),
                shape: tensor.shape().to_vec(),
                actual: bytes.len(),
            });
        }
        let data = chunks
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        Ok(U64Tensor {
            shape: tensor.shape().to_vec(),
            data,
        })
    }

    /// Instantiate the tokenizer declared by this model bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer data cannot be read or initialized.
    pub fn load_tokenizer(&self) -> Result<RuntimeTokenizer, RuntimeTokenizerError> {
        let tokenizer_path = self.root.join(&self.manifest.tokenizer.path);
        let bytes = std::fs::read(&tokenizer_path).map_err(|source| RuntimeTokenizerError::Io {
            path: tokenizer_path,
            source,
        })?;
        match self.manifest.tokenizer.kind {
            TokenizerKind::Regex => Ok(RuntimeTokenizer::Regex(Box::new(
                RegexTokenizer::from_json(&bytes)?,
            ))),
            TokenizerKind::Sudachi => Ok(RuntimeTokenizer::Sudachi(
                JapaneseTokenizer::from_bundle_json(&self.root, &bytes)?,
            )),
        }
    }
}

impl F32Tensor {
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

impl U64Tensor {
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.data
    }

    #[must_use]
    pub fn into_data(self) -> Vec<u64> {
        self.data
    }
}

fn is_safe_relative_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::{BundleManifest, ManifestError};

    const MANIFEST: &str = r#"{
      "format_version": 1,
      "source": {
        "spacy_version": "3.8.13",
        "model_name": "en_core_web_sm",
        "model_version": "3.8.0",
        "lang": "en"
      },
      "runtime": {
        "min_runtime_version": "0.0.1",
        "requires_python": false
      },
      "tokenizer": {
        "kind": "regex",
        "path": "tokenizer.json"
      },
      "pipeline": [{
        "name": "tagger",
        "factory": "tagger",
        "kind": "trainable",
        "root_node": 0,
        "nodes": [{
          "index": 0,
          "name": "with_array(softmax)",
          "children": [],
          "dims": {"nO": 50},
          "refs": {},
          "params": {
            "W": {"key": "components.tagger.nodes.0.W", "dtype": "F32", "shape": [50, 96]}
          }
        }]
      }]
    }"#;

    #[test]
    fn accepts_python_free_manifest() {
        let manifest = BundleManifest::from_json(MANIFEST.as_bytes()).unwrap();
        assert_eq!(manifest.pipeline[0].name, "tagger");
    }

    #[test]
    fn rejects_python_runtime_dependency() {
        let input = MANIFEST.replace(r#""requires_python": false"#, r#""requires_python": true"#);
        assert!(matches!(
            BundleManifest::from_json(input.as_bytes()),
            Err(ManifestError::RequiresPython)
        ));
    }

    #[test]
    fn rejects_parent_directory_state_path() {
        let input = MANIFEST.replace(
            r#""nodes": [{"#,
            r#""state_path": "../state.bin", "nodes": [{"#,
        );
        assert!(matches!(
            BundleManifest::from_json(input.as_bytes()),
            Err(ManifestError::UnsafeStatePath { .. })
        ));
    }
}
