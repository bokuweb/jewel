//! Schema and validation for exported `spacy-rs` model bundles.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use spacy_core::Doc;
#[cfg(feature = "delarocha-tokenizer")]
use spacy_tokenizer::{DelarochaTokenizer, DelarochaTokenizerError};
#[cfg(feature = "sudachi-tokenizer")]
use spacy_tokenizer::{JapaneseTokenizer, JapaneseTokenizerError};
use spacy_tokenizer::{RegexTokenizer, TokenizeError, Tokenizer, TokenizerError, TokenizerSession};
use thiserror::Error;

pub const CURRENT_FORMAT_VERSION: u32 = 1;

/// Resource limits applied while parsing and loading a model bundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleLimits {
    pub max_manifest_bytes: u64,
    pub max_weights_bytes: u64,
    pub max_tokenizer_bytes: u64,
    pub max_component_state_bytes: u64,
    pub max_components: usize,
    pub max_nodes_per_component: usize,
    pub max_tensors: usize,
    pub max_tensor_rank: usize,
}

impl Default for BundleLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 16 * 1024 * 1024,
            max_weights_bytes: 1024 * 1024 * 1024,
            max_tokenizer_bytes: 64 * 1024 * 1024,
            max_component_state_bytes: 256 * 1024 * 1024,
            max_components: 128,
            max_nodes_per_component: 32 * 1024,
            max_tensors: 128 * 1024,
            max_tensor_rank: 8,
        }
    }
}

/// Bundle resource guarded by [`BundleLimits`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BundleLimitResource {
    ManifestBytes,
    WeightsBytes,
    TokenizerBytes,
    ComponentStateBytes,
    Components,
    ComponentNodes,
    Tensors,
    TensorRank,
}

impl BundleLimitResource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ManifestBytes => "manifest_bytes",
            Self::WeightsBytes => "weights_bytes",
            Self::TokenizerBytes => "tokenizer_bytes",
            Self::ComponentStateBytes => "component_state_bytes",
            Self::Components => "components",
            Self::ComponentNodes => "component_nodes",
            Self::Tensors => "tensors",
            Self::TensorRank => "tensor_rank",
        }
    }
}

impl std::fmt::Display for BundleLimitResource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Details for a model bundle resource limit failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("bundle resource {resource} has {actual} units, exceeding limit {limit}")]
pub struct BundleLimitError {
    pub resource: BundleLimitResource,
    pub actual: u64,
    pub limit: u64,
    pub component: Option<String>,
    pub path: Option<PathBuf>,
}

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
    Delarocha,
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
    pub settings: BTreeMap<String, serde_json::Value>,
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
    limits: BundleLimits,
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
    #[error(transparent)]
    Limit(#[from] BundleLimitError),
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
    #[error(transparent)]
    Limit(#[from] BundleLimitError),
}

pub enum RuntimeTokenizer {
    #[cfg(feature = "delarocha-tokenizer")]
    Delarocha(Box<DelarochaTokenizer>),
    Regex(Box<RegexTokenizer>),
    #[cfg(feature = "sudachi-tokenizer")]
    Sudachi(JapaneseTokenizer),
}

#[derive(Debug, Error)]
pub enum RuntimeTokenizerError {
    #[error("tokenizer backend {0:?} is not enabled in this Jewel build")]
    BackendDisabled(&'static str),
    #[cfg(feature = "delarocha-tokenizer")]
    #[error(transparent)]
    Delarocha(#[from] DelarochaTokenizerError),
    #[cfg(feature = "sudachi-tokenizer")]
    #[error(transparent)]
    Japanese(#[from] JapaneseTokenizerError),
    #[error("could not read tokenizer file {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Regex(#[from] TokenizerError),
    #[error(transparent)]
    Limit(#[from] BundleLimitError),
}

impl RuntimeTokenizer {
    /// Tokenize text using the language implementation selected by the bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected tokenizer backend fails.
    pub fn tokenize(&self, text: &str) -> Result<Doc, RuntimeTokenizerError> {
        match self {
            #[cfg(feature = "delarocha-tokenizer")]
            Self::Delarocha(tokenizer) => Ok(tokenizer.tokenize(text)?),
            Self::Regex(tokenizer) => Ok(tokenizer.tokenize(text)?),
            #[cfg(feature = "sudachi-tokenizer")]
            Self::Sudachi(tokenizer) => Ok(tokenizer.tokenize(text)?),
        }
    }
}

impl Tokenizer for RuntimeTokenizer {
    fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError> {
        RuntimeTokenizer::tokenize(self, text).map_err(TokenizeError::new)
    }

    fn session(&self) -> Box<dyn TokenizerSession + '_> {
        match self {
            #[cfg(feature = "delarocha-tokenizer")]
            Self::Delarocha(tokenizer) => tokenizer.reusable_session(),
            Self::Regex(tokenizer) => Tokenizer::session(tokenizer.as_ref()),
            #[cfg(feature = "sudachi-tokenizer")]
            Self::Sudachi(tokenizer) => Tokenizer::session(tokenizer),
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
        Self::from_json_with_limits(bytes, &BundleLimits::default())
    }

    /// Parse and validate a bundle manifest with caller-selected limits.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] when parsing, structural validation, or a
    /// configured resource limit fails.
    pub fn from_json_with_limits(
        bytes: &[u8],
        limits: &BundleLimits,
    ) -> Result<Self, ManifestError> {
        check_limit(
            BundleLimitResource::ManifestBytes,
            bytes.len(),
            limits.max_manifest_bytes,
            None,
            None,
        )?;
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate_with_limits(limits)?;
        Ok(manifest)
    }

    /// Validate the runtime compatibility contract.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] if this runtime cannot safely execute the
    /// declared bundle structure.
    pub fn validate(&self) -> Result<(), ManifestError> {
        self.validate_with_limits(&BundleLimits::default())
    }

    /// Validate the runtime compatibility contract and structural limits.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError`] if the runtime cannot execute the declared
    /// structure or a configured resource limit is exceeded.
    pub fn validate_with_limits(&self, limits: &BundleLimits) -> Result<(), ManifestError> {
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
        check_limit(
            BundleLimitResource::Components,
            self.pipeline.len(),
            limits.max_components,
            None,
            None,
        )?;

        let mut component_names = BTreeSet::new();
        let mut tensor_count = 0_usize;
        for component in &self.pipeline {
            if component.name.is_empty() || component.factory.is_empty() {
                return Err(ManifestError::EmptyComponentIdentity);
            }
            if !component_names.insert(component.name.as_str()) {
                return Err(ManifestError::DuplicateComponent(component.name.clone()));
            }
            check_limit(
                BundleLimitResource::ComponentNodes,
                component.nodes.len(),
                limits.max_nodes_per_component,
                Some(component.name.clone()),
                None,
            )?;

            let mut node_indices = BTreeSet::new();
            for node in &component.nodes {
                if !node_indices.insert(node.index) {
                    return Err(ManifestError::DuplicateNodeIndex {
                        component: component.name.clone(),
                        index: node.index,
                    });
                }
                tensor_count = tensor_count.checked_add(node.params.len()).ok_or_else(|| {
                    limit_error(
                        BundleLimitResource::Tensors,
                        u64::MAX,
                        limits.max_tensors,
                        Some(component.name.clone()),
                        None,
                    )
                })?;
                check_limit(
                    BundleLimitResource::Tensors,
                    tensor_count,
                    limits.max_tensors,
                    Some(component.name.clone()),
                    None,
                )?;
                for tensor in node.params.values() {
                    check_limit(
                        BundleLimitResource::TensorRank,
                        tensor.shape.len(),
                        limits.max_tensor_rank,
                        Some(component.name.clone()),
                        None,
                    )?;
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
        if let Some(vectors) = &self.vectors {
            for tensor in [&vectors.data, &vectors.keys, &vectors.rows] {
                tensor_count = tensor_count.checked_add(1).ok_or_else(|| {
                    limit_error(
                        BundleLimitResource::Tensors,
                        u64::MAX,
                        limits.max_tensors,
                        Some("vectors".to_owned()),
                        None,
                    )
                })?;
                check_limit(
                    BundleLimitResource::Tensors,
                    tensor_count,
                    limits.max_tensors,
                    Some("vectors".to_owned()),
                    None,
                )?;
                check_limit(
                    BundleLimitResource::TensorRank,
                    tensor.shape.len(),
                    limits.max_tensor_rank,
                    Some("vectors".to_owned()),
                    None,
                )?;
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
        Self::load_with_limits(root, BundleLimits::default())
    }

    /// Load a Python-free model bundle with caller-selected resource limits.
    ///
    /// # Errors
    ///
    /// Returns [`BundleError`] if validation fails, a required resource is
    /// missing, or a configured resource limit is exceeded.
    pub fn load_with_limits(
        root: impl AsRef<Path>,
        limits: BundleLimits,
    ) -> Result<Self, BundleError> {
        let root = root.as_ref();
        let manifest_path = root.join("manifest.json");
        let bytes = read_limited(
            &manifest_path,
            limits.max_manifest_bytes,
            BundleLimitResource::ManifestBytes,
            None,
        )
        .map_err(BundleError::from)?;
        let manifest = BundleManifest::from_json_with_limits(&bytes, &limits)?;

        let weights_path = root.join("weights.safetensors");
        if !weights_path.is_file() {
            return Err(BundleError::MissingFile(weights_path));
        }
        let weight_bytes = read_limited(
            &weights_path,
            limits.max_weights_bytes,
            BundleLimitResource::WeightsBytes,
            None,
        )
        .map_err(BundleError::from)?;
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
                check_path_limit(
                    &state_path,
                    limits.max_component_state_bytes,
                    BundleLimitResource::ComponentStateBytes,
                    Some(component.name.clone()),
                )
                .map_err(BundleError::from)?;
            }
        }

        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            weight_bytes,
            limits,
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

    #[must_use]
    pub fn limits(&self) -> &BundleLimits {
        &self.limits
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
        let bytes = read_limited(
            &tokenizer_path,
            self.limits.max_tokenizer_bytes,
            BundleLimitResource::TokenizerBytes,
            Some("tokenizer".to_owned()),
        )
        .map_err(RuntimeTokenizerError::from)?;
        match self.manifest.tokenizer.kind {
            TokenizerKind::Delarocha => {
                #[cfg(feature = "delarocha-tokenizer")]
                {
                    Ok(RuntimeTokenizer::Delarocha(Box::new(
                        DelarochaTokenizer::from_bundle_json(&self.root, &bytes)?,
                    )))
                }
                #[cfg(not(feature = "delarocha-tokenizer"))]
                {
                    Err(RuntimeTokenizerError::BackendDisabled("delarocha"))
                }
            }
            TokenizerKind::Regex => Ok(RuntimeTokenizer::Regex(Box::new(
                RegexTokenizer::from_json(&bytes)?,
            ))),
            TokenizerKind::Sudachi => {
                #[cfg(feature = "sudachi-tokenizer")]
                {
                    Ok(RuntimeTokenizer::Sudachi(
                        JapaneseTokenizer::from_bundle_json(&self.root, &bytes)?,
                    ))
                }
                #[cfg(not(feature = "sudachi-tokenizer"))]
                {
                    Err(RuntimeTokenizerError::BackendDisabled("sudachi"))
                }
            }
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

#[derive(Debug, Error)]
enum LimitedReadError {
    #[error("could not read model bundle file {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Limit(#[from] BundleLimitError),
}

impl From<LimitedReadError> for BundleError {
    fn from(error: LimitedReadError) -> Self {
        match error {
            LimitedReadError::Io { path, source } => Self::Io { path, source },
            LimitedReadError::Limit(error) => Self::Limit(error),
        }
    }
}

impl From<LimitedReadError> for RuntimeTokenizerError {
    fn from(error: LimitedReadError) -> Self {
        match error {
            LimitedReadError::Io { path, source } => Self::Io { path, source },
            LimitedReadError::Limit(error) => Self::Limit(error),
        }
    }
}

fn read_limited(
    path: &Path,
    limit: u64,
    resource: BundleLimitResource,
    component: Option<String>,
) -> Result<Vec<u8>, LimitedReadError> {
    check_path_limit(path, limit, resource, component.clone())?;
    let file = std::fs::File::open(path).map_err(|source| LimitedReadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| LimitedReadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    check_limit(
        resource,
        bytes.len(),
        limit,
        component,
        Some(path.to_path_buf()),
    )?;
    Ok(bytes)
}

fn check_path_limit(
    path: &Path,
    limit: u64,
    resource: BundleLimitResource,
    component: Option<String>,
) -> Result<(), LimitedReadError> {
    let metadata = std::fs::metadata(path).map_err(|source| LimitedReadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() > limit {
        return Err(limit_error(
            resource,
            metadata.len(),
            limit,
            component,
            Some(path.to_path_buf()),
        )
        .into());
    }
    Ok(())
}

fn check_limit(
    resource: BundleLimitResource,
    actual: usize,
    limit: impl TryInto<u64>,
    component: Option<String>,
    path: Option<PathBuf>,
) -> Result<(), BundleLimitError> {
    let actual = u64::try_from(actual).unwrap_or(u64::MAX);
    let limit = limit.try_into().unwrap_or(u64::MAX);
    if actual > limit {
        Err(limit_error(resource, actual, limit, component, path))
    } else {
        Ok(())
    }
}

fn limit_error(
    resource: BundleLimitResource,
    actual: u64,
    limit: impl TryInto<u64>,
    component: Option<String>,
    path: Option<PathBuf>,
) -> BundleLimitError {
    BundleLimitError {
        resource,
        actual,
        limit: limit.try_into().unwrap_or(u64::MAX),
        component,
        path,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{
        read_limited, BundleLimitResource, BundleLimits, BundleManifest, LimitedReadError,
        ManifestError, TokenizerKind,
    };

    static TEMPORARY_FILE_ID: AtomicU64 = AtomicU64::new(0);

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
    fn accepts_delarocha_tokenizer_manifests_in_every_build() {
        let input = MANIFEST.replace(r#""kind": "regex""#, r#""kind": "delarocha""#);
        let manifest = BundleManifest::from_json(input.as_bytes()).unwrap();
        assert_eq!(manifest.tokenizer.kind, TokenizerKind::Delarocha);
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

    #[test]
    fn rejects_manifest_bytes_over_the_configured_limit() {
        let limits = BundleLimits {
            max_manifest_bytes: u64::try_from(MANIFEST.len() - 1).unwrap(),
            ..BundleLimits::default()
        };
        assert!(matches!(
            BundleManifest::from_json_with_limits(MANIFEST.as_bytes(), &limits),
            Err(ManifestError::Limit(error))
                if error.resource == BundleLimitResource::ManifestBytes
                    && error.actual == u64::try_from(MANIFEST.len()).unwrap()
        ));
    }

    #[test]
    fn rejects_component_node_and_tensor_limits() {
        for limits in [
            BundleLimits {
                max_components: 0,
                ..BundleLimits::default()
            },
            BundleLimits {
                max_nodes_per_component: 0,
                ..BundleLimits::default()
            },
            BundleLimits {
                max_tensors: 0,
                ..BundleLimits::default()
            },
            BundleLimits {
                max_tensor_rank: 1,
                ..BundleLimits::default()
            },
        ] {
            assert!(matches!(
                BundleManifest::from_json_with_limits(MANIFEST.as_bytes(), &limits),
                Err(ManifestError::Limit(_))
            ));
        }
    }

    #[test]
    fn bounded_file_reader_rejects_before_reading_an_oversized_file() {
        let id = TEMPORARY_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "jewel-bounded-read-{}-{id}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [0_u8; 16]).unwrap();
        let result = read_limited(&path, 8, BundleLimitResource::WeightsBytes, None);
        std::fs::remove_file(&path).unwrap();

        assert!(matches!(
            result,
            Err(LimitedReadError::Limit(error))
                if error.resource == BundleLimitResource::WeightsBytes
                    && error.actual == 16
                    && error.limit == 8
                    && error.path.as_deref() == Some(path.as_path())
        ));
    }
}
