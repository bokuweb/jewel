use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    Bundle, BundleError, BundleLimitError, BundleLimitResource, DependencyParserError,
    EntityRecognizerError, ManifestError, ModelOpError, NerPipeline, PipelineError,
    RuntimeTokenizerError, SourceManifest, Tok2VecError, TransitionScorerError,
};

/// Schema version of [`NerCompatibilityReport`].
pub const COMPATIBILITY_REPORT_VERSION: u32 = 1;

/// The part of a spaCy model bundle that caused a compatibility failure.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityArea {
    Manifest,
    File,
    Component,
    GraphNode,
    Attribute,
    Tensor,
    Tokenizer,
    TokenizerFeature,
    Language,
}

/// A stable, machine-readable description of one incompatibility.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompatibilityDiagnostic {
    /// Stable identifier suitable for programmatic branching.
    pub code: String,
    pub area: CompatibilityArea,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    pub message: String,
}

/// Result of loading a bundle and constructing Jewel's extraction pipeline.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NerCompatibilityReport {
    pub report_version: u32,
    pub compatible: bool,
    pub bundle_path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceManifest>,
    pub diagnostics: Vec<CompatibilityDiagnostic>,
}

impl CompatibilityDiagnostic {
    /// Convert bundle loading failure details to a stable diagnostic.
    #[must_use]
    pub fn from_bundle_error(error: &BundleError) -> Self {
        match error {
            BundleError::Io { path, .. } => Self::new("bundle_io", CompatibilityArea::File, error)
                .with_item(path.display().to_string()),
            BundleError::Manifest(error) => Self::from_manifest_error(error),
            BundleError::MissingFile(path) => {
                Self::new("missing_file", CompatibilityArea::File, error)
                    .with_item(path.display().to_string())
            }
            BundleError::Weights(_) => {
                Self::new("invalid_weights", CompatibilityArea::Tensor, error)
            }
            BundleError::MissingTensor(key) => {
                Self::new("missing_tensor", CompatibilityArea::Tensor, error).with_tensor_key(key)
            }
            BundleError::DuplicateTensor(key) => {
                Self::new("duplicate_tensor", CompatibilityArea::Tensor, error).with_tensor_key(key)
            }
            BundleError::TensorShape { key, .. } => {
                Self::new("tensor_shape", CompatibilityArea::Tensor, error).with_tensor_key(key)
            }
            BundleError::TensorDtype { key, .. } => {
                Self::new("tensor_dtype", CompatibilityArea::Tensor, error).with_tensor_key(key)
            }
            BundleError::InvalidF32Data { key, .. } | BundleError::InvalidU64Data { key, .. } => {
                Self::new("invalid_tensor_data", CompatibilityArea::Tensor, error)
                    .with_tensor_key(key)
            }
            BundleError::Limit(error) => Self::from_bundle_limit(error),
        }
    }

    /// Convert extraction pipeline loading failure details to a stable diagnostic.
    #[must_use]
    pub fn from_pipeline_error(error: &PipelineError) -> Self {
        match error {
            PipelineError::Tokenizer(error) => Self::from_tokenizer_error(error),
            PipelineError::Tokenization(_) => {
                Self::new("tokenization_failed", CompatibilityArea::Tokenizer, error)
            }
            PipelineError::Tok2Vec(error) => Self::from_tok2vec_error(error),
            PipelineError::Tagger(error) => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("tagger")
            }
            PipelineError::Parser(error) => Self::from_parser_error(error),
            PipelineError::Ner(error) => Self::from_ner_error(error),
            PipelineError::Sentencizer(crate::SentencizerError::MissingComponent(component)) => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
            PipelineError::Sentencizer(crate::SentencizerError::InvalidSetting { name }) => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("sentencizer")
                    .with_item((*name).to_owned())
            }
            PipelineError::SentenceRecognizer(error) => Self::from_sentence_recognizer_error(error),
            PipelineError::MultipleSentenceBoundaryComponents => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("sentence_boundary")
            }
            PipelineError::MultipleComponents { factory, .. } => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component(*factory)
            }
            PipelineError::MissingRequiredComponent { factory } => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(*factory)
            }
            PipelineError::InvalidUpstreamTok2Vec {
                component,
                upstream,
            } => Self::new(
                "missing_upstream_tok2vec",
                CompatibilityArea::Component,
                error,
            )
            .with_component(component.clone())
            .with_item(upstream.clone()),
            PipelineError::ConflictingUpstreamTok2Vec { .. } => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("tok2vec")
            }
            PipelineError::Language { actual, .. }
            | PipelineError::UnsupportedLanguage { actual } => {
                Self::new("unsupported_language", CompatibilityArea::Language, error)
                    .with_item(actual.clone())
            }
        }
    }

    fn from_manifest_error(error: &ManifestError) -> Self {
        match error {
            ManifestError::UnsupportedFormat { actual, .. } => Self::new(
                "unsupported_bundle_format",
                CompatibilityArea::Manifest,
                error,
            )
            .with_item(actual.to_string()),
            ManifestError::RequiresPython => {
                Self::new("python_required", CompatibilityArea::Manifest, error)
            }
            ManifestError::DuplicateComponent(component)
            | ManifestError::MissingRootNode(component) => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
            ManifestError::EmptyComponentIdentity => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
            }
            ManifestError::DuplicateNodeIndex { component, index } => {
                Self::new("duplicate_graph_node", CompatibilityArea::GraphNode, error)
                    .with_component(component.clone())
                    .with_node(*index)
            }
            ManifestError::MissingChild {
                component, node, ..
            } => Self::new("missing_graph_node", CompatibilityArea::GraphNode, error)
                .with_component(component.clone())
                .with_node(*node),
            ManifestError::UnsafeStatePath { component, path } => {
                Self::new("unsafe_bundle_path", CompatibilityArea::File, error)
                    .with_component(component.clone())
                    .with_item(path.clone())
            }
            ManifestError::Json(_) => {
                Self::new("invalid_manifest_json", CompatibilityArea::Manifest, error)
            }
            ManifestError::Limit(error) => Self::from_bundle_limit(error),
        }
    }

    fn from_tokenizer_error(error: &RuntimeTokenizerError) -> Self {
        match error {
            RuntimeTokenizerError::BackendDisabled(backend) => Self::new(
                "tokenizer_backend_disabled",
                CompatibilityArea::Tokenizer,
                error,
            )
            .with_item(*backend),
            #[cfg(feature = "delarocha-tokenizer")]
            RuntimeTokenizerError::Delarocha(error) => {
                use crate::DelarochaTokenizerError;

                let mut diagnostic = match error {
                    DelarochaTokenizerError::UnsupportedFeature(feature) => Self::new(
                        "unsupported_tokenizer_feature",
                        CompatibilityArea::TokenizerFeature,
                        error,
                    )
                    .with_item(feature.clone()),
                    DelarochaTokenizerError::MissingPos(pos) => Self::new(
                        "missing_tokenizer_pos",
                        CompatibilityArea::TokenizerFeature,
                        error,
                    )
                    .with_item(pos.clone()),
                    DelarochaTokenizerError::UnsupportedFormat { actual, .. } => Self::new(
                        "unsupported_tokenizer_format",
                        CompatibilityArea::Tokenizer,
                        error,
                    )
                    .with_item(actual.to_string()),
                    _ => Self::new(
                        "invalid_tokenizer_configuration",
                        CompatibilityArea::Tokenizer,
                        error,
                    ),
                };
                diagnostic.component = Some("tokenizer".to_owned());
                diagnostic
            }
            #[cfg(feature = "sudachi-tokenizer")]
            RuntimeTokenizerError::Japanese(error) => {
                use crate::JapaneseTokenizerError;

                let mut diagnostic = match error {
                    JapaneseTokenizerError::MissingPos(pos) => Self::new(
                        "missing_tokenizer_pos",
                        CompatibilityArea::TokenizerFeature,
                        error,
                    )
                    .with_item(pos.clone()),
                    JapaneseTokenizerError::UnsupportedFormat { actual, .. } => Self::new(
                        "unsupported_tokenizer_format",
                        CompatibilityArea::Tokenizer,
                        error,
                    )
                    .with_item(actual.to_string()),
                    _ => Self::new(
                        "invalid_tokenizer_configuration",
                        CompatibilityArea::Tokenizer,
                        error,
                    ),
                };
                diagnostic.component = Some("tokenizer".to_owned());
                diagnostic
            }
            RuntimeTokenizerError::Io { path, .. } => {
                Self::new("tokenizer_io", CompatibilityArea::File, error)
                    .with_component("tokenizer")
                    .with_item(path.display().to_string())
            }
            RuntimeTokenizerError::Regex(error) => {
                use crate::TokenizerError;

                match error {
                    TokenizerError::UnsupportedFormat { actual, .. } => Self::new(
                        "unsupported_tokenizer_format",
                        CompatibilityArea::Tokenizer,
                        error,
                    )
                    .with_component("tokenizer")
                    .with_item(actual.to_string()),
                    _ => Self::new(
                        "invalid_tokenizer_configuration",
                        CompatibilityArea::Tokenizer,
                        error,
                    )
                    .with_component("tokenizer"),
                }
            }
            RuntimeTokenizerError::Limit(error) => Self::from_bundle_limit(error),
        }
    }

    fn from_bundle_limit(error: &BundleLimitError) -> Self {
        let area = match error.resource {
            BundleLimitResource::ManifestBytes => CompatibilityArea::Manifest,
            BundleLimitResource::WeightsBytes
            | BundleLimitResource::Tensors
            | BundleLimitResource::TensorRank => CompatibilityArea::Tensor,
            BundleLimitResource::TokenizerBytes => CompatibilityArea::Tokenizer,
            BundleLimitResource::ComponentStateBytes => CompatibilityArea::File,
            BundleLimitResource::Components => CompatibilityArea::Component,
            BundleLimitResource::ComponentNodes => CompatibilityArea::GraphNode,
        };
        let mut diagnostic =
            Self::new("bundle_limit_exceeded", area, error).with_item(error.resource.as_str());
        diagnostic.component.clone_from(&error.component);
        if let Some(path) = &error.path {
            diagnostic.message = format!("{} ({})", diagnostic.message, path.display());
        }
        diagnostic
    }

    fn from_tok2vec_error(error: &Tok2VecError) -> Self {
        match error {
            Tok2VecError::Model(error) => Self::from_model_error(error, "tok2vec"),
            Tok2VecError::Kernel(_) | Tok2VecError::InvalidGraph(_) => {
                Self::new("unsupported_graph", CompatibilityArea::GraphNode, error)
                    .with_component("tok2vec")
            }
            Tok2VecError::UnsupportedFeatureColumn(column) => Self::new(
                "unsupported_tok2vec_feature",
                CompatibilityArea::Attribute,
                error,
            )
            .with_component("tok2vec")
            .with_item(column.clone()),
            Tok2VecError::MissingComponent(component) => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
        }
    }

    fn from_parser_error(error: &DependencyParserError) -> Self {
        match error {
            DependencyParserError::Scorer(error) => Self::from_scorer_error(error, "parser"),
            DependencyParserError::MissingComponent(component) => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
            DependencyParserError::InvalidModel(_) | DependencyParserError::InvalidMove(_) => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("parser")
            }
            DependencyParserError::NoValidMove { .. }
            | DependencyParserError::StepLimit { .. }
            | DependencyParserError::HeadOverflow { .. } => Self::new(
                "parser_execution_failed",
                CompatibilityArea::Component,
                error,
            )
            .with_component("parser"),
        }
    }

    fn from_ner_error(error: &EntityRecognizerError) -> Self {
        match error {
            EntityRecognizerError::Tok2Vec(error) => Self::from_tok2vec_error(error),
            EntityRecognizerError::Scorer(error) => Self::from_scorer_error(error, "ner"),
            EntityRecognizerError::MissingComponent(component) => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
            EntityRecognizerError::InvalidModel(_) | EntityRecognizerError::InvalidMove(_) => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("ner")
            }
            EntityRecognizerError::ExternalTok2VecRequired => Self::new(
                "missing_upstream_tok2vec",
                CompatibilityArea::Component,
                error,
            )
            .with_component("ner"),
            EntityRecognizerError::NoValidMove { .. } => {
                Self::new("ner_execution_failed", CompatibilityArea::Component, error)
                    .with_component("ner")
            }
        }
    }

    fn from_sentence_recognizer_error(error: &crate::SentenceRecognizerError) -> Self {
        match error {
            crate::SentenceRecognizerError::Tok2Vec(error) => Self::from_tok2vec_error(error),
            crate::SentenceRecognizerError::Classifier(crate::TaggerError::Model(error)) => {
                Self::from_model_error(error, "senter")
            }
            crate::SentenceRecognizerError::Classifier(_)
            | crate::SentenceRecognizerError::InvalidLabels(_) => {
                Self::new("invalid_component", CompatibilityArea::Component, error)
                    .with_component("senter")
            }
            crate::SentenceRecognizerError::MissingComponent(component) => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
            crate::SentenceRecognizerError::InvalidSetting { name } => {
                let mut diagnostic =
                    Self::new("invalid_component", CompatibilityArea::Component, error)
                        .with_component("senter");
                diagnostic.item = Some((*name).to_owned());
                diagnostic
            }
            crate::SentenceRecognizerError::ExternalTok2VecRequired => Self::new(
                "missing_upstream_tok2vec",
                CompatibilityArea::Component,
                error,
            )
            .with_component("senter"),
        }
    }

    fn from_scorer_error(error: &TransitionScorerError, component: &str) -> Self {
        match error {
            TransitionScorerError::Model(error) => Self::from_model_error(error, component),
            TransitionScorerError::MissingComponent(component) => {
                Self::new("missing_component", CompatibilityArea::Component, error)
                    .with_component(component.clone())
            }
            TransitionScorerError::InvalidGraph(_) => {
                Self::new("unsupported_graph", CompatibilityArea::GraphNode, error)
                    .with_component(component)
            }
        }
    }

    fn from_model_error(error: &ModelOpError, component: &str) -> Self {
        match error {
            ModelOpError::Bundle(error) => {
                let mut diagnostic = Self::from_bundle_error(error);
                diagnostic.component = Some(component.to_owned());
                diagnostic
            }
            ModelOpError::Kernel(_) => {
                Self::new("invalid_graph_shape", CompatibilityArea::GraphNode, error)
                    .with_component(component)
            }
            ModelOpError::InvalidNode {
                node, operation, ..
            } => Self::new(
                "unsupported_graph_node",
                CompatibilityArea::GraphNode,
                error,
            )
            .with_component(component)
            .with_node(*node)
            .with_item(*operation),
            ModelOpError::MissingDimension { node, name } => Self::new(
                "missing_graph_dimension",
                CompatibilityArea::Attribute,
                error,
            )
            .with_component(component)
            .with_node(*node)
            .with_item(name.clone()),
            ModelOpError::InvalidAttribute { node, name, .. } => Self::new(
                "invalid_graph_attribute",
                CompatibilityArea::Attribute,
                error,
            )
            .with_component(component)
            .with_node(*node)
            .with_item(name.clone()),
            ModelOpError::InvalidParameter { node, name, .. } => {
                Self::new("invalid_graph_parameter", CompatibilityArea::Tensor, error)
                    .with_component(component)
                    .with_node(*node)
                    .with_item(name.clone())
            }
        }
    }

    fn new(
        code: impl Into<String>,
        area: CompatibilityArea,
        error: &impl std::fmt::Display,
    ) -> Self {
        Self {
            code: code.into(),
            area,
            component: None,
            node: None,
            item: None,
            message: error.to_string(),
        }
    }

    fn with_component(mut self, component: impl Into<String>) -> Self {
        self.component = Some(component.into());
        self
    }

    fn with_node(mut self, node: usize) -> Self {
        self.node = Some(node);
        self
    }

    fn with_item(mut self, item: impl Into<String>) -> Self {
        self.item = Some(item.into());
        self
    }

    fn with_tensor_key(mut self, key: &str) -> Self {
        self.item = Some(key.to_owned());
        self.component = key
            .strip_prefix("components.")
            .and_then(|suffix| suffix.split_once('.'))
            .map(|(component, _)| component.to_owned());
        self
    }
}

impl NerCompatibilityReport {
    /// Load a bundle and construct the language-aware extraction pipeline.
    ///
    /// This function always returns a report. Compatibility failures are
    /// represented by stable diagnostics rather than returned as Rust errors.
    #[must_use]
    pub fn inspect(root: impl AsRef<Path>) -> Self {
        let bundle_path = root.as_ref().to_path_buf();
        let bundle = match Bundle::load(&bundle_path) {
            Ok(bundle) => bundle,
            Err(error) => {
                return Self {
                    report_version: COMPATIBILITY_REPORT_VERSION,
                    compatible: false,
                    bundle_path,
                    source: None,
                    diagnostics: vec![CompatibilityDiagnostic::from_bundle_error(&error)],
                };
            }
        };
        let source = Some(bundle.manifest().source.clone());
        match NerPipeline::load(&bundle) {
            Ok(_) => Self {
                report_version: COMPATIBILITY_REPORT_VERSION,
                compatible: true,
                bundle_path,
                source,
                diagnostics: Vec::new(),
            },
            Err(error) => Self {
                report_version: COMPATIBILITY_REPORT_VERSION,
                compatible: false,
                bundle_path,
                source,
                diagnostics: vec![CompatibilityDiagnostic::from_pipeline_error(&error)],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CompatibilityArea, CompatibilityDiagnostic, COMPATIBILITY_REPORT_VERSION};
    use crate::{
        BundleError, BundleLimitError, BundleLimitResource, ModelOpError, PipelineError,
        SentencizerError, Tok2VecError,
    };

    #[test]
    fn tensor_diagnostic_preserves_the_tensor_key() {
        let error = BundleError::TensorShape {
            key: "components.ner.nodes.4.W".to_owned(),
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        let diagnostic = CompatibilityDiagnostic::from_bundle_error(&error);
        assert_eq!(diagnostic.code, "tensor_shape");
        assert_eq!(diagnostic.area, CompatibilityArea::Tensor);
        assert_eq!(diagnostic.component.as_deref(), Some("ner"));
        assert_eq!(diagnostic.item.as_deref(), Some("components.ner.nodes.4.W"));
    }

    #[test]
    fn graph_diagnostic_preserves_component_node_and_operation() {
        let error = PipelineError::Tok2Vec(Tok2VecError::Model(ModelOpError::InvalidNode {
            node: 12,
            operation: "maxout",
            message: "missing dimension nO".to_owned(),
        }));
        let diagnostic = CompatibilityDiagnostic::from_pipeline_error(&error);
        assert_eq!(diagnostic.code, "unsupported_graph_node");
        assert_eq!(diagnostic.area, CompatibilityArea::GraphNode);
        assert_eq!(diagnostic.component.as_deref(), Some("tok2vec"));
        assert_eq!(diagnostic.node, Some(12));
        assert_eq!(diagnostic.item.as_deref(), Some("maxout"));
    }

    #[test]
    fn attribute_diagnostic_preserves_attribute_name() {
        let error = PipelineError::Tok2Vec(Tok2VecError::Model(ModelOpError::InvalidAttribute {
            node: 8,
            name: "seed".to_owned(),
            message: "expected an unsigned integer".to_owned(),
        }));
        let diagnostic = CompatibilityDiagnostic::from_pipeline_error(&error);
        assert_eq!(diagnostic.code, "invalid_graph_attribute");
        assert_eq!(diagnostic.area, CompatibilityArea::Attribute);
        assert_eq!(diagnostic.component.as_deref(), Some("tok2vec"));
        assert_eq!(diagnostic.node, Some(8));
        assert_eq!(diagnostic.item.as_deref(), Some("seed"));
    }

    #[test]
    fn unsupported_tok2vec_feature_has_a_stable_diagnostic() {
        let error =
            PipelineError::Tok2Vec(Tok2VecError::UnsupportedFeatureColumn("LEMMA".to_owned()));
        let diagnostic = CompatibilityDiagnostic::from_pipeline_error(&error);
        assert_eq!(diagnostic.code, "unsupported_tok2vec_feature");
        assert_eq!(diagnostic.area, CompatibilityArea::Attribute);
        assert_eq!(diagnostic.component.as_deref(), Some("tok2vec"));
        assert_eq!(diagnostic.item.as_deref(), Some("LEMMA"));
    }

    #[test]
    fn invalid_sentencizer_setting_has_a_stable_diagnostic() {
        let error = PipelineError::Sentencizer(SentencizerError::InvalidSetting {
            name: "punct_chars",
        });
        let diagnostic = CompatibilityDiagnostic::from_pipeline_error(&error);
        assert_eq!(diagnostic.code, "invalid_component");
        assert_eq!(diagnostic.area, CompatibilityArea::Component);
        assert_eq!(diagnostic.component.as_deref(), Some("sentencizer"));
        assert_eq!(diagnostic.item.as_deref(), Some("punct_chars"));
    }

    #[test]
    fn invalid_sentence_recognizer_setting_has_a_stable_diagnostic() {
        let error =
            PipelineError::SentenceRecognizer(crate::SentenceRecognizerError::InvalidSetting {
                name: "overwrite",
            });
        let diagnostic = CompatibilityDiagnostic::from_pipeline_error(&error);
        assert_eq!(diagnostic.code, "invalid_component");
        assert_eq!(diagnostic.area, CompatibilityArea::Component);
        assert_eq!(diagnostic.component.as_deref(), Some("senter"));
        assert_eq!(diagnostic.item.as_deref(), Some("overwrite"));
    }

    #[test]
    fn invalid_named_upstream_has_a_stable_diagnostic() {
        let error = PipelineError::InvalidUpstreamTok2Vec {
            component: "entities".to_owned(),
            upstream: "encoder".to_owned(),
        };
        let diagnostic = CompatibilityDiagnostic::from_pipeline_error(&error);
        assert_eq!(diagnostic.code, "missing_upstream_tok2vec");
        assert_eq!(diagnostic.area, CompatibilityArea::Component);
        assert_eq!(diagnostic.component.as_deref(), Some("entities"));
        assert_eq!(diagnostic.item.as_deref(), Some("encoder"));
    }

    #[test]
    fn limit_diagnostic_preserves_resource_and_component() {
        let error = BundleError::Limit(BundleLimitError {
            resource: BundleLimitResource::ComponentNodes,
            actual: 33,
            limit: 32,
            component: Some("ner".to_owned()),
            path: None,
        });
        let diagnostic = CompatibilityDiagnostic::from_bundle_error(&error);
        assert_eq!(diagnostic.code, "bundle_limit_exceeded");
        assert_eq!(diagnostic.area, CompatibilityArea::GraphNode);
        assert_eq!(diagnostic.component.as_deref(), Some("ner"));
        assert_eq!(diagnostic.item.as_deref(), Some("component_nodes"));
    }

    #[test]
    fn report_schema_version_is_stable() {
        assert_eq!(COMPATIBILITY_REPORT_VERSION, 1);
    }
}
