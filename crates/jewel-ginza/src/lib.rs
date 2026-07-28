//! GiNZA-specific model validation and entity label adaptation.
//!
//! The standard CNN model executes through `jewel-core`. Transformer-backed
//! GiNZA models use the optional `transformers` integration boundary.

use jewel_core::{Bundle, BundleManifest, NamedEntity, NerPipeline, PipelineError, TokenizerKind};
use thiserror::Error;

#[cfg(feature = "transformers")]
pub use jewel_transformers::{
    validate_token_vectors, TransformerEncoder, TransformerError, TransformerSpec,
};

/// GiNZA model architecture detected from an exported bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GinzaModelFamily {
    Standard,
    Electra,
}

/// Entity returned with both GiNZA ENE and coarse extraction labels.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GinzaEntity {
    pub entity: NamedEntity,
    pub coarse_label: Option<&'static str>,
}

impl GinzaEntity {
    #[must_use]
    pub fn ene_label(&self) -> &str {
        &self.entity.label
    }
}

/// Loaded standard GiNZA extraction pipeline.
pub struct GinzaPipeline {
    inner: NerPipeline,
}

/// GiNZA adapter validation or inference failure.
#[derive(Debug, Error)]
pub enum GinzaError {
    #[error("bundle model {model:?} is not a GiNZA model")]
    UnsupportedModel { model: String },
    #[error("GiNZA requires a Japanese bundle, got language {actual:?}")]
    Language { actual: String },
    #[error("GiNZA Electra requires the jewel-transformers execution path")]
    TransformersRequired,
    #[error("exact standard GiNZA compatibility requires the Sudachi tokenizer, got {actual:?}")]
    Tokenizer { actual: TokenizerKind },
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
}

impl GinzaPipeline {
    /// Load a standard GiNZA CNN bundle with its source-compatible tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error for non-GiNZA, non-Japanese, Electra, or non-Sudachi
    /// bundles, and for models unsupported by `jewel-core`.
    pub fn load(bundle: &Bundle) -> Result<Self, GinzaError> {
        if ginza_model_family(bundle.manifest())? == GinzaModelFamily::Electra {
            return Err(GinzaError::TransformersRequired);
        }
        if bundle.manifest().tokenizer.kind != TokenizerKind::Sudachi {
            return Err(GinzaError::Tokenizer {
                actual: bundle.manifest().tokenizer.kind,
            });
        }
        Ok(Self {
            inner: NerPipeline::load(bundle)?,
        })
    }

    /// Return the underlying language-aware Jewel pipeline.
    #[must_use]
    pub const fn core(&self) -> &NerPipeline {
        &self.inner
    }

    /// Extract raw ENE labels and their coarse extraction mappings.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<GinzaEntity>, GinzaError> {
        Ok(self
            .inner
            .extract_entities(text)?
            .into_iter()
            .map(|entity| GinzaEntity {
                coarse_label: coarse_label(&entity.label),
                entity,
            })
            .collect())
    }
}

/// Validate and classify a GiNZA bundle manifest.
///
/// # Errors
///
/// Returns an error when the manifest is not a supported GiNZA source.
pub fn ginza_model_family(manifest: &BundleManifest) -> Result<GinzaModelFamily, GinzaError> {
    if manifest.source.lang != "ja" {
        return Err(GinzaError::Language {
            actual: manifest.source.lang.clone(),
        });
    }
    let normalized_name = manifest.source.model_name.replace('-', "_");
    if !normalized_name.contains("ginza") {
        return Err(GinzaError::UnsupportedModel {
            model: manifest.source.model_name.clone(),
        });
    }
    let electra = normalized_name.contains("electra")
        || manifest
            .pipeline
            .iter()
            .any(|component| component.factory.contains("transformer"));
    Ok(if electra {
        GinzaModelFamily::Electra
    } else {
        GinzaModelFamily::Standard
    })
}

/// Map GiNZA's extraction-relevant ENE labels to coarse labels.
#[must_use]
pub fn coarse_label(label: &str) -> Option<&'static str> {
    match label {
        "Person" | "God" => Some("PERSON"),
        "Organization"
        | "Organization_Other"
        | "International_Organization"
        | "Political_Organization"
        | "Political_Organization_Other"
        | "Political_Party"
        | "Juridical_Person"
        | "Juridical_Person_Other"
        | "Corporation_Other"
        | "Nonprofit_Organization"
        | "Company"
        | "Company_Group"
        | "Government" => Some("ORG"),
        "GPE" | "GPE_Other" | "City" | "Province" | "Country" => Some("GPE"),
        "Address" | "Address_Other" | "Postal_Address" => Some("ADDRESS"),
        "Title" | "Title_Other" | "Position_Vocation" => Some("TITLE"),
        "Currency" | "Money" => Some("MONEY"),
        "Phone_Number" => Some("PHONE"),
        "Email" => Some("EMAIL"),
        "URL" => Some("URL"),
        "Date" | "Period_Day" | "Period_Week" | "Period_Month" | "Period_Year" => Some("DATE"),
        "Time" | "Period_Time" => Some("TIME"),
        "Percent" => Some("PERCENT"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use jewel_core::{
        BundleManifest, ComponentKind, ComponentManifest, RuntimeManifest, SourceManifest,
        TokenizerKind, TokenizerManifest,
    };

    use super::{coarse_label, ginza_model_family, GinzaError, GinzaModelFamily};

    fn manifest(model_name: &str, lang: &str, factory: &str) -> BundleManifest {
        BundleManifest {
            format_version: 1,
            source: SourceManifest {
                spacy_version: "3.7.0".to_owned(),
                model_name: model_name.to_owned(),
                model_version: "5.2.0".to_owned(),
                lang: lang.to_owned(),
            },
            runtime: RuntimeManifest {
                min_runtime_version: "0.0.4".to_owned(),
                requires_python: false,
            },
            tokenizer: TokenizerManifest {
                kind: TokenizerKind::Sudachi,
                path: "tokenizer.json".to_owned(),
            },
            vectors: None,
            pipeline: vec![ComponentManifest {
                name: factory.to_owned(),
                factory: factory.to_owned(),
                kind: ComponentKind::Trainable,
                root_node: None,
                settings: BTreeMap::new(),
                nodes: Vec::new(),
                state_path: None,
                labels: Vec::new(),
                moves: Vec::new(),
            }],
        }
    }

    #[test]
    fn maps_contract_extraction_labels() {
        assert_eq!(coarse_label("Person"), Some("PERSON"));
        assert_eq!(coarse_label("Company"), Some("ORG"));
        assert_eq!(coarse_label("Money"), Some("MONEY"));
        assert_eq!(coarse_label("Postal_Address"), Some("ADDRESS"));
        assert_eq!(coarse_label("City"), Some("GPE"));
        assert_eq!(coarse_label("Position_Vocation"), Some("TITLE"));
        assert_eq!(coarse_label("Email"), Some("EMAIL"));
        assert_eq!(coarse_label("Period_Time"), Some("TIME"));
        assert_eq!(coarse_label("Unknown_Category"), None);
    }

    #[test]
    fn distinguishes_standard_and_electra_models() {
        assert_eq!(
            ginza_model_family(&manifest("ginza", "ja", "tok2vec")).unwrap(),
            GinzaModelFamily::Standard
        );
        assert_eq!(
            ginza_model_family(&manifest("ginza_electra", "ja", "transformer_custom")).unwrap(),
            GinzaModelFamily::Electra
        );
    }

    #[test]
    fn rejects_non_ginza_and_non_japanese_manifests() {
        assert!(matches!(
            ginza_model_family(&manifest("core_web_sm", "ja", "tok2vec")),
            Err(GinzaError::UnsupportedModel { .. })
        ));
        assert!(matches!(
            ginza_model_family(&manifest("ginza", "en", "tok2vec")),
            Err(GinzaError::Language { .. })
        ));
    }
}
