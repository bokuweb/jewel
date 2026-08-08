//! GiNZA-specific model validation and entity label adaptation.
//!
//! The standard CNN model executes through `jewel-core`. Transformer-backed
//! GiNZA models use the optional `transformers` integration boundary.

use std::collections::BTreeMap;

use jewel_core::{
    Bundle, BundleManifest, Doc, EntityConstraint, EntityLabelFilter, EntityLabelSelection,
    EntityRecognizerError, NamedEntity, NerBatchInput, NerPipeline, PipelineError, TokenizerKind,
};
use thiserror::Error;

#[cfg(feature = "transformers")]
use jewel_core::{
    apply_entity_constraints, ComponentManifest, DependencyParser, DependencyParserError,
    EntityRecognizer, EntityRuler, EntityRulerError, Matrix, RuntimeTokenizer,
    RuntimeTokenizerError, SentenceRecognizer, Sentencizer, Tokenizer,
};
#[cfg(feature = "transformers")]
pub use jewel_transformers::{
    validate_token_vectors, CandleElectraEncoder, TransformerEncoder, TransformerError,
    TransformerSpec,
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
    coarse_labels: CoarseLabelMap,
}

/// Loaded GiNZA Electra pipeline with a caller-selected transformer backend.
#[cfg(feature = "transformers")]
pub struct GinzaElectraPipeline<E> {
    tokenizer: RuntimeTokenizer,
    encoder: E,
    parser: Option<DependencyParser>,
    sentence_recognizer: Option<SentenceRecognizer>,
    sentencizer: Option<Sentencizer>,
    ner: EntityRecognizer,
    pre_ner_entity_rulers: Vec<EntityRuler>,
    post_ner_entity_rulers: Vec<EntityRuler>,
    spec: TransformerSpec,
    coarse_labels: CoarseLabelMap,
}

type CoarseLabelMap = BTreeMap<String, &'static str>;

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
    #[cfg(feature = "transformers")]
    #[error(transparent)]
    Transformer(#[from] TransformerError),
    #[cfg(feature = "transformers")]
    #[error(transparent)]
    RuntimeTokenizer(#[from] RuntimeTokenizerError),
    #[cfg(feature = "transformers")]
    #[error(transparent)]
    Parser(#[from] DependencyParserError),
    #[cfg(feature = "transformers")]
    #[error(transparent)]
    EntityRuler(#[from] EntityRulerError),
    #[error(transparent)]
    Ner(#[from] EntityRecognizerError),
    #[cfg(feature = "transformers")]
    #[error("GiNZA Electra bundle requires exactly one NER component")]
    ElectraNerComponent,
    #[cfg(feature = "transformers")]
    #[error("GiNZA Electra bundle contains multiple parser components")]
    ElectraParserComponents,
    #[cfg(feature = "transformers")]
    #[error("GiNZA Electra component {component:?} has invalid transformer upstream {upstream:?}")]
    InvalidElectraTransformerUpstream { component: String, upstream: String },
    #[cfg(feature = "transformers")]
    #[error(
        "GiNZA Electra component {component:?} listens to transformer {actual:?}, expected {expected:?}"
    )]
    ElectraTransformerUpstream {
        component: String,
        expected: String,
        actual: String,
    },
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
            coarse_labels: exported_coarse_labels(bundle.manifest()),
        })
    }

    /// Return the underlying language-aware Jewel pipeline.
    #[must_use]
    pub const fn core(&self) -> &NerPipeline {
        &self.inner
    }

    /// Return whether the standard pipeline includes dependency parsing.
    #[must_use]
    pub const fn has_dependency_parser(&self) -> bool {
        self.inner.has_dependency_parser()
    }

    /// Return whether the standard pipeline includes a rule-based sentencizer.
    #[must_use]
    pub const fn has_sentencizer(&self) -> bool {
        self.inner.has_sentencizer()
    }

    /// Return whether the standard pipeline includes a trainable senter.
    #[must_use]
    pub const fn has_sentence_recognizer(&self) -> bool {
        self.inner.has_sentence_recognizer()
    }

    /// Return whether the standard pipeline includes an entity ruler.
    #[must_use]
    pub fn has_entity_ruler(&self) -> bool {
        self.inner.has_entity_ruler()
    }

    /// Return labels declared by the statistical model or entity rulers.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.inner.supported_entity_labels()
    }

    /// Return whether the model or an entity ruler declares a label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        self.inner.supports_entity_label(label)
    }

    /// Compile requested labels against the model and entity ruler labels.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        self.inner.select_entity_labels(labels)
    }

    /// Return GiNZA entities already attached to a processed document.
    #[must_use]
    pub fn entities(&self, doc: &Doc) -> Vec<GinzaEntity> {
        adapt_ginza_entities(self.inner.entities(doc), &self.coarse_labels)
    }

    /// Return attached GiNZA entities whose ENE labels are in `labels`.
    #[must_use]
    pub fn entities_by_labels(&self, doc: &Doc, labels: &[&str]) -> Vec<GinzaEntity> {
        self.entities_with_filter(doc, &EntityLabelFilter::new(labels))
    }

    /// Return attached GiNZA entities accepted by a reusable ENE filter.
    #[must_use]
    pub fn entities_with_filter(&self, doc: &Doc, filter: &EntityLabelFilter) -> Vec<GinzaEntity> {
        adapt_ginza_entities(
            self.inner.entities_with_filter(doc, filter),
            &self.coarse_labels,
        )
    }

    /// Return attached GiNZA entities with one ENE label.
    #[must_use]
    pub fn entities_by_label(&self, doc: &Doc, label: &str) -> Vec<GinzaEntity> {
        self.entities_by_labels(doc, &[label])
    }

    /// Extract raw ENE labels and their coarse extraction mappings.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<GinzaEntity>, GinzaError> {
        Ok(adapt_ginza_entities(
            self.inner.extract_entities(text)?,
            &self.coarse_labels,
        ))
    }

    /// Extract only entities whose raw ENE labels are included in `labels`.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities_by_labels(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<GinzaEntity>, GinzaError> {
        self.extract_entities_with_filter(text, &EntityLabelFilter::new(labels))
    }

    /// Extract entities accepted by a reusable raw ENE label filter.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities_with_filter(
        &self,
        text: &str,
        filter: &EntityLabelFilter,
    ) -> Result<Vec<GinzaEntity>, GinzaError> {
        Ok(adapt_ginza_entities(
            self.inner.extract_entities_with_filter(text, filter)?,
            &self.coarse_labels,
        ))
    }

    /// Extract entities using GiNZA's exported ENE-to-OntoNotes mapping.
    ///
    /// ENE labels introduced by a ruler map to `OTHERS`.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the bundle has no mapping.
    pub fn extract_entities_ontonotes(&self, text: &str) -> Result<Vec<NamedEntity>, GinzaError> {
        let doc = self.inner.process(text)?;
        self.entities_ontonotes(&doc)
    }

    /// Extract OntoNotes-mapped entities while enforcing NER constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints, failed inference, or a
    /// missing mapping.
    pub fn extract_entities_ontonotes_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Vec<NamedEntity>, GinzaError> {
        let doc = self.process_with_constraints(text, constraints)?;
        self.entities_ontonotes(&doc)
    }

    /// Map entities already attached to a document to OntoNotes labels.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle has no exported mapping.
    pub fn entities_ontonotes(&self, doc: &Doc) -> Result<Vec<NamedEntity>, GinzaError> {
        Ok(self
            .inner
            .entities_with_mapping_or(doc, "ontonotes", "OTHERS")?)
    }

    /// Return token-aligned B/I/O labels using GiNZA's OntoNotes mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the bundle has no mapping.
    pub fn token_labels_ontonotes(&self, text: &str) -> Result<Vec<String>, GinzaError> {
        let doc = self.inner.process(text)?;
        Ok(self
            .inner
            .token_labels_with_mapping_or(&doc, "ontonotes", "OTHERS")?)
    }

    /// Return constrained token-aligned B/I/O labels using OntoNotes mapping.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints, failed inference, or a
    /// missing mapping.
    pub fn token_labels_ontonotes_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Vec<String>, GinzaError> {
        let doc = self.process_with_constraints(text, constraints)?;
        Ok(self
            .inner
            .token_labels_with_mapping_or(&doc, "ontonotes", "OTHERS")?)
    }

    /// Tokenize text and attach standard GiNZA sentence and entity annotations.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, GinzaError> {
        Ok(self.inner.process(text)?)
    }

    /// Run standard GiNZA NER with spaCy-compatible preset entity, blocked,
    /// missing, and outside constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed inference.
    pub fn process_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Doc, GinzaError> {
        Ok(self.inner.process_with_constraints(text, constraints)?)
    }

    /// Extract standard GiNZA entities while enforcing NER constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed inference.
    pub fn extract_entities_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Vec<GinzaEntity>, GinzaError> {
        let doc = self.process_with_constraints(text, constraints)?;
        Ok(adapt_ginza_entities(
            self.inner.entities(&doc),
            &self.coarse_labels,
        ))
    }

    /// Process a standard GiNZA document batch with one tokenizer session.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn process_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Doc>, GinzaError> {
        Ok(self.inner.process_batch(texts)?)
    }

    /// Process standard GiNZA texts with document-specific preset constraints.
    ///
    /// Token-index and Unicode character constraints may be mixed between
    /// documents.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn process_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Doc>, GinzaError> {
        Ok(self.inner.process_batch_with_constraints(inputs)?)
    }

    /// Extract raw ENE labels from a standard GiNZA document batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(adapt_ginza_batches(
            self.inner.extract_entities_batch(texts)?,
            &self.coarse_labels,
        ))
    }

    /// Extract selected raw ENE labels from a standard GiNZA batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_by_labels_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        labels: &[&str],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        self.extract_entities_with_filter_batch(texts, &EntityLabelFilter::new(labels))
    }

    /// Extract standard GiNZA entities accepted by a reusable ENE filter.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_with_filter_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(adapt_ginza_batches(
            self.inner
                .extract_entities_with_filter_batch(texts, filter)?,
            &self.coarse_labels,
        ))
    }

    /// Extract raw ENE labels from a constrained standard GiNZA batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn extract_entities_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(adapt_ginza_batches(
            self.inner.extract_entities_batch_with_constraints(inputs)?,
            &self.coarse_labels,
        ))
    }

    /// Extract selected raw ENE labels from a constrained standard GiNZA batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn extract_entities_by_labels_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
        labels: &[&str],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        self.extract_entities_with_filter_batch_with_constraints(
            inputs,
            &EntityLabelFilter::new(labels),
        )
    }

    /// Extract constrained standard GiNZA entities accepted by a reusable ENE
    /// filter.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn extract_entities_with_filter_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(self
            .process_batch_with_constraints(inputs)?
            .iter()
            .map(|doc| self.entities_with_filter(doc, filter))
            .collect())
    }

    /// Extract OntoNotes-mapped spans from a standard GiNZA batch.
    ///
    /// Post-NER labels absent from the exported GiNZA mapping use `OTHERS`.
    ///
    /// # Errors
    ///
    /// Returns the first inference error or a missing mapping error.
    pub fn extract_entities_ontonotes_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, GinzaError> {
        let docs = self.process_batch(texts)?;
        self.entities_ontonotes_batch(&docs)
    }

    /// Extract OntoNotes-mapped spans from a constrained standard GiNZA batch.
    ///
    /// # Errors
    ///
    /// Returns the first constraint or inference error, or a missing mapping
    /// error.
    pub fn extract_entities_ontonotes_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Vec<NamedEntity>>, GinzaError> {
        let docs = self.process_batch_with_constraints(inputs)?;
        self.entities_ontonotes_batch(&docs)
    }

    /// Map entity spans in already processed documents to OntoNotes labels.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle has no exported mapping.
    pub fn entities_ontonotes_batch(
        &self,
        docs: &[Doc],
    ) -> Result<Vec<Vec<NamedEntity>>, GinzaError> {
        docs.iter()
            .map(|doc| self.entities_ontonotes(doc))
            .collect()
    }

    /// Return token-aligned OntoNotes B/I/O labels for a standard GiNZA batch.
    ///
    /// # Errors
    ///
    /// Returns the first inference error or a missing mapping error.
    pub fn token_labels_ontonotes_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<String>>, GinzaError> {
        let docs = self.process_batch(texts)?;
        self.token_labels_ontonotes_batch_from_docs(&docs)
    }

    /// Return token-aligned OntoNotes labels for a constrained standard batch.
    ///
    /// # Errors
    ///
    /// Returns the first constraint or inference error, or a missing mapping
    /// error.
    pub fn token_labels_ontonotes_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Vec<String>>, GinzaError> {
        let docs = self.process_batch_with_constraints(inputs)?;
        self.token_labels_ontonotes_batch_from_docs(&docs)
    }

    /// Map token labels in already processed documents to OntoNotes B/I/O.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle has no exported mapping.
    pub fn token_labels_ontonotes_batch_from_docs(
        &self,
        docs: &[Doc],
    ) -> Result<Vec<Vec<String>>, GinzaError> {
        docs.iter()
            .map(|doc| {
                Ok(self
                    .inner
                    .token_labels_with_mapping_or(doc, "ontonotes", "OTHERS")?)
            })
            .collect()
    }
}

fn adapt_ginza_entities(
    entities: Vec<NamedEntity>,
    coarse_labels: &CoarseLabelMap,
) -> Vec<GinzaEntity> {
    entities
        .into_iter()
        .map(|entity| GinzaEntity {
            coarse_label: resolve_coarse_label(coarse_labels, &entity.label),
            entity,
        })
        .collect()
}

fn adapt_ginza_batches(
    batches: Vec<Vec<NamedEntity>>,
    coarse_labels: &CoarseLabelMap,
) -> Vec<Vec<GinzaEntity>> {
    batches
        .into_iter()
        .map(|entities| adapt_ginza_entities(entities, coarse_labels))
        .collect()
}

#[cfg(feature = "transformers")]
fn entity_ruler_names(manifest: &BundleManifest, ner_index: usize) -> (Vec<&str>, Vec<&str>) {
    let (before, after): (Vec<_>, Vec<_>) = manifest
        .pipeline
        .iter()
        .enumerate()
        .filter(|(_, component)| component.factory == "entity_ruler")
        .partition(|(index, _)| *index < ner_index);
    (
        before
            .into_iter()
            .map(|(_, component)| component.name.as_str())
            .collect(),
        after
            .into_iter()
            .map(|(_, component)| component.name.as_str())
            .collect(),
    )
}

#[cfg(feature = "transformers")]
fn optional_component_name<'a>(
    manifest: &'a BundleManifest,
    factory: &'static str,
) -> Result<Option<&'a str>, PipelineError> {
    let names = manifest
        .pipeline
        .iter()
        .filter(|component| component.factory == factory)
        .map(|component| component.name.as_str())
        .collect::<Vec<_>>();
    match names.as_slice() {
        [] => Ok(None),
        [name] => Ok(Some(*name)),
        _ => Err(PipelineError::MultipleComponents {
            factory,
            names: names.into_iter().map(str::to_owned).collect(),
        }),
    }
}

#[cfg(feature = "transformers")]
fn sentence_boundary_component_names(
    manifest: &BundleManifest,
) -> Result<(Option<&str>, Option<&str>), PipelineError> {
    let senter = optional_component_name(manifest, "senter")?;
    let sentencizer = optional_component_name(manifest, "sentencizer")?;
    if senter.is_some() && sentencizer.is_some() {
        return Err(PipelineError::MultipleSentenceBoundaryComponents);
    }
    Ok((senter, sentencizer))
}

#[cfg(feature = "transformers")]
fn validate_transformer_upstream(
    component: &ComponentManifest,
    transformer_name: &str,
    requires_external_vectors: bool,
) -> Result<(), GinzaError> {
    let has_transformer_listener = component
        .nodes
        .iter()
        .any(|node| node.name == "transformer-listener");
    if requires_external_vectors && !has_transformer_listener {
        return Err(GinzaError::InvalidElectraTransformerUpstream {
            component: component.name.clone(),
            upstream: "missing transformer-listener".to_owned(),
        });
    }
    if !has_transformer_listener {
        return Ok(());
    }
    let upstream = match component.settings.get("transformer_upstream") {
        None => "transformer",
        Some(value) => value
            .as_str()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| GinzaError::InvalidElectraTransformerUpstream {
                component: component.name.clone(),
                upstream: value.to_string(),
            })?,
    };
    if upstream != transformer_name {
        return Err(GinzaError::ElectraTransformerUpstream {
            component: component.name.clone(),
            expected: transformer_name.to_owned(),
            actual: upstream.to_owned(),
        });
    }
    Ok(())
}

#[cfg(feature = "transformers")]
fn has_external_vector_listener(component: &ComponentManifest) -> bool {
    component.nodes.iter().any(|node| {
        matches!(
            node.name.as_str(),
            "tok2vec-listener" | "transformer-listener"
        )
    })
}

#[cfg(feature = "transformers")]
struct LoadedElectraComponents {
    tokenizer: RuntimeTokenizer,
    parser: Option<DependencyParser>,
    sentence_recognizer: Option<SentenceRecognizer>,
    sentencizer: Option<Sentencizer>,
    ner: EntityRecognizer,
    pre_ner_entity_rulers: Vec<EntityRuler>,
    post_ner_entity_rulers: Vec<EntityRuler>,
    spec: TransformerSpec,
    coarse_labels: CoarseLabelMap,
}

#[cfg(feature = "transformers")]
fn electra_spec(bundle: &Bundle) -> Result<TransformerSpec, GinzaError> {
    if ginza_model_family(bundle.manifest())? != GinzaModelFamily::Electra {
        return Err(GinzaError::TransformersRequired);
    }
    if bundle.manifest().tokenizer.kind != TokenizerKind::Sudachi {
        return Err(GinzaError::Tokenizer {
            actual: bundle.manifest().tokenizer.kind,
        });
    }
    Ok(TransformerSpec::from_bundle(bundle)?)
}

#[cfg(feature = "transformers")]
fn load_electra_components(bundle: &Bundle) -> Result<LoadedElectraComponents, GinzaError> {
    let spec = electra_spec(bundle)?;
    load_electra_components_with_spec(bundle, spec)
}

#[cfg(feature = "transformers")]
fn load_electra_components_with_spec(
    bundle: &Bundle,
    spec: TransformerSpec,
) -> Result<LoadedElectraComponents, GinzaError> {
    let transformer_name = bundle
        .manifest()
        .pipeline
        .iter()
        .find(|component| component.factory.contains("transformer"))
        .expect("TransformerSpec validated exactly one transformer component")
        .name
        .as_str();
    let ner_components = bundle
        .manifest()
        .pipeline
        .iter()
        .enumerate()
        .filter(|(_, component)| component.factory == "ner")
        .collect::<Vec<_>>();
    let (ner_index, ner_component) = match ner_components.as_slice() {
        [(index, component)] => (*index, *component),
        _ => return Err(GinzaError::ElectraNerComponent),
    };
    let parser_components = bundle
        .manifest()
        .pipeline
        .iter()
        .filter(|component| component.factory == "parser")
        .collect::<Vec<_>>();
    let parser_component = match parser_components.as_slice() {
        [] => None,
        [component] => Some(*component),
        _ => return Err(GinzaError::ElectraParserComponents),
    };
    validate_transformer_upstream(ner_component, transformer_name, true)?;
    if let Some(component) = parser_component {
        validate_transformer_upstream(component, transformer_name, true)?;
    }
    let (senter_name, sentencizer_name) = sentence_boundary_component_names(bundle.manifest())?;
    if let Some(name) = senter_name {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == name)
            .expect("selected sentence recognizer belongs to the manifest");
        validate_transformer_upstream(
            component,
            transformer_name,
            has_external_vector_listener(component),
        )?;
    }
    let parser = parser_component
        .map(|component| DependencyParser::load(bundle, &component.name))
        .transpose()?;
    let sentence_recognizer = senter_name
        .map(|name| SentenceRecognizer::load(bundle, name))
        .transpose()
        .map_err(PipelineError::from)?;
    let sentencizer = sentencizer_name
        .map(|name| Sentencizer::load(bundle, name))
        .transpose()
        .map_err(PipelineError::from)?;
    let (pre_ner_entity_ruler_names, post_ner_entity_ruler_names) =
        entity_ruler_names(bundle.manifest(), ner_index);
    let pre_ner_entity_rulers = pre_ner_entity_ruler_names
        .iter()
        .map(|name| EntityRuler::load(bundle, name))
        .collect::<Result<Vec<_>, _>>()?;
    let post_ner_entity_rulers = post_ner_entity_ruler_names
        .iter()
        .map(|name| EntityRuler::load(bundle, name))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ner = EntityRecognizer::load(bundle, &ner_component.name)?;
    for ruler in pre_ner_entity_rulers.iter().chain(&post_ner_entity_rulers) {
        ner.register_entity_ruler(ruler);
    }
    Ok(LoadedElectraComponents {
        tokenizer: bundle.load_tokenizer()?,
        parser,
        sentence_recognizer,
        sentencizer,
        ner,
        pre_ner_entity_rulers,
        post_ner_entity_rulers,
        spec,
        coarse_labels: exported_coarse_labels(bundle.manifest()),
    })
}

/// Validate all assets and execution components required by GiNZA Electra.
///
/// This performs the same tokenizer, transformer contract, parser or sentence
/// boundary, NER, and entity ruler loading used by
/// [`GinzaElectraPipeline::load`] without constructing a transformer encoder.
///
/// # Errors
///
/// Returns an error when the bundle is not executable by the Electra pipeline.
#[cfg(feature = "transformers")]
pub fn validate_electra_bundle(bundle: &Bundle) -> Result<TransformerSpec, GinzaError> {
    Ok(load_electra_components(bundle)?.spec)
}

#[cfg(feature = "transformers")]
impl<E: TransformerEncoder> GinzaElectraPipeline<E> {
    /// Load an Electra bundle with an initialized native encoder.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle, exported transformer assets, encoder
    /// specification, parser, NER scorer, or entity ruler is incompatible.
    pub fn load(bundle: &Bundle, encoder: E) -> Result<Self, GinzaError> {
        let spec = electra_spec(bundle)?;
        if encoder.spec() != &spec {
            return Err(TransformerError::InvalidSpec(
                "encoder specification does not match the exported bundle".to_owned(),
            )
            .into());
        }
        let components = load_electra_components_with_spec(bundle, spec)?;
        Ok(Self {
            tokenizer: components.tokenizer,
            encoder,
            parser: components.parser,
            sentence_recognizer: components.sentence_recognizer,
            sentencizer: components.sentencizer,
            ner: components.ner,
            pre_ner_entity_rulers: components.pre_ner_entity_rulers,
            post_ner_entity_rulers: components.post_ner_entity_rulers,
            spec: components.spec,
            coarse_labels: components.coarse_labels,
        })
    }

    /// Return the exported transformer contract validated at load time.
    #[must_use]
    pub const fn spec(&self) -> &TransformerSpec {
        &self.spec
    }

    /// Return whether the Electra pipeline includes dependency parsing.
    #[must_use]
    pub const fn has_dependency_parser(&self) -> bool {
        self.parser.is_some()
    }

    /// Return whether the Electra pipeline includes a rule-based sentencizer.
    #[must_use]
    pub const fn has_sentencizer(&self) -> bool {
        self.sentencizer.is_some()
    }

    /// Return whether the Electra pipeline includes a trainable senter.
    #[must_use]
    pub const fn has_sentence_recognizer(&self) -> bool {
        self.sentence_recognizer.is_some()
    }

    /// Return whether the Electra pipeline includes an entity ruler.
    #[must_use]
    pub fn has_entity_ruler(&self) -> bool {
        !self.pre_ner_entity_rulers.is_empty() || !self.post_ner_entity_rulers.is_empty()
    }

    /// Return labels declared by the statistical model or entity rulers.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.ner.supported_entity_labels()
    }

    /// Return whether the model or an entity ruler declares a label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        self.ner.supports_entity_label(label)
    }

    /// Compile requested labels against the model and entity ruler labels.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        self.ner.select_entity_labels(labels)
    }

    fn annotate_sentence_boundaries(
        &self,
        doc: &mut Doc,
        vectors: &Matrix,
    ) -> Result<(), GinzaError> {
        if let Some(sentence_recognizer) = &self.sentence_recognizer {
            if sentence_recognizer.requires_external_tok2vec() {
                sentence_recognizer
                    .annotate_with_tok2vec(doc, vectors)
                    .map_err(PipelineError::from)?;
            } else {
                sentence_recognizer
                    .annotate(doc)
                    .map_err(PipelineError::from)?;
            }
        } else if let Some(sentencizer) = &self.sentencizer {
            sentencizer.annotate(doc);
        } else if let Some(first) = doc.tokens_mut().first_mut() {
            first.sent_start = 1;
        }
        Ok(())
    }

    /// Tokenize text and run Electra, sentence annotation, and GiNZA NER.
    ///
    /// # Errors
    ///
    /// Returns an error for tokenization, transformer inference, sentence
    /// annotation, or entity-recognition failures.
    pub fn process(&self, text: &str) -> Result<Doc, GinzaError> {
        self.process_with_constraints(text, &[])
    }

    /// Run Electra, sentence annotation, and NER with spaCy-compatible
    /// constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints, transformer inference,
    /// sentence annotation, or entity recognition.
    pub fn process_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Doc, GinzaError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        let vectors = self.encoder.encode(&doc)?;
        validate_token_vectors(&doc, &self.spec, &vectors)?;
        if let Some(parser) = &self.parser {
            parser.annotate(&mut doc, &vectors)?;
        } else {
            self.annotate_sentence_boundaries(&mut doc, &vectors)?;
        }
        for ruler in &self.pre_ner_entity_rulers {
            ruler.annotate(&mut doc)?;
        }
        apply_entity_constraints(&mut doc, constraints)?;
        self.ner.annotate_with_tok2vec(&mut doc, &vectors)?;
        for ruler in &self.post_ner_entity_rulers {
            ruler.annotate(&mut doc)?;
        }
        Ok(doc)
    }

    fn annotate_batch(
        &self,
        docs: &mut [Doc],
        constraints: &[&[EntityConstraint]],
    ) -> Result<(), GinzaError> {
        let vectors = self.encoder.encode_batch(docs)?;
        if vectors.len() != docs.len() {
            return Err(TransformerError::InvalidBatchOutput {
                expected: docs.len(),
                actual: vectors.len(),
            }
            .into());
        }
        debug_assert_eq!(constraints.len(), docs.len());
        for ((doc, vectors), constraints) in docs
            .iter_mut()
            .zip(vectors.iter())
            .zip(constraints.iter().copied())
        {
            validate_token_vectors(doc, &self.spec, vectors)?;
            if let Some(parser) = &self.parser {
                parser.annotate(doc, vectors)?;
            } else {
                self.annotate_sentence_boundaries(doc, vectors)?;
            }
            for ruler in &self.pre_ner_entity_rulers {
                ruler.annotate(doc)?;
            }
            apply_entity_constraints(doc, constraints)?;
            self.ner.annotate_with_tok2vec(doc, vectors)?;
            for ruler in &self.post_ner_entity_rulers {
                ruler.annotate(doc)?;
            }
        }
        Ok(())
    }

    /// Process a GiNZA Electra document batch through the encoder batch hook.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, transformer, sentence, or NER error.
    pub fn process_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Doc>, GinzaError> {
        let mut docs = {
            let mut tokenizer = self.tokenizer.session();
            texts
                .iter()
                .map(|text| tokenizer.tokenize(text.as_ref()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PipelineError::from)?
        };
        let constraints = vec![&[][..]; docs.len()];
        self.annotate_batch(&mut docs, &constraints)?;
        Ok(docs)
    }

    /// Process GiNZA Electra texts with document-specific preset constraints.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, transformer, constraint, sentence, or
    /// NER error.
    pub fn process_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Doc>, GinzaError> {
        let mut docs = {
            let mut tokenizer = self.tokenizer.session();
            inputs
                .iter()
                .map(|input| tokenizer.tokenize(input.text))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PipelineError::from)?
        };
        let constraints = inputs
            .iter()
            .map(|input| input.constraints)
            .collect::<Vec<_>>();
        self.annotate_batch(&mut docs, &constraints)?;
        Ok(docs)
    }

    /// Extract raw ENE labels and their coarse extraction mappings.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<GinzaEntity>, GinzaError> {
        let doc = self.process(text)?;
        Ok(self.entities(&doc))
    }

    /// Extract only entities whose raw ENE labels are included in `labels`.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities_by_labels(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<GinzaEntity>, GinzaError> {
        self.extract_entities_with_filter(text, &EntityLabelFilter::new(labels))
    }

    /// Extract Electra entities accepted by a reusable raw ENE label filter.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenization or inference fails.
    pub fn extract_entities_with_filter(
        &self,
        text: &str,
        filter: &EntityLabelFilter,
    ) -> Result<Vec<GinzaEntity>, GinzaError> {
        let doc = self.process(text)?;
        Ok(self.entities_with_filter(&doc, filter))
    }

    /// Extract entities using GiNZA's exported ENE-to-OntoNotes mapping.
    ///
    /// ENE labels introduced by a ruler map to `OTHERS`.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the bundle has no mapping.
    pub fn extract_entities_ontonotes(&self, text: &str) -> Result<Vec<NamedEntity>, GinzaError> {
        let doc = self.process(text)?;
        self.entities_ontonotes(&doc)
    }

    /// Extract OntoNotes-mapped entities while enforcing NER constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints, failed inference, or a
    /// missing mapping.
    pub fn extract_entities_ontonotes_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Vec<NamedEntity>, GinzaError> {
        let doc = self.process_with_constraints(text, constraints)?;
        self.entities_ontonotes(&doc)
    }

    /// Return token-aligned B/I/O labels using GiNZA's OntoNotes mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the bundle has no mapping.
    pub fn token_labels_ontonotes(&self, text: &str) -> Result<Vec<String>, GinzaError> {
        let doc = self.process(text)?;
        Ok(self
            .ner
            .token_labels_with_mapping_or(&doc, "ontonotes", "OTHERS")?)
    }

    /// Return constrained token-aligned B/I/O labels using OntoNotes mapping.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints, failed inference, or a
    /// missing mapping.
    pub fn token_labels_ontonotes_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Vec<String>, GinzaError> {
        let doc = self.process_with_constraints(text, constraints)?;
        Ok(self
            .ner
            .token_labels_with_mapping_or(&doc, "ontonotes", "OTHERS")?)
    }

    /// Extract Electra GiNZA entities while enforcing NER constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed inference.
    pub fn extract_entities_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Vec<GinzaEntity>, GinzaError> {
        let doc = self.process_with_constraints(text, constraints)?;
        Ok(self.entities(&doc))
    }

    /// Extract raw ENE and coarse labels from a GiNZA Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(self
            .process_batch(texts)?
            .iter()
            .map(|doc| self.entities(doc))
            .collect())
    }

    /// Extract selected raw ENE labels from a GiNZA Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_by_labels_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        labels: &[&str],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        self.extract_entities_with_filter_batch(texts, &EntityLabelFilter::new(labels))
    }

    /// Extract Electra entities accepted by a reusable raw ENE label filter.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_with_filter_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(self
            .process_batch(texts)?
            .iter()
            .map(|doc| self.entities_with_filter(doc, filter))
            .collect())
    }

    /// Extract OntoNotes-mapped spans from a GiNZA Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first inference error or a missing mapping error.
    pub fn extract_entities_ontonotes_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, GinzaError> {
        let docs = self.process_batch(texts)?;
        self.entities_ontonotes_batch(&docs)
    }

    /// Extract OntoNotes-mapped spans from a constrained Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first constraint or inference error, or a missing mapping
    /// error.
    pub fn extract_entities_ontonotes_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Vec<NamedEntity>>, GinzaError> {
        let docs = self.process_batch_with_constraints(inputs)?;
        self.entities_ontonotes_batch(&docs)
    }

    /// Extract raw ENE and coarse labels from a constrained Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn extract_entities_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(self
            .process_batch_with_constraints(inputs)?
            .iter()
            .map(|doc| self.entities(doc))
            .collect())
    }

    /// Extract selected raw ENE labels from a constrained GiNZA Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn extract_entities_by_labels_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
        labels: &[&str],
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        self.extract_entities_with_filter_batch_with_constraints(
            inputs,
            &EntityLabelFilter::new(labels),
        )
    }

    /// Extract constrained GiNZA Electra entities accepted by a reusable ENE
    /// filter.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization, constraint, or inference error.
    pub fn extract_entities_with_filter_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<GinzaEntity>>, GinzaError> {
        Ok(self
            .process_batch_with_constraints(inputs)?
            .iter()
            .map(|doc| self.entities_with_filter(doc, filter))
            .collect())
    }

    /// Return GiNZA entities already attached to a processed document.
    #[must_use]
    pub fn entities(&self, doc: &Doc) -> Vec<GinzaEntity> {
        adapt_ginza_entities(self.ner.entities(doc), &self.coarse_labels)
    }

    /// Return attached Electra entities whose ENE labels are in `labels`.
    #[must_use]
    pub fn entities_by_labels(&self, doc: &Doc, labels: &[&str]) -> Vec<GinzaEntity> {
        self.entities_with_filter(doc, &EntityLabelFilter::new(labels))
    }

    /// Return attached Electra entities accepted by a reusable ENE filter.
    #[must_use]
    pub fn entities_with_filter(&self, doc: &Doc, filter: &EntityLabelFilter) -> Vec<GinzaEntity> {
        adapt_ginza_entities(
            self.ner.entities_with_filter(doc, filter),
            &self.coarse_labels,
        )
    }

    /// Return attached Electra entities with one ENE label.
    #[must_use]
    pub fn entities_by_label(&self, doc: &Doc, label: &str) -> Vec<GinzaEntity> {
        self.entities_by_labels(doc, &[label])
    }

    /// Map entities already attached to a document to OntoNotes labels.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle has no exported mapping.
    pub fn entities_ontonotes(&self, doc: &Doc) -> Result<Vec<NamedEntity>, GinzaError> {
        Ok(self
            .ner
            .entities_with_mapping_or(doc, "ontonotes", "OTHERS")?)
    }

    /// Map Electra entity spans in processed documents to OntoNotes labels.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle has no exported mapping.
    pub fn entities_ontonotes_batch(
        &self,
        docs: &[Doc],
    ) -> Result<Vec<Vec<NamedEntity>>, GinzaError> {
        docs.iter()
            .map(|doc| self.entities_ontonotes(doc))
            .collect()
    }

    /// Return token-aligned OntoNotes B/I/O labels for an Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first inference error or a missing mapping error.
    pub fn token_labels_ontonotes_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<String>>, GinzaError> {
        let docs = self.process_batch(texts)?;
        self.token_labels_ontonotes_batch_from_docs(&docs)
    }

    /// Return token-aligned OntoNotes labels for a constrained Electra batch.
    ///
    /// # Errors
    ///
    /// Returns the first constraint or inference error, or a missing mapping
    /// error.
    pub fn token_labels_ontonotes_batch_with_constraints(
        &self,
        inputs: &[NerBatchInput<'_>],
    ) -> Result<Vec<Vec<String>>, GinzaError> {
        let docs = self.process_batch_with_constraints(inputs)?;
        self.token_labels_ontonotes_batch_from_docs(&docs)
    }

    /// Map token labels in processed Electra documents to OntoNotes B/I/O.
    ///
    /// # Errors
    ///
    /// Returns an error when the bundle has no exported mapping.
    pub fn token_labels_ontonotes_batch_from_docs(
        &self,
        docs: &[Doc],
    ) -> Result<Vec<Vec<String>>, GinzaError> {
        docs.iter()
            .map(|doc| {
                Ok(self
                    .ner
                    .token_labels_with_mapping_or(doc, "ontonotes", "OTHERS")?)
            })
            .collect()
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

fn exported_coarse_labels(manifest: &BundleManifest) -> CoarseLabelMap {
    manifest
        .pipeline
        .iter()
        .filter(|component| component.factory == "ner")
        .filter_map(|component| component.settings.get("label_mappings"))
        .filter_map(|mappings| mappings.as_object())
        .filter_map(|mappings| mappings.get("ontonotes"))
        .filter_map(|mapping| mapping.as_object())
        .flat_map(|mapping| mapping.iter())
        .filter_map(|(label, mapped)| {
            let mapped = mapped.as_str()?;
            coarse_label(label)
                .or_else(|| coarse_ontonotes_label(mapped))
                .map(|coarse| (label.clone(), coarse))
        })
        .collect()
}

fn resolve_coarse_label(exported: &CoarseLabelMap, label: &str) -> Option<&'static str> {
    coarse_label(label).or_else(|| exported.get(label).copied())
}

fn coarse_ontonotes_label(label: &str) -> Option<&'static str> {
    match label {
        "ANIMAL" => Some("ANIMAL"),
        "CARDINAL" => Some("CARDINAL"),
        "DATE" => Some("DATE"),
        "EMAIL" => Some("EMAIL"),
        "EVENT" => Some("EVENT"),
        "FAC" => Some("FAC"),
        "GPE" => Some("GPE"),
        "LANGUAGE" => Some("LANGUAGE"),
        "LAW" => Some("LAW"),
        "LOC" => Some("LOC"),
        "MONEY" => Some("MONEY"),
        "NORP" => Some("NORP"),
        "ORDINAL" => Some("ORDINAL"),
        "ORG" => Some("ORG"),
        "PERCENT" => Some("PERCENT"),
        "PERSON" => Some("PERSON"),
        "PHONE" => Some("PHONE"),
        "PRODUCT" => Some("PRODUCT"),
        "QUANTITY" => Some("QUANTITY"),
        "TIME" => Some("TIME"),
        "URL" => Some("URL"),
        "WORK_OF_ART" => Some("WORK_OF_ART"),
        _ => None,
    }
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
        | "Government"
        | "Military"
        | "Pro_Sports_Organization"
        | "Public_Institution"
        | "Research_Institute"
        | "School"
        | "Show_Organization"
        | "Sports_League"
        | "Sports_Organization_Other" => Some("ORG"),
        "GPE" | "GPE_Other" | "City" | "County" | "Province" | "Country" | "Domestic_Region"
        | "Region_Other" | "Location_Other" => Some("GPE"),
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

    #[cfg(feature = "transformers")]
    use jewel_core::NodeManifest;
    use jewel_core::{
        BundleManifest, ComponentKind, ComponentManifest, NamedEntity, RuntimeManifest,
        SourceManifest, TokenizerKind, TokenizerManifest,
    };
    use serde_json::json;

    use super::{
        adapt_ginza_batches, coarse_label, exported_coarse_labels, ginza_model_family,
        resolve_coarse_label, GinzaError, GinzaModelFamily,
    };
    #[cfg(feature = "transformers")]
    use super::{
        entity_ruler_names, sentence_boundary_component_names, validate_transformer_upstream,
    };

    fn component(name: &str, factory: &str) -> ComponentManifest {
        ComponentManifest {
            name: name.to_owned(),
            factory: factory.to_owned(),
            kind: ComponentKind::Trainable,
            root_node: None,
            settings: BTreeMap::new(),
            nodes: Vec::new(),
            state_path: None,
            labels: Vec::new(),
            moves: Vec::new(),
        }
    }

    #[cfg(feature = "transformers")]
    fn transformer_listener(index: usize) -> NodeManifest {
        NodeManifest {
            index,
            name: "transformer-listener".to_owned(),
            children: Vec::new(),
            dims: BTreeMap::new(),
            refs: BTreeMap::new(),
            params: BTreeMap::new(),
            attrs: BTreeMap::new(),
            omitted_attrs: Vec::new(),
        }
    }

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
            pipeline: vec![component(factory, factory)],
        }
    }

    #[test]
    fn maps_contract_extraction_labels() {
        assert_eq!(coarse_label("Person"), Some("PERSON"));
        assert_eq!(coarse_label("Company"), Some("ORG"));
        assert_eq!(coarse_label("Public_Institution"), Some("ORG"));
        assert_eq!(coarse_label("Research_Institute"), Some("ORG"));
        assert_eq!(coarse_label("Sports_League"), Some("ORG"));
        assert_eq!(coarse_label("Money"), Some("MONEY"));
        assert_eq!(coarse_label("Postal_Address"), Some("ADDRESS"));
        assert_eq!(coarse_label("City"), Some("GPE"));
        assert_eq!(coarse_label("Domestic_Region"), Some("GPE"));
        assert_eq!(coarse_label("Position_Vocation"), Some("TITLE"));
        assert_eq!(coarse_label("Email"), Some("EMAIL"));
        assert_eq!(coarse_label("Period_Time"), Some("TIME"));
        assert_eq!(coarse_label("Unknown_Category"), None);
    }

    #[test]
    fn batch_adapter_preserves_documents_and_adds_coarse_labels() {
        let coarse_labels = BTreeMap::new();
        let batches = adapt_ginza_batches(
            vec![
                vec![NamedEntity {
                    text: "株式会社青空".to_owned(),
                    label: "Company".to_owned(),
                    ent_id: None,
                    start_token: 0,
                    end_token: 2,
                    start_char: 0,
                    end_char: 6,
                }],
                Vec::new(),
            ],
            &coarse_labels,
        );

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0][0].ene_label(), "Company");
        assert_eq!(batches[0][0].coarse_label, Some("ORG"));
        assert!(batches[1].is_empty());
    }

    #[test]
    fn extends_coarse_labels_from_exported_ontonotes_mapping() {
        let mut manifest = manifest("ginza", "ja", "ner");
        manifest.pipeline[0].settings.insert(
            "label_mappings".to_owned(),
            json!({
                "ontonotes": {
                    "Product_Other": "PRODUCT",
                    "Point": "QUANTITY",
                    "Public_Institution": "FAC",
                    "Position_Vocation": "OTHERS",
                    "Unknown_Category": "OTHERS"
                }
            }),
        );

        let coarse_labels = exported_coarse_labels(&manifest);
        assert_eq!(
            resolve_coarse_label(&coarse_labels, "Product_Other"),
            Some("PRODUCT")
        );
        assert_eq!(
            resolve_coarse_label(&coarse_labels, "Point"),
            Some("QUANTITY")
        );
        assert_eq!(
            resolve_coarse_label(&coarse_labels, "Public_Institution"),
            Some("ORG")
        );
        assert_eq!(
            resolve_coarse_label(&coarse_labels, "Position_Vocation"),
            Some("TITLE")
        );
        assert_eq!(
            resolve_coarse_label(&coarse_labels, "Unknown_Category"),
            None
        );
    }

    #[cfg(feature = "transformers")]
    #[test]
    fn electra_entity_rulers_preserve_pre_and_post_ner_order() {
        let mut manifest = manifest("ginza_electra", "ja", "known_parties");
        manifest.pipeline[0].factory = "entity_ruler".to_owned();
        manifest.pipeline.push(component("ner", "ner"));
        manifest
            .pipeline
            .push(component("contract_terms", "entity_ruler"));
        assert_eq!(
            entity_ruler_names(&manifest, 1),
            (vec!["known_parties"], vec!["contract_terms"])
        );
    }

    #[cfg(feature = "transformers")]
    #[test]
    fn electra_sentence_boundary_components_are_unambiguous() {
        let mut manifest = manifest("ginza_electra", "ja", "ner");
        manifest.pipeline.push(component("sentences", "senter"));
        assert_eq!(
            sentence_boundary_component_names(&manifest).unwrap(),
            (Some("sentences"), None)
        );

        manifest
            .pipeline
            .push(component("rule_sentences", "sentencizer"));
        assert!(matches!(
            sentence_boundary_component_names(&manifest),
            Err(jewel_core::PipelineError::MultipleSentenceBoundaryComponents)
        ));
    }

    #[cfg(feature = "transformers")]
    #[test]
    fn electra_components_validate_their_transformer_upstream() {
        let mut entities = component("entities", "ner");
        entities.nodes.push(transformer_listener(0));
        entities
            .settings
            .insert("transformer_upstream".to_owned(), json!("context_encoder"));
        validate_transformer_upstream(&entities, "context_encoder", true).unwrap();

        assert!(matches!(
            validate_transformer_upstream(&entities, "transformer", true),
            Err(GinzaError::ElectraTransformerUpstream {
                component,
                expected,
                actual
            }) if component == "entities"
                && expected == "transformer"
                && actual == "context_encoder"
        ));

        let parser = component("dependencies", "parser");
        assert!(matches!(
            validate_transformer_upstream(&parser, "transformer", true),
            Err(GinzaError::InvalidElectraTransformerUpstream { component, .. })
                if component == "dependencies"
        ));
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
