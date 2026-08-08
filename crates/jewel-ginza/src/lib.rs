//! GiNZA-specific model validation and entity label adaptation.
//!
//! The standard CNN model executes through `jewel-core`. Transformer-backed
//! GiNZA models use the optional `transformers` integration boundary.

use std::collections::BTreeMap;

use jewel_core::{
    Bundle, BundleManifest, Doc, EntityConstraint, EntityLabelSelection, EntityRecognizerError,
    NamedEntity, NerBatchInput, NerPipeline, PipelineError, TokenizerKind,
};
use thiserror::Error;

#[cfg(feature = "transformers")]
use jewel_core::{
    apply_entity_constraints, DependencyParser, DependencyParserError, EntityRecognizer,
    EntityRuler, EntityRulerError, Matrix, RuntimeTokenizer, RuntimeTokenizerError,
    SentenceRecognizer, Sentencizer,
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
    entity_rulers: Vec<EntityRuler>,
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

    /// Return whether the standard pipeline includes a post-NER entity ruler.
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

    /// Extract entities using GiNZA's exported ENE-to-OntoNotes mapping.
    ///
    /// ENE labels introduced by a post-NER ruler map to `OTHERS`.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the bundle has no mapping.
    pub fn extract_entities_ontonotes(&self, text: &str) -> Result<Vec<NamedEntity>, GinzaError> {
        let doc = self.inner.process(text)?;
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
fn post_ner_entity_ruler_names<'a>(
    manifest: &'a BundleManifest,
    ner_index: usize,
    ner_name: &str,
) -> Result<Vec<&'a str>, PipelineError> {
    manifest
        .pipeline
        .iter()
        .enumerate()
        .filter(|(_, component)| component.factory == "entity_ruler")
        .map(|(index, component)| {
            if index < ner_index {
                Err(PipelineError::UnsupportedComponentOrder {
                    component: component.name.clone(),
                    after: ner_name.to_owned(),
                })
            } else {
                Ok(component.name.as_str())
            }
        })
        .collect()
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
struct LoadedElectraComponents {
    tokenizer: RuntimeTokenizer,
    parser: Option<DependencyParser>,
    sentence_recognizer: Option<SentenceRecognizer>,
    sentencizer: Option<Sentencizer>,
    ner: EntityRecognizer,
    entity_rulers: Vec<EntityRuler>,
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
    let ner_components = bundle
        .manifest()
        .pipeline
        .iter()
        .enumerate()
        .filter(|(_, component)| component.factory == "ner")
        .map(|(index, component)| (index, component.name.as_str()))
        .collect::<Vec<_>>();
    let (ner_index, ner_name) = match ner_components.as_slice() {
        [(index, name)] => (*index, *name),
        _ => return Err(GinzaError::ElectraNerComponent),
    };
    let parser_names = bundle
        .manifest()
        .pipeline
        .iter()
        .filter(|component| component.factory == "parser")
        .map(|component| component.name.as_str())
        .collect::<Vec<_>>();
    let parser = match parser_names.as_slice() {
        [] => None,
        [name] => Some(DependencyParser::load(bundle, name)?),
        _ => return Err(GinzaError::ElectraParserComponents),
    };
    let (senter_name, sentencizer_name) = sentence_boundary_component_names(bundle.manifest())?;
    let sentence_recognizer = senter_name
        .map(|name| SentenceRecognizer::load(bundle, name))
        .transpose()
        .map_err(PipelineError::from)?;
    let sentencizer = sentencizer_name
        .map(|name| Sentencizer::load(bundle, name))
        .transpose()
        .map_err(PipelineError::from)?;
    let entity_rulers = post_ner_entity_ruler_names(bundle.manifest(), ner_index, ner_name)?
        .iter()
        .map(|name| EntityRuler::load(bundle, name))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ner = EntityRecognizer::load(bundle, ner_name)?;
    for ruler in &entity_rulers {
        ner.register_entity_ruler(ruler);
    }
    Ok(LoadedElectraComponents {
        tokenizer: bundle.load_tokenizer()?,
        parser,
        sentence_recognizer,
        sentencizer,
        ner,
        entity_rulers,
        spec,
        coarse_labels: exported_coarse_labels(bundle.manifest()),
    })
}

/// Validate all assets and execution components required by GiNZA Electra.
///
/// This performs the same tokenizer, transformer contract, parser or sentence
/// boundary, NER, and post-NER entity ruler loading used by
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
            entity_rulers: components.entity_rulers,
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

    /// Return whether the Electra pipeline includes a post-NER entity ruler.
    #[must_use]
    pub fn has_entity_ruler(&self) -> bool {
        !self.entity_rulers.is_empty()
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
        apply_entity_constraints(&mut doc, constraints)?;
        self.ner.annotate_with_tok2vec(&mut doc, &vectors)?;
        for ruler in &self.entity_rulers {
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
            apply_entity_constraints(doc, constraints)?;
            self.ner.annotate_with_tok2vec(doc, vectors)?;
            for ruler in &self.entity_rulers {
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
        let mut docs = texts
            .iter()
            .map(|text| self.tokenizer.tokenize(text.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
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
        let mut docs = inputs
            .iter()
            .map(|input| self.tokenizer.tokenize(input.text))
            .collect::<Result<Vec<_>, _>>()?;
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

    /// Extract entities using GiNZA's exported ENE-to-OntoNotes mapping.
    ///
    /// ENE labels introduced by a post-NER ruler map to `OTHERS`.
    ///
    /// # Errors
    ///
    /// Returns an error when inference fails or the bundle has no mapping.
    pub fn extract_entities_ontonotes(&self, text: &str) -> Result<Vec<NamedEntity>, GinzaError> {
        let doc = self.process(text)?;
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

    /// Return GiNZA entities already attached to a processed document.
    #[must_use]
    pub fn entities(&self, doc: &Doc) -> Vec<GinzaEntity> {
        adapt_ginza_entities(self.ner.entities(doc), &self.coarse_labels)
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
    use super::{post_ner_entity_ruler_names, sentence_boundary_component_names};

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
    fn electra_entity_rulers_must_follow_ner() {
        let mut manifest = manifest("ginza_electra", "ja", "ner");
        manifest
            .pipeline
            .push(component("contract_terms", "entity_ruler"));
        assert_eq!(
            post_ner_entity_ruler_names(&manifest, 0, "ner").unwrap(),
            vec!["contract_terms"]
        );

        manifest.pipeline.swap(0, 1);
        assert!(matches!(
            post_ner_entity_ruler_names(&manifest, 1, "ner"),
            Err(jewel_core::PipelineError::UnsupportedComponentOrder {
                component,
                after
            }) if component == "contract_terms" && after == "ner"
        ));
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
