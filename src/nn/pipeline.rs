use std::sync::Arc;

use serde::{Deserialize, Serialize};
use spacy_core::Doc;
use spacy_model::{Bundle, ComponentManifest, RuntimeTokenizerError};
use spacy_tokenizer::{SharedTokenizer, TokenizeError};
use thiserror::Error;

use crate::{
    apply_entity_constraints, DependencyParser, DependencyParserError, EntityConstraint,
    EntityLabelFilter, EntityLabelSelection, EntityRecognizer, EntityRecognizerError, EntityRuler,
    EntityRulerError, NamedEntity, SentenceRecognizer, SentenceRecognizerError, Sentencizer,
    SentencizerError, Tagger, TaggerError, Tok2Vec, Tok2VecError,
};

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Tokenizer(#[from] RuntimeTokenizerError),
    #[error(transparent)]
    Tokenization(#[from] TokenizeError),
    #[error(transparent)]
    Tok2Vec(#[from] Tok2VecError),
    #[error(transparent)]
    Tagger(#[from] TaggerError),
    #[error(transparent)]
    Parser(#[from] DependencyParserError),
    #[error(transparent)]
    Ner(#[from] EntityRecognizerError),
    #[error(transparent)]
    Sentencizer(#[from] SentencizerError),
    #[error(transparent)]
    SentenceRecognizer(#[from] SentenceRecognizerError),
    #[error(transparent)]
    EntityRuler(#[from] EntityRulerError),
    #[error("pipeline contains both trainable and rule-based sentence boundary components")]
    MultipleSentenceBoundaryComponents,
    #[error("pipeline contains multiple {factory:?} components: {names:?}")]
    MultipleComponents {
        factory: &'static str,
        names: Vec<String>,
    },
    #[error("pipeline requires exactly one {factory:?} component")]
    MissingRequiredComponent { factory: &'static str },
    #[error("component {component:?} has invalid upstream tok2vec setting {upstream:?}")]
    InvalidUpstreamTok2Vec { component: String, upstream: String },
    #[error(
        "components require different upstream tok2vec components: {expected:?} and {actual:?}"
    )]
    ConflictingUpstreamTok2Vec { expected: String, actual: String },
    #[error("component {component:?} must appear after {after:?}")]
    UnsupportedComponentOrder { component: String, after: String },
    #[error("pipeline language is {actual:?}, expected {expected:?}")]
    Language {
        expected: &'static str,
        actual: String,
    },
    #[error("unsupported NER pipeline language {actual:?}; expected \"en\" or \"ja\"")]
    UnsupportedLanguage { actual: String },
}

/// Language selected by a language-aware [`NerPipeline`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum NerLanguage {
    #[serde(rename = "en")]
    English,
    #[serde(rename = "ja")]
    Japanese,
}

impl NerLanguage {
    /// Return the ISO 639-1 language code used in Jewel bundle manifests.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Japanese => "ja",
        }
    }
}

/// Python-free English tokenization, `tok2vec`, and fine-grained POS tagging.
pub struct EnglishTaggerPipeline {
    tokenizer: SharedTokenizer,
    tok2vec: Tok2Vec,
    tagger: Tagger,
}

/// Python-free English tagging, dependency parsing, and entity recognition.
pub struct EnglishPipeline {
    tokenizer: SharedTokenizer,
    tok2vec: Tok2Vec,
    tagger: Tagger,
    parser: DependencyParser,
    ner: EntityRecognizer,
}

/// Python-free English tokenization, sentence parsing, and NER without
/// loading tagger or lemmatizer components that entity extraction does not use.
pub struct EnglishNerPipeline {
    tokenizer: SharedTokenizer,
    upstream: Option<NerUpstream>,
    sentence_recognizer: Option<SentenceRecognizer>,
    sentencizer: Option<Sentencizer>,
    ner: EntityRecognizer,
    entity_rulers: Vec<EntityRuler>,
}

/// Python-free Japanese tokenization and named-entity recognition.
pub struct JapaneseNerPipeline {
    tokenizer: SharedTokenizer,
    upstream: Option<NerUpstream>,
    sentence_recognizer: Option<SentenceRecognizer>,
    sentencizer: Option<Sentencizer>,
    ner: EntityRecognizer,
    entity_rulers: Vec<EntityRuler>,
}

struct NerUpstream {
    tok2vec: Tok2Vec,
    parser: Option<DependencyParser>,
}

struct UpstreamRequirement {
    name: String,
    component: String,
}

impl NerUpstream {
    fn load_optional(
        bundle: &Bundle,
        requirement: Option<&UpstreamRequirement>,
        parser_name: Option<&str>,
    ) -> Result<Option<Self>, PipelineError> {
        if requirement.is_none() && parser_name.is_none() {
            return Ok(None);
        }
        let requirement =
            requirement.ok_or(PipelineError::MissingRequiredComponent { factory: "tok2vec" })?;
        let upstream_name = requirement.name.as_str();
        let upstream_component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == upstream_name)
            .ok_or_else(|| PipelineError::InvalidUpstreamTok2Vec {
                component: requirement.component.clone(),
                upstream: upstream_name.to_owned(),
            })?;
        if upstream_component.factory != "tok2vec" {
            return Err(PipelineError::InvalidUpstreamTok2Vec {
                component: requirement.component.clone(),
                upstream: upstream_name.to_owned(),
            });
        }
        Ok(Some(Self {
            tok2vec: Tok2Vec::load(bundle, upstream_name)?,
            parser: parser_name
                .map(|name| DependencyParser::load(bundle, name))
                .transpose()?,
        }))
    }

    fn vectors(&self, doc: &mut Doc) -> Result<crate::Matrix, PipelineError> {
        let vectors = self.tok2vec.forward(doc)?;
        if let Some(parser) = &self.parser {
            parser.annotate(doc, &vectors)?;
        }
        Ok(vectors)
    }

    const fn has_dependency_parser(&self) -> bool {
        self.parser.is_some()
    }
}

struct NerComponents {
    upstream: Option<NerUpstream>,
    sentence_recognizer: Option<SentenceRecognizer>,
    sentencizer: Option<Sentencizer>,
    ner: EntityRecognizer,
    entity_rulers: Vec<EntityRuler>,
}

fn unique_component<'a>(
    bundle: &'a Bundle,
    factory: &'static str,
    required: bool,
) -> Result<Option<&'a ComponentManifest>, PipelineError> {
    let components = bundle
        .manifest()
        .pipeline
        .iter()
        .filter(|component| component.factory == factory)
        .collect::<Vec<_>>();
    match components.as_slice() {
        [] if required => Err(PipelineError::MissingRequiredComponent { factory }),
        [] => Ok(None),
        [component] => Ok(Some(*component)),
        _ => Err(PipelineError::MultipleComponents {
            factory,
            names: components
                .into_iter()
                .map(|component| component.name.clone())
                .collect(),
        }),
    }
}

fn listener_upstream(
    component: &ComponentManifest,
    required: bool,
) -> Result<Option<String>, PipelineError> {
    let has_listener = component
        .nodes
        .iter()
        .any(|node| node.name == "tok2vec-listener");
    if !required && !has_listener {
        return Ok(None);
    }
    match component.settings.get("tok2vec_upstream") {
        None => Ok(Some("tok2vec".to_owned())),
        Some(value) => value
            .as_str()
            .filter(|name| !name.is_empty())
            .map(|name| Some(name.to_owned()))
            .ok_or_else(|| PipelineError::InvalidUpstreamTok2Vec {
                component: component.name.clone(),
                upstream: value.to_string(),
            }),
    }
}

fn merge_upstream(
    selected: &mut Option<UpstreamRequirement>,
    component: &str,
    candidate: Option<String>,
) -> Result<(), PipelineError> {
    let Some(candidate) = candidate else {
        return Ok(());
    };
    if let Some(expected) = selected {
        if expected.name != candidate {
            return Err(PipelineError::ConflictingUpstreamTok2Vec {
                expected: expected.name.clone(),
                actual: candidate,
            });
        }
    } else {
        *selected = Some(UpstreamRequirement {
            name: candidate,
            component: component.to_owned(),
        });
    }
    Ok(())
}

fn load_ner_components(bundle: &Bundle) -> Result<NerComponents, PipelineError> {
    let ner_component = unique_component(bundle, "ner", true)?
        .expect("required component is returned after validation");
    let parser_component = unique_component(bundle, "parser", false)?;
    let senter_component = unique_component(bundle, "senter", false)?;
    let sentencizer_component = unique_component(bundle, "sentencizer", false)?;
    if senter_component.is_some() && sentencizer_component.is_some() {
        return Err(PipelineError::MultipleSentenceBoundaryComponents);
    }

    let ner_index = bundle
        .manifest()
        .pipeline
        .iter()
        .position(|component| component.name == ner_component.name)
        .expect("selected component belongs to the manifest");
    let entity_ruler_components = bundle
        .manifest()
        .pipeline
        .iter()
        .enumerate()
        .filter(|(_, component)| component.factory == "entity_ruler")
        .map(|(index, component)| {
            if index < ner_index {
                Err(PipelineError::UnsupportedComponentOrder {
                    component: component.name.clone(),
                    after: ner_component.name.clone(),
                })
            } else {
                Ok(component)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut ner = EntityRecognizer::load(bundle, &ner_component.name)?;
    let entity_rulers = entity_ruler_components
        .iter()
        .map(|component| EntityRuler::load(bundle, &component.name))
        .collect::<Result<Vec<_>, _>>()?;
    for ruler in &entity_rulers {
        ner.register_labels(ruler.labels());
    }
    let sentence_recognizer = senter_component
        .map(|component| SentenceRecognizer::load(bundle, &component.name))
        .transpose()?;
    let sentencizer = sentencizer_component
        .map(|component| Sentencizer::load(bundle, &component.name))
        .transpose()?;

    let mut upstream_name = None;
    merge_upstream(
        &mut upstream_name,
        &ner_component.name,
        listener_upstream(ner_component, ner.requires_external_tok2vec())?,
    )?;
    if parser_component.is_none() {
        if let (Some(component), Some(recognizer)) =
            (senter_component, sentence_recognizer.as_ref())
        {
            merge_upstream(
                &mut upstream_name,
                &component.name,
                listener_upstream(component, recognizer.requires_external_tok2vec())?,
            )?;
        }
    }
    if let Some(component) = parser_component {
        merge_upstream(
            &mut upstream_name,
            &component.name,
            listener_upstream(component, true)?,
        )?;
    }
    let upstream = NerUpstream::load_optional(
        bundle,
        upstream_name.as_ref(),
        parser_component.map(|component| component.name.as_str()),
    )?;

    Ok(NerComponents {
        upstream,
        sentence_recognizer,
        sentencizer,
        ner,
        entity_rulers,
    })
}

fn ensure_document_start(doc: &mut Doc) {
    if let Some(first) = doc.tokens_mut().first_mut() {
        first.sent_start = 1;
    }
}

fn annotate_ner(
    upstream: Option<&NerUpstream>,
    sentence_recognizer: Option<&SentenceRecognizer>,
    sentencizer: Option<&Sentencizer>,
    ner: &EntityRecognizer,
    entity_rulers: &[EntityRuler],
    doc: &mut Doc,
) -> Result<(), PipelineError> {
    annotate_ner_with_constraints(
        upstream,
        sentence_recognizer,
        sentencizer,
        ner,
        entity_rulers,
        doc,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn annotate_ner_with_constraints(
    upstream: Option<&NerUpstream>,
    sentence_recognizer: Option<&SentenceRecognizer>,
    sentencizer: Option<&Sentencizer>,
    ner: &EntityRecognizer,
    entity_rulers: &[EntityRuler],
    doc: &mut Doc,
    constraints: &[EntityConstraint],
) -> Result<(), PipelineError> {
    let vectors = if let Some(upstream) = upstream {
        Some(upstream.vectors(doc)?)
    } else {
        None
    };
    let has_dependency_parser = upstream.is_some_and(NerUpstream::has_dependency_parser);
    if !has_dependency_parser {
        if let Some(sentence_recognizer) = sentence_recognizer {
            if sentence_recognizer.requires_external_tok2vec() {
                let vectors = vectors
                    .as_ref()
                    .ok_or(SentenceRecognizerError::ExternalTok2VecRequired)?;
                sentence_recognizer.annotate_with_tok2vec(doc, vectors)?;
            } else {
                sentence_recognizer.annotate(doc)?;
            }
        } else if let Some(sentencizer) = sentencizer {
            sentencizer.annotate(doc);
        } else {
            ensure_document_start(doc);
        }
    }
    apply_entity_constraints(doc, constraints)?;
    if ner.requires_external_tok2vec() {
        let vectors = vectors
            .as_ref()
            .ok_or(EntityRecognizerError::ExternalTok2VecRequired)?;
        ner.annotate_with_tok2vec(doc, vectors)?;
    } else {
        ner.annotate(doc)?;
    }
    for ruler in entity_rulers {
        ruler.annotate(doc)?;
    }
    Ok(())
}

/// Language-aware extraction pipeline for supported English and Japanese bundles.
pub enum NerPipeline {
    English(EnglishNerPipeline),
    Japanese(JapaneseNerPipeline),
}

impl NerPipeline {
    /// Load an extraction pipeline based on the bundle manifest language.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported languages or incompatible model data.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        let tokenizer: SharedTokenizer = Arc::new(bundle.load_tokenizer()?);
        Self::load_with_tokenizer(bundle, tokenizer)
    }

    /// Load an extraction pipeline with a tokenizer owned by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported languages or incompatible model data.
    pub fn load_with_tokenizer(
        bundle: &Bundle,
        tokenizer: SharedTokenizer,
    ) -> Result<Self, PipelineError> {
        match bundle.manifest().source.lang.as_str() {
            "en" => Ok(Self::English(EnglishNerPipeline::load_with_tokenizer(
                bundle, tokenizer,
            )?)),
            "ja" => Ok(Self::Japanese(JapaneseNerPipeline::load_with_tokenizer(
                bundle, tokenizer,
            )?)),
            actual => Err(PipelineError::UnsupportedLanguage {
                actual: actual.to_owned(),
            }),
        }
    }

    /// Return the language implementation selected from the bundle.
    #[must_use]
    pub const fn language(&self) -> NerLanguage {
        match self {
            Self::English(_) => NerLanguage::English,
            Self::Japanese(_) => NerLanguage::Japanese,
        }
    }

    /// Return whether this extraction pipeline includes dependency parsing.
    #[must_use]
    pub const fn has_dependency_parser(&self) -> bool {
        match self {
            Self::English(pipeline) => pipeline.has_dependency_parser(),
            Self::Japanese(pipeline) => pipeline.has_dependency_parser(),
        }
    }

    /// Return whether this extraction pipeline includes rule-based sentence
    /// segmentation.
    #[must_use]
    pub const fn has_sentencizer(&self) -> bool {
        match self {
            Self::English(pipeline) => pipeline.has_sentencizer(),
            Self::Japanese(pipeline) => pipeline.has_sentencizer(),
        }
    }

    /// Return whether this extraction pipeline includes trainable sentence
    /// recognition.
    #[must_use]
    pub const fn has_sentence_recognizer(&self) -> bool {
        match self {
            Self::English(pipeline) => pipeline.has_sentence_recognizer(),
            Self::Japanese(pipeline) => pipeline.has_sentence_recognizer(),
        }
    }

    /// Return whether this extraction pipeline includes one or more
    /// post-NER exact phrase rulers.
    #[must_use]
    pub fn has_entity_ruler(&self) -> bool {
        match self {
            Self::English(pipeline) => pipeline.has_entity_ruler(),
            Self::Japanese(pipeline) => pipeline.has_entity_ruler(),
        }
    }

    /// Return entity labels declared by the statistical model or phrase rulers.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::English(pipeline) => pipeline.ner.supported_entity_labels(),
            Self::Japanese(pipeline) => pipeline.ner.supported_entity_labels(),
        }
    }

    /// Return whether the model or a phrase ruler declares an entity label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        match self {
            Self::English(pipeline) => pipeline.ner.supports_entity_label(label),
            Self::Japanese(pipeline) => pipeline.ner.supports_entity_label(label),
        }
    }

    /// Compile requested labels against the model and phrase ruler labels.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        match self {
            Self::English(pipeline) => pipeline.ner.select_entity_labels(labels),
            Self::Japanese(pipeline) => pipeline.ner.select_entity_labels(labels),
        }
    }

    /// Tokenize text and attach sentence boundaries and named entities.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.process(text),
            Self::Japanese(pipeline) => pipeline.process(text),
        }
    }

    /// Tokenize text and run NER with spaCy-compatible preset entity,
    /// blocked-span, and outside-span constraints.
    ///
    /// Constraint offsets are token indexes after this pipeline's tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed inference.
    pub fn process_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Doc, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.process_with_constraints(text, constraints),
            Self::Japanese(pipeline) => pipeline.process_with_constraints(text, constraints),
        }
    }

    /// Return entity spans already attached to a processed document.
    #[must_use]
    pub fn entities(&self, doc: &Doc) -> Vec<NamedEntity> {
        match self {
            Self::English(pipeline) => pipeline.ner.entities(doc),
            Self::Japanese(pipeline) => pipeline.ner.entities(doc),
        }
    }

    /// Return entity spans with labels converted by an exported mapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested mapping is absent.
    pub fn entities_with_mapping_or(
        &self,
        doc: &Doc,
        mapping_name: &str,
        fallback: &str,
    ) -> Result<Vec<NamedEntity>, EntityRecognizerError> {
        match self {
            Self::English(pipeline) => {
                pipeline
                    .ner
                    .entities_with_mapping_or(doc, mapping_name, fallback)
            }
            Self::Japanese(pipeline) => {
                pipeline
                    .ner
                    .entities_with_mapping_or(doc, mapping_name, fallback)
            }
        }
    }

    /// Return token-aligned mapped B/I/O labels from a processed document.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested mapping is absent.
    pub fn token_labels_with_mapping_or(
        &self,
        doc: &Doc,
        mapping_name: &str,
        fallback: &str,
    ) -> Result<Vec<String>, EntityRecognizerError> {
        match self {
            Self::English(pipeline) => {
                pipeline
                    .ner
                    .token_labels_with_mapping_or(doc, mapping_name, fallback)
            }
            Self::Japanese(pipeline) => {
                pipeline
                    .ner
                    .token_labels_with_mapping_or(doc, mapping_name, fallback)
            }
        }
    }

    /// Extract all recognized entity spans.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.extract_entities(text),
            Self::Japanese(pipeline) => pipeline.extract_entities(text),
        }
    }

    /// Extract only entity spans whose labels are included in `labels`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn extract_entities_by_labels(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        self.extract_entities_with_filter(text, &EntityLabelFilter::new(labels))
    }

    /// Extract entity spans accepted by a reusable label filter.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn extract_entities_with_filter(
        &self,
        text: &str,
        filter: &EntityLabelFilter,
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.extract_entities_with_filter(text, filter),
            Self::Japanese(pipeline) => pipeline.extract_entities_with_filter(text, filter),
        }
    }

    /// Extract spans labeled `PERSON`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or inference fails.
    pub fn extract_people(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.extract_people(text),
            Self::Japanese(pipeline) => pipeline.extract_people(text),
        }
    }

    /// Process multiple texts while reusing the loaded tokenizer and model.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn process_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Doc>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.process_batch(texts),
            Self::Japanese(pipeline) => pipeline.process_batch(texts),
        }
    }

    /// Extract all entity spans from multiple texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.extract_entities_batch(texts),
            Self::Japanese(pipeline) => pipeline.extract_entities_batch(texts),
        }
    }

    /// Extract selected entity labels from multiple texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_by_labels_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        labels: &[&str],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        self.extract_entities_with_filter_batch(texts, &EntityLabelFilter::new(labels))
    }

    /// Extract entities accepted by a reusable label filter from multiple texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_with_filter_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.extract_entities_with_filter_batch(texts, filter),
            Self::Japanese(pipeline) => pipeline.extract_entities_with_filter_batch(texts, filter),
        }
    }

    /// Extract `PERSON` spans from multiple texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_people_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        match self {
            Self::English(pipeline) => pipeline.extract_people_batch(texts),
            Self::Japanese(pipeline) => pipeline.extract_people_batch(texts),
        }
    }
}

impl EnglishTaggerPipeline {
    /// Construct the supported portion of `en_core_web_sm`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer data, neural graph structure, weights, or
    /// tagger labels are missing or incompatible.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        let tokenizer: SharedTokenizer = Arc::new(bundle.load_tokenizer()?);
        Self::load_with_tokenizer(bundle, tokenizer)
    }

    /// Construct the supported English tagger with a caller-owned tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the language or model data is incompatible.
    pub fn load_with_tokenizer(
        bundle: &Bundle,
        tokenizer: SharedTokenizer,
    ) -> Result<Self, PipelineError> {
        if bundle.manifest().source.lang != "en" {
            return Err(PipelineError::Language {
                expected: "en",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        Ok(Self {
            tokenizer,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
            tagger: Tagger::load(bundle, "tagger")?,
        })
    }

    /// Tokenize text and attach the tagger's fine-grained `TAG` annotation.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or neural execution fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        let vectors = self.tok2vec.forward(&doc)?;
        let scores = self.tagger.scores(&vectors)?;
        self.tagger.annotate(&mut doc, &scores)?;
        Ok(doc)
    }
}

impl EnglishPipeline {
    /// Construct the tagging and parsing subset of `en_core_web_sm`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer data or a required model graph is
    /// incompatible.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        let tokenizer: SharedTokenizer = Arc::new(bundle.load_tokenizer()?);
        Self::load_with_tokenizer(bundle, tokenizer)
    }

    /// Construct the English pipeline with a caller-owned tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the language or model data is incompatible.
    pub fn load_with_tokenizer(
        bundle: &Bundle,
        tokenizer: SharedTokenizer,
    ) -> Result<Self, PipelineError> {
        if bundle.manifest().source.lang != "en" {
            return Err(PipelineError::Language {
                expected: "en",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        Ok(Self {
            tokenizer,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
            tagger: Tagger::load(bundle, "tagger")?,
            parser: DependencyParser::load(bundle, "parser")?,
            ner: EntityRecognizer::load(bundle, "ner")?,
        })
    }

    /// Return the entity labels declared by the loaded English model.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.ner.supported_entity_labels()
    }

    /// Return whether the loaded English model declares an entity label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        self.ner.supports_entity_label(label)
    }

    /// Compile requested labels against those declared by the loaded English model.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        self.ner.select_entity_labels(labels)
    }

    /// Tokenize text and attach tags, dependencies, and named entities.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or neural execution fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        self.process_with_constraints(text, &[])
    }

    /// Run the full English pipeline with spaCy-compatible NER constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed neural execution.
    pub fn process_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        let vectors = self.tok2vec.forward(&doc)?;
        let tag_scores = self.tagger.scores(&vectors)?;
        self.tagger.annotate(&mut doc, &tag_scores)?;
        self.parser.annotate(&mut doc, &vectors)?;
        apply_entity_constraints(&mut doc, constraints)?;
        self.ner.annotate(&mut doc)?;
        Ok(doc)
    }

    /// Extract all recognized English entity spans.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, tagging, parsing, or NER inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities(&doc))
    }

    /// Extract only English entity spans whose labels are included in `labels`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_entities_by_labels(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        self.extract_entities_with_filter(text, &EntityLabelFilter::new(labels))
    }

    /// Extract English entity spans accepted by a reusable label filter.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, tagging, parsing, or NER inference fails.
    pub fn extract_entities_with_filter(
        &self,
        text: &str,
        filter: &EntityLabelFilter,
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_with_filter(&doc, filter))
    }

    /// Extract spans labeled `PERSON` by the English model.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, tagging, parsing, or NER inference fails.
    pub fn extract_people(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_by_label(&doc, "PERSON"))
    }
}

impl EnglishNerPipeline {
    /// Construct an extraction-only English pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer or NER data is missing or incompatible.
    /// When a parser component is present, its `tok2vec` data is also required.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        let tokenizer: SharedTokenizer = Arc::new(bundle.load_tokenizer()?);
        Self::load_with_tokenizer(bundle, tokenizer)
    }

    /// Construct the extraction-only English pipeline with a caller-owned
    /// tokenizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the language or model data is incompatible.
    pub fn load_with_tokenizer(
        bundle: &Bundle,
        tokenizer: SharedTokenizer,
    ) -> Result<Self, PipelineError> {
        if bundle.manifest().source.lang != "en" {
            return Err(PipelineError::Language {
                expected: "en",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        let components = load_ner_components(bundle)?;
        Ok(Self {
            tokenizer,
            upstream: components.upstream,
            sentence_recognizer: components.sentence_recognizer,
            sentencizer: components.sentencizer,
            ner: components.ner,
            entity_rulers: components.entity_rulers,
        })
    }

    /// Return whether this extraction pipeline includes dependency parsing.
    #[must_use]
    pub const fn has_dependency_parser(&self) -> bool {
        match &self.upstream {
            Some(upstream) => upstream.has_dependency_parser(),
            None => false,
        }
    }

    /// Return whether this extraction pipeline includes rule-based sentence
    /// segmentation.
    #[must_use]
    pub const fn has_sentencizer(&self) -> bool {
        self.sentencizer.is_some()
    }

    /// Return whether this extraction pipeline includes trainable sentence
    /// recognition.
    #[must_use]
    pub const fn has_sentence_recognizer(&self) -> bool {
        self.sentence_recognizer.is_some()
    }

    /// Return whether this pipeline includes one or more post-NER exact phrase
    /// rulers.
    #[must_use]
    pub fn has_entity_ruler(&self) -> bool {
        !self.entity_rulers.is_empty()
    }

    /// Return labels declared by the English model or phrase rulers.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.ner.supported_entity_labels()
    }

    /// Return whether the English model or a phrase ruler declares a label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        self.ner.supports_entity_label(label)
    }

    /// Compile requested labels against the English model and phrase rulers.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        self.ner.select_entity_labels(labels)
    }

    /// Tokenize English text and attach sentence boundaries and named entities.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        self.process_with_constraints(text, &[])
    }

    /// Tokenize English text and run NER with spaCy-compatible constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed inference.
    pub fn process_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        self.annotate_with_constraints(&mut doc, constraints)?;
        Ok(doc)
    }

    fn annotate(&self, doc: &mut Doc) -> Result<(), PipelineError> {
        annotate_ner(
            self.upstream.as_ref(),
            self.sentence_recognizer.as_ref(),
            self.sentencizer.as_ref(),
            &self.ner,
            &self.entity_rulers,
            doc,
        )
    }

    fn annotate_with_constraints(
        &self,
        doc: &mut Doc,
        constraints: &[EntityConstraint],
    ) -> Result<(), PipelineError> {
        annotate_ner_with_constraints(
            self.upstream.as_ref(),
            self.sentence_recognizer.as_ref(),
            self.sentencizer.as_ref(),
            &self.ner,
            &self.entity_rulers,
            doc,
            constraints,
        )
    }

    /// Extract all recognized English entity spans.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities(&doc))
    }

    /// Extract only English entity spans whose labels are included in `labels`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_entities_by_labels(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        self.extract_entities_with_filter(text, &EntityLabelFilter::new(labels))
    }

    /// Extract English entity spans accepted by a reusable label filter.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_entities_with_filter(
        &self,
        text: &str,
        filter: &EntityLabelFilter,
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_with_filter(&doc, filter))
    }

    /// Extract spans labeled `PERSON` by the English model.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_people(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_by_label(&doc, "PERSON"))
    }

    /// Process multiple English texts while reusing the tokenizer and model.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn process_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Doc>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(document)
            })
            .collect()
    }

    /// Extract all entity spans from multiple English texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(self.ner.entities(&document))
            })
            .collect()
    }

    /// Extract selected entity labels from multiple English texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_by_labels_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        labels: &[&str],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        self.extract_entities_with_filter_batch(texts, &EntityLabelFilter::new(labels))
    }

    /// Extract entities accepted by a reusable label filter from multiple English texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_with_filter_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(self.ner.entities_with_filter(&document, filter))
            })
            .collect()
    }

    /// Extract `PERSON` spans from multiple English texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_people_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(self.ner.entities_by_label(&document, "PERSON"))
            })
            .collect()
    }
}

impl JapaneseNerPipeline {
    /// Load the tokenizer and NER model from `ja_core_news_sm`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer assets or the NER model are incompatible.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        let tokenizer: SharedTokenizer = Arc::new(bundle.load_tokenizer()?);
        Self::load_with_tokenizer(bundle, tokenizer)
    }

    /// Load the Japanese NER model with a tokenizer owned by the caller.
    ///
    /// The tokenizer must reproduce the token attributes and boundaries used
    /// to train the exported model.
    ///
    /// # Errors
    ///
    /// Returns an error if the language or NER model is incompatible.
    pub fn load_with_tokenizer(
        bundle: &Bundle,
        tokenizer: SharedTokenizer,
    ) -> Result<Self, PipelineError> {
        if bundle.manifest().source.lang != "ja" {
            return Err(PipelineError::Language {
                expected: "ja",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        let components = load_ner_components(bundle)?;
        Ok(Self {
            tokenizer,
            upstream: components.upstream,
            sentence_recognizer: components.sentence_recognizer,
            sentencizer: components.sentencizer,
            ner: components.ner,
            entity_rulers: components.entity_rulers,
        })
    }

    /// Return whether this extraction pipeline includes dependency parsing.
    #[must_use]
    pub const fn has_dependency_parser(&self) -> bool {
        match &self.upstream {
            Some(upstream) => upstream.has_dependency_parser(),
            None => false,
        }
    }

    /// Return whether this extraction pipeline includes rule-based sentence
    /// segmentation.
    #[must_use]
    pub const fn has_sentencizer(&self) -> bool {
        self.sentencizer.is_some()
    }

    /// Return whether this extraction pipeline includes trainable sentence
    /// recognition.
    #[must_use]
    pub const fn has_sentence_recognizer(&self) -> bool {
        self.sentence_recognizer.is_some()
    }

    /// Return whether this pipeline includes one or more post-NER exact phrase
    /// rulers.
    #[must_use]
    pub fn has_entity_ruler(&self) -> bool {
        !self.entity_rulers.is_empty()
    }

    /// Return labels declared by the Japanese model or phrase rulers.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.ner.supported_entity_labels()
    }

    /// Return whether the Japanese model or a phrase ruler declares a label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        self.ner.supports_entity_label(label)
    }

    /// Compile requested labels against the Japanese model and phrase rulers.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        self.ner.select_entity_labels(labels)
    }

    /// Tokenize Japanese text and attach `ENT_IOB`/`ENT_TYPE` annotations.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or NER inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        self.process_with_constraints(text, &[])
    }

    /// Tokenize Japanese text and run NER with spaCy-compatible constraints.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid constraints or failed inference.
    pub fn process_with_constraints(
        &self,
        text: &str,
        constraints: &[EntityConstraint],
    ) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        self.annotate_with_constraints(&mut doc, constraints)?;
        Ok(doc)
    }

    fn annotate(&self, doc: &mut Doc) -> Result<(), PipelineError> {
        annotate_ner(
            self.upstream.as_ref(),
            self.sentence_recognizer.as_ref(),
            self.sentencizer.as_ref(),
            &self.ner,
            &self.entity_rulers,
            doc,
        )
    }

    fn annotate_with_constraints(
        &self,
        doc: &mut Doc,
        constraints: &[EntityConstraint],
    ) -> Result<(), PipelineError> {
        annotate_ner_with_constraints(
            self.upstream.as_ref(),
            self.sentence_recognizer.as_ref(),
            self.sentencizer.as_ref(),
            &self.ner,
            &self.entity_rulers,
            doc,
            constraints,
        )
    }

    /// Extract all recognized entity spans.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or NER inference fails.
    pub fn extract_entities(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities(&doc))
    }

    /// Extract only Japanese entity spans whose labels are included in `labels`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or NER inference fails.
    pub fn extract_entities_by_labels(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        self.extract_entities_with_filter(text, &EntityLabelFilter::new(labels))
    }

    /// Extract Japanese entity spans accepted by a reusable label filter.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_entities_with_filter(
        &self,
        text: &str,
        filter: &EntityLabelFilter,
    ) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_with_filter(&doc, filter))
    }

    /// Extract spans labeled `PERSON` by the Japanese model.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or NER inference fails.
    pub fn extract_people(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_by_label(&doc, "PERSON"))
    }

    /// Process multiple Japanese texts while reusing the loaded dictionary and
    /// neural model.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn process_batch<S: AsRef<str>>(&self, texts: &[S]) -> Result<Vec<Doc>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(document)
            })
            .collect()
    }

    /// Extract all entity spans from multiple texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(self.ner.entities(&document))
            })
            .collect()
    }

    /// Extract selected entity labels from multiple Japanese texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_by_labels_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        labels: &[&str],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        self.extract_entities_with_filter_batch(texts, &EntityLabelFilter::new(labels))
    }

    /// Extract entities accepted by a reusable label filter from multiple Japanese texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_entities_with_filter_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        filter: &EntityLabelFilter,
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(self.ner.entities_with_filter(&document, filter))
            })
            .collect()
    }

    /// Extract `PERSON` spans from multiple texts.
    ///
    /// # Errors
    ///
    /// Returns the first tokenization or inference error.
    pub fn extract_people_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<NamedEntity>>, PipelineError> {
        let mut tokenizer = self.tokenizer.session();
        texts
            .iter()
            .map(|text| {
                let mut document = tokenizer.tokenize(text.as_ref())?;
                self.annotate(&mut document)?;
                Ok(self.ner.entities_by_label(&document, "PERSON"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_document_start, listener_upstream, merge_upstream, JapaneseNerPipeline, NerLanguage,
        PipelineError,
    };
    use spacy_core::Doc;
    use spacy_model::{Bundle, ComponentManifest};
    use spacy_tokenizer::SharedTokenizer;

    #[test]
    fn ner_language_uses_bundle_language_codes_in_json() {
        assert_eq!(NerLanguage::English.code(), "en");
        assert_eq!(NerLanguage::Japanese.code(), "ja");
        assert_eq!(
            serde_json::to_string(&NerLanguage::English).unwrap(),
            "\"en\""
        );
        assert_eq!(
            serde_json::to_string(&NerLanguage::Japanese).unwrap(),
            "\"ja\""
        );
    }

    #[test]
    fn japanese_pipeline_exposes_tokenizer_injection_constructor() {
        let _: fn(&Bundle, SharedTokenizer) -> Result<JapaneseNerPipeline, PipelineError> =
            JapaneseNerPipeline::load_with_tokenizer;
    }

    #[test]
    fn parserless_ner_marks_the_document_start() {
        let mut doc = Doc::from_words(&["Acme", "hired", "Alice"], &[true, true, false]).unwrap();
        ensure_document_start(&mut doc);
        assert_eq!(doc.tokens()[0].sent_start, 1);
        assert_eq!(doc.tokens()[1].sent_start, 0);
        assert_eq!(doc.tokens()[2].sent_start, 0);

        let mut empty = Doc::default();
        ensure_document_start(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn listener_uses_the_exported_custom_upstream_name() {
        let component: ComponentManifest = serde_json::from_value(serde_json::json!({
            "name": "sentence_model",
            "factory": "senter",
            "kind": "trainable",
            "root_node": 0,
            "settings": {"tok2vec_upstream": "encoder"},
            "nodes": [{
                "index": 0,
                "name": "tok2vec-listener",
                "dims": {},
                "refs": {},
                "params": {}
            }]
        }))
        .unwrap();
        assert_eq!(
            listener_upstream(&component, true).unwrap().as_deref(),
            Some("encoder")
        );
    }

    #[test]
    fn listener_defaults_to_the_legacy_tok2vec_name() {
        let component: ComponentManifest = serde_json::from_value(serde_json::json!({
            "name": "ner",
            "factory": "ner",
            "kind": "trainable",
            "root_node": 0,
            "nodes": [{
                "index": 0,
                "name": "tok2vec-listener",
                "dims": {},
                "refs": {},
                "params": {}
            }]
        }))
        .unwrap();
        assert_eq!(
            listener_upstream(&component, true).unwrap().as_deref(),
            Some("tok2vec")
        );
    }

    #[test]
    fn conflicting_listener_upstreams_are_rejected() {
        let mut selected = None;
        merge_upstream(&mut selected, "entities", Some("encoder_a".to_owned())).unwrap();
        let error = merge_upstream(
            &mut selected,
            "sentence_model",
            Some("encoder_b".to_owned()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PipelineError::ConflictingUpstreamTok2Vec { expected, actual }
                if expected == "encoder_a" && actual == "encoder_b"
        ));
    }
}
