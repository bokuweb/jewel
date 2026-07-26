use std::sync::Arc;

use serde::{Deserialize, Serialize};
use spacy_core::Doc;
use spacy_model::{Bundle, RuntimeTokenizerError};
use spacy_tokenizer::{SharedTokenizer, TokenizeError};
use thiserror::Error;

use crate::{
    DependencyParser, DependencyParserError, EntityLabelFilter, EntityLabelSelection,
    EntityRecognizer, EntityRecognizerError, NamedEntity, Tagger, TaggerError, Tok2Vec,
    Tok2VecError,
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
    tok2vec: Tok2Vec,
    parser: DependencyParser,
    ner: EntityRecognizer,
}

/// Python-free Japanese tokenization and named-entity recognition.
pub struct JapaneseNerPipeline {
    tokenizer: SharedTokenizer,
    tok2vec: Tok2Vec,
    parser: DependencyParser,
    ner: EntityRecognizer,
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

    /// Return the entity labels declared by the loaded model.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::English(pipeline) => pipeline.ner.supported_entity_labels(),
            Self::Japanese(pipeline) => pipeline.ner.supported_entity_labels(),
        }
    }

    /// Return whether the loaded model declares an entity label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        match self {
            Self::English(pipeline) => pipeline.ner.supports_entity_label(label),
            Self::Japanese(pipeline) => pipeline.ner.supports_entity_label(label),
        }
    }

    /// Compile requested labels against those declared by the loaded model.
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
        let mut doc = self.tokenizer.tokenize(text)?;
        let vectors = self.tok2vec.forward(&doc)?;
        let tag_scores = self.tagger.scores(&vectors)?;
        self.tagger.annotate(&mut doc, &tag_scores)?;
        self.parser.annotate(&mut doc, &vectors)?;
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
    /// Construct the extraction-only subset of `en_core_web_sm`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer, `tok2vec`, parser, or NER data is missing
    /// or incompatible.
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
        Ok(Self {
            tokenizer,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
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

    /// Tokenize English text and attach sentence boundaries and named entities.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        self.annotate(&mut doc)?;
        Ok(doc)
    }

    fn annotate(&self, doc: &mut Doc) -> Result<(), PipelineError> {
        let vectors = self.tok2vec.forward(doc)?;
        self.parser.annotate(doc, &vectors)?;
        self.ner.annotate(doc)?;
        Ok(())
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
        Ok(Self {
            tokenizer,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
            parser: DependencyParser::load(bundle, "parser")?,
            ner: EntityRecognizer::load(bundle, "ner")?,
        })
    }

    /// Return the entity labels declared by the loaded Japanese model.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.ner.supported_entity_labels()
    }

    /// Return whether the loaded Japanese model declares an entity label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        self.ner.supports_entity_label(label)
    }

    /// Compile requested labels against those declared by the loaded Japanese model.
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
        let mut doc = self.tokenizer.tokenize(text)?;
        self.annotate(&mut doc)?;
        Ok(doc)
    }

    fn annotate(&self, doc: &mut Doc) -> Result<(), PipelineError> {
        let vectors = self.tok2vec.forward(doc)?;
        self.parser.annotate(doc, &vectors)?;
        self.ner.annotate(doc)?;
        Ok(())
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
    use super::{JapaneseNerPipeline, NerLanguage, PipelineError};
    use spacy_model::Bundle;
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
}
