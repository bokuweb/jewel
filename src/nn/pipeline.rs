use spacy_core::Doc;
use spacy_model::{Bundle, RuntimeTokenizer, RuntimeTokenizerError};
use thiserror::Error;

use crate::{
    DependencyParser, DependencyParserError, EntityRecognizer, EntityRecognizerError, NamedEntity,
    Tagger, TaggerError, Tok2Vec, Tok2VecError,
};

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Tokenizer(#[from] RuntimeTokenizerError),
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
}

/// Python-free English tokenization, `tok2vec`, and fine-grained POS tagging.
pub struct EnglishTaggerPipeline {
    tokenizer: RuntimeTokenizer,
    tok2vec: Tok2Vec,
    tagger: Tagger,
}

/// Python-free English tagging, dependency parsing, and entity recognition.
pub struct EnglishPipeline {
    tokenizer: RuntimeTokenizer,
    tok2vec: Tok2Vec,
    tagger: Tagger,
    parser: DependencyParser,
    ner: EntityRecognizer,
}

/// Python-free English tokenization, sentence parsing, and NER without
/// loading tagger or lemmatizer components that entity extraction does not use.
pub struct EnglishNerPipeline {
    tokenizer: RuntimeTokenizer,
    tok2vec: Tok2Vec,
    parser: DependencyParser,
    ner: EntityRecognizer,
}

/// Python-free Japanese tokenization and named-entity recognition.
pub struct JapaneseNerPipeline {
    tokenizer: RuntimeTokenizer,
    tok2vec: Tok2Vec,
    parser: DependencyParser,
    ner: EntityRecognizer,
}

impl EnglishTaggerPipeline {
    /// Construct the supported portion of `en_core_web_sm`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenizer data, neural graph structure, weights, or
    /// tagger labels are missing or incompatible.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        if bundle.manifest().source.lang != "en" {
            return Err(PipelineError::Language {
                expected: "en",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        Ok(Self {
            tokenizer: bundle.load_tokenizer()?,
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
        if bundle.manifest().source.lang != "en" {
            return Err(PipelineError::Language {
                expected: "en",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        Ok(Self {
            tokenizer: bundle.load_tokenizer()?,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
            tagger: Tagger::load(bundle, "tagger")?,
            parser: DependencyParser::load(bundle, "parser")?,
            ner: EntityRecognizer::load(bundle, "ner")?,
        })
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
        if bundle.manifest().source.lang != "en" {
            return Err(PipelineError::Language {
                expected: "en",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        Ok(Self {
            tokenizer: bundle.load_tokenizer()?,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
            parser: DependencyParser::load(bundle, "parser")?,
            ner: EntityRecognizer::load(bundle, "ner")?,
        })
    }

    /// Tokenize English text and attach sentence boundaries and named entities.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        let vectors = self.tok2vec.forward(&doc)?;
        self.parser.annotate(&mut doc, &vectors)?;
        self.ner.annotate(&mut doc)?;
        Ok(doc)
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

    /// Extract spans labeled `PERSON` by the English model.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization, parsing, or NER inference fails.
    pub fn extract_people(&self, text: &str) -> Result<Vec<NamedEntity>, PipelineError> {
        let doc = self.process(text)?;
        Ok(self.ner.entities_by_label(&doc, "PERSON"))
    }
}

impl JapaneseNerPipeline {
    /// Load the tokenizer and NER model from `ja_core_news_sm`.
    ///
    /// # Errors
    ///
    /// Returns an error if Sudachi assets or the NER model are incompatible.
    pub fn load(bundle: &Bundle) -> Result<Self, PipelineError> {
        if bundle.manifest().source.lang != "ja" {
            return Err(PipelineError::Language {
                expected: "ja",
                actual: bundle.manifest().source.lang.clone(),
            });
        }
        Ok(Self {
            tokenizer: bundle.load_tokenizer()?,
            tok2vec: Tok2Vec::load(bundle, "tok2vec")?,
            parser: DependencyParser::load(bundle, "parser")?,
            ner: EntityRecognizer::load(bundle, "ner")?,
        })
    }

    /// Tokenize Japanese text and attach `ENT_IOB`/`ENT_TYPE` annotations.
    ///
    /// # Errors
    ///
    /// Returns an error if Sudachi tokenization or NER inference fails.
    pub fn process(&self, text: &str) -> Result<Doc, PipelineError> {
        let mut doc = self.tokenizer.tokenize(text)?;
        let vectors = self.tok2vec.forward(&doc)?;
        self.parser.annotate(&mut doc, &vectors)?;
        self.ner.annotate(&mut doc)?;
        Ok(doc)
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
        texts
            .iter()
            .map(|text| self.process(text.as_ref()))
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
        self.process_batch(texts)
            .map(|documents| documents.iter().map(|doc| self.ner.entities(doc)).collect())
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
        self.process_batch(texts).map(|documents| {
            documents
                .iter()
                .map(|doc| self.ner.entities_by_label(doc, "PERSON"))
                .collect()
        })
    }
}
