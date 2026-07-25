use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use spacy_core::{Doc, StringStore, TokenData};
use sudachi::analysis::stateless_tokenizer::StatelessTokenizer;
use sudachi::analysis::{Mode, Tokenize};
use sudachi::config::ConfigBuilder;
use sudachi::dic::dictionary::JapaneseDictionary;
use thiserror::Error;

const CURRENT_FORMAT_VERSION: u32 = 1;
const GAP_TAG: &str = "空白";
const EMPTY_MORPH: u64 = 456;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum SplitMode {
    #[default]
    A,
    B,
    C,
}

impl SplitMode {
    fn sudachi(self) -> Mode {
        match self {
            Self::A => Mode::A,
            Self::B => Mode::B,
            Self::C => Mode::C,
        }
    }
}

impl FromStr for SplitMode {
    type Err = JapaneseTokenizerError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "A" | "a" => Ok(Self::A),
            "B" | "b" => Ok(Self::B),
            "C" | "c" => Ok(Self::C),
            _ => Err(JapaneseTokenizerError::InvalidSplitMode(value.to_owned())),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JapaneseTokenizerConfig {
    pub format_version: u32,
    pub language: String,
    pub split_mode: Option<String>,
    pub config_path: String,
    pub dictionary_path: String,
    pub sudachipy_version: Option<String>,
    pub dictionary_version: Option<String>,
    pub dictionary_sha256: Option<String>,
    #[serde(default)]
    pub tag_map: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    pub tag_orth_map: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>>,
    #[serde(default)]
    pub tag_bigram_map: Vec<TagBigramRule>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TagBigramRule {
    pub tag: String,
    pub next_tag: String,
    pub pos: Option<u64>,
    pub next_pos: Option<u64>,
}

pub struct JapaneseTokenizer {
    dictionary: Arc<JapaneseDictionary>,
    split_mode: SplitMode,
    tag_map: std::collections::BTreeMap<String, u64>,
    tag_orth_map: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>>,
    tag_bigram_map: std::collections::BTreeMap<(String, String), (Option<u64>, Option<u64>)>,
}

#[derive(Debug, Error)]
pub enum JapaneseTokenizerError {
    #[error("Japanese tokenizer configuration is invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported tokenizer format {actual}; this runtime supports {supported}")]
    UnsupportedFormat { actual: u32, supported: u32 },
    #[error("Japanese tokenizer language must be \"ja\", got {0:?}")]
    InvalidLanguage(String),
    #[error("invalid Sudachi split mode {0:?}")]
    InvalidSplitMode(String),
    #[error("unsafe path in Japanese tokenizer bundle: {0:?}")]
    UnsafePath(String),
    #[error("Japanese tokenizer bundle file does not exist: {0}")]
    MissingFile(PathBuf),
    #[error("could not read Sudachi dictionary {path}: {source}")]
    ReadDictionary {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Sudachi dictionary checksum is {actual}, expected {expected}")]
    DictionaryChecksum { expected: String, actual: String },
    #[error("could not read Japanese tokenizer configuration {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Sudachi initialization failed: {0}")]
    SudachiInit(String),
    #[error("Sudachi tokenization failed: {0}")]
    SudachiTokenize(String),
    #[error("Sudachi tokens cannot be aligned to input text at character offset {offset}")]
    Alignment { offset: usize },
    #[error("Sudachi POS tag {0:?} is absent from the exported spaCy mapping")]
    MissingPos(String),
}

#[derive(Clone, Debug)]
struct DetailedToken {
    surface: String,
    tag: String,
    inflection: String,
    lemma: String,
    norm: String,
    reading: Option<String>,
}

impl JapaneseTokenizer {
    /// Load the Japanese tokenizer and its external Sudachi dictionary from a
    /// Python-free model bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if the exported paths are unsafe or missing, the
    /// configuration is invalid, or Sudachi cannot initialize the dictionary.
    pub fn from_bundle_json(
        bundle_root: impl AsRef<Path>,
        bytes: &[u8],
    ) -> Result<Self, JapaneseTokenizerError> {
        let config: JapaneseTokenizerConfig = serde_json::from_slice(bytes)?;
        Self::from_config(bundle_root, config)
    }

    /// Load a decoded Japanese tokenizer configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if bundle validation or Sudachi initialization fails.
    pub fn from_config(
        bundle_root: impl AsRef<Path>,
        config: JapaneseTokenizerConfig,
    ) -> Result<Self, JapaneseTokenizerError> {
        if config.format_version != CURRENT_FORMAT_VERSION {
            return Err(JapaneseTokenizerError::UnsupportedFormat {
                actual: config.format_version,
                supported: CURRENT_FORMAT_VERSION,
            });
        }
        if config.language != "ja" {
            return Err(JapaneseTokenizerError::InvalidLanguage(config.language));
        }
        let split_mode = config
            .split_mode
            .as_deref()
            .unwrap_or("A")
            .parse::<SplitMode>()?;
        let root = bundle_root.as_ref();
        let config_path = resolve_bundle_path(root, &config.config_path)?;
        let dictionary_path = resolve_bundle_path(root, &config.dictionary_path)?;
        for path in [&config_path, &dictionary_path] {
            if !path.is_file() {
                return Err(JapaneseTokenizerError::MissingFile(path.clone()));
            }
        }
        if let Some(expected) = &config.dictionary_sha256 {
            let actual = sha256_file(&dictionary_path)?;
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(JapaneseTokenizerError::DictionaryChecksum {
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        let builder = ConfigBuilder::from_file(&config_path).map_err(|error| {
            JapaneseTokenizerError::ReadConfig {
                path: config_path.clone(),
                source: match error {
                    sudachi::config::ConfigError::Io(source) => source,
                    other => {
                        return JapaneseTokenizerError::SudachiInit(other.to_string());
                    }
                },
            }
        })?;
        let sudachi_config = builder.system_dict(dictionary_path).build();
        let dictionary = JapaneseDictionary::from_cfg(&sudachi_config)
            .map_err(|error| JapaneseTokenizerError::SudachiInit(error.to_string()))?;
        let tag_bigram_map = config
            .tag_bigram_map
            .into_iter()
            .map(|rule| ((rule.tag, rule.next_tag), (rule.pos, rule.next_pos)))
            .collect();

        Ok(Self {
            dictionary: Arc::new(dictionary),
            split_mode,
            tag_map: config.tag_map,
            tag_orth_map: config.tag_orth_map,
            tag_bigram_map,
        })
    }

    #[must_use]
    pub fn split_mode(&self) -> SplitMode {
        self.split_mode
    }

    /// Tokenize Japanese text with the bundled dictionary and spaCy's
    /// whitespace/alignment conventions.
    ///
    /// # Errors
    ///
    /// Returns an error if Sudachi analysis fails, its surfaces cannot be
    /// aligned to the source text, or an exported POS mapping is incomplete.
    pub fn tokenize(&self, text: &str) -> Result<Doc, JapaneseTokenizerError> {
        if text.is_empty() {
            return Ok(Doc::default());
        }
        let tokenizer = StatelessTokenizer::new(self.dictionary.clone());
        let morphemes = tokenizer
            .tokenize(text, self.split_mode.sudachi(), false)
            .map_err(|error| JapaneseTokenizerError::SudachiTokenize(error.to_string()))?;

        let mut detailed = Vec::with_capacity(morphemes.len());
        for morpheme in morphemes.iter() {
            let surface = morpheme.surface().to_string();
            if surface.is_empty() {
                continue;
            }
            let pos = morpheme.part_of_speech();
            let tag = pos[..pos.len().min(4)]
                .iter()
                .filter(|part| part.as_str() != "*")
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join("-");
            let inflection = pos[pos.len().min(4)..]
                .iter()
                .filter(|part| part.as_str() != "*")
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(";");
            detailed.push(DetailedToken {
                surface,
                tag,
                inflection,
                lemma: morpheme.dictionary_form().to_owned(),
                norm: morpheme.normalized_form().to_owned(),
                reading: Some(morpheme.reading_form().to_owned()),
            });
        }
        detailed = collapse_space_tokens(detailed);
        self.align_to_doc(text, &detailed)
    }

    fn align_to_doc(
        &self,
        text: &str,
        detailed: &[DetailedToken],
    ) -> Result<Doc, JapaneseTokenizerError> {
        if detailed.is_empty() {
            return Ok(Doc::default());
        }
        if detailed.iter().all(|token| is_whitespace(&token.surface)) {
            let mut token = TokenData::new(text, false, 0);
            let gap_pos = self.resolve_pos(text, GAP_TAG, None)?.0;
            annotate_gap(&mut token, text, gap_pos);
            return Ok(Doc::new(vec![token]));
        }

        let mut tokens = Vec::with_capacity(detailed.len());
        let mut text_byte = 0;
        let mut text_char = 0;
        let mut next_pos = None;
        for (index, item) in detailed.iter().enumerate() {
            if is_whitespace(&item.surface) {
                continue;
            }
            let remaining = &text[text_byte..];
            let Some(relative_byte) = remaining.find(&item.surface) else {
                return Err(JapaneseTokenizerError::Alignment { offset: text_char });
            };
            if relative_byte > 0 {
                let gap = &remaining[..relative_byte];
                let mut token = TokenData::new(gap, false, text_char);
                let gap_pos = self.resolve_pos(gap, GAP_TAG, None)?.0;
                annotate_gap(&mut token, gap, gap_pos);
                text_char += gap.chars().count();
                text_byte += relative_byte;
                tokens.push(token);
            }

            let next_tag = detailed.get(index + 1).map(|token| token.tag.as_str());
            let (pos, following_pos) = if let Some(pos) = next_pos.take() {
                (pos, None)
            } else {
                self.resolve_pos(&item.surface, &item.tag, next_tag)?
            };
            next_pos = following_pos;

            let mut has_space = false;
            if detailed
                .get(index + 1)
                .is_some_and(|token| token.surface == " ")
            {
                has_space = true;
            }
            let mut token = TokenData::new(&item.surface, has_space, text_char);
            token.tag = StringStore::id(&item.tag);
            token.pos = pos;
            token.lemma = StringStore::id(if item.lemma.is_empty() {
                &item.surface
            } else {
                &item.lemma
            });
            token.norm = StringStore::id(&item.norm);
            let morph = morph_string(item);
            token.morph = if morph.is_empty() {
                EMPTY_MORPH
            } else {
                StringStore::id(&morph)
            };

            let consumed_chars = item.surface.chars().count() + usize::from(has_space);
            let consumed_bytes = item.surface.len() + usize::from(has_space);
            text_char += consumed_chars;
            text_byte += consumed_bytes;
            tokens.push(token);
        }

        if text_byte < text.len() {
            let gap = &text[text_byte..];
            let mut token = TokenData::new(gap, false, text_char);
            let gap_pos = self.resolve_pos(gap, GAP_TAG, None)?.0;
            annotate_gap(&mut token, gap, gap_pos);
            tokens.push(token);
        }
        Ok(Doc::new(tokens))
    }

    fn resolve_pos(
        &self,
        orth: &str,
        tag: &str,
        next_tag: Option<&str>,
    ) -> Result<(u64, Option<u64>), JapaneseTokenizerError> {
        if let Some(pos) = self
            .tag_orth_map
            .get(tag)
            .and_then(|orth_map| orth_map.get(orth))
        {
            return Ok((*pos, None));
        }
        if let Some(next_tag) = next_tag {
            if let Some((current, next)) = self
                .tag_bigram_map
                .get(&(tag.to_owned(), next_tag.to_owned()))
            {
                let current = match current {
                    Some(pos) => *pos,
                    None => self.unigram_pos(tag)?,
                };
                return Ok((current, *next));
            }
        }
        Ok((self.unigram_pos(tag)?, None))
    }

    fn unigram_pos(&self, tag: &str) -> Result<u64, JapaneseTokenizerError> {
        self.tag_map
            .get(tag)
            .copied()
            .ok_or_else(|| JapaneseTokenizerError::MissingPos(tag.to_owned()))
    }
}

fn resolve_bundle_path(root: &Path, relative: &str) -> Result<PathBuf, JapaneseTokenizerError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(JapaneseTokenizerError::UnsafePath(relative.to_owned()));
    }
    Ok(root.join(path))
}

fn sha256_file(path: &Path) -> Result<String, JapaneseTokenizerError> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|source| JapaneseTokenizerError::ReadDictionary {
            path: path.to_path_buf(),
            source,
        })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count =
            file.read(&mut buffer)
                .map_err(|source| JapaneseTokenizerError::ReadDictionary {
                    path: path.to_path_buf(),
                    source,
                })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn collapse_space_tokens(tokens: Vec<DetailedToken>) -> Vec<DetailedToken> {
    let mut result = Vec::with_capacity(tokens.len());
    for token in tokens {
        let duplicate_space = token.tag == GAP_TAG
            && is_whitespace(&token.surface)
            && result.last().is_some_and(|previous: &DetailedToken| {
                previous.tag == GAP_TAG && is_whitespace(&previous.surface)
            });
        if !duplicate_space {
            result.push(token);
        }
    }
    result
}

fn is_whitespace(text: &str) -> bool {
    !text.is_empty() && text.chars().all(char::is_whitespace)
}

fn annotate_gap(token: &mut TokenData, text: &str, pos: u64) {
    token.tag = StringStore::id(GAP_TAG);
    token.pos = pos;
    token.lemma = StringStore::id(text);
    token.norm = StringStore::id(text);
    token.morph = EMPTY_MORPH;
}

fn morph_string(token: &DetailedToken) -> String {
    let mut features = Vec::with_capacity(2);
    if !token.inflection.is_empty() {
        features.push(format!("Inflection={}", token.inflection));
    }
    if let Some(reading) = token
        .reading
        .as_deref()
        .filter(|reading| !reading.is_empty())
    {
        features.push(format!("Reading={}", reading.replace(['=', '|'], "_")));
    }
    features.join("|")
}

#[cfg(test)]
mod tests {
    use super::DetailedToken;
    use super::{collapse_space_tokens, morph_string, SplitMode};

    fn token(surface: &str, tag: &str) -> DetailedToken {
        DetailedToken {
            surface: surface.to_owned(),
            tag: tag.to_owned(),
            inflection: String::new(),
            lemma: surface.to_owned(),
            norm: surface.to_owned(),
            reading: None,
        }
    }

    #[test]
    fn split_mode_is_spacy_compatible() {
        assert_eq!("A".parse::<SplitMode>().unwrap(), SplitMode::A);
        assert_eq!("b".parse::<SplitMode>().unwrap(), SplitMode::B);
        assert!("D".parse::<SplitMode>().is_err());
    }

    #[test]
    fn continuous_sudachi_space_tokens_are_collapsed() {
        let tokens = vec![
            token("私", "代名詞"),
            token(" ", "空白"),
            token(" ", "空白"),
            token("Rust", "名詞-普通名詞-一般"),
        ];
        let result = collapse_space_tokens(tokens);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn japanese_morph_features_match_spacy_serialization() {
        let mut value = token("行っ", "動詞-非自立可能");
        value.inflection = "五段-カ行;連用形-促音便".to_owned();
        value.reading = Some("イッ|異体".to_owned());
        assert_eq!(
            morph_string(&value),
            "Inflection=五段-カ行;連用形-促音便|Reading=イッ_異体"
        );
    }
}
