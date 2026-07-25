use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use delarocha::{VibratoSystemDictionary, VibratoSystemTokenizer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use spacy_core::{Doc, StringStore, TokenData};
use thiserror::Error;

use super::TagBigramRule;

const CURRENT_FORMAT_VERSION: u32 = 1;
const GAP_TAG: &str = "空白";
const EMPTY_MORPH: u64 = 456;
const DEFAULT_MAX_GROUPING_LEN: usize = 24;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DelarochaFeatureSchema {
    #[default]
    Ipadic,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DelarochaTokenizerConfig {
    pub format_version: u32,
    pub language: String,
    pub dictionary_path: String,
    pub dictionary_sha256: Option<String>,
    #[serde(default)]
    pub feature_schema: DelarochaFeatureSchema,
    #[serde(default = "default_ignore_space")]
    pub ignore_space: bool,
    #[serde(default = "default_max_grouping_len")]
    pub max_grouping_len: usize,
    #[serde(default)]
    pub merge_formatted_numbers: bool,
    #[serde(default)]
    pub merge_address_towns: bool,
    #[serde(default)]
    pub compatibility_rules: Vec<DelarochaCompatibilityRule>,
    #[serde(default)]
    pub tag_map: BTreeMap<String, u64>,
    #[serde(default)]
    pub tag_orth_map: BTreeMap<String, BTreeMap<String, u64>>,
    #[serde(default)]
    pub tag_bigram_map: Vec<TagBigramRule>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DelarochaCompatibilityRule {
    pub text: String,
    pub tokens: Vec<DelarochaRuleToken>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DelarochaRuleToken {
    pub surface: String,
    pub tag: String,
    pub inflection: String,
    pub lemma: String,
    pub norm: String,
    pub reading: Option<String>,
}

pub struct DelarochaTokenizer {
    tokenizer: VibratoSystemTokenizer,
    feature_schema: DelarochaFeatureSchema,
    tag_map: BTreeMap<String, u64>,
    tag_orth_map: BTreeMap<String, BTreeMap<String, u64>>,
    tag_bigram_map: BTreeMap<(String, String), (Option<u64>, Option<u64>)>,
    merge_formatted_numbers: bool,
    merge_address_towns: bool,
    compatibility_rules: Vec<DelarochaCompatibilityRule>,
}

#[derive(Debug, Error)]
pub enum DelarochaTokenizerError {
    #[error("delarocha tokenizer configuration is invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported tokenizer format {actual}; this runtime supports {supported}")]
    UnsupportedFormat { actual: u32, supported: u32 },
    #[error("delarocha tokenizer language must be \"ja\", got {0:?}")]
    InvalidLanguage(String),
    #[error("delarocha max_grouping_len must be greater than zero")]
    InvalidMaxGroupingLen,
    #[error("invalid delarocha compatibility rule for {0:?}")]
    InvalidCompatibilityRule(String),
    #[error("unsafe path in delarocha tokenizer bundle: {0:?}")]
    UnsafePath(String),
    #[error("delarocha tokenizer bundle file does not exist: {0}")]
    MissingFile(PathBuf),
    #[error("could not read delarocha dictionary {path}: {source}")]
    ReadDictionary {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("delarocha dictionary checksum is {actual}, expected {expected}")]
    DictionaryChecksum { expected: String, actual: String },
    #[error("delarocha initialization failed: {0}")]
    Initialization(String),
    #[error("delarocha tokenization failed: {0}")]
    Tokenize(String),
    #[error("delarocha returned an invalid byte range {start}..{end} for input length {length}")]
    InvalidRange {
        start: usize,
        end: usize,
        length: usize,
    },
    #[error("unsupported IPADIC feature {0:?}")]
    UnsupportedFeature(String),
    #[error("mapped POS tag {0:?} is absent from the exported spaCy mapping")]
    MissingPos(String),
}

#[derive(Clone, Debug)]
struct DetailedToken {
    surface: String,
    start_byte: usize,
    end_byte: usize,
    start_char: usize,
    tag: String,
    inflection: String,
    lemma: String,
    norm: String,
    reading: Option<String>,
}

impl DelarochaTokenizer {
    /// Load a delarocha tokenizer and its Vibrato system dictionary from a
    /// Python-free model bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when configuration, bundle paths, checksum validation,
    /// or dictionary initialization fails.
    pub fn from_bundle_json(
        bundle_root: impl AsRef<Path>,
        bytes: &[u8],
    ) -> Result<Self, DelarochaTokenizerError> {
        let config: DelarochaTokenizerConfig = serde_json::from_slice(bytes)?;
        Self::from_config(bundle_root, config)
    }

    /// Load a decoded delarocha tokenizer configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration or dictionary is invalid.
    pub fn from_config(
        bundle_root: impl AsRef<Path>,
        config: DelarochaTokenizerConfig,
    ) -> Result<Self, DelarochaTokenizerError> {
        if config.format_version != CURRENT_FORMAT_VERSION {
            return Err(DelarochaTokenizerError::UnsupportedFormat {
                actual: config.format_version,
                supported: CURRENT_FORMAT_VERSION,
            });
        }
        if config.language != "ja" {
            return Err(DelarochaTokenizerError::InvalidLanguage(config.language));
        }
        if config.max_grouping_len == 0 {
            return Err(DelarochaTokenizerError::InvalidMaxGroupingLen);
        }
        for rule in &config.compatibility_rules {
            if rule.text.is_empty()
                || rule.tokens.is_empty()
                || rule.tokens.iter().any(|token| token.surface.is_empty())
                || rule
                    .tokens
                    .iter()
                    .map(|token| token.surface.as_str())
                    .collect::<String>()
                    != rule.text
            {
                return Err(DelarochaTokenizerError::InvalidCompatibilityRule(
                    rule.text.clone(),
                ));
            }
        }

        let dictionary_path = resolve_bundle_path(bundle_root.as_ref(), &config.dictionary_path)?;
        if !dictionary_path.is_file() {
            return Err(DelarochaTokenizerError::MissingFile(dictionary_path));
        }
        if let Some(expected) = &config.dictionary_sha256 {
            let actual = sha256_file(&dictionary_path)?;
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(DelarochaTokenizerError::DictionaryChecksum {
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        let dictionary = VibratoSystemDictionary::from_path(&dictionary_path)
            .map_err(|error| DelarochaTokenizerError::Initialization(error.to_string()))?;
        let tokenizer = dictionary
            .into_tokenizer()
            .ignore_space(config.ignore_space)
            .map_err(|error| DelarochaTokenizerError::Initialization(error.to_string()))?
            .max_grouping_len(config.max_grouping_len);
        let tag_bigram_map = config
            .tag_bigram_map
            .into_iter()
            .map(|rule| ((rule.tag, rule.next_tag), (rule.pos, rule.next_pos)))
            .collect();

        let mut compatibility_rules = config.compatibility_rules;
        compatibility_rules.sort_by_key(|rule| std::cmp::Reverse(rule.text.len()));
        Ok(Self {
            tokenizer,
            feature_schema: config.feature_schema,
            tag_map: config.tag_map,
            tag_orth_map: config.tag_orth_map,
            tag_bigram_map,
            merge_formatted_numbers: config.merge_formatted_numbers,
            merge_address_towns: config.merge_address_towns,
            compatibility_rules,
        })
    }

    /// Tokenize Japanese text with delarocha and adapt IPADIC features to
    /// spaCy's Japanese token attributes.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported dictionary features, missing exported
    /// POS mappings, invalid offsets, or tokenization failures.
    pub fn tokenize(&self, text: &str) -> Result<Doc, DelarochaTokenizerError> {
        if text.is_empty() {
            return Ok(Doc::default());
        }
        let raw = self
            .tokenizer
            .tokenize(text)
            .map_err(|error| DelarochaTokenizerError::Tokenize(error.to_string()))?;
        let mut detailed = Vec::with_capacity(raw.len());
        for token in raw {
            let range = token.range_byte();
            if range.start > range.end || range.end > text.len() {
                return Err(DelarochaTokenizerError::InvalidRange {
                    start: range.start,
                    end: range.end,
                    length: text.len(),
                });
            }
            let attributes = parse_feature(self.feature_schema, token.feature())?;
            detailed.push(DetailedToken {
                surface: token.surface().to_owned(),
                start_byte: range.start,
                end_byte: range.end,
                start_char: token.range_char().start,
                tag: attributes.tag,
                inflection: attributes.inflection,
                lemma: attributes
                    .lemma
                    .unwrap_or_else(|| token.surface().to_owned()),
                norm: attributes
                    .norm
                    .unwrap_or_else(|| token.surface().to_owned()),
                reading: attributes.reading,
            });
        }
        detailed = apply_compatibility_rules(text, detailed, &self.compatibility_rules);
        if self.merge_formatted_numbers {
            detailed = merge_formatted_numbers(detailed);
        }
        if self.merge_address_towns {
            detailed = merge_address_towns(detailed);
        }
        self.to_doc(text, &detailed)
    }

    fn to_doc(
        &self,
        text: &str,
        detailed: &[DetailedToken],
    ) -> Result<Doc, DelarochaTokenizerError> {
        if detailed.is_empty() {
            let mut token = TokenData::new(text, false, 0);
            annotate_gap(&mut token, text, self.unigram_pos(GAP_TAG)?);
            return Ok(Doc::new(vec![token]));
        }

        let mut tokens: Vec<TokenData> = Vec::with_capacity(detailed.len());
        let mut previous_byte = 0;
        let mut previous_char = 0;
        let mut next_pos = None;
        for (index, item) in detailed.iter().enumerate() {
            if item.start_byte < previous_byte || item.end_byte > text.len() {
                return Err(DelarochaTokenizerError::InvalidRange {
                    start: item.start_byte,
                    end: item.end_byte,
                    length: text.len(),
                });
            }
            if item.start_byte > previous_byte {
                let gap = &text[previous_byte..item.start_byte];
                append_gap(&mut tokens, gap, previous_char, self.unigram_pos(GAP_TAG)?);
            }

            let next_tag = detailed.get(index + 1).map(|token| token.tag.as_str());
            let (pos, following_pos) = if let Some(pos) = next_pos.take() {
                (pos, None)
            } else {
                self.resolve_pos(&item.surface, &item.tag, next_tag)?
            };
            next_pos = following_pos;

            let mut token = TokenData::new(&item.surface, false, item.start_char);
            token.tag = StringStore::id(&item.tag);
            token.pos = pos;
            token.lemma = StringStore::id(&item.lemma);
            token.norm = StringStore::id(&item.norm);
            let morph = morph_string(item);
            token.morph = if morph.is_empty() {
                EMPTY_MORPH
            } else {
                StringStore::id(&morph)
            };
            tokens.push(token);
            previous_byte = item.end_byte;
            previous_char = item.start_char + item.surface.chars().count();
        }

        if previous_byte < text.len() {
            append_gap(
                &mut tokens,
                &text[previous_byte..],
                previous_char,
                self.unigram_pos(GAP_TAG)?,
            );
        }
        Ok(Doc::new(tokens))
    }

    fn resolve_pos(
        &self,
        orth: &str,
        tag: &str,
        next_tag: Option<&str>,
    ) -> Result<(u64, Option<u64>), DelarochaTokenizerError> {
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

    fn unigram_pos(&self, tag: &str) -> Result<u64, DelarochaTokenizerError> {
        self.tag_map
            .get(tag)
            .copied()
            .ok_or_else(|| DelarochaTokenizerError::MissingPos(tag.to_owned()))
    }
}

#[derive(Debug, Eq, PartialEq)]
struct FeatureAttributes {
    tag: String,
    inflection: String,
    lemma: Option<String>,
    norm: Option<String>,
    reading: Option<String>,
}

fn parse_feature(
    schema: DelarochaFeatureSchema,
    feature: &str,
) -> Result<FeatureAttributes, DelarochaTokenizerError> {
    match schema {
        DelarochaFeatureSchema::Ipadic => parse_ipadic_feature(feature),
    }
}

fn parse_ipadic_feature(feature: &str) -> Result<FeatureAttributes, DelarochaTokenizerError> {
    let fields = feature.split(',').collect::<Vec<_>>();
    if fields.is_empty() || fields[0].is_empty() {
        return Err(DelarochaTokenizerError::UnsupportedFeature(
            feature.to_owned(),
        ));
    }
    let value = |index: usize| {
        fields
            .get(index)
            .copied()
            .filter(|value| !value.is_empty() && *value != "*")
    };
    let tag = ipadic_to_unidic_tag(value(0).unwrap_or_default(), value(1), value(2), value(3))
        .ok_or_else(|| DelarochaTokenizerError::UnsupportedFeature(feature.to_owned()))?;
    let inflection = [value(4), value(5)]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(";");
    let lemma = value(6).map(str::to_owned);
    let reading = value(7).map(str::to_owned);

    Ok(FeatureAttributes {
        tag: tag.to_owned(),
        inflection,
        norm: lemma.clone(),
        lemma,
        reading,
    })
}

fn ipadic_to_unidic_tag<'a>(
    pos: &'a str,
    sub1: Option<&str>,
    sub2: Option<&str>,
    sub3: Option<&str>,
) -> Option<&'a str> {
    match (pos, sub1, sub2, sub3) {
        ("名詞", Some("固有名詞"), Some("人名"), Some("姓")) => {
            Some("名詞-固有名詞-人名-姓")
        }
        ("名詞", Some("固有名詞"), Some("人名"), Some("名")) => {
            Some("名詞-固有名詞-人名-名")
        }
        ("名詞", Some("固有名詞"), Some("人名"), _) => Some("名詞-固有名詞-人名-一般"),
        ("名詞", Some("固有名詞"), Some("地域"), Some("国")) => {
            Some("名詞-固有名詞-地名-国")
        }
        ("名詞", Some("固有名詞"), Some("地域"), _) => Some("名詞-固有名詞-地名-一般"),
        ("名詞", Some("固有名詞"), _, _) => Some("名詞-固有名詞-一般"),
        ("名詞", Some("代名詞"), _, _) => Some("代名詞"),
        ("名詞", Some("数"), _, _) => Some("名詞-数詞"),
        ("名詞", Some("サ変接続"), _, _) => Some("名詞-普通名詞-サ変可能"),
        ("名詞", Some("形容動詞語幹"), _, _) => Some("名詞-普通名詞-形状詞可能"),
        ("名詞", Some("副詞可能"), _, _) => Some("名詞-普通名詞-副詞可能"),
        ("名詞", Some("ナイ形容詞語幹"), _, _) => Some("形状詞-助動詞語幹"),
        ("名詞", Some("接尾"), Some("助数詞"), _) => Some("接尾辞-名詞的-助数詞"),
        ("名詞", Some("接尾"), Some("副詞可能"), _) => Some("接尾辞-名詞的-副詞可能"),
        ("名詞", Some("接尾"), Some("サ変接続"), _) => Some("接尾辞-名詞的-サ変可能"),
        ("名詞", Some("接尾"), _, _) => Some("接尾辞-名詞的-一般"),
        ("名詞", Some("非自立"), Some("助動詞語幹"), _) => Some("名詞-助動詞語幹"),
        ("名詞", Some("非自立"), Some("形容動詞語幹"), _) => {
            Some("名詞-普通名詞-形状詞可能")
        }
        ("名詞", Some("非自立"), Some("副詞可能"), _) => Some("名詞-普通名詞-副詞可能"),
        ("名詞", Some("一般" | "非自立"), _, _) => Some("名詞-普通名詞-一般"),
        ("動詞", Some("自立"), _, _) => Some("動詞-一般"),
        ("動詞", Some("非自立" | "接尾"), _, _) => Some("動詞-非自立可能"),
        ("形容詞", Some("自立"), _, _) => Some("形容詞-一般"),
        ("形容詞", Some("非自立" | "接尾"), _, _) => Some("形容詞-非自立可能"),
        ("助詞", Some("格助詞"), _, _) => Some("助詞-格助詞"),
        ("助詞", Some("連体化" | "並立助詞"), _, _) => Some("助詞-格助詞"),
        ("助詞", Some("係助詞"), _, _) => Some("助詞-係助詞"),
        ("助詞", Some("終助詞"), _, _) => Some("助詞-終助詞"),
        ("助詞", Some("接続助詞"), _, _) => Some("助詞-接続助詞"),
        ("助詞", Some("副助詞" | "副助詞／並立助詞／終助詞"), _, _) => {
            Some("助詞-副助詞")
        }
        ("助詞", Some("副詞化"), _, _) => Some("助詞-副助詞"),
        ("助詞", Some("特殊"), _, _) => Some("助詞-終助詞"),
        ("記号", Some("空白"), _, _) => Some("空白"),
        ("記号", Some("句点"), _, _) => Some("補助記号-句点"),
        ("記号", Some("読点"), _, _) => Some("補助記号-読点"),
        ("記号", Some("括弧開"), _, _) => Some("補助記号-括弧開"),
        ("記号", Some("括弧閉"), _, _) => Some("補助記号-括弧閉"),
        ("記号", Some("アルファベット"), _, _) => Some("記号-文字"),
        ("記号", _, _, _) => Some("補助記号-一般"),
        ("感動詞", _, _, _) => Some("感動詞-一般"),
        ("フィラー", _, _, _) => Some("感動詞-フィラー"),
        ("接頭詞", _, _, _) => Some("接頭辞"),
        ("副詞", _, _, _) => Some("副詞"),
        ("助動詞", _, _, _) => Some("助動詞"),
        ("接続詞", _, _, _) => Some("接続詞"),
        ("連体詞", _, _, _) => Some("連体詞"),
        _ => None,
    }
}

fn apply_compatibility_rules(
    text: &str,
    tokens: Vec<DetailedToken>,
    rules: &[DelarochaCompatibilityRule],
) -> Vec<DetailedToken> {
    if rules.is_empty() {
        return tokens;
    }
    let mut output = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        let mut matched = None;
        for rule in rules {
            let start_byte = tokens[index].start_byte;
            let Some(remaining) = text.get(start_byte..) else {
                continue;
            };
            if !remaining.starts_with(&rule.text) {
                continue;
            }
            let target_end = start_byte + rule.text.len();
            let mut end = index;
            let mut cursor = start_byte;
            while end < tokens.len()
                && tokens[end].start_byte == cursor
                && tokens[end].end_byte <= target_end
            {
                cursor = tokens[end].end_byte;
                end += 1;
            }
            if cursor == target_end {
                matched = Some((rule, end));
                break;
            }
        }

        let Some((rule, end)) = matched else {
            output.push(tokens[index].clone());
            index += 1;
            continue;
        };
        let mut start_byte = tokens[index].start_byte;
        let mut start_char = tokens[index].start_char;
        for piece in &rule.tokens {
            let end_byte = start_byte + piece.surface.len();
            output.push(DetailedToken {
                surface: piece.surface.clone(),
                start_byte,
                end_byte,
                start_char,
                tag: piece.tag.clone(),
                inflection: piece.inflection.clone(),
                lemma: piece.lemma.clone(),
                norm: piece.norm.clone(),
                reading: piece.reading.clone(),
            });
            start_byte = end_byte;
            start_char += piece.surface.chars().count();
        }
        index = end;
    }
    output
}

fn merge_formatted_numbers(tokens: Vec<DetailedToken>) -> Vec<DetailedToken> {
    let mut output = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if tokens[index].tag != "名詞-数詞" || !is_ascii_digits(&tokens[index].surface) {
            output.push(tokens[index].clone());
            index += 1;
            continue;
        }

        let mut end = index;
        while end + 2 < tokens.len() {
            let separator = &tokens[end + 1];
            let following = &tokens[end + 2];
            let contiguous = tokens[end].end_byte == separator.start_byte
                && separator.end_byte == following.start_byte;
            let valid_separator = separator.surface == "."
                || (separator.surface == "," && following.surface.len() == 3);
            if !contiguous
                || !valid_separator
                || following.tag != "名詞-数詞"
                || !is_ascii_digits(&following.surface)
            {
                break;
            }
            end += 2;
        }
        if end == index {
            output.push(tokens[index].clone());
            index += 1;
            continue;
        }

        let surface = tokens[index..=end]
            .iter()
            .map(|token| token.surface.as_str())
            .collect::<String>();
        output.push(DetailedToken {
            surface: surface.clone(),
            start_byte: tokens[index].start_byte,
            end_byte: tokens[end].end_byte,
            start_char: tokens[index].start_char,
            tag: "名詞-数詞".to_owned(),
            inflection: String::new(),
            lemma: surface.clone(),
            norm: surface.replace(',', ""),
            reading: None,
        });
        index = end + 1;
    }
    output
}

fn merge_address_towns(tokens: Vec<DetailedToken>) -> Vec<DetailedToken> {
    let mut output = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        let address_pattern = index + 3 < tokens.len()
            && tokens[index].tag.starts_with("名詞-固有名詞-")
            && tokens[index + 1].surface == "町"
            && tokens[index + 2].tag == "名詞-数詞"
            && is_ascii_digits(&tokens[index + 2].surface)
            && tokens[index + 3].surface == "丁目"
            && tokens[index].end_byte == tokens[index + 1].start_byte
            && tokens[index + 1].end_byte == tokens[index + 2].start_byte
            && tokens[index + 2].end_byte == tokens[index + 3].start_byte;
        if !address_pattern {
            output.push(tokens[index].clone());
            index += 1;
            continue;
        }

        let surface = format!("{}{}", tokens[index].surface, tokens[index + 1].surface);
        let reading = match (
            tokens[index].reading.as_deref(),
            tokens[index + 1].reading.as_deref(),
        ) {
            (Some(first), Some(second)) => Some(format!("{first}{second}")),
            _ => None,
        };
        output.push(DetailedToken {
            surface: surface.clone(),
            start_byte: tokens[index].start_byte,
            end_byte: tokens[index + 1].end_byte,
            start_char: tokens[index].start_char,
            tag: "名詞-固有名詞-地名-一般".to_owned(),
            inflection: String::new(),
            lemma: surface.clone(),
            norm: surface,
            reading,
        });
        index += 2;
    }
    output
}

fn is_ascii_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

fn resolve_bundle_path(root: &Path, relative: &str) -> Result<PathBuf, DelarochaTokenizerError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(DelarochaTokenizerError::UnsafePath(relative.to_owned()));
    }
    Ok(root.join(path))
}

fn sha256_file(path: &Path) -> Result<String, DelarochaTokenizerError> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|source| DelarochaTokenizerError::ReadDictionary {
            path: path.to_path_buf(),
            source,
        })?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count =
            file.read(&mut buffer)
                .map_err(|source| DelarochaTokenizerError::ReadDictionary {
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

fn append_gap(tokens: &mut Vec<TokenData>, gap: &str, char_offset: usize, pos: u64) {
    if gap == " " && !tokens.is_empty() {
        if let Some(last) = tokens.last_mut() {
            last.has_space = true;
        }
        return;
    }
    let mut token = TokenData::new(gap, false, char_offset);
    annotate_gap(&mut token, gap, pos);
    tokens.push(token);
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

const fn default_ignore_space() -> bool {
    true
}

const fn default_max_grouping_len() -> usize {
    DEFAULT_MAX_GROUPING_LEN
}

#[cfg(test)]
mod tests {
    use super::{
        apply_compatibility_rules, ipadic_to_unidic_tag, merge_address_towns,
        merge_formatted_numbers, parse_ipadic_feature, DelarochaCompatibilityRule,
        DelarochaFeatureSchema, DelarochaRuleToken, DelarochaTokenizer, DelarochaTokenizerConfig,
        DetailedToken, FeatureAttributes,
    };
    use delarocha::VibratoSystemDictionary;
    use std::collections::BTreeMap;

    #[test]
    fn maps_contract_entities_to_spacy_tags() {
        assert_eq!(
            ipadic_to_unidic_tag("名詞", Some("固有名詞"), Some("人名"), Some("姓")),
            Some("名詞-固有名詞-人名-姓")
        );
        assert_eq!(
            ipadic_to_unidic_tag("名詞", Some("数"), None, None),
            Some("名詞-数詞")
        );
        assert_eq!(
            ipadic_to_unidic_tag("名詞", Some("固有名詞"), Some("組織"), None),
            Some("名詞-固有名詞-一般")
        );
    }

    #[test]
    fn parses_ipadic_lemma_reading_and_inflection() {
        assert_eq!(
            parse_ipadic_feature("動詞,自立,*,*,五段・ラ行,基本形,支払う,シハラウ,シハラウ")
                .unwrap(),
            FeatureAttributes {
                tag: "動詞-一般".to_owned(),
                inflection: "五段・ラ行;基本形".to_owned(),
                lemma: Some("支払う".to_owned()),
                norm: Some("支払う".to_owned()),
                reading: Some("シハラウ".to_owned()),
            }
        );
    }

    #[test]
    fn config_defaults_are_explicit_and_stable() {
        let config: DelarochaTokenizerConfig = serde_json::from_str(
            r#"{"format_version":1,"language":"ja","dictionary_path":"dictionary.dic.zst","dictionary_sha256":null}"#,
        )
        .unwrap();
        assert!(config.ignore_space);
        assert_eq!(config.max_grouping_len, 24);
    }

    #[test]
    fn exported_rule_restores_sudachi_compound_boundaries() {
        let rule = DelarochaCompatibilityRule {
            text: "株式会社".to_owned(),
            tokens: vec![
                rule_token("株式", "カブシキ"),
                rule_token("会社", "ガイシャ"),
            ],
        };
        let result = apply_compatibility_rules(
            "株式会社青空",
            vec![
                detailed_token("株式会社", 0, 0, "名詞-普通名詞-一般"),
                detailed_token("青空", 12, 4, "名詞-普通名詞-一般"),
            ],
            &[rule],
        );

        assert_eq!(
            result
                .iter()
                .map(|token| token.surface.as_str())
                .collect::<Vec<_>>(),
            ["株式", "会社", "青空"]
        );
        assert_eq!(result[1].start_byte, 6);
        assert_eq!(result[1].start_char, 2);
    }

    #[test]
    fn formatted_number_merge_restores_sudachi_amount_boundary() {
        let result = merge_formatted_numbers(vec![
            detailed_token("12", 0, 0, "名詞-数詞"),
            detailed_token(",", 2, 2, "補助記号-読点"),
            detailed_token("500", 3, 3, "名詞-数詞"),
            detailed_token("円", 6, 6, "名詞-普通名詞-助数詞可能"),
        ]);

        assert_eq!(result[0].surface, "12,500");
        assert_eq!(result[0].norm, "12500");
        assert_eq!(result[0].end_byte, 6);
        assert_eq!(result[1].surface, "円");
    }

    #[test]
    fn address_town_merge_is_limited_to_numbered_chome_context() {
        let result = merge_address_towns(vec![
            detailed_token("山下", 0, 0, "名詞-固有名詞-人名-姓"),
            detailed_token("町", 6, 2, "接尾辞-名詞的-一般"),
            detailed_token("1", 9, 3, "名詞-数詞"),
            detailed_token("丁目", 10, 4, "名詞-普通名詞-助数詞可能"),
        ]);

        assert_eq!(result[0].surface, "山下町");
        assert_eq!(result[0].tag, "名詞-固有名詞-地名-一般");
        assert_eq!(result[1].surface, "1");
    }

    #[test]
    fn tokenizes_contract_text_with_real_ipadic_when_configured() {
        let Some(dictionary_path) =
            std::env::var_os("VIBRATO_SYSTEM_DIC").map(std::path::PathBuf::from)
        else {
            return;
        };
        let text = "甲株式会社の代表取締役山田太郎は、乙株式会社に対し、違約金12,500円及び遅延損害金を支払う。";
        let raw_tokenizer = VibratoSystemDictionary::from_path(&dictionary_path)
            .unwrap()
            .into_tokenizer()
            .ignore_space(true)
            .unwrap()
            .max_grouping_len(24);
        let mut tag_map = BTreeMap::from([("空白".to_owned(), 1)]);
        for token in raw_tokenizer.tokenize(text).unwrap() {
            let attributes = parse_ipadic_feature(token.feature()).unwrap();
            tag_map.insert(attributes.tag, 1);
        }

        let root = dictionary_path.parent().unwrap();
        let config = DelarochaTokenizerConfig {
            format_version: 1,
            language: "ja".to_owned(),
            dictionary_path: dictionary_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            dictionary_sha256: None,
            feature_schema: DelarochaFeatureSchema::Ipadic,
            ignore_space: true,
            max_grouping_len: 24,
            merge_formatted_numbers: true,
            merge_address_towns: true,
            compatibility_rules: Vec::new(),
            tag_map,
            tag_orth_map: BTreeMap::new(),
            tag_bigram_map: Vec::new(),
        };
        let tokenizer = DelarochaTokenizer::from_config(root, config).unwrap();
        let doc = tokenizer.tokenize(text).unwrap();

        assert_eq!(doc.text(), text);
        assert!(!doc.is_empty());
    }

    fn detailed_token(
        surface: &str,
        start_byte: usize,
        start_char: usize,
        tag: &str,
    ) -> DetailedToken {
        DetailedToken {
            surface: surface.to_owned(),
            start_byte,
            end_byte: start_byte + surface.len(),
            start_char,
            tag: tag.to_owned(),
            inflection: String::new(),
            lemma: surface.to_owned(),
            norm: surface.to_owned(),
            reading: None,
        }
    }

    fn rule_token(surface: &str, reading: &str) -> DelarochaRuleToken {
        DelarochaRuleToken {
            surface: surface.to_owned(),
            tag: "名詞-普通名詞-一般".to_owned(),
            inflection: String::new(),
            lemma: surface.to_owned(),
            norm: surface.to_owned(),
            reading: Some(reading.to_owned()),
        }
    }
}
