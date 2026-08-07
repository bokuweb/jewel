use std::{collections::HashSet, sync::OnceLock};

use fancy_regex::Regex;
use spacy_core::{Doc, StringStore, TokenData};
use spacy_model::Bundle;
use thiserror::Error;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Error)]
pub enum EntityRulerError {
    #[error("entity ruler component {0:?} is missing")]
    MissingComponent(String),
    #[error("entity ruler setting {name:?} is missing or invalid")]
    InvalidSetting { name: &'static str },
    #[error("entity ruler phrase matcher attribute {0:?} is unsupported")]
    UnsupportedPhraseMatcherAttribute(String),
    #[error("entity ruler pattern {index} is invalid: {message}")]
    InvalidPattern { index: usize, message: String },
    #[error("entity ruler regular expression failed: {0}")]
    Regex(String),
}

#[derive(Clone, Copy)]
enum PhraseAttribute {
    Id(IdAttribute),
    Boolean(BooleanAttribute),
    EntIob,
    Length,
}

impl PhraseAttribute {
    fn parse(value: &str) -> Result<Self, EntityRulerError> {
        match value {
            "ORTH" | "TEXT" => Ok(Self::Id(IdAttribute::Orth)),
            "LOWER" => Ok(Self::Id(IdAttribute::Lower)),
            "NORM" => Ok(Self::Id(IdAttribute::Norm)),
            "SHAPE" => Ok(Self::Id(IdAttribute::Shape)),
            "LENGTH" => Ok(Self::Length),
            "ENT_IOB" => Ok(Self::EntIob),
            "ENT_TYPE" => Ok(Self::Id(IdAttribute::EntType)),
            "ENT_ID" => Ok(Self::Id(IdAttribute::EntId)),
            "ENT_KB_ID" => Ok(Self::Id(IdAttribute::EntKbId)),
            value if BooleanAttribute::parse(value).is_some() => Ok(Self::Boolean(
                BooleanAttribute::parse(value).expect("checked phrase Boolean attribute"),
            )),
            value => Err(EntityRulerError::UnsupportedPhraseMatcherAttribute(
                value.to_owned(),
            )),
        }
    }

    fn value(
        self,
        token: &TokenData,
        language: RulerLanguage,
        stop_word_ids: &HashSet<u64>,
    ) -> u64 {
        match self {
            Self::Id(attribute) => attribute.value(token),
            Self::Boolean(attribute) => u64::from(attribute.value(token, language, stop_word_ids)),
            Self::EntIob => u64::from(token.ent_iob),
            Self::Length => token.text.chars().count() as u64,
        }
    }
}

struct PhrasePattern {
    label_id: u64,
    ent_id: u64,
    token_ids: Vec<u64>,
}

#[derive(Clone, Copy)]
enum IdAttribute {
    Orth,
    Lower,
    Norm,
    Prefix,
    Suffix,
    Shape,
    Lemma,
    Pos,
    Tag,
    Dep,
    Morph,
    EntType,
    EntId,
    EntKbId,
}

impl IdAttribute {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "ORTH" | "TEXT" => Some(Self::Orth),
            "LOWER" => Some(Self::Lower),
            "NORM" => Some(Self::Norm),
            "PREFIX" => Some(Self::Prefix),
            "SUFFIX" => Some(Self::Suffix),
            "SHAPE" => Some(Self::Shape),
            "LEMMA" => Some(Self::Lemma),
            "POS" => Some(Self::Pos),
            "TAG" => Some(Self::Tag),
            "DEP" => Some(Self::Dep),
            "MORPH" => Some(Self::Morph),
            "ENT_TYPE" => Some(Self::EntType),
            "ENT_ID" => Some(Self::EntId),
            "ENT_KB_ID" => Some(Self::EntKbId),
            _ => None,
        }
    }

    fn value(self, token: &TokenData) -> u64 {
        match self {
            Self::Orth => token.orth,
            Self::Lower => StringStore::id(&token.text.to_lowercase()),
            Self::Norm => {
                if token.norm == 0 {
                    StringStore::id(&token.text.to_lowercase())
                } else {
                    token.norm
                }
            }
            Self::Prefix => StringStore::id(&prefix(&token.text)),
            Self::Suffix => StringStore::id(&suffix(&token.text)),
            Self::Shape => StringStore::id(&word_shape(&token.text)),
            Self::Lemma => token.lemma,
            Self::Pos => token.pos,
            Self::Tag => token.tag,
            Self::Dep => token.dep,
            Self::Morph => token.morph,
            Self::EntType => token.ent_type,
            Self::EntId => token.ent_id,
            Self::EntKbId => token.ent_kb_id,
        }
    }
}

#[derive(Clone, Copy)]
enum TextAttribute {
    Orth,
    Lower,
    Prefix,
    Suffix,
    Shape,
}

impl TextAttribute {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "ORTH" | "TEXT" => Some(Self::Orth),
            "LOWER" => Some(Self::Lower),
            "PREFIX" => Some(Self::Prefix),
            "SUFFIX" => Some(Self::Suffix),
            "SHAPE" => Some(Self::Shape),
            _ => None,
        }
    }

    fn value(self, token: &TokenData) -> String {
        match self {
            Self::Orth => token.text.to_string(),
            Self::Lower => token.text.to_lowercase(),
            Self::Prefix => prefix(&token.text),
            Self::Suffix => suffix(&token.text),
            Self::Shape => word_shape(&token.text),
        }
    }
}

#[derive(Clone, Copy)]
enum BooleanAttribute {
    IsAlpha,
    IsAscii,
    IsBracket,
    IsCurrency,
    IsDigit,
    IsLeftPunct,
    IsLower,
    IsPunct,
    IsQuote,
    IsRightPunct,
    IsSpace,
    IsStop,
    IsTitle,
    IsUpper,
    LikeEmail,
    LikeNum,
    LikeUrl,
    SentStart,
    Spacy,
}

#[derive(Clone, Copy)]
enum RulerLanguage {
    English,
    Other,
}

impl RulerLanguage {
    fn parse(value: &str) -> Self {
        if value == "en" {
            Self::English
        } else {
            Self::Other
        }
    }
}

impl BooleanAttribute {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "IS_ALPHA" => Some(Self::IsAlpha),
            "IS_ASCII" => Some(Self::IsAscii),
            "IS_BRACKET" => Some(Self::IsBracket),
            "IS_CURRENCY" => Some(Self::IsCurrency),
            "IS_DIGIT" => Some(Self::IsDigit),
            "IS_LEFT_PUNCT" => Some(Self::IsLeftPunct),
            "IS_LOWER" => Some(Self::IsLower),
            "IS_PUNCT" => Some(Self::IsPunct),
            "IS_QUOTE" => Some(Self::IsQuote),
            "IS_RIGHT_PUNCT" => Some(Self::IsRightPunct),
            "IS_SENT_START" => Some(Self::SentStart),
            "IS_SPACE" => Some(Self::IsSpace),
            "IS_STOP" => Some(Self::IsStop),
            "IS_TITLE" => Some(Self::IsTitle),
            "IS_UPPER" => Some(Self::IsUpper),
            "LIKE_EMAIL" => Some(Self::LikeEmail),
            "LIKE_NUM" => Some(Self::LikeNum),
            "LIKE_URL" => Some(Self::LikeUrl),
            "SENT_START" => Some(Self::SentStart),
            "SPACY" => Some(Self::Spacy),
            _ => None,
        }
    }

    fn value(
        self,
        token: &TokenData,
        language: RulerLanguage,
        stop_word_ids: &HashSet<u64>,
    ) -> bool {
        let text = token.text.as_ref();
        match self {
            Self::IsAlpha => !text.is_empty() && text.chars().all(char::is_alphabetic),
            Self::IsAscii => text.is_ascii(),
            Self::IsBracket => matches!(text, "(" | ")" | "[" | "]" | "{" | "}" | "<" | ">"),
            Self::IsCurrency => {
                !text.is_empty() && text.chars().all(UnicodeCategories::is_symbol_currency)
            }
            Self::IsDigit => !text.is_empty() && text.chars().all(is_digit),
            Self::IsLeftPunct => matches!(
                text,
                "(" | "["
                    | "{"
                    | "<"
                    | "\""
                    | "'"
                    | "«"
                    | "‘"
                    | "‚"
                    | "‛"
                    | "“"
                    | "„"
                    | "‟"
                    | "‹"
                    | "❮"
                    | "``"
            ),
            Self::IsLower => is_lower(text),
            Self::IsPunct => {
                !text.is_empty() && text.chars().all(UnicodeCategories::is_punctuation)
            }
            Self::IsQuote => matches!(
                text,
                "\"" | "'"
                    | "`"
                    | "«"
                    | "»"
                    | "‘"
                    | "’"
                    | "‚"
                    | "‛"
                    | "“"
                    | "”"
                    | "„"
                    | "‟"
                    | "‹"
                    | "›"
                    | "❮"
                    | "❯"
                    | "''"
                    | "``"
            ),
            Self::IsRightPunct => matches!(
                text,
                ")" | "]" | "}" | ">" | "\"" | "'" | "»" | "’" | "”" | "›" | "❯" | "''"
            ),
            Self::IsSpace => !text.is_empty() && text.chars().all(char::is_whitespace),
            Self::IsStop => stop_word_ids.contains(&StringStore::id(&text.to_lowercase())),
            Self::IsTitle => is_title(text),
            Self::IsUpper => is_upper(text),
            Self::LikeEmail => like_email(text),
            Self::LikeNum => like_num(text, language),
            Self::LikeUrl => like_url(text),
            Self::SentStart => token.sent_start == 1 || token.idx == 0,
            Self::Spacy => token.has_space,
        }
    }
}

#[derive(Clone, Copy)]
enum NumericComparison {
    Equal,
    NotEqual,
    GreaterOrEqual,
    LessOrEqual,
    Greater,
    Less,
}

impl NumericComparison {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "==" => Some(Self::Equal),
            "!=" => Some(Self::NotEqual),
            ">=" => Some(Self::GreaterOrEqual),
            "<=" => Some(Self::LessOrEqual),
            ">" => Some(Self::Greater),
            "<" => Some(Self::Less),
            _ => None,
        }
    }

    fn matches(self, actual: f64, expected: f64) -> bool {
        match self {
            Self::Equal => actual == expected,
            Self::NotEqual => actual != expected,
            Self::GreaterOrEqual => actual >= expected,
            Self::LessOrEqual => actual <= expected,
            Self::Greater => actual > expected,
            Self::Less => actual < expected,
        }
    }
}

#[derive(Clone, Copy)]
enum SetRelation {
    Subset,
    Superset,
    Intersects,
}

impl SetRelation {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "is_subset" => Some(Self::Subset),
            "is_superset" => Some(Self::Superset),
            "intersects" => Some(Self::Intersects),
            _ => None,
        }
    }

    fn matches(self, actual: &[u64], expected: &[u64]) -> bool {
        match self {
            Self::Subset => actual
                .iter()
                .all(|feature| expected.binary_search(feature).is_ok()),
            Self::Superset => expected
                .iter()
                .all(|feature| actual.binary_search(feature).is_ok()),
            Self::Intersects => actual
                .iter()
                .any(|feature| expected.binary_search(feature).is_ok()),
        }
    }

    fn matches_scalar(self, actual: u64, expected: &[u64]) -> bool {
        match self {
            Self::Subset | Self::Intersects => expected.binary_search(&actual).is_ok(),
            Self::Superset => expected.is_empty() || (expected.len() == 1 && expected[0] == actual),
        }
    }
}

enum TokenConstraint {
    Equal(IdAttribute, Vec<u64>),
    In(IdAttribute, Vec<u64>),
    NotIn(IdAttribute, Vec<u64>),
    Regex(TextAttribute, Regex),
    RegexSet(TextAttribute, Vec<Regex>, bool),
    Fuzzy(TextAttribute, String, Option<usize>),
    FuzzySet(TextAttribute, Vec<String>, Option<usize>, bool),
    EntIob(Vec<u8>, bool),
    Boolean(BooleanAttribute, bool),
    Length(NumericComparison, f64),
    LengthSet(Vec<i64>, bool),
    IdSetRelation(IdAttribute, SetRelation, Vec<u64>),
    MorphSet(SetRelation, Vec<u64>),
}

impl TokenConstraint {
    fn matches(
        &self,
        token: &TokenData,
        language: RulerLanguage,
        stop_word_ids: &HashSet<u64>,
    ) -> Result<bool, EntityRulerError> {
        match self {
            Self::Equal(attribute, values) | Self::In(attribute, values) => {
                Ok(values.contains(&attribute.value(token)))
            }
            Self::NotIn(attribute, values) => Ok(!values.contains(&attribute.value(token))),
            Self::Regex(attribute, regex) => regex
                .is_match(&attribute.value(token))
                .map_err(|error| EntityRulerError::Regex(error.to_string())),
            Self::RegexSet(attribute, regexes, negate) => {
                let value = attribute.value(token);
                let mut matched = false;
                for regex in regexes {
                    if regex
                        .is_match(&value)
                        .map_err(|error| EntityRulerError::Regex(error.to_string()))?
                    {
                        matched = true;
                        break;
                    }
                }
                Ok(matched != *negate)
            }
            Self::Fuzzy(attribute, pattern, max_edits) => {
                Ok(fuzzy_matches(&attribute.value(token), pattern, *max_edits))
            }
            Self::FuzzySet(attribute, patterns, max_edits, negate) => {
                let value = attribute.value(token);
                let matched = patterns
                    .iter()
                    .any(|pattern| fuzzy_matches(&value, pattern, *max_edits));
                Ok(matched != *negate)
            }
            Self::EntIob(values, negate) => Ok(values.contains(&token.ent_iob) != *negate),
            Self::Boolean(attribute, expected) => {
                Ok(attribute.value(token, language, stop_word_ids) == *expected)
            }
            Self::Length(comparison, expected) => {
                Ok(comparison.matches(token.text.chars().count() as f64, *expected))
            }
            Self::LengthSet(values, negate) => {
                let matched = i64::try_from(token.text.chars().count())
                    .is_ok_and(|length| values.contains(&length));
                Ok(matched != *negate)
            }
            Self::IdSetRelation(attribute, relation, expected) => {
                Ok(relation.matches_scalar(attribute.value(token), expected))
            }
            Self::MorphSet(comparison, expected) => {
                Ok(comparison.matches(&token.morph_features, expected))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Quantifier {
    Repeat {
        minimum: usize,
        maximum: Option<usize>,
    },
    Negate,
}

impl Quantifier {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "1" => Some(Self::Repeat {
                minimum: 1,
                maximum: Some(1),
            }),
            "!" => Some(Self::Negate),
            "?" => Some(Self::Repeat {
                minimum: 0,
                maximum: Some(1),
            }),
            "*" => Some(Self::Repeat {
                minimum: 0,
                maximum: None,
            }),
            "+" => Some(Self::Repeat {
                minimum: 1,
                maximum: None,
            }),
            _ => parse_repetition(value),
        }
    }
}

fn parse_repetition(value: &str) -> Option<Quantifier> {
    let bounds = value.strip_prefix('{')?.strip_suffix('}')?;
    let (minimum, maximum) = if let Some((minimum, maximum)) = bounds.split_once(',') {
        if maximum.contains(',') || (minimum.is_empty() && maximum.is_empty()) {
            return None;
        }
        let minimum = if minimum.is_empty() {
            0
        } else {
            minimum.parse().ok()?
        };
        let maximum = if maximum.is_empty() {
            None
        } else {
            Some(maximum.parse().ok()?)
        };
        (minimum, maximum)
    } else {
        let exact = bounds.parse().ok()?;
        (exact, Some(exact))
    };
    if maximum.is_some_and(|maximum| minimum > maximum) {
        return None;
    }
    Some(Quantifier::Repeat { minimum, maximum })
}

struct TokenStep {
    quantifier: Quantifier,
    constraints: Vec<TokenConstraint>,
}

impl TokenStep {
    fn matches(
        &self,
        token: &TokenData,
        language: RulerLanguage,
        stop_word_ids: &HashSet<u64>,
    ) -> Result<bool, EntityRulerError> {
        for constraint in &self.constraints {
            if !constraint.matches(token, language, stop_word_ids)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

struct TokenPattern {
    label_id: u64,
    ent_id: u64,
    steps: Vec<TokenStep>,
}

struct RulerMatch {
    label_id: u64,
    ent_id: u64,
    priority: usize,
    start: usize,
    end: usize,
}

/// Supported subset of spaCy's `EntityRuler`.
///
/// Jewel supports post-NER phrase rulers matching spaCy's lexical, Boolean,
/// sentence, whitespace, and upstream entity attributes.
/// Token patterns support text, normalized and structural string attributes,
/// length comparisons, lexical Boolean attributes, upstream entity attributes,
/// `IN`, `NOT_IN`, `REGEX`, direct or set-valued `FUZZY` predicates, wildcard
/// tokens, and simple or bounded repetition operators.
pub struct EntityRuler {
    language: RulerLanguage,
    attribute: PhraseAttribute,
    overwrite: bool,
    patterns: Vec<PhrasePattern>,
    token_patterns: Vec<TokenPattern>,
    stop_word_ids: HashSet<u64>,
    labels: Vec<String>,
    entity_ids: Vec<String>,
}

impl EntityRuler {
    /// Load one named entity ruler component.
    ///
    /// # Errors
    ///
    /// Returns an error when settings or patterns are incompatible.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, EntityRulerError> {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| EntityRulerError::MissingComponent(component_name.to_owned()))?;
        let overwrite = component
            .settings
            .get("overwrite_ents")
            .and_then(serde_json::Value::as_bool)
            .ok_or(EntityRulerError::InvalidSetting {
                name: "overwrite_ents",
            })?;
        let attribute = component
            .settings
            .get("phrase_matcher_attr")
            .and_then(serde_json::Value::as_str)
            .ok_or(EntityRulerError::InvalidSetting {
                name: "phrase_matcher_attr",
            })
            .and_then(PhraseAttribute::parse)?;
        let phrase_values = component
            .settings
            .get("patterns")
            .and_then(serde_json::Value::as_array)
            .ok_or(EntityRulerError::InvalidSetting { name: "patterns" })?;
        let token_values = component
            .settings
            .get("token_patterns")
            .map(|value| {
                value.as_array().ok_or(EntityRulerError::InvalidSetting {
                    name: "token_patterns",
                })
            })
            .transpose()?;
        let stop_word_ids = component
            .settings
            .get("stop_word_ids")
            .map(|value| {
                value
                    .as_array()
                    .ok_or(EntityRulerError::InvalidSetting {
                        name: "stop_word_ids",
                    })?
                    .iter()
                    .map(|value| {
                        value.as_u64().ok_or(EntityRulerError::InvalidSetting {
                            name: "stop_word_ids",
                        })
                    })
                    .collect::<Result<HashSet<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        let mut patterns = Vec::with_capacity(phrase_values.len());
        let mut token_patterns = Vec::with_capacity(token_values.map_or(0, std::vec::Vec::len));
        let mut labels = Vec::new();
        let mut entity_ids = Vec::new();
        for (index, value) in phrase_values.iter().enumerate() {
            let label = value
                .get("label")
                .and_then(serde_json::Value::as_str)
                .filter(|label| !label.is_empty())
                .ok_or_else(|| EntityRulerError::InvalidPattern {
                    index,
                    message: "label is missing or empty".to_owned(),
                })?;
            let token_ids = value
                .get("token_ids")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| EntityRulerError::InvalidPattern {
                    index,
                    message: "token_ids is missing".to_owned(),
                })?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| EntityRulerError::InvalidPattern {
                            index,
                            message: "token_ids must contain unsigned integers".to_owned(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if token_ids.is_empty() {
                return Err(EntityRulerError::InvalidPattern {
                    index,
                    message: "token_ids must not be empty".to_owned(),
                });
            }
            if !labels.iter().any(|known| known == label) {
                labels.push(label.to_owned());
            }
            let ent_id = parse_pattern_id(value, index)?;
            if let Some(id) = value
                .get("id")
                .and_then(serde_json::Value::as_str)
                .filter(|id| !id.is_empty())
            {
                if !entity_ids.iter().any(|known| known == id) {
                    entity_ids.push(id.to_owned());
                }
            }
            patterns.push(PhrasePattern {
                label_id: StringStore::id(label),
                ent_id,
                token_ids,
            });
        }
        for (index, value) in token_values.into_iter().flatten().enumerate() {
            let pattern = parse_token_pattern(value, index)?;
            let label = value
                .get("label")
                .and_then(serde_json::Value::as_str)
                .expect("validated token pattern label");
            if !labels.iter().any(|known| known == label) {
                labels.push(label.to_owned());
            }
            if let Some(id) = value
                .get("id")
                .and_then(serde_json::Value::as_str)
                .filter(|id| !id.is_empty())
            {
                if !entity_ids.iter().any(|known| known == id) {
                    entity_ids.push(id.to_owned());
                }
            }
            token_patterns.push(pattern);
        }
        Ok(Self {
            language: RulerLanguage::parse(&bundle.manifest().source.lang),
            attribute,
            overwrite,
            patterns,
            token_patterns,
            stop_word_ids,
            labels,
            entity_ids,
        })
    }

    /// Return labels declared by phrase and token patterns.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.labels.iter().map(String::as_str)
    }

    /// Return non-empty pattern IDs declared by phrase and token patterns.
    pub fn entity_ids(&self) -> impl Iterator<Item = &str> {
        self.entity_ids.iter().map(String::as_str)
    }

    /// Match phrases and update entity annotations.
    pub fn annotate(&self, doc: &mut Doc) -> Result<(), EntityRulerError> {
        let token_ids = doc
            .tokens()
            .iter()
            .map(|token| {
                self.attribute
                    .value(token, self.language, &self.stop_word_ids)
            })
            .collect::<Vec<_>>();
        let mut matches = Vec::new();
        let mut unique = HashSet::new();
        for (pattern_index, pattern) in self.patterns.iter().enumerate() {
            if pattern.token_ids.len() > token_ids.len() {
                continue;
            }
            for start in 0..=token_ids.len() - pattern.token_ids.len() {
                let end = start + pattern.token_ids.len();
                if token_ids[start..end] == pattern.token_ids
                    && unique.insert((pattern.label_id, start, end))
                {
                    matches.push(RulerMatch {
                        label_id: pattern.label_id,
                        ent_id: pattern.ent_id,
                        priority: pattern_index,
                        start,
                        end,
                    });
                }
            }
        }
        for (pattern_index, pattern) in self.token_patterns.iter().enumerate() {
            for start in 0..doc.len() {
                for end in token_pattern_ends(
                    pattern,
                    doc.tokens(),
                    start,
                    self.language,
                    &self.stop_word_ids,
                )? {
                    if unique.insert((pattern.label_id, start, end)) {
                        matches.push(RulerMatch {
                            label_id: pattern.label_id,
                            ent_id: pattern.ent_id,
                            priority: self.patterns.len() + pattern_index,
                            start,
                            end,
                        });
                    }
                }
            }
        }
        matches.sort_by(|left, right| {
            (right.end - right.start)
                .cmp(&(left.end - left.start))
                .then_with(|| left.start.cmp(&right.start))
                .then_with(|| left.priority.cmp(&right.priority))
        });

        let existing = entity_ranges(doc);
        let mut keep_existing = vec![true; existing.len()];
        let mut seen_tokens = vec![false; doc.len()];
        let mut accepted = Vec::new();
        for found in matches {
            if !self.overwrite
                && doc.tokens()[found.start..found.end]
                    .iter()
                    .any(|token| token.ent_type != 0)
            {
                continue;
            }
            if seen_tokens[found.start] || seen_tokens[found.end - 1] {
                continue;
            }
            for (keep, &(start, end)) in keep_existing.iter_mut().zip(&existing) {
                if start < found.end && end > found.start {
                    *keep = false;
                }
            }
            seen_tokens[found.start..found.end].fill(true);
            accepted.push(found);
        }

        for (&(start, end), keep) in existing.iter().zip(keep_existing) {
            if !keep {
                for token in &mut doc.tokens_mut()[start..end] {
                    token.ent_iob = 2;
                    token.ent_type = 0;
                    token.ent_id = 0;
                    token.ent_kb_id = 0;
                }
            }
        }
        for found in accepted {
            for (offset, token) in doc.tokens_mut()[found.start..found.end]
                .iter_mut()
                .enumerate()
            {
                token.ent_iob = if offset == 0 { 3 } else { 1 };
                token.ent_type = found.label_id;
                token.ent_id = found.ent_id;
                token.ent_kb_id = 0;
            }
        }
        Ok(())
    }
}

fn token_pattern_ends(
    pattern: &TokenPattern,
    tokens: &[TokenData],
    start: usize,
    language: RulerLanguage,
    stop_word_ids: &HashSet<u64>,
) -> Result<Vec<usize>, EntityRulerError> {
    let mut positions = vec![start];
    for step in &pattern.steps {
        let mut next = Vec::new();
        for position in positions {
            match step.quantifier {
                Quantifier::Negate => {
                    if position < tokens.len()
                        && !step.matches(&tokens[position], language, stop_word_ids)?
                    {
                        next.push(position + 1);
                    }
                }
                Quantifier::Repeat { minimum, maximum } => {
                    let mut cursor = position;
                    let mut count = 0;
                    if minimum == 0 {
                        next.push(cursor);
                    }
                    while maximum.is_none_or(|maximum| count < maximum)
                        && cursor < tokens.len()
                        && step.matches(&tokens[cursor], language, stop_word_ids)?
                    {
                        cursor += 1;
                        count += 1;
                        if count >= minimum {
                            next.push(cursor);
                        }
                    }
                }
            }
        }
        next.sort_unstable();
        next.dedup();
        if next.is_empty() {
            return Ok(next);
        }
        positions = next;
    }
    positions.retain(|end| *end > start);
    Ok(positions)
}

fn is_digit(character: char) -> bool {
    character.is_number_decimal_digit()
        || character
            .to_string()
            .nfkc()
            .all(|normalized| normalized.is_ascii_digit())
}

fn prefix(text: &str) -> String {
    text.chars().next().into_iter().collect()
}

fn suffix(text: &str) -> String {
    let mut characters = text.chars().rev().take(3).collect::<Vec<_>>();
    characters.reverse();
    characters.into_iter().collect()
}

fn word_shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".to_owned();
    }
    let mut shape = String::new();
    let mut last = None;
    let mut sequence = 0;
    for character in text.chars() {
        let shape_character = if character.is_alphabetic() {
            if character.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if is_digit(character) {
            'd'
        } else {
            character
        };
        if last == Some(shape_character) {
            sequence += 1;
        } else {
            sequence = 0;
            last = Some(shape_character);
        }
        if sequence < 4 {
            shape.push(shape_character);
        }
    }
    shape
}

fn is_lower(text: &str) -> bool {
    text.chars().any(char::is_lowercase) && !text.chars().any(char::is_uppercase)
}

fn is_upper(text: &str) -> bool {
    text.chars().any(char::is_uppercase) && !text.chars().any(char::is_lowercase)
}

fn is_title(text: &str) -> bool {
    let mut has_cased = false;
    let mut previous_cased = false;
    for character in text.chars() {
        if character.is_uppercase() {
            if previous_cased {
                return false;
            }
            has_cased = true;
            previous_cased = true;
        } else if character.is_lowercase() {
            if !previous_cased {
                return false;
            }
            has_cased = true;
            previous_cased = true;
        } else {
            previous_cased = false;
        }
    }
    has_cased
}

fn like_num(text: &str, language: RulerLanguage) -> bool {
    let stripped = text
        .chars()
        .next()
        .filter(|character| matches!(character, '+' | '-' | '±' | '~'))
        .map_or(text, |character| &text[character.len_utf8()..]);
    let compact = stripped.replace([',', '.'], "");
    if !compact.is_empty() && compact.chars().all(is_digit) {
        return true;
    }
    if let Some((numerator, denominator)) = compact.split_once('/') {
        if !numerator.is_empty()
            && !denominator.is_empty()
            && !denominator.contains('/')
            && numerator.chars().all(is_digit)
            && denominator.chars().all(is_digit)
        {
            return true;
        }
    }
    matches!(language, RulerLanguage::English) && english_like_num(&compact)
}

fn english_like_num(text: &str) -> bool {
    const CARDINALS: &[&str] = &[
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "quadrillion",
        "quintillion",
        "sextillion",
        "septillion",
        "octillion",
        "nonillion",
        "decillion",
        "gajillion",
        "bazillion",
    ];
    const ORDINALS: &[&str] = &[
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
        "nineteenth",
        "twentieth",
        "thirtieth",
        "fortieth",
        "fiftieth",
        "sixtieth",
        "seventieth",
        "eightieth",
        "ninetieth",
        "hundredth",
        "thousandth",
        "millionth",
        "billionth",
        "trillionth",
        "quadrillionth",
        "quintillionth",
        "sextillionth",
        "septillionth",
        "octillionth",
        "nonillionth",
        "decillionth",
        "gajillionth",
        "bazillionth",
    ];
    let lower = text.to_ascii_lowercase();
    if CARDINALS.contains(&lower.as_str()) || ORDINALS.contains(&lower.as_str()) {
        return true;
    }
    ["st", "nd", "rd", "th"].iter().any(|suffix| {
        lower
            .strip_suffix(suffix)
            .is_some_and(|prefix| !prefix.is_empty() && prefix.chars().all(is_digit))
    })
}

fn like_email(text: &str) -> bool {
    static EMAIL: OnceLock<Regex> = OnceLock::new();
    EMAIL
        .get_or_init(|| {
            Regex::new(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
                .expect("spaCy-compatible email regex is valid")
        })
        .is_match(text)
        .unwrap_or(false)
}

fn like_url(text: &str) -> bool {
    static URL: OnceLock<Regex> = OnceLock::new();
    if text.starts_with("http://") || text.starts_with("https://") {
        return true;
    }
    if text.starts_with("www.") && text.chars().count() >= 5 {
        return true;
    }
    if text.is_empty() || text.starts_with('.') || text.ends_with('.') || text.contains('@') {
        return false;
    }
    if like_public_ipv4_url(text) {
        return true;
    }
    URL.get_or_init(|| {
        Regex::new(
            r"^(?:[\w+.-]{2,}://)?(?:[A-Za-z0-9\u{00a1}-\u{ffff}][A-Za-z0-9\u{00a1}-\u{ffff}_-]{0,62}\.)+[a-z\u{00df}-\u{00f6}\u{00f8}-\u{00ff}]{2,63}(?::\d{2,5})?(?:[/?#]\S*)?$",
        )
        .expect("spaCy-compatible URL regex is valid")
    })
    .is_match(text)
    .unwrap_or(false)
}

fn like_public_ipv4_url(text: &str) -> bool {
    let authority = text.split(['/', '?', '#']).next().unwrap_or(text);
    let (host, port) = authority
        .split_once(':')
        .map_or((authority, None), |(host, port)| (host, Some(port)));
    if port.is_some_and(|port| {
        !(2..=5).contains(&port.len()) || !port.chars().all(|character| character.is_ascii_digit())
    }) {
        return false;
    }
    let octets = host
        .split('.')
        .map(str::parse::<u8>)
        .collect::<Result<Vec<_>, _>>();
    let Ok(octets) = octets else {
        return false;
    };
    if octets.len() != 4 || octets[0] == 0 || octets[0] >= 224 || octets[3] == 0 || octets[3] == 255
    {
        return false;
    }
    !matches!(octets.as_slice(), [10 | 127, ..])
        && !matches!(octets.as_slice(), [169, 254, ..])
        && !matches!(octets.as_slice(), [192, 168, ..])
        && !matches!(octets.as_slice(), [172, 16..=31, ..])
}

fn fuzzy_matches(input: &str, pattern: &str, max_edits: Option<usize>) -> bool {
    if input == pattern {
        return true;
    }
    let pattern_length = pattern.chars().count();
    let max_edits = max_edits.unwrap_or_else(|| default_fuzzy_edits(pattern_length));
    bounded_levenshtein(input, pattern, max_edits).is_some()
}

fn default_fuzzy_edits(pattern_length: usize) -> usize {
    let scaled = pattern_length.saturating_mul(3);
    let quotient = scaled / 10;
    let remainder = scaled % 10;
    let rounded = if remainder > 5 || (remainder == 5 && quotient % 2 == 1) {
        quotient + 1
    } else {
        quotient
    };
    rounded.max(2)
}

fn bounded_levenshtein(left: &str, right: &str, max_edits: usize) -> Option<usize> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > max_edits {
        return None;
    }
    if right.is_empty() {
        return (left.len() <= max_edits).then_some(left.len());
    }
    let outside = max_edits + 1;
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![outside; right.len() + 1];
    for (left_index, left_character) in left.iter().enumerate() {
        let row = left_index + 1;
        current.fill(outside);
        current[0] = row;
        let start = row.saturating_sub(max_edits).max(1);
        let end = row.saturating_add(max_edits).min(right.len());
        if start > end {
            return None;
        }
        let mut row_minimum = if start == 1 { current[0] } else { outside };
        for column in start..=end {
            let substitution =
                previous[column - 1] + usize::from(*left_character != right[column - 1]);
            let insertion = current[column - 1].saturating_add(1);
            let deletion = previous[column].saturating_add(1);
            current[column] = substitution.min(insertion).min(deletion);
            row_minimum = row_minimum.min(current[column]);
        }
        if row_minimum > max_edits {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= max_edits).then_some(previous[right.len()])
}

fn parse_token_pattern(
    value: &serde_json::Value,
    index: usize,
) -> Result<TokenPattern, EntityRulerError> {
    let label = value
        .get("label")
        .and_then(serde_json::Value::as_str)
        .filter(|label| !label.is_empty())
        .ok_or_else(|| invalid_pattern(index, "label is missing or empty"))?;
    let tokens = value
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .filter(|tokens| !tokens.is_empty())
        .ok_or_else(|| invalid_pattern(index, "tokens is missing or empty"))?;
    let mut steps = Vec::with_capacity(tokens.len());
    for token in tokens {
        let quantifier = token
            .get("op")
            .and_then(serde_json::Value::as_str)
            .and_then(Quantifier::parse)
            .ok_or_else(|| invalid_pattern(index, "token has an invalid op"))?;
        let constraints = token
            .get("constraints")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_pattern(index, "token constraints are missing"))?
            .iter()
            .map(|constraint| parse_token_constraint(constraint, index))
            .collect::<Result<Vec<_>, _>>()?;
        steps.push(TokenStep {
            quantifier,
            constraints,
        });
    }
    Ok(TokenPattern {
        label_id: StringStore::id(label),
        ent_id: parse_pattern_id(value, index)?,
        steps,
    })
}

fn parse_pattern_id(value: &serde_json::Value, index: usize) -> Result<u64, EntityRulerError> {
    match value.get("id") {
        None => Ok(0),
        Some(value) => value
            .as_str()
            .map(StringStore::id)
            .ok_or_else(|| invalid_pattern(index, "id must be a string")),
    }
}

fn parse_token_constraint(
    value: &serde_json::Value,
    index: usize,
) -> Result<TokenConstraint, EntityRulerError> {
    let attribute_name = value
        .get("attribute")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_pattern(index, "constraint attribute is missing"))?;
    let kind = value
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| invalid_pattern(index, "constraint kind is missing"))?;
    if kind == "boolean" {
        let attribute = BooleanAttribute::parse(attribute_name)
            .ok_or_else(|| invalid_pattern(index, "boolean attribute is unsupported"))?;
        let expected = value
            .get("value")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| invalid_pattern(index, "boolean value is missing"))?;
        return Ok(TokenConstraint::Boolean(attribute, expected));
    }
    if kind == "iob" {
        if attribute_name != "ENT_IOB" {
            return Err(invalid_pattern(index, "IOB attribute is unsupported"));
        }
        let values = value
            .get("values")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_pattern(index, "IOB values are missing"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .filter(|value| *value <= 3)
                    .map(|value| value as u8)
                    .ok_or_else(|| invalid_pattern(index, "IOB values are invalid"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let negate = parse_negate(value, index, "IOB")?;
        return Ok(TokenConstraint::EntIob(values, negate));
    }
    if kind == "regex" {
        let attribute = TextAttribute::parse(attribute_name)
            .ok_or_else(|| invalid_pattern(index, "regex attribute is unsupported"))?;
        let pattern = value
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_pattern(index, "regex pattern is missing"))?;
        let regex = Regex::new(pattern)
            .map_err(|error| invalid_pattern(index, format!("invalid regex: {error}")))?;
        return Ok(TokenConstraint::Regex(attribute, regex));
    }
    if kind == "regex_set" {
        let attribute = TextAttribute::parse(attribute_name)
            .ok_or_else(|| invalid_pattern(index, "regex attribute is unsupported"))?;
        let regexes = parse_string_patterns(value, index, "regex")?
            .map(|pattern| {
                Regex::new(pattern)
                    .map_err(|error| invalid_pattern(index, format!("invalid regex: {error}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let negate = parse_negate(value, index, "regex")?;
        return Ok(TokenConstraint::RegexSet(attribute, regexes, negate));
    }
    if kind == "fuzzy" || kind == "fuzzy_set" {
        let attribute = TextAttribute::parse(attribute_name)
            .ok_or_else(|| invalid_pattern(index, "fuzzy attribute is unsupported"))?;
        let max_edits = parse_fuzzy_max_edits(value, index)?;
        if kind == "fuzzy_set" {
            let patterns = parse_string_patterns(value, index, "fuzzy")?
                .map(str::to_owned)
                .collect();
            let negate = parse_negate(value, index, "fuzzy")?;
            return Ok(TokenConstraint::FuzzySet(
                attribute, patterns, max_edits, negate,
            ));
        }
        let pattern = value
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_pattern(index, "fuzzy pattern is missing"))?;
        return Ok(TokenConstraint::Fuzzy(
            attribute,
            pattern.to_owned(),
            max_edits,
        ));
    }
    if kind == "numeric" {
        if attribute_name != "LENGTH" {
            return Err(invalid_pattern(index, "numeric attribute is unsupported"));
        }
        let comparison = value
            .get("comparison")
            .and_then(serde_json::Value::as_str)
            .and_then(NumericComparison::parse)
            .ok_or_else(|| invalid_pattern(index, "numeric comparison is unsupported"))?;
        let expected = value
            .get("value")
            .and_then(serde_json::Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid_pattern(index, "numeric value is invalid"))?;
        return Ok(TokenConstraint::Length(comparison, expected));
    }
    if kind == "numeric_set" {
        if attribute_name != "LENGTH" {
            return Err(invalid_pattern(
                index,
                "numeric set attribute is unsupported",
            ));
        }
        let values = value
            .get("values")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_pattern(index, "numeric set values are missing"))?
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| invalid_pattern(index, "numeric set values must be integers"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let negate = parse_negate(value, index, "numeric set")?;
        return Ok(TokenConstraint::LengthSet(values, negate));
    }
    if kind == "morph_set" {
        if attribute_name != "MORPH" {
            return Err(invalid_pattern(
                index,
                "morphology set attribute is unsupported",
            ));
        }
        let comparison = value
            .get("comparison")
            .and_then(serde_json::Value::as_str)
            .and_then(SetRelation::parse)
            .ok_or_else(|| invalid_pattern(index, "morphology set comparison is unsupported"))?;
        let mut features = value
            .get("features")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_pattern(index, "morphology set features are missing"))?
            .iter()
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    invalid_pattern(index, "morphology set features must be unsigned")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        features.sort_unstable();
        features.dedup();
        return Ok(TokenConstraint::MorphSet(comparison, features));
    }
    if kind == "id_set_relation" {
        let attribute = IdAttribute::parse(attribute_name)
            .ok_or_else(|| invalid_pattern(index, "ID set attribute is unsupported"))?;
        let relation = value
            .get("comparison")
            .and_then(serde_json::Value::as_str)
            .and_then(SetRelation::parse)
            .ok_or_else(|| invalid_pattern(index, "ID set comparison is unsupported"))?;
        let mut values = value
            .get("values")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid_pattern(index, "ID set values are missing"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| invalid_pattern(index, "ID set values must be unsigned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        values.sort_unstable();
        values.dedup();
        return Ok(TokenConstraint::IdSetRelation(attribute, relation, values));
    }
    let attribute = IdAttribute::parse(attribute_name)
        .ok_or_else(|| invalid_pattern(index, "ID attribute is unsupported"))?;
    let values = value
        .get("values")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid_pattern(index, "constraint values are missing"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| invalid_pattern(index, "constraint values must be unsigned"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    match kind {
        "equal" => Ok(TokenConstraint::Equal(attribute, values)),
        "in" => Ok(TokenConstraint::In(attribute, values)),
        "not_in" => Ok(TokenConstraint::NotIn(attribute, values)),
        _ => Err(invalid_pattern(index, "constraint kind is unsupported")),
    }
}

fn parse_string_patterns<'a>(
    value: &'a serde_json::Value,
    index: usize,
    predicate: &str,
) -> Result<impl Iterator<Item = &'a str>, EntityRulerError> {
    value
        .get("patterns")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid_pattern(index, format!("{predicate} patterns are missing")))?
        .iter()
        .map(move |pattern| {
            pattern.as_str().ok_or_else(|| {
                invalid_pattern(index, format!("{predicate} patterns must be strings"))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_iter)
}

fn parse_negate(
    value: &serde_json::Value,
    index: usize,
    predicate: &str,
) -> Result<bool, EntityRulerError> {
    value
        .get("negate")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| invalid_pattern(index, format!("{predicate} negate is missing")))
}

fn parse_fuzzy_max_edits(
    value: &serde_json::Value,
    index: usize,
) -> Result<Option<usize>, EntityRulerError> {
    match value.get("max_edits").and_then(serde_json::Value::as_i64) {
        Some(-1) => Ok(None),
        Some(value @ 1..=9) => Ok(Some(value as usize)),
        _ => Err(invalid_pattern(index, "fuzzy max_edits is invalid")),
    }
}

fn invalid_pattern(index: usize, message: impl Into<String>) -> EntityRulerError {
    EntityRulerError::InvalidPattern {
        index,
        message: message.into(),
    }
}

fn entity_ranges(doc: &Doc) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < doc.len() {
        let first = &doc.tokens()[start];
        if first.ent_iob != 3 || first.ent_type == 0 {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < doc.len()
            && doc.tokens()[end].ent_iob == 1
            && doc.tokens()[end].ent_type == first.ent_type
        {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use spacy_core::{Doc, StringStore};

    use super::{
        bounded_levenshtein, default_fuzzy_edits, entity_ranges, parse_repetition,
        parse_token_pattern, BooleanAttribute, EntityRuler, IdAttribute, PhraseAttribute,
        PhrasePattern, Quantifier, RulerLanguage, SetRelation,
    };

    #[derive(Deserialize)]
    struct Fixture {
        spacy_version: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        phrase_matcher_attr: String,
        overwrite_ents: bool,
        words: Vec<String>,
        spaces: Vec<bool>,
        token_ids: Vec<u64>,
        patterns: Vec<Pattern>,
        initial_entities: Vec<Entity>,
        entities: Vec<Entity>,
    }

    #[derive(Deserialize)]
    struct Pattern {
        label: String,
        token_ids: Vec<u64>,
    }

    #[derive(Deserialize)]
    struct Entity {
        start: usize,
        end: usize,
        label: String,
    }

    #[derive(Deserialize)]
    struct TokenFixture {
        spacy_version: String,
        cases: Vec<TokenCase>,
    }

    #[derive(Deserialize)]
    struct TokenCase {
        words: Vec<String>,
        spaces: Vec<bool>,
        #[serde(default)]
        norm_ids: Vec<u64>,
        #[serde(default)]
        morphs: Vec<String>,
        #[serde(default)]
        pos_ids: Vec<u64>,
        overwrite_ents: bool,
        patterns: Vec<serde_json::Value>,
        initial_entities: Vec<Entity>,
        entities: Vec<Entity>,
    }

    fn ruler(patterns: &[(&str, &[&str])], overwrite: bool) -> EntityRuler {
        let patterns = patterns
            .iter()
            .map(|(label, words)| PhrasePattern {
                label_id: StringStore::id(label),
                ent_id: 0,
                token_ids: words.iter().map(|word| StringStore::id(word)).collect(),
            })
            .collect();
        EntityRuler {
            language: RulerLanguage::English,
            attribute: PhraseAttribute::Id(IdAttribute::Orth),
            overwrite,
            patterns,
            token_patterns: Vec::new(),
            stop_word_ids: Default::default(),
            labels: Vec::new(),
            entity_ids: Vec::new(),
        }
    }

    #[test]
    fn longest_phrase_wins_and_repeated_matches_are_retained() {
        let mut doc = Doc::from_words(
            &["Acme", "Corp", "and", "Acme", "Corp"],
            &[true, true, true, true, false],
        )
        .unwrap();
        ruler(&[("ORG", &["Acme"]), ("ORG", &["Acme", "Corp"])], false)
            .annotate(&mut doc)
            .unwrap();
        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| token.ent_iob)
                .collect::<Vec<_>>(),
            [3, 1, 0, 3, 1]
        );
    }

    #[test]
    fn accepts_spacy_3_8_phrase_matcher_attributes() {
        let supported = [
            "IS_ALPHA",
            "IS_ASCII",
            "IS_DIGIT",
            "IS_LOWER",
            "IS_PUNCT",
            "IS_SPACE",
            "IS_TITLE",
            "IS_UPPER",
            "LIKE_URL",
            "LIKE_NUM",
            "LIKE_EMAIL",
            "IS_STOP",
            "IS_BRACKET",
            "IS_QUOTE",
            "IS_LEFT_PUNCT",
            "IS_RIGHT_PUNCT",
            "IS_CURRENCY",
            "ORTH",
            "TEXT",
            "LOWER",
            "NORM",
            "SHAPE",
            "LENGTH",
            "ENT_IOB",
            "ENT_TYPE",
            "SENT_START",
            "IS_SENT_START",
            "SPACY",
            "ENT_KB_ID",
            "ENT_ID",
        ];
        for attribute in supported {
            assert!(PhraseAttribute::parse(attribute).is_ok(), "{attribute}");
        }
        for attribute in ["PREFIX", "SUFFIX", "LEMMA", "POS", "TAG", "DEP", "MORPH"] {
            assert!(PhraseAttribute::parse(attribute).is_err(), "{attribute}");
        }
    }

    #[test]
    fn set_relations_match_spacy_empty_and_scalar_semantics() {
        let feature = StringStore::id("Number=Sing");
        assert!(SetRelation::Subset.matches(&[], &[feature]));
        assert!(!SetRelation::Subset.matches(&[feature], &[]));
        assert!(SetRelation::Superset.matches(&[feature], &[]));
        assert!(!SetRelation::Superset.matches(&[], &[feature]));
        assert!(!SetRelation::Intersects.matches(&[], &[feature]));

        assert!(SetRelation::Subset.matches_scalar(feature, &[feature]));
        assert!(SetRelation::Intersects.matches_scalar(feature, &[feature]));
        assert!(SetRelation::Superset.matches_scalar(feature, &[]));
        assert!(SetRelation::Superset.matches_scalar(feature, &[feature]));
        assert!(!SetRelation::Superset.matches_scalar(feature, &[feature, feature + 1]));
    }

    #[test]
    fn punctuation_flags_match_spacy_3_8_lexical_attributes() {
        let stop_word_ids = Default::default();
        let cases = [
            ("(", true, false, true, false),
            (")", true, false, false, true),
            ("[", true, false, true, false),
            ("]", true, false, false, true),
            ("「", false, false, false, false),
            ("」", false, false, false, false),
            ("“", false, true, true, false),
            ("”", false, true, false, true),
            ("\"", false, true, true, true),
            ("'", false, true, true, true),
            ("—", false, false, false, false),
            ("«", false, true, true, false),
            ("»", false, true, false, true),
        ];
        for (text, bracket, quote, left, right) in cases {
            let doc = Doc::from_words(&[text], &[false]).unwrap();
            let token = &doc.tokens()[0];
            assert_eq!(
                BooleanAttribute::IsBracket.value(token, RulerLanguage::English, &stop_word_ids),
                bracket,
                "{text:?}"
            );
            assert_eq!(
                BooleanAttribute::IsQuote.value(token, RulerLanguage::English, &stop_word_ids),
                quote,
                "{text:?}"
            );
            assert_eq!(
                BooleanAttribute::IsLeftPunct.value(token, RulerLanguage::English, &stop_word_ids),
                left,
                "{text:?}"
            );
            assert_eq!(
                BooleanAttribute::IsRightPunct.value(token, RulerLanguage::English, &stop_word_ids),
                right,
                "{text:?}"
            );
        }
    }

    #[test]
    fn matches_exported_stop_words_case_insensitively() {
        let token_pattern = parse_token_pattern(
            &serde_json::json!({
                "label": "STOP",
                "tokens": [{
                    "op": "1",
                    "constraints": [{
                        "attribute": "IS_STOP",
                        "kind": "boolean",
                        "value": true
                    }]
                }]
            }),
            0,
        )
        .unwrap();
        let ruler = EntityRuler {
            language: RulerLanguage::English,
            attribute: PhraseAttribute::Id(IdAttribute::Orth),
            overwrite: false,
            patterns: Vec::new(),
            token_patterns: vec![token_pattern],
            stop_word_ids: [StringStore::id("the")].into_iter().collect(),
            labels: vec!["STOP".to_owned()],
            entity_ids: Vec::new(),
        };
        let mut doc = Doc::from_words(&["The", "contract"], &[true, false]).unwrap();

        ruler.annotate(&mut doc).unwrap();

        assert_eq!(doc.tokens()[0].ent_type, StringStore::id("STOP"));
        assert_eq!(doc.tokens()[1].ent_type, 0);
    }

    #[test]
    fn assigns_and_clears_spacy_entity_ruler_pattern_ids() {
        let phrase_id = StringStore::id("acme-org");
        let token_id = StringStore::id("widget-product");
        let token_pattern = parse_token_pattern(
            &serde_json::json!({
                "label": "PRODUCT",
                "id": "widget-product",
                "tokens": [{
                    "op": "1",
                    "constraints": [{
                        "attribute": "LOWER",
                        "kind": "equal",
                        "values": [StringStore::id("widget")]
                    }]
                }]
            }),
            0,
        )
        .unwrap();
        let ent_id_pattern = parse_token_pattern(
            &serde_json::json!({
                "label": "MIGRATED",
                "id": "migrated-id",
                "tokens": [{
                    "op": "1",
                    "constraints": [{
                        "attribute": "ENT_ID",
                        "kind": "equal",
                        "values": [StringStore::id("old-id")]
                    }, {
                        "attribute": "ENT_KB_ID",
                        "kind": "equal",
                        "values": [StringStore::id("Q123")]
                    }]
                }]
            }),
            1,
        )
        .unwrap();
        let linguistic_pattern = parse_token_pattern(
            &serde_json::json!({
                "label": "SIGNED_ACTION",
                "tokens": [{
                    "op": "1",
                    "constraints": [
                        {
                            "attribute": "LEMMA",
                            "kind": "equal",
                            "values": [StringStore::id("sign")]
                        },
                        {
                            "attribute": "POS",
                            "kind": "equal",
                            "values": [StringStore::id("VERB")]
                        },
                        {
                            "attribute": "TAG",
                            "kind": "equal",
                            "values": [StringStore::id("VBD")]
                        },
                        {
                            "attribute": "DEP",
                            "kind": "equal",
                            "values": [StringStore::id("ROOT")]
                        },
                        {
                            "attribute": "MORPH",
                            "kind": "equal",
                            "values": [StringStore::id("Tense=Past|VerbForm=Fin")]
                        },
                        {
                            "attribute": "SENT_START",
                            "kind": "boolean",
                            "value": false
                        },
                        {
                            "attribute": "SPACY",
                            "kind": "boolean",
                            "value": false
                        }
                    ]
                }]
            }),
            2,
        )
        .unwrap();
        let ruler = EntityRuler {
            language: RulerLanguage::English,
            attribute: PhraseAttribute::Id(IdAttribute::Orth),
            overwrite: true,
            patterns: vec![
                PhrasePattern {
                    label_id: StringStore::id("ORG"),
                    ent_id: phrase_id,
                    token_ids: vec![StringStore::id("Acme"), StringStore::id("Corp")],
                },
                PhrasePattern {
                    label_id: StringStore::id("TERM"),
                    ent_id: 0,
                    token_ids: vec![StringStore::id("plain")],
                },
            ],
            token_patterns: vec![token_pattern, ent_id_pattern, linguistic_pattern],
            stop_word_ids: Default::default(),
            labels: vec![
                "ORG".to_owned(),
                "TERM".to_owned(),
                "PRODUCT".to_owned(),
                "MIGRATED".to_owned(),
                "SIGNED_ACTION".to_owned(),
            ],
            entity_ids: vec![
                "acme-org".to_owned(),
                "widget-product".to_owned(),
                "migrated-id".to_owned(),
            ],
        };
        let mut doc = Doc::from_words(
            &["Acme", "Corp", "Widget", "plain", "Legacy", "Signed"],
            &[true, true, true, true, true, false],
        )
        .unwrap();
        for (offset, token) in doc.tokens_mut()[..2].iter_mut().enumerate() {
            token.ent_iob = if offset == 0 { 3 } else { 1 };
            token.ent_type = StringStore::id("OLD");
            token.ent_id = StringStore::id("old-id");
        }
        doc.tokens_mut()[4].ent_iob = 3;
        doc.tokens_mut()[4].ent_type = StringStore::id("OLD");
        doc.tokens_mut()[4].ent_id = StringStore::id("old-id");
        doc.tokens_mut()[4].ent_kb_id = StringStore::id("Q123");
        doc.tokens_mut()[5].lemma = StringStore::id("sign");
        doc.tokens_mut()[5].pos = StringStore::id("VERB");
        doc.tokens_mut()[5].tag = StringStore::id("VBD");
        doc.tokens_mut()[5].dep = StringStore::id("ROOT");
        doc.tokens_mut()[5].morph = StringStore::id("Tense=Past|VerbForm=Fin");

        ruler.annotate(&mut doc).unwrap();

        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| token.ent_id)
                .collect::<Vec<_>>(),
            [
                phrase_id,
                phrase_id,
                token_id,
                0,
                StringStore::id("migrated-id"),
                0
            ]
        );
        assert_eq!(doc.tokens()[5].ent_type, StringStore::id("SIGNED_ACTION"));
        assert_eq!(doc.tokens()[4].ent_kb_id, 0);
        let stop_word_ids = Default::default();
        assert!(BooleanAttribute::SentStart.value(
            &doc.tokens()[0],
            RulerLanguage::English,
            &stop_word_ids
        ));
        assert!(BooleanAttribute::parse("IS_SENT_START").unwrap().value(
            &doc.tokens()[0],
            RulerLanguage::English,
            &stop_word_ids
        ));
        assert!(BooleanAttribute::Spacy.value(
            &doc.tokens()[0],
            RulerLanguage::English,
            &stop_word_ids
        ));
        assert_eq!(
            ruler.entity_ids().collect::<Vec<_>>(),
            ["acme-org", "widget-product", "migrated-id"]
        );
    }

    #[test]
    fn fuzzy_distance_is_unicode_aware_and_uses_spacy_rounding() {
        assert_eq!(bounded_levenshtein("kitten", "sitting", 3), Some(3));
        assert_eq!(bounded_levenshtein("kitten", "sitting", 2), None);
        assert_eq!(bounded_levenshtein("株式会社", "株式会杜", 1), Some(1));
        assert_eq!(bounded_levenshtein("", "ab", 2), Some(2));
        assert_eq!(bounded_levenshtein("ab", "", 2), Some(2));
        assert_eq!(default_fuzzy_edits(4), 2);
        assert_eq!(default_fuzzy_edits(15), 4);
        assert_eq!(default_fuzzy_edits(25), 8);
    }

    #[test]
    fn parses_spacy_bounded_repetition_operators() {
        assert_eq!(
            parse_repetition("{2}"),
            Some(Quantifier::Repeat {
                minimum: 2,
                maximum: Some(2),
            })
        );
        assert_eq!(
            parse_repetition("{1,3}"),
            Some(Quantifier::Repeat {
                minimum: 1,
                maximum: Some(3),
            })
        );
        assert_eq!(
            parse_repetition("{2,}"),
            Some(Quantifier::Repeat {
                minimum: 2,
                maximum: None,
            })
        );
        assert_eq!(
            parse_repetition("{,2}"),
            Some(Quantifier::Repeat {
                minimum: 0,
                maximum: Some(2),
            })
        );
        assert_eq!(parse_repetition("{3,2}"), None);
        assert_eq!(parse_repetition("{,}"), None);
    }

    #[test]
    fn overwrite_controls_existing_entities() {
        let mut preserved =
            Doc::from_words(&["Acme", "Corp", "Japan"], &[true, true, false]).unwrap();
        let old = StringStore::id("GPE");
        for (offset, token) in preserved.tokens_mut()[1..3].iter_mut().enumerate() {
            token.ent_iob = if offset == 0 { 3 } else { 1 };
            token.ent_type = old;
        }
        ruler(&[("ORG", &["Acme", "Corp"])], false)
            .annotate(&mut preserved)
            .unwrap();
        assert_eq!(preserved.tokens()[0].ent_type, 0);
        assert_eq!(preserved.tokens()[1].ent_type, old);

        let mut replaced = preserved.clone();
        ruler(&[("ORG", &["Acme", "Corp"])], true)
            .annotate(&mut replaced)
            .unwrap();
        assert_eq!(replaced.tokens()[0].ent_type, StringStore::id("ORG"));
        assert_eq!(replaced.tokens()[1].ent_type, StringStore::id("ORG"));
        assert_eq!(replaced.tokens()[2].ent_type, 0);
    }

    #[test]
    fn matches_spacy_3_8_golden_phrase_ruler_annotations() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/entity_ruler_spacy_3_8.json"
        ))
        .unwrap();
        assert_eq!(fixture.spacy_version, "3.8.13");
        for case in fixture.cases {
            let attribute = PhraseAttribute::parse(&case.phrase_matcher_attr).unwrap();
            let mut doc = Doc::from_words(&case.words, &case.spaces).unwrap();
            for token in doc.tokens_mut() {
                token.ent_iob = 2;
            }
            if matches!(attribute, PhraseAttribute::Id(IdAttribute::Norm)) {
                for (token, value) in doc.tokens_mut().iter_mut().zip(&case.token_ids) {
                    token.norm = *value;
                }
            }
            for entity in &case.initial_entities {
                let entity_type = StringStore::id(&entity.label);
                for (offset, token) in doc.tokens_mut()[entity.start..entity.end]
                    .iter_mut()
                    .enumerate()
                {
                    token.ent_iob = if offset == 0 { 3 } else { 1 };
                    token.ent_type = entity_type;
                }
            }
            let stop_word_ids = [StringStore::id("the"), StringStore::id("and")]
                .into_iter()
                .collect();
            assert_eq!(
                doc.tokens()
                    .iter()
                    .map(|token| { attribute.value(token, RulerLanguage::English, &stop_word_ids) })
                    .collect::<Vec<_>>(),
                case.token_ids
            );
            let labels = case
                .patterns
                .iter()
                .map(|pattern| pattern.label.clone())
                .collect();
            let patterns = case
                .patterns
                .into_iter()
                .map(|pattern| PhrasePattern {
                    label_id: StringStore::id(&pattern.label),
                    ent_id: 0,
                    token_ids: pattern.token_ids,
                })
                .collect();
            EntityRuler {
                language: RulerLanguage::English,
                attribute,
                overwrite: case.overwrite_ents,
                patterns,
                token_patterns: Vec::new(),
                stop_word_ids,
                labels,
                entity_ids: Vec::new(),
            }
            .annotate(&mut doc)
            .unwrap();
            let actual = entity_ranges(&doc)
                .into_iter()
                .map(|(start, end)| (start, end, doc.tokens()[start].ent_type))
                .collect::<Vec<_>>();
            let expected = case
                .entities
                .into_iter()
                .map(|entity| (entity.start, entity.end, StringStore::id(&entity.label)))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn matches_spacy_3_8_golden_token_ruler_annotations() {
        let fixture: TokenFixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/entity_ruler_token_spacy_3_8.json"
        ))
        .unwrap();
        assert_eq!(fixture.spacy_version, "3.8.13");
        for case in fixture.cases {
            let mut doc = Doc::from_words(&case.words, &case.spaces).unwrap();
            for token in doc.tokens_mut() {
                token.ent_iob = 2;
            }
            for (token, norm) in doc.tokens_mut().iter_mut().zip(&case.norm_ids) {
                token.norm = *norm;
            }
            for (token, morph) in doc.tokens_mut().iter_mut().zip(&case.morphs) {
                token.morph = StringStore::id(morph);
                token.set_morph_features(morph);
            }
            for (token, pos) in doc.tokens_mut().iter_mut().zip(&case.pos_ids) {
                token.pos = *pos;
            }
            for entity in case.initial_entities {
                let entity_type = StringStore::id(&entity.label);
                for (offset, token) in doc.tokens_mut()[entity.start..entity.end]
                    .iter_mut()
                    .enumerate()
                {
                    token.ent_iob = if offset == 0 { 3 } else { 1 };
                    token.ent_type = entity_type;
                }
            }
            let labels = case
                .patterns
                .iter()
                .map(|pattern| pattern["label"].as_str().unwrap().to_owned())
                .collect();
            let token_patterns = case
                .patterns
                .iter()
                .enumerate()
                .map(|(index, pattern)| parse_token_pattern(pattern, index))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            EntityRuler {
                language: RulerLanguage::English,
                attribute: PhraseAttribute::Id(IdAttribute::Orth),
                overwrite: case.overwrite_ents,
                patterns: Vec::new(),
                token_patterns,
                stop_word_ids: Default::default(),
                labels,
                entity_ids: Vec::new(),
            }
            .annotate(&mut doc)
            .unwrap();
            let actual = entity_ranges(&doc)
                .into_iter()
                .map(|(start, end)| (start, end, doc.tokens()[start].ent_type))
                .collect::<Vec<_>>();
            let expected = case
                .entities
                .into_iter()
                .map(|entity| (entity.start, entity.end, StringStore::id(&entity.label)))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
}
