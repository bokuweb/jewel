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
    Orth,
    Lower,
    Norm,
}

impl PhraseAttribute {
    fn parse(value: &str) -> Result<Self, EntityRulerError> {
        match value {
            "ORTH" => Ok(Self::Orth),
            "LOWER" => Ok(Self::Lower),
            "NORM" => Ok(Self::Norm),
            value => Err(EntityRulerError::UnsupportedPhraseMatcherAttribute(
                value.to_owned(),
            )),
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
        }
    }
}

struct PhrasePattern {
    label_id: u64,
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
        }
    }
}

#[derive(Clone, Copy)]
enum TextAttribute {
    Orth,
    Lower,
}

impl TextAttribute {
    fn value(self, token: &TokenData) -> String {
        match self {
            Self::Orth => token.text.to_string(),
            Self::Lower => token.text.to_lowercase(),
        }
    }
}

#[derive(Clone, Copy)]
enum BooleanAttribute {
    IsAlpha,
    IsAscii,
    IsCurrency,
    IsDigit,
    IsLower,
    IsPunct,
    IsSpace,
    IsTitle,
    IsUpper,
    LikeEmail,
    LikeNum,
    LikeUrl,
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
            "IS_CURRENCY" => Some(Self::IsCurrency),
            "IS_DIGIT" => Some(Self::IsDigit),
            "IS_LOWER" => Some(Self::IsLower),
            "IS_PUNCT" => Some(Self::IsPunct),
            "IS_SPACE" => Some(Self::IsSpace),
            "IS_TITLE" => Some(Self::IsTitle),
            "IS_UPPER" => Some(Self::IsUpper),
            "LIKE_EMAIL" => Some(Self::LikeEmail),
            "LIKE_NUM" => Some(Self::LikeNum),
            "LIKE_URL" => Some(Self::LikeUrl),
            _ => None,
        }
    }

    fn value(self, text: &str, language: RulerLanguage) -> bool {
        match self {
            Self::IsAlpha => !text.is_empty() && text.chars().all(char::is_alphabetic),
            Self::IsAscii => text.is_ascii(),
            Self::IsCurrency => {
                !text.is_empty() && text.chars().all(UnicodeCategories::is_symbol_currency)
            }
            Self::IsDigit => !text.is_empty() && text.chars().all(is_digit),
            Self::IsLower => is_lower(text),
            Self::IsPunct => {
                !text.is_empty() && text.chars().all(UnicodeCategories::is_punctuation)
            }
            Self::IsSpace => !text.is_empty() && text.chars().all(char::is_whitespace),
            Self::IsTitle => is_title(text),
            Self::IsUpper => is_upper(text),
            Self::LikeEmail => like_email(text),
            Self::LikeNum => like_num(text, language),
            Self::LikeUrl => like_url(text),
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

enum TokenConstraint {
    Equal(IdAttribute, Vec<u64>),
    In(IdAttribute, Vec<u64>),
    NotIn(IdAttribute, Vec<u64>),
    Regex(TextAttribute, Regex),
    Boolean(BooleanAttribute, bool),
    Length(NumericComparison, f64),
}

impl TokenConstraint {
    fn matches(
        &self,
        token: &TokenData,
        language: RulerLanguage,
    ) -> Result<bool, EntityRulerError> {
        match self {
            Self::Equal(attribute, values) | Self::In(attribute, values) => {
                Ok(values.contains(&attribute.value(token)))
            }
            Self::NotIn(attribute, values) => Ok(!values.contains(&attribute.value(token))),
            Self::Regex(attribute, regex) => regex
                .is_match(&attribute.value(token))
                .map_err(|error| EntityRulerError::Regex(error.to_string())),
            Self::Boolean(attribute, expected) => {
                Ok(attribute.value(&token.text, language) == *expected)
            }
            Self::Length(comparison, expected) => {
                Ok(comparison.matches(token.text.chars().count() as f64, *expected))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Quantifier {
    One,
    Negate,
    ZeroOrOne,
    ZeroOrMore,
    OneOrMore,
}

impl Quantifier {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "1" => Some(Self::One),
            "!" => Some(Self::Negate),
            "?" => Some(Self::ZeroOrOne),
            "*" => Some(Self::ZeroOrMore),
            "+" => Some(Self::OneOrMore),
            _ => None,
        }
    }
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
    ) -> Result<bool, EntityRulerError> {
        for constraint in &self.constraints {
            if !constraint.matches(token, language)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

struct TokenPattern {
    label_id: u64,
    steps: Vec<TokenStep>,
}

struct RulerMatch {
    label_id: u64,
    priority: usize,
    start: usize,
    end: usize,
}

/// Supported subset of spaCy's `EntityRuler`.
///
/// Jewel supports post-NER phrase rulers matching `ORTH`, `LOWER`, or `NORM`.
/// Token patterns support text, normalized and structural string attributes,
/// length comparisons, lexical Boolean attributes, `IN`, `NOT_IN`, `REGEX`,
/// and the `!`, `?`, `*`, and `+` operators.
pub struct EntityRuler {
    language: RulerLanguage,
    attribute: PhraseAttribute,
    overwrite: bool,
    patterns: Vec<PhrasePattern>,
    token_patterns: Vec<TokenPattern>,
    labels: Vec<String>,
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
        let mut patterns = Vec::with_capacity(phrase_values.len());
        let mut token_patterns = Vec::with_capacity(token_values.map_or(0, std::vec::Vec::len));
        let mut labels = Vec::new();
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
            patterns.push(PhrasePattern {
                label_id: StringStore::id(label),
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
            token_patterns.push(pattern);
        }
        Ok(Self {
            language: RulerLanguage::parse(&bundle.manifest().source.lang),
            attribute,
            overwrite,
            patterns,
            token_patterns,
            labels,
        })
    }

    /// Return labels declared by phrase and token patterns.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.labels.iter().map(String::as_str)
    }

    /// Match phrases and update entity annotations.
    pub fn annotate(&self, doc: &mut Doc) -> Result<(), EntityRulerError> {
        let token_ids = doc
            .tokens()
            .iter()
            .map(|token| self.attribute.value(token))
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
                        priority: pattern_index,
                        start,
                        end,
                    });
                }
            }
        }
        for (pattern_index, pattern) in self.token_patterns.iter().enumerate() {
            for start in 0..doc.len() {
                for end in token_pattern_ends(pattern, doc.tokens(), start, self.language)? {
                    if unique.insert((pattern.label_id, start, end)) {
                        matches.push(RulerMatch {
                            label_id: pattern.label_id,
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
) -> Result<Vec<usize>, EntityRulerError> {
    let mut positions = vec![start];
    for step in &pattern.steps {
        let mut next = Vec::new();
        for position in positions {
            match step.quantifier {
                Quantifier::One => {
                    if position < tokens.len() && step.matches(&tokens[position], language)? {
                        next.push(position + 1);
                    }
                }
                Quantifier::Negate => {
                    if position < tokens.len() && !step.matches(&tokens[position], language)? {
                        next.push(position + 1);
                    }
                }
                Quantifier::ZeroOrOne => {
                    next.push(position);
                    if position < tokens.len() && step.matches(&tokens[position], language)? {
                        next.push(position + 1);
                    }
                }
                Quantifier::ZeroOrMore | Quantifier::OneOrMore => {
                    let mut cursor = position;
                    if matches!(step.quantifier, Quantifier::ZeroOrMore) {
                        next.push(cursor);
                    }
                    while cursor < tokens.len() && step.matches(&tokens[cursor], language)? {
                        cursor += 1;
                        next.push(cursor);
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
            .filter(|constraints| !constraints.is_empty())
            .ok_or_else(|| invalid_pattern(index, "token has no constraints"))?
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
        steps,
    })
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
    if kind == "regex" {
        let attribute = match attribute_name {
            "ORTH" | "TEXT" => TextAttribute::Orth,
            "LOWER" => TextAttribute::Lower,
            _ => return Err(invalid_pattern(index, "regex attribute is unsupported")),
        };
        let pattern = value
            .get("pattern")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid_pattern(index, "regex pattern is missing"))?;
        let regex = Regex::new(pattern)
            .map_err(|error| invalid_pattern(index, format!("invalid regex: {error}")))?;
        return Ok(TokenConstraint::Regex(attribute, regex));
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
        entity_ranges, parse_token_pattern, EntityRuler, PhraseAttribute, PhrasePattern,
        RulerLanguage,
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
                token_ids: words.iter().map(|word| StringStore::id(word)).collect(),
            })
            .collect();
        EntityRuler {
            language: RulerLanguage::English,
            attribute: PhraseAttribute::Orth,
            overwrite,
            patterns,
            token_patterns: Vec::new(),
            labels: Vec::new(),
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
            if matches!(attribute, PhraseAttribute::Norm) {
                for (token, value) in doc.tokens_mut().iter_mut().zip(&case.token_ids) {
                    token.norm = *value;
                }
            }
            assert_eq!(
                doc.tokens()
                    .iter()
                    .map(|token| attribute.value(token))
                    .collect::<Vec<_>>(),
                case.token_ids
            );
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
                .map(|pattern| pattern.label.clone())
                .collect();
            let patterns = case
                .patterns
                .into_iter()
                .map(|pattern| PhrasePattern {
                    label_id: StringStore::id(&pattern.label),
                    token_ids: pattern.token_ids,
                })
                .collect();
            EntityRuler {
                language: RulerLanguage::English,
                attribute,
                overwrite: case.overwrite_ents,
                patterns,
                token_patterns: Vec::new(),
                labels,
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
            for (token, norm) in doc.tokens_mut().iter_mut().zip(&case.norm_ids) {
                token.norm = *norm;
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
                attribute: PhraseAttribute::Orth,
                overwrite: case.overwrite_ents,
                patterns: Vec::new(),
                token_patterns,
                labels,
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
