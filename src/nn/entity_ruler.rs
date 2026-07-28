use std::collections::HashSet;

use spacy_core::{Doc, StringStore, TokenData};
use spacy_model::Bundle;
use thiserror::Error;

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

struct PhraseMatch {
    pattern: usize,
    start: usize,
    end: usize,
}

/// Exact phrase subset of spaCy's `EntityRuler`.
///
/// Jewel supports post-NER phrase rulers matching `ORTH`, `LOWER`, or `NORM`.
/// Token-pattern rules are rejected by the exporter instead of approximated.
pub struct EntityRuler {
    attribute: PhraseAttribute,
    overwrite: bool,
    patterns: Vec<PhrasePattern>,
    labels: Vec<String>,
}

impl EntityRuler {
    /// Load one named entity ruler component.
    ///
    /// # Errors
    ///
    /// Returns an error when settings or phrase patterns are incompatible.
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
        let values = component
            .settings
            .get("patterns")
            .and_then(serde_json::Value::as_array)
            .ok_or(EntityRulerError::InvalidSetting { name: "patterns" })?;
        let mut patterns = Vec::with_capacity(values.len());
        let mut labels = Vec::new();
        for (index, value) in values.iter().enumerate() {
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
        Ok(Self {
            attribute,
            overwrite,
            patterns,
            labels,
        })
    }

    /// Return labels declared by exact phrase patterns.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.labels.iter().map(String::as_str)
    }

    /// Match phrases and update entity annotations.
    pub fn annotate(&self, doc: &mut Doc) {
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
                    matches.push(PhraseMatch {
                        pattern: pattern_index,
                        start,
                        end,
                    });
                }
            }
        }
        matches.sort_by(|left, right| {
            (right.end - right.start)
                .cmp(&(left.end - left.start))
                .then_with(|| left.start.cmp(&right.start))
                .then_with(|| left.pattern.cmp(&right.pattern))
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
            let pattern = &self.patterns[found.pattern];
            for (offset, token) in doc.tokens_mut()[found.start..found.end]
                .iter_mut()
                .enumerate()
            {
                token.ent_iob = if offset == 0 { 3 } else { 1 };
                token.ent_type = pattern.label_id;
            }
        }
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

    use super::{entity_ranges, EntityRuler, PhraseAttribute, PhrasePattern};

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

    fn ruler(patterns: &[(&str, &[&str])], overwrite: bool) -> EntityRuler {
        let patterns = patterns
            .iter()
            .map(|(label, words)| PhrasePattern {
                label_id: StringStore::id(label),
                token_ids: words.iter().map(|word| StringStore::id(word)).collect(),
            })
            .collect();
        EntityRuler {
            attribute: PhraseAttribute::Orth,
            overwrite,
            patterns,
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
        ruler(&[("ORG", &["Acme"]), ("ORG", &["Acme", "Corp"])], false).annotate(&mut doc);
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
        ruler(&[("ORG", &["Acme", "Corp"])], false).annotate(&mut preserved);
        assert_eq!(preserved.tokens()[0].ent_type, 0);
        assert_eq!(preserved.tokens()[1].ent_type, old);

        let mut replaced = preserved.clone();
        ruler(&[("ORG", &["Acme", "Corp"])], true).annotate(&mut replaced);
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
                attribute,
                overwrite: case.overwrite_ents,
                patterns,
                labels,
            }
            .annotate(&mut doc);
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
