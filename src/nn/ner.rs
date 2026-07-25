use serde::{Deserialize, Serialize};
use spacy_core::{Doc, StringStore};
use spacy_model::Bundle;
use thiserror::Error;

use crate::{Tok2Vec, Tok2VecError, TransitionScorer, TransitionScorerError};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NamedEntity {
    pub text: String,
    pub label: String,
    pub start_token: usize,
    pub end_token: usize,
    pub start_char: usize,
    pub end_char: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NerAction {
    Begin(String),
    In(String),
    Last(String),
    Unit(String),
    Out,
}

impl NerAction {
    /// Parse one class name from spaCy's BILUO pushdown transition system.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown transition prefix.
    pub fn parse(name: &str) -> Result<Self, EntityRecognizerError> {
        if name == "O" {
            return Ok(Self::Out);
        }
        let (prefix, label) = name.split_once('-').ok_or_else(|| {
            EntityRecognizerError::InvalidMove(format!("invalid NER move {name:?}"))
        })?;
        match prefix {
            "B" => Ok(Self::Begin(label.to_owned())),
            "I" => Ok(Self::In(label.to_owned())),
            "L" => Ok(Self::Last(label.to_owned())),
            "U" => Ok(Self::Unit(label.to_owned())),
            _ => Err(EntityRecognizerError::InvalidMove(format!(
                "unsupported NER move prefix {prefix:?}"
            ))),
        }
    }
}

#[derive(Debug, Error)]
pub enum EntityRecognizerError {
    #[error(transparent)]
    Tok2Vec(#[from] Tok2VecError),
    #[error(transparent)]
    Scorer(#[from] TransitionScorerError),
    #[error("NER component {0:?} is missing")]
    MissingComponent(String),
    #[error("NER model is invalid: {0}")]
    InvalidModel(String),
    #[error("invalid NER move: {0}")]
    InvalidMove(String),
    #[error("no valid NER transition for token {token}")]
    NoValidMove { token: usize },
}

/// Inference state for spaCy's BILUO entity transition system.
#[derive(Clone, Debug)]
pub struct NerState {
    length: usize,
    buffer: usize,
    open: Option<(usize, String)>,
    entities: Vec<(usize, usize, String)>,
    sent_starts: Vec<bool>,
    is_space: Vec<bool>,
}

impl NerState {
    #[must_use]
    pub fn new(doc: &Doc) -> Self {
        Self {
            length: doc.len(),
            buffer: 0,
            open: None,
            entities: Vec::new(),
            sent_starts: doc
                .tokens()
                .iter()
                .map(|token| token.sent_start == 1)
                .collect(),
            is_space: doc
                .tokens()
                .iter()
                .map(|token| token.text.chars().all(char::is_whitespace))
                .collect(),
        }
    }

    #[must_use]
    pub fn is_final(&self) -> bool {
        self.buffer >= self.length
    }

    #[must_use]
    pub fn features(&self) -> [i32; 3] {
        let current = i32::try_from(self.buffer)
            .ok()
            .filter(|_| self.buffer < self.length)
            .unwrap_or(-1);
        let start = self
            .open
            .as_ref()
            .and_then(|(start, _)| i32::try_from(*start).ok())
            .unwrap_or(-1);
        let previous = if current >= 0 && start >= 0 {
            current - 1
        } else {
            -1
        };
        [current, start, previous]
    }

    #[must_use]
    pub fn is_valid(&self, action: &NerAction) -> bool {
        if self.is_final() {
            return false;
        }
        let remaining = self.length - self.buffer;
        let current_space = self.is_space[self.buffer];
        let next_is_space = self.is_space.get(self.buffer + 1).copied().unwrap_or(false);
        let next_starts_sentence = self
            .sent_starts
            .get(self.buffer + 1)
            .copied()
            .unwrap_or(false);
        match action {
            NerAction::Begin(label) => {
                self.open.is_none()
                    && remaining >= 2
                    && !label.is_empty()
                    && !next_starts_sentence
                    && !next_is_space
                    && !current_space
            }
            NerAction::In(label) => {
                remaining >= 2
                    && !next_starts_sentence
                    && !next_is_space
                    && !current_space
                    && self
                        .open
                        .as_ref()
                        .is_some_and(|(_, open_label)| open_label == label)
            }
            NerAction::Last(label) => self.open.as_ref().is_some_and(|(_, open_label)| {
                !label.is_empty() && open_label == label && !current_space
            }),
            NerAction::Unit(label) => self.open.is_none() && !label.is_empty() && !current_space,
            NerAction::Out => self.open.is_none(),
        }
    }

    /// Apply one valid entity transition and advance by one token.
    ///
    /// # Errors
    ///
    /// Returns an error when the transition is invalid for the current state.
    pub fn apply(&mut self, action: &NerAction) -> Result<(), EntityRecognizerError> {
        if !self.is_valid(action) {
            return Err(EntityRecognizerError::InvalidMove(format!(
                "move {action:?} is invalid at token {}",
                self.buffer
            )));
        }
        match action {
            NerAction::Begin(label) => {
                self.open = Some((self.buffer, label.clone()));
            }
            NerAction::In(_) | NerAction::Out => {}
            NerAction::Last(_) => {
                let Some((start, label)) = self.open.take() else {
                    return Err(EntityRecognizerError::InvalidMove(
                        "last move requires an open entity".to_owned(),
                    ));
                };
                self.entities.push((start, self.buffer + 1, label));
            }
            NerAction::Unit(label) => {
                self.entities
                    .push((self.buffer, self.buffer + 1, label.clone()));
            }
        }
        self.buffer += 1;
        Ok(())
    }

    #[must_use]
    pub fn entities(&self) -> &[(usize, usize, String)] {
        &self.entities
    }
}

/// Python-free `EntityRecognizer` for an exported spaCy NER component.
pub struct EntityRecognizer {
    encoder: Tok2Vec,
    scorer: TransitionScorer,
    actions: Vec<NerAction>,
    labels: Vec<(u64, String)>,
}

impl EntityRecognizer {
    /// Load the private NER encoder, scorer, and ordered transition classes.
    ///
    /// # Errors
    ///
    /// Returns an error when model nodes or move metadata are incompatible.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, EntityRecognizerError> {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| EntityRecognizerError::MissingComponent(component_name.to_owned()))?;
        let scorer = TransitionScorer::load(bundle, component_name)?;
        if component.moves.len() != scorer.class_count() {
            return Err(EntityRecognizerError::InvalidModel(format!(
                "model has {} score classes but {} exported moves",
                scorer.class_count(),
                component.moves.len()
            )));
        }
        Ok(Self {
            encoder: Tok2Vec::load(bundle, component_name)?,
            scorer,
            actions: component
                .moves
                .iter()
                .map(|name| NerAction::parse(name))
                .collect::<Result<Vec<_>, _>>()?,
            labels: component
                .labels
                .iter()
                .map(|label| (StringStore::id(label), label.clone()))
                .collect(),
        })
    }

    /// Recognize entities and attach spaCy-compatible `ENT_IOB`/`ENT_TYPE`.
    ///
    /// # Errors
    ///
    /// Returns an error if neural inference or transition decoding fails.
    pub fn annotate(&self, doc: &mut Doc) -> Result<Vec<usize>, EntityRecognizerError> {
        let vectors = self.encoder.forward(doc)?;
        let cache = self.scorer.precompute(&vectors)?;
        let mut state = NerState::new(doc);
        let mut history = Vec::with_capacity(doc.len());
        while !state.is_final() {
            let scores = self.scorer.score(&cache, &state.features())?;
            let mut best = None;
            for (index, action) in self.actions.iter().enumerate() {
                if state.is_valid(action)
                    && best.is_none_or(|previous| scores.row(0)[index] > scores.row(0)[previous])
                {
                    best = Some(index);
                }
            }
            let action_index = best.ok_or(EntityRecognizerError::NoValidMove {
                token: history.len(),
            })?;
            state.apply(&self.actions[action_index])?;
            history.push(action_index);
        }
        for token in doc.tokens_mut() {
            if token.ent_iob == 0 {
                token.ent_iob = 2;
            }
        }
        for (start, end, label) in state.entities {
            let entity_type = StringStore::id(&label);
            for (offset, token) in doc.tokens_mut()[start..end].iter_mut().enumerate() {
                token.ent_iob = if offset == 0 { 3 } else { 1 };
                token.ent_type = entity_type;
            }
        }
        Ok(history)
    }

    #[must_use]
    pub fn actions(&self) -> &[NerAction] {
        &self.actions
    }

    /// Return the entity spans currently attached to a document.
    #[must_use]
    pub fn entities(&self, doc: &Doc) -> Vec<NamedEntity> {
        let mut entities = Vec::new();
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
            let last = &doc.tokens()[end - 1];
            let label = self
                .labels
                .iter()
                .find(|(id, _)| *id == first.ent_type)
                .map_or_else(|| first.ent_type.to_string(), |(_, label)| label.clone());
            let mut text = String::new();
            for (offset, token) in doc.tokens()[start..end].iter().enumerate() {
                text.push_str(&token.text);
                if token.has_space && offset + 1 < end - start {
                    text.push(' ');
                }
            }
            entities.push(NamedEntity {
                text,
                label,
                start_token: start,
                end_token: end,
                start_char: first.idx,
                end_char: last.idx + last.text.chars().count(),
            });
            start = end;
        }
        entities
    }

    /// Return entity spans with the requested label.
    #[must_use]
    pub fn entities_by_label(&self, doc: &Doc, label: &str) -> Vec<NamedEntity> {
        self.entities(doc)
            .into_iter()
            .filter(|entity| entity.label == label)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use spacy_core::{Doc, TokenData};

    use super::{NamedEntity, NerAction, NerState};

    #[test]
    fn named_entity_round_trips_as_json() {
        let entity = NamedEntity {
            text: "山田太郎".to_owned(),
            label: "PERSON".to_owned(),
            start_token: 3,
            end_token: 5,
            start_char: 7,
            end_char: 11,
        };
        let json = serde_json::to_string(&entity).unwrap();
        assert_eq!(serde_json::from_str::<NamedEntity>(&json).unwrap(), entity);
    }

    #[test]
    fn biluo_state_tracks_open_and_closed_entities() {
        let doc = Doc::from_words(&["New", "York", "works"], &[true, true, false]).unwrap();
        let mut state = NerState::new(&doc);
        assert_eq!(state.features(), [0, -1, -1]);
        state.apply(&NerAction::Begin("GPE".to_owned())).unwrap();
        assert_eq!(state.features(), [1, 0, 0]);
        state.apply(&NerAction::Last("GPE".to_owned())).unwrap();
        state.apply(&NerAction::Out).unwrap();
        assert!(state.is_final());
        assert_eq!(state.entities(), &[(0, 2, "GPE".to_owned())]);
    }

    #[test]
    fn biluo_state_closes_before_explicit_whitespace_tokens() {
        let doc = Doc::new(vec![
            TokenData::new("山田", true, 0),
            TokenData::new("太郎", false, 3),
            TokenData::new("\n", false, 5),
            TokenData::new("受託者", false, 6),
        ]);
        let mut state = NerState::new(&doc);

        assert!(state.is_valid(&NerAction::Begin("PERSON".to_owned())));
        state.apply(&NerAction::Begin("PERSON".to_owned())).unwrap();
        assert!(!state.is_valid(&NerAction::In("PERSON".to_owned())));
        assert!(state.is_valid(&NerAction::Last("PERSON".to_owned())));
        state.apply(&NerAction::Last("PERSON".to_owned())).unwrap();
        assert!(!state.is_valid(&NerAction::Unit("ORG".to_owned())));
        assert!(state.is_valid(&NerAction::Out));
    }
}
