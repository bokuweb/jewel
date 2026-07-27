use serde::{Deserialize, Serialize};
use spacy_core::{Doc, StringStore};
use spacy_model::Bundle;
use thiserror::Error;

use crate::{Matrix, Tok2Vec, Tok2VecError, TransitionScorer, TransitionScorerError};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NamedEntity {
    pub text: String,
    pub label: String,
    pub start_token: usize,
    pub end_token: usize,
    pub start_char: usize,
    pub end_char: usize,
}

/// Reusable entity-label filter backed by spaCy-compatible string IDs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EntityLabelFilter {
    entity_types: Vec<u64>,
}

impl EntityLabelFilter {
    /// Compile label names once for repeated single-document or batch extraction.
    #[must_use]
    pub fn new(labels: &[&str]) -> Self {
        let mut entity_types = labels
            .iter()
            .filter(|label| !label.is_empty())
            .map(|label| StringStore::id(label))
            .collect::<Vec<_>>();
        entity_types.sort_unstable();
        entity_types.dedup();
        Self { entity_types }
    }

    /// Return whether the filter contains no labels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entity_types.is_empty()
    }

    /// Return the number of distinct, non-empty labels in the filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entity_types.len()
    }

    /// Return whether a label is included in the filter.
    #[must_use]
    pub fn contains(&self, label: &str) -> bool {
        !label.is_empty() && self.matches(StringStore::id(label))
    }

    fn matches(&self, entity_type: u64) -> bool {
        self.entity_types.binary_search(&entity_type).is_ok()
    }
}

/// Model-aware entity-label selection with a reusable extraction filter.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EntityLabelSelection {
    filter: EntityLabelFilter,
    selected_labels: Vec<String>,
    missing_labels: Vec<String>,
}

impl EntityLabelSelection {
    pub(crate) fn compile(
        requested_labels: &[&str],
        mut supports: impl FnMut(&str) -> bool,
    ) -> Self {
        let mut selected_labels = Vec::new();
        let mut missing_labels = Vec::new();
        for label in requested_labels {
            if label.is_empty()
                || selected_labels.iter().any(|selected| selected == label)
                || missing_labels.iter().any(|missing| missing == label)
            {
                continue;
            }
            if supports(label) {
                selected_labels.push((*label).to_owned());
            } else {
                missing_labels.push((*label).to_owned());
            }
        }
        let filter = EntityLabelFilter::new(
            &selected_labels
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        Self {
            filter,
            selected_labels,
            missing_labels,
        }
    }

    /// Return the reusable filter containing labels declared by the model.
    #[must_use]
    pub fn filter(&self) -> &EntityLabelFilter {
        &self.filter
    }

    /// Return requested labels declared by the model, in request order.
    #[must_use]
    pub fn selected_labels(&self) -> &[String] {
        &self.selected_labels
    }

    /// Return requested labels absent from the model, in request order.
    #[must_use]
    pub fn missing_labels(&self) -> &[String] {
        &self.missing_labels
    }

    /// Return whether every distinct, non-empty requested label is supported.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.missing_labels.is_empty()
    }
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
    #[error("NER model requires vectors from an upstream tok2vec component")]
    ExternalTok2VecRequired,
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
    encoder: Option<Tok2Vec>,
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
            encoder: if uses_external_tok2vec(component) {
                None
            } else {
                Some(Tok2Vec::load(bundle, component_name)?)
            },
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

    /// Return whether this component consumes vectors from an upstream
    /// `tok2vec` component.
    #[must_use]
    pub const fn requires_external_tok2vec(&self) -> bool {
        self.encoder.is_none()
    }

    /// Return the entity labels declared by the loaded model.
    ///
    /// The order matches the exported component manifest. Callers can use this
    /// to report model capabilities before compiling downstream label filters.
    pub fn supported_entity_labels(&self) -> impl Iterator<Item = &str> {
        self.labels.iter().map(|(_, label)| label.as_str())
    }

    /// Return whether the loaded model declares an entity label.
    #[must_use]
    pub fn supports_entity_label(&self, label: &str) -> bool {
        !label.is_empty() && self.labels.iter().any(|(_, supported)| supported == label)
    }

    /// Compile requested labels against those declared by the loaded model.
    #[must_use]
    pub fn select_entity_labels(&self, labels: &[&str]) -> EntityLabelSelection {
        EntityLabelSelection::compile(labels, |label| self.supports_entity_label(label))
    }

    pub(crate) fn register_labels<'a>(&mut self, labels: impl IntoIterator<Item = &'a str>) {
        for label in labels {
            if label.is_empty() || self.labels.iter().any(|(_, known)| known == label) {
                continue;
            }
            self.labels.push((StringStore::id(label), label.to_owned()));
        }
    }

    /// Recognize entities and attach spaCy-compatible `ENT_IOB`/`ENT_TYPE`.
    ///
    /// # Errors
    ///
    /// Returns an error if neural inference or transition decoding fails.
    pub fn annotate(&self, doc: &mut Doc) -> Result<Vec<usize>, EntityRecognizerError> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or(EntityRecognizerError::ExternalTok2VecRequired)?;
        let vectors = encoder.forward(doc)?;
        self.annotate_with_tok2vec(doc, &vectors)
    }

    /// Recognize entities using vectors produced by the source pipeline's
    /// upstream `tok2vec` component.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector shape, transition scorer, or decoding
    /// state is incompatible.
    pub fn annotate_with_tok2vec(
        &self,
        doc: &mut Doc,
        vectors: &Matrix,
    ) -> Result<Vec<usize>, EntityRecognizerError> {
        if vectors.rows() != doc.len() {
            return Err(EntityRecognizerError::InvalidModel(format!(
                "document has {} tokens but tok2vec has {} rows",
                doc.len(),
                vectors.rows()
            )));
        }
        let cache = self.scorer.precompute(vectors)?;
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
        collect_entities(doc, &self.labels, |_| true)
    }

    /// Return entity spans whose labels are included in `labels`.
    ///
    /// An empty label list returns no entities. Entity text is allocated only
    /// for matching spans, which is useful when a downstream application needs
    /// a small subset such as people and organizations.
    #[must_use]
    pub fn entities_by_labels(&self, doc: &Doc, labels: &[&str]) -> Vec<NamedEntity> {
        self.entities_with_filter(doc, &EntityLabelFilter::new(labels))
    }

    /// Return entity spans accepted by a reusable label filter.
    #[must_use]
    pub fn entities_with_filter(&self, doc: &Doc, filter: &EntityLabelFilter) -> Vec<NamedEntity> {
        collect_entities(doc, &self.labels, |entity_type| filter.matches(entity_type))
    }

    /// Return entity spans with the requested label.
    #[must_use]
    pub fn entities_by_label(&self, doc: &Doc, label: &str) -> Vec<NamedEntity> {
        self.entities_by_labels(doc, &[label])
    }
}

fn uses_external_tok2vec(component: &spacy_model::ComponentManifest) -> bool {
    component
        .nodes
        .iter()
        .any(|node| node.name == "tok2vec-listener")
}

fn collect_entities(
    doc: &Doc,
    labels: &[(u64, String)],
    mut matches_entity_type: impl FnMut(u64) -> bool,
) -> Vec<NamedEntity> {
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
        if !matches_entity_type(first.ent_type) {
            start = end;
            continue;
        }
        let label = labels
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

#[cfg(test)]
mod tests {
    use spacy_core::{Doc, StringStore, TokenData};
    use spacy_model::ComponentManifest;

    use super::{
        collect_entities, uses_external_tok2vec, EntityLabelFilter, EntityLabelSelection,
        NamedEntity, NerAction, NerState,
    };

    #[test]
    fn detects_an_upstream_tok2vec_listener() {
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
        assert!(uses_external_tok2vec(&component));
    }

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

    #[test]
    fn entity_collection_builds_only_selected_labels() {
        let person = StringStore::id("PERSON");
        let organization = StringStore::id("ORG");
        let mut tokens = vec![
            TokenData::new("Jane", true, 0),
            TokenData::new("Smith", false, 5),
            TokenData::new("Acme", false, 11),
        ];
        tokens[0].ent_iob = 3;
        tokens[0].ent_type = person;
        tokens[1].ent_iob = 1;
        tokens[1].ent_type = person;
        tokens[2].ent_iob = 3;
        tokens[2].ent_type = organization;
        let doc = Doc::new(tokens);
        let labels = vec![
            (person, "PERSON".to_owned()),
            (organization, "ORG".to_owned()),
        ];

        let filter = EntityLabelFilter::new(&["", "ORG", "PERSON", "PERSON"]);
        assert!(!filter.is_empty());
        assert_eq!(filter.len(), 2);
        assert!(filter.contains("PERSON"));
        assert!(filter.contains("ORG"));
        assert!(!filter.contains("GPE"));
        assert!(!filter.contains(""));
        assert!(EntityLabelFilter::new(&[""]).is_empty());
        let selected = collect_entities(&doc, &labels, |entity_type| {
            entity_type == StringStore::id("PERSON")
        });
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].text, "Jane Smith");
        assert_eq!(selected[0].start_char, 0);
        assert_eq!(selected[0].end_char, 10);
        assert!(collect_entities(&doc, &labels, |_| false).is_empty());
        assert!(filter.matches(person));
        assert!(filter.matches(organization));
        assert!(!EntityLabelFilter::default().matches(person));
    }

    #[test]
    fn model_aware_selection_deduplicates_and_reports_missing_labels() {
        let selection = EntityLabelSelection::compile(
            &["", "PERSON", "ORG", "PERSON", "PRODUCT", "PRODUCT"],
            |label| matches!(label, "PERSON" | "ORG"),
        );

        assert_eq!(selection.selected_labels(), ["PERSON", "ORG"]);
        assert_eq!(selection.missing_labels(), ["PRODUCT"]);
        assert!(selection.filter().contains("PERSON"));
        assert!(selection.filter().contains("ORG"));
        assert!(!selection.filter().contains("PRODUCT"));
        assert!(!selection.is_complete());

        let empty = EntityLabelSelection::compile(&["", ""], |_| true);
        assert!(empty.selected_labels().is_empty());
        assert!(empty.missing_labels().is_empty());
        assert!(empty.filter().is_empty());
        assert!(empty.is_complete());
    }
}
