use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use spacy_core::{CharSpanAlignment, Doc, StringStore};
use spacy_model::Bundle;
use thiserror::Error;

use crate::{Matrix, Tok2Vec, Tok2VecError, TransitionScorer, TransitionScorerError};

use super::entity_ruler::EntityRuler;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NamedEntity {
    pub text: String,
    pub label: String,
    /// Optional spaCy `Span.ent_id_`, typically assigned by `EntityRuler`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ent_id: Option<String>,
    pub start_token: usize,
    pub end_token: usize,
    pub start_char: usize,
    pub end_char: usize,
}

/// A preset spaCy NER annotation that constrains inference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EntityConstraint {
    /// Assign a spaCy `Doc.set_ents` default to every uncovered token.
    Default(EntityConstraintDefault),
    /// Force an entity over the half-open token range.
    Entity {
        start: usize,
        end: usize,
        label: String,
    },
    /// Prevent an entity over the half-open token range.
    Blocked { start: usize, end: usize },
    /// Preset spaCy's `O` annotation over the half-open token range.
    Outside { start: usize, end: usize },
    /// Clear the entity annotation over the half-open token range.
    Missing { start: usize, end: usize },
    /// Force an entity over an aligned Unicode character range.
    EntityChars {
        start: usize,
        end: usize,
        label: String,
        alignment: CharSpanAlignment,
    },
    /// Prevent an entity over an aligned Unicode character range.
    BlockedChars {
        start: usize,
        end: usize,
        alignment: CharSpanAlignment,
    },
    /// Preset spaCy's `O` annotation over an aligned character range.
    OutsideChars {
        start: usize,
        end: usize,
        alignment: CharSpanAlignment,
    },
    /// Clear annotations over an aligned Unicode character range.
    MissingChars {
        start: usize,
        end: usize,
        alignment: CharSpanAlignment,
    },
}

/// Annotation assigned to tokens not covered by an explicit constraint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EntityConstraintDefault {
    /// Preserve each uncovered token's existing annotation.
    #[default]
    Unmodified,
    /// Prevent entities on every uncovered token.
    Blocked,
    /// Clear annotations on every uncovered token.
    Missing,
    /// Preset spaCy's `O` annotation on every uncovered token.
    Outside,
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
    #[error("invalid entity constraint range {start}..{end} for document length {len}")]
    InvalidConstraintRange {
        start: usize,
        end: usize,
        len: usize,
    },
    #[error("entity constraints overlap at token {token}")]
    OverlappingConstraint { token: usize },
    #[error("entity constraints specify more than one default")]
    MultipleConstraintDefaults,
    #[error("entity constraint labels must not be empty")]
    EmptyConstraintLabel,
    #[error("NER label mapping {0:?} is not present in the exported component")]
    MissingLabelMapping(String),
    #[error("NER label {label:?} is absent from mapping {mapping:?}")]
    MissingMappedLabel { mapping: String, label: String },
    #[error(
        "character constraint range {start}..{end} does not align to a non-empty token span with {alignment:?}"
    )]
    UnalignedCharacterConstraint {
        start: usize,
        end: usize,
        alignment: CharSpanAlignment,
    },
}

/// Attach preset entity, blocked, missing, and outside annotations to a document.
///
/// The resulting `ENT_IOB`/`ENT_TYPE` values use the same representation as
/// `Doc.set_ents(..., default="unmodified")` and are consumed by
/// [`EntityRecognizer`] as preset transition annotations. Token constraints
/// use token indexes. `*Chars` constraints use Unicode code-point offsets and
/// spaCy-compatible `Doc.char_span` alignment. A `Default` constraint assigns
/// the selected annotation to every otherwise uncovered token.
///
/// # Errors
///
/// Returns an error for empty, out-of-bounds, or overlapping spans, or an
/// entity with an empty label.
pub fn apply_entity_constraints(
    doc: &mut Doc,
    constraints: &[EntityConstraint],
) -> Result<(), EntityRecognizerError> {
    apply_entity_constraints_impl(doc, constraints, None)
}

/// Attach constraints and assign `default` to every uncovered token.
///
/// This reproduces spaCy's `Doc.set_ents(..., default=...)` choices. Explicit
/// constraints always take precedence over the selected default. The
/// constraint slice must not also contain an [`EntityConstraint::Default`].
///
/// # Errors
///
/// Returns an error for empty, out-of-bounds, unaligned, or overlapping spans,
/// or an entity with an empty label.
pub fn apply_entity_constraints_with_default(
    doc: &mut Doc,
    constraints: &[EntityConstraint],
    default: EntityConstraintDefault,
) -> Result<(), EntityRecognizerError> {
    apply_entity_constraints_impl(doc, constraints, Some(default))
}

fn apply_entity_constraints_impl(
    doc: &mut Doc,
    constraints: &[EntityConstraint],
    mut default: Option<EntityConstraintDefault>,
) -> Result<(), EntityRecognizerError> {
    enum ConstraintKind<'a> {
        Entity(&'a str),
        Blocked,
        Outside,
        Missing,
    }

    let mut resolved = Vec::with_capacity(constraints.len());
    for constraint in constraints {
        if let EntityConstraint::Default(value) = constraint {
            if default.replace(*value).is_some() {
                return Err(EntityRecognizerError::MultipleConstraintDefaults);
            }
            continue;
        }
        if matches!(
            constraint,
            EntityConstraint::Entity { label, .. }
                | EntityConstraint::EntityChars { label, .. }
                if label.is_empty()
        ) {
            return Err(EntityRecognizerError::EmptyConstraintLabel);
        }
        let (start, end, kind) = match constraint {
            EntityConstraint::Default(_) => unreachable!(),
            EntityConstraint::Entity { start, end, label } => {
                (*start, *end, ConstraintKind::Entity(label))
            }
            EntityConstraint::Blocked { start, end } => (*start, *end, ConstraintKind::Blocked),
            EntityConstraint::Outside { start, end } => (*start, *end, ConstraintKind::Outside),
            EntityConstraint::Missing { start, end } => (*start, *end, ConstraintKind::Missing),
            EntityConstraint::EntityChars {
                start,
                end,
                label,
                alignment,
            } => {
                let span = doc
                    .char_span(*start..*end, *alignment)
                    .filter(|span| !span.is_empty())
                    .ok_or(EntityRecognizerError::UnalignedCharacterConstraint {
                        start: *start,
                        end: *end,
                        alignment: *alignment,
                    })?;
                (span.start(), span.end(), ConstraintKind::Entity(label))
            }
            EntityConstraint::BlockedChars {
                start,
                end,
                alignment,
            } => {
                let span = doc
                    .char_span(*start..*end, *alignment)
                    .filter(|span| !span.is_empty())
                    .ok_or(EntityRecognizerError::UnalignedCharacterConstraint {
                        start: *start,
                        end: *end,
                        alignment: *alignment,
                    })?;
                (span.start(), span.end(), ConstraintKind::Blocked)
            }
            EntityConstraint::OutsideChars {
                start,
                end,
                alignment,
            } => {
                let span = doc
                    .char_span(*start..*end, *alignment)
                    .filter(|span| !span.is_empty())
                    .ok_or(EntityRecognizerError::UnalignedCharacterConstraint {
                        start: *start,
                        end: *end,
                        alignment: *alignment,
                    })?;
                (span.start(), span.end(), ConstraintKind::Outside)
            }
            EntityConstraint::MissingChars {
                start,
                end,
                alignment,
            } => {
                let span = doc
                    .char_span(*start..*end, *alignment)
                    .filter(|span| !span.is_empty())
                    .ok_or(EntityRecognizerError::UnalignedCharacterConstraint {
                        start: *start,
                        end: *end,
                        alignment: *alignment,
                    })?;
                (span.start(), span.end(), ConstraintKind::Missing)
            }
        };
        resolved.push((start, end, kind));
    }

    let mut claimed = vec![false; doc.len()];
    for (start, end, _) in &resolved {
        if start >= end || *end > doc.len() {
            return Err(EntityRecognizerError::InvalidConstraintRange {
                start: *start,
                end: *end,
                len: doc.len(),
            });
        }
        for (offset, is_claimed) in claimed[*start..*end].iter_mut().enumerate() {
            if *is_claimed {
                return Err(EntityRecognizerError::OverlappingConstraint {
                    token: *start + offset,
                });
            }
            *is_claimed = true;
        }
    }
    for (start, end, kind) in resolved {
        match kind {
            ConstraintKind::Entity(label) => {
                let entity_type = StringStore::id(label);
                for (offset, token) in doc.tokens_mut()[start..end].iter_mut().enumerate() {
                    token.ent_iob = if offset == 0 { 3 } else { 1 };
                    token.ent_type = entity_type;
                    token.ent_id = 0;
                    token.ent_kb_id = 0;
                }
            }
            ConstraintKind::Blocked => {
                for token in &mut doc.tokens_mut()[start..end] {
                    token.ent_iob = 3;
                    token.ent_type = 0;
                    token.ent_id = 0;
                    token.ent_kb_id = 0;
                }
            }
            ConstraintKind::Outside => {
                for token in &mut doc.tokens_mut()[start..end] {
                    token.ent_iob = 2;
                    token.ent_type = 0;
                    token.ent_id = 0;
                    token.ent_kb_id = 0;
                }
            }
            ConstraintKind::Missing => {
                for token in &mut doc.tokens_mut()[start..end] {
                    token.ent_iob = 0;
                    token.ent_type = 0;
                    token.ent_id = 0;
                    token.ent_kb_id = 0;
                }
            }
        }
    }
    let default = default.unwrap_or_default();
    if default != EntityConstraintDefault::Unmodified {
        for (token, is_claimed) in doc.tokens_mut().iter_mut().zip(claimed) {
            if is_claimed {
                continue;
            }
            match default {
                EntityConstraintDefault::Unmodified => unreachable!(),
                EntityConstraintDefault::Blocked => token.ent_iob = 3,
                EntityConstraintDefault::Missing => token.ent_iob = 0,
                EntityConstraintDefault::Outside => token.ent_iob = 2,
            }
            token.ent_type = 0;
            token.ent_id = 0;
            token.ent_kb_id = 0;
        }
    }
    Ok(())
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
    preset_iob: Vec<u8>,
    preset_type: Vec<u64>,
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
            preset_iob: doc.tokens().iter().map(|token| token.ent_iob).collect(),
            preset_type: doc.tokens().iter().map(|token| token.ent_type).collect(),
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
        let current_iob = self.preset_iob[self.buffer];
        let current_type = self.preset_type[self.buffer];
        let next_iob = self.preset_iob.get(self.buffer + 1).copied().unwrap_or(0);
        let next_starts_sentence = self
            .sent_starts
            .get(self.buffer + 1)
            .copied()
            .unwrap_or(false);
        match action {
            NerAction::Begin(label) => {
                if self.open.is_some() || remaining < 2 || label.is_empty() || current_iob == 1 {
                    false
                } else if current_iob == 3 {
                    StringStore::id(label) == current_type && next_iob == 1
                } else {
                    next_iob != 3 && !next_starts_sentence && !current_space
                }
            }
            NerAction::In(label) => {
                if remaining < 2
                    || label.is_empty()
                    || self
                        .open
                        .as_ref()
                        .is_none_or(|(_, open_label)| open_label != label)
                    || current_iob == 3
                    || next_iob == 3
                {
                    false
                } else if current_iob == 1 {
                    !matches!(next_iob, 0 | 2) && StringStore::id(label) == current_type
                } else {
                    !next_starts_sentence
                }
            }
            NerAction::Last(label) => {
                if label.is_empty() || self.open.is_none() {
                    false
                } else if current_iob == 1 && next_iob != 1 {
                    StringStore::id(label) == current_type
                } else {
                    self.open
                        .as_ref()
                        .is_some_and(|(_, open_label)| open_label == label)
                        && next_iob != 1
                }
            }
            NerAction::Unit(label) => {
                if label.is_empty() {
                    current_iob == 3 && current_type == 0
                } else if self.open.is_some() || next_iob == 1 {
                    false
                } else if current_iob == 3 {
                    StringStore::id(label) == current_type
                } else {
                    !current_space
                }
            }
            NerAction::Out => self.open.is_none() && !matches!(current_iob, 1 | 3),
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
                if !label.is_empty() {
                    self.entities
                        .push((self.buffer, self.buffer + 1, label.clone()));
                }
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
    entity_ids: Vec<(u64, String)>,
    label_mappings: BTreeMap<String, BTreeMap<String, String>>,
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
            encoder: if uses_external_vectors(component) {
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
            entity_ids: Vec::new(),
            label_mappings: load_label_mappings(component)?,
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

    pub(crate) fn register_entity_ids<'a>(&mut self, ids: impl IntoIterator<Item = &'a str>) {
        for id in ids {
            if id.is_empty() || self.entity_ids.iter().any(|(_, known)| known == id) {
                continue;
            }
            self.entity_ids.push((StringStore::id(id), id.to_owned()));
        }
    }

    /// Register labels and pattern IDs produced by a post-NER entity ruler.
    ///
    /// Custom pipeline orchestrators must call this before extracting entities
    /// so ruler-only labels and `Span.ent_id_` values can be resolved.
    pub fn register_entity_ruler(&mut self, ruler: &EntityRuler) {
        self.register_labels(ruler.labels());
        self.register_entity_ids(ruler.entity_ids());
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
                token.ent_id = 0;
                token.ent_kb_id = 0;
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
        collect_entities(doc, &self.labels, &self.entity_ids, |_| true)
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
        collect_entities(doc, &self.labels, &self.entity_ids, |entity_type| {
            filter.matches(entity_type)
        })
    }

    /// Return entity spans with the requested label.
    #[must_use]
    pub fn entities_by_label(&self, doc: &Doc, label: &str) -> Vec<NamedEntity> {
        self.entities_by_labels(doc, &[label])
    }

    /// Return entity spans with labels converted by an exported mapping.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested mapping or an observed source label
    /// is absent.
    pub fn entities_with_mapping(
        &self,
        doc: &Doc,
        mapping_name: &str,
    ) -> Result<Vec<NamedEntity>, EntityRecognizerError> {
        self.entities_with_mapping_impl(doc, mapping_name, None)
    }

    /// Return mapped spans, replacing unknown labels with `fallback`.
    ///
    /// This is useful when post-NER rulers introduce labels that are not part
    /// of the statistical component's exported mapping.
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
        self.entities_with_mapping_impl(doc, mapping_name, Some(fallback))
    }

    /// Return token-aligned B/I/O labels converted by an exported mapping.
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
        let mapping = self
            .label_mappings
            .get(mapping_name)
            .ok_or_else(|| EntityRecognizerError::MissingLabelMapping(mapping_name.to_owned()))?;
        Ok(mapped_token_labels(doc, &self.labels, mapping, fallback))
    }

    fn entities_with_mapping_impl(
        &self,
        doc: &Doc,
        mapping_name: &str,
        fallback: Option<&str>,
    ) -> Result<Vec<NamedEntity>, EntityRecognizerError> {
        let mapping = self
            .label_mappings
            .get(mapping_name)
            .ok_or_else(|| EntityRecognizerError::MissingLabelMapping(mapping_name.to_owned()))?;
        self.entities(doc)
            .into_iter()
            .map(|mut entity| {
                entity.label = if let Some(mapped) = mapping.get(&entity.label) {
                    mapped.clone()
                } else if let Some(fallback) = fallback {
                    fallback.to_owned()
                } else {
                    return Err(EntityRecognizerError::MissingMappedLabel {
                        mapping: mapping_name.to_owned(),
                        label: entity.label,
                    });
                };
                Ok(entity)
            })
            .collect()
    }
}

fn load_label_mappings(
    component: &spacy_model::ComponentManifest,
) -> Result<BTreeMap<String, BTreeMap<String, String>>, EntityRecognizerError> {
    let Some(value) = component.settings.get("label_mappings") else {
        return Ok(BTreeMap::new());
    };
    let mappings: BTreeMap<String, BTreeMap<String, String>> =
        serde_json::from_value(value.clone()).map_err(|error| {
            EntityRecognizerError::InvalidModel(format!(
                "component label mappings are invalid: {error}"
            ))
        })?;
    for (name, mapping) in &mappings {
        if name.is_empty() {
            return Err(EntityRecognizerError::InvalidModel(
                "component label mapping name must not be empty".to_owned(),
            ));
        }
        for label in &component.labels {
            match mapping.get(label) {
                None => {
                    return Err(EntityRecognizerError::InvalidModel(format!(
                        "mapping {name:?} is missing component label {label:?}"
                    )));
                }
                Some(target) if target.is_empty() => {
                    return Err(EntityRecognizerError::InvalidModel(format!(
                        "mapping {name:?} has an empty target for label {label:?}"
                    )));
                }
                Some(_) => {}
            }
        }
        if let Some(source) = mapping
            .keys()
            .find(|source| !component.labels.contains(source))
        {
            return Err(EntityRecognizerError::InvalidModel(format!(
                "mapping {name:?} contains unknown source label {source:?}"
            )));
        }
    }
    Ok(mappings)
}

fn mapped_token_labels(
    doc: &Doc,
    labels: &[(u64, String)],
    mapping: &BTreeMap<String, String>,
    fallback: &str,
) -> Vec<String> {
    doc.tokens()
        .iter()
        .map(|token| match token.ent_iob {
            3 | 1 => {
                let source = labels
                    .iter()
                    .find(|(id, _)| *id == token.ent_type)
                    .map_or("", |(_, label)| label.as_str());
                let mapped = mapping.get(source).map_or(fallback, String::as_str);
                format!("{}-{mapped}", if token.ent_iob == 3 { "B" } else { "I" })
            }
            2 => "O".to_owned(),
            _ => String::new(),
        })
        .collect()
}

fn uses_external_vectors(component: &spacy_model::ComponentManifest) -> bool {
    component.nodes.iter().any(|node| {
        matches!(
            node.name.as_str(),
            "tok2vec-listener" | "transformer-listener"
        )
    })
}

fn collect_entities(
    doc: &Doc,
    labels: &[(u64, String)],
    entity_ids: &[(u64, String)],
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
        let ent_id = (first.ent_id != 0).then(|| {
            entity_ids
                .iter()
                .find(|(id, _)| *id == first.ent_id)
                .map_or_else(|| first.ent_id.to_string(), |(_, id)| id.clone())
        });
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
            ent_id,
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
    use std::collections::BTreeMap;

    use spacy_core::{CharSpanAlignment, Doc, StringStore, TokenData};
    use spacy_model::ComponentManifest;

    use super::{
        apply_entity_constraints, apply_entity_constraints_with_default, collect_entities,
        load_label_mappings, mapped_token_labels, uses_external_vectors, EntityConstraint,
        EntityConstraintDefault, EntityLabelFilter, EntityLabelSelection, EntityRecognizerError,
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
        assert!(uses_external_vectors(&component));
    }

    #[test]
    fn detects_an_upstream_transformer_listener() {
        let component: ComponentManifest = serde_json::from_value(serde_json::json!({
            "name": "ner",
            "factory": "ner",
            "kind": "trainable",
            "root_node": 0,
            "nodes": [{
                "index": 0,
                "name": "transformer-listener",
                "dims": {},
                "refs": {},
                "params": {}
            }]
        }))
        .unwrap();
        assert!(uses_external_vectors(&component));
    }

    #[test]
    fn loads_complete_exported_label_mappings() {
        let component: ComponentManifest = serde_json::from_value(serde_json::json!({
            "name": "ner",
            "factory": "ner",
            "kind": "trainable",
            "settings": {
                "label_mappings": {
                    "ontonotes": {
                        "Company": "ORG",
                        "Person": "PERSON"
                    }
                }
            },
            "labels": ["Company", "Person"]
        }))
        .unwrap();
        let mappings = load_label_mappings(&component).unwrap();
        assert_eq!(mappings["ontonotes"]["Company"], "ORG");
        assert_eq!(mappings["ontonotes"]["Person"], "PERSON");
    }

    #[test]
    fn rejects_incomplete_exported_label_mappings() {
        let component: ComponentManifest = serde_json::from_value(serde_json::json!({
            "name": "ner",
            "factory": "ner",
            "kind": "trainable",
            "settings": {
                "label_mappings": {
                    "ontonotes": {
                        "Company": "ORG"
                    }
                }
            },
            "labels": ["Company", "Person"]
        }))
        .unwrap();
        assert!(matches!(
            load_label_mappings(&component),
            Err(EntityRecognizerError::InvalidModel(message))
                if message.contains("Person")
        ));
    }

    #[test]
    fn maps_token_aligned_bio_labels_with_ginza_fallback() {
        let mut doc = Doc::from_words(&["東京", "と", "独自"], &[false; 3]).unwrap();
        doc.tokens_mut()[0].ent_iob = 3;
        doc.tokens_mut()[0].ent_type = StringStore::id("City");
        doc.tokens_mut()[1].ent_iob = 2;
        doc.tokens_mut()[2].ent_iob = 3;
        doc.tokens_mut()[2].ent_type = StringStore::id("Custom");
        let labels = vec![
            (StringStore::id("City"), "City".to_owned()),
            (StringStore::id("Custom"), "Custom".to_owned()),
        ];
        let mapping = BTreeMap::from([("City".to_owned(), "GPE".to_owned())]);

        assert_eq!(
            mapped_token_labels(&doc, &labels, &mapping, "OTHERS"),
            ["B-GPE", "O", "B-OTHERS"]
        );
    }

    #[test]
    fn named_entity_round_trips_as_json() {
        let entity = NamedEntity {
            text: "山田太郎".to_owned(),
            label: "PERSON".to_owned(),
            ent_id: Some("contract-party".to_owned()),
            start_token: 3,
            end_token: 5,
            start_char: 7,
            end_char: 11,
        };
        let json = serde_json::to_string(&entity).unwrap();
        assert_eq!(serde_json::from_str::<NamedEntity>(&json).unwrap(), entity);
        assert_eq!(
            serde_json::from_str::<NamedEntity>(
                r#"{"text":"山田太郎","label":"PERSON","start_token":3,"end_token":5,"start_char":7,"end_char":11}"#
            )
            .unwrap()
            .ent_id,
            None
        );
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
    fn preset_multi_token_entity_forces_matching_biluo_moves() {
        let mut doc = Doc::from_words(&["New", "York"], &[true, false]).unwrap();
        doc.tokens_mut()[0].ent_iob = 3;
        doc.tokens_mut()[0].ent_type = StringStore::id("GPE");
        doc.tokens_mut()[1].ent_iob = 1;
        doc.tokens_mut()[1].ent_type = StringStore::id("GPE");
        let mut state = NerState::new(&doc);

        assert!(state.is_valid(&NerAction::Begin("GPE".to_owned())));
        assert!(!state.is_valid(&NerAction::Begin("ORG".to_owned())));
        assert!(!state.is_valid(&NerAction::Unit("GPE".to_owned())));
        assert!(!state.is_valid(&NerAction::Out));
        state.apply(&NerAction::Begin("GPE".to_owned())).unwrap();
        assert!(state.is_valid(&NerAction::Last("GPE".to_owned())));
        assert!(!state.is_valid(&NerAction::In("GPE".to_owned())));
    }

    #[test]
    fn blocked_tokens_force_empty_unit_move() {
        let mut doc = Doc::from_words(&["secret"], &[false]).unwrap();
        doc.tokens_mut()[0].ent_iob = 3;
        let mut state = NerState::new(&doc);

        assert!(state.is_valid(&NerAction::Unit(String::new())));
        assert!(!state.is_valid(&NerAction::Unit("ORG".to_owned())));
        assert!(!state.is_valid(&NerAction::Out));
        state.apply(&NerAction::Unit(String::new())).unwrap();
        assert!(state.entities().is_empty());
    }

    #[test]
    fn preset_entity_can_cross_sentence_and_whitespace_boundaries() {
        let mut doc = Doc::from_words(&["A", " ", "B"], &[false, false, false]).unwrap();
        for (index, token) in doc.tokens_mut().iter_mut().enumerate() {
            token.ent_iob = if index == 0 { 3 } else { 1 };
            token.ent_type = StringStore::id("ORG");
        }
        doc.tokens_mut()[1].sent_start = 1;
        let mut state = NerState::new(&doc);

        assert!(state.is_valid(&NerAction::Begin("ORG".to_owned())));
        state.apply(&NerAction::Begin("ORG".to_owned())).unwrap();
        assert!(state.is_valid(&NerAction::In("ORG".to_owned())));
        state.apply(&NerAction::In("ORG".to_owned())).unwrap();
        assert!(state.is_valid(&NerAction::Last("ORG".to_owned())));
    }

    #[test]
    fn entity_constraints_match_spacy_iob_representation() {
        let mut doc =
            Doc::from_words(&["preset", "entity", "blocked", "outside"], &[true; 4]).unwrap();
        apply_entity_constraints(
            &mut doc,
            &[
                EntityConstraint::Entity {
                    start: 0,
                    end: 2,
                    label: "ORG".to_owned(),
                },
                EntityConstraint::Blocked { start: 2, end: 3 },
                EntityConstraint::Outside { start: 3, end: 4 },
            ],
        )
        .unwrap();

        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| (token.ent_iob, token.ent_type))
                .collect::<Vec<_>>(),
            vec![
                (3, StringStore::id("ORG")),
                (1, StringStore::id("ORG")),
                (3, 0),
                (2, 0),
            ]
        );
    }

    #[test]
    fn character_constraints_align_unicode_offsets_before_ner() {
        let mut doc =
            Doc::from_words(&["株式会社", "青空", "は", "東京", "です"], &[false; 5]).unwrap();
        doc.tokens_mut()[4].ent_iob = 3;
        doc.tokens_mut()[4].ent_type = StringStore::id("OLD");
        apply_entity_constraints(
            &mut doc,
            &[
                EntityConstraint::EntityChars {
                    start: 0,
                    end: 6,
                    label: "Company".to_owned(),
                    alignment: CharSpanAlignment::Strict,
                },
                EntityConstraint::BlockedChars {
                    start: 6,
                    end: 7,
                    alignment: CharSpanAlignment::Strict,
                },
                EntityConstraint::OutsideChars {
                    start: 7,
                    end: 9,
                    alignment: CharSpanAlignment::Strict,
                },
                EntityConstraint::MissingChars {
                    start: 9,
                    end: 11,
                    alignment: CharSpanAlignment::Strict,
                },
            ],
        )
        .unwrap();

        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| (token.ent_iob, token.ent_type))
                .collect::<Vec<_>>(),
            vec![
                (3, StringStore::id("Company")),
                (1, StringStore::id("Company")),
                (3, 0),
                (2, 0),
                (0, 0),
            ]
        );
    }

    #[test]
    fn constraint_defaults_match_spacy_3_8_set_ents() {
        let cases = [
            (
                EntityConstraintDefault::Unmodified,
                (3, StringStore::id("OLD")),
            ),
            (EntityConstraintDefault::Blocked, (3, 0)),
            (EntityConstraintDefault::Missing, (0, 0)),
            (EntityConstraintDefault::Outside, (2, 0)),
        ];
        for (default, expected_last) in cases {
            let mut doc = Doc::from_words(
                &[
                    "entity", "inside", "blocked", "missing", "outside", "default",
                ],
                &[true, true, true, true, true, false],
            )
            .unwrap();
            doc.tokens_mut()[5].ent_iob = 3;
            doc.tokens_mut()[5].ent_type = StringStore::id("OLD");

            apply_entity_constraints_with_default(
                &mut doc,
                &[
                    EntityConstraint::Entity {
                        start: 0,
                        end: 2,
                        label: "ORG".to_owned(),
                    },
                    EntityConstraint::Blocked { start: 2, end: 3 },
                    EntityConstraint::Missing { start: 3, end: 4 },
                    EntityConstraint::Outside { start: 4, end: 5 },
                ],
                default,
            )
            .unwrap();

            assert_eq!(
                doc.tokens()
                    .iter()
                    .map(|token| (token.ent_iob, token.ent_type))
                    .collect::<Vec<_>>(),
                vec![
                    (3, StringStore::id("ORG")),
                    (1, StringStore::id("ORG")),
                    (3, 0),
                    (0, 0),
                    (2, 0),
                    expected_last,
                ]
            );
        }
    }

    #[test]
    fn default_constraint_applies_through_existing_pipeline_inputs() {
        let mut doc = Doc::from_words(&["Acme", "and", "Example"], &[true, true, false]).unwrap();
        apply_entity_constraints(
            &mut doc,
            &[
                EntityConstraint::Default(EntityConstraintDefault::Blocked),
                EntityConstraint::Entity {
                    start: 0,
                    end: 1,
                    label: "ORG".to_owned(),
                },
            ],
        )
        .unwrap();

        assert_eq!(
            doc.tokens()
                .iter()
                .map(|token| (token.ent_iob, token.ent_type))
                .collect::<Vec<_>>(),
            [(3, StringStore::id("ORG")), (3, 0), (3, 0),]
        );
    }

    #[test]
    fn multiple_constraint_defaults_are_rejected() {
        let mut doc = Doc::from_words(&["Acme"], &[false]).unwrap();
        let error = apply_entity_constraints(
            &mut doc,
            &[
                EntityConstraint::Default(EntityConstraintDefault::Missing),
                EntityConstraint::Default(EntityConstraintDefault::Outside),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EntityRecognizerError::MultipleConstraintDefaults
        ));

        let error = apply_entity_constraints_with_default(
            &mut doc,
            &[EntityConstraint::Default(
                EntityConstraintDefault::Unmodified,
            )],
            EntityConstraintDefault::Outside,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            EntityRecognizerError::MultipleConstraintDefaults
        ));
    }

    #[test]
    fn character_constraints_report_strict_alignment_failures() {
        let mut doc = Doc::from_words(&["株式会社", "青空"], &[false; 2]).unwrap();
        let error = apply_entity_constraints(
            &mut doc,
            &[EntityConstraint::EntityChars {
                start: 1,
                end: 6,
                label: "Company".to_owned(),
                alignment: CharSpanAlignment::Strict,
            }],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            EntityRecognizerError::UnalignedCharacterConstraint {
                start: 1,
                end: 6,
                alignment: CharSpanAlignment::Strict,
            }
        ));
    }

    #[test]
    fn overlapping_entity_constraints_are_rejected() {
        let mut doc = Doc::from_words(&["one", "two"], &[true, false]).unwrap();
        let error = apply_entity_constraints(
            &mut doc,
            &[
                EntityConstraint::Blocked { start: 0, end: 2 },
                EntityConstraint::Outside { start: 1, end: 2 },
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            super::EntityRecognizerError::OverlappingConstraint { token: 1 }
        ));
    }

    #[test]
    fn biluo_state_allows_explicit_whitespace_inside_entities() {
        let doc = Doc::new(vec![
            TokenData::new("山田", true, 0),
            TokenData::new("太郎", false, 3),
            TokenData::new("\n", false, 5),
            TokenData::new("受託者", false, 6),
        ]);
        let mut state = NerState::new(&doc);

        assert!(state.is_valid(&NerAction::Begin("PERSON".to_owned())));
        state.apply(&NerAction::Begin("PERSON".to_owned())).unwrap();
        assert!(state.is_valid(&NerAction::In("PERSON".to_owned())));
        state.apply(&NerAction::In("PERSON".to_owned())).unwrap();
        assert!(state.is_valid(&NerAction::Last("PERSON".to_owned())));
        state.apply(&NerAction::Last("PERSON".to_owned())).unwrap();
        assert_eq!(state.entities(), &[(0, 3, "PERSON".to_owned())]);
        assert!(state.is_valid(&NerAction::Unit("ORG".to_owned())));
    }

    #[test]
    fn biluo_state_does_not_start_entities_on_explicit_whitespace() {
        let doc = Doc::new(vec![
            TokenData::new("\n", false, 0),
            TokenData::new("受託者", false, 1),
        ]);
        let state = NerState::new(&doc);

        assert!(!state.is_valid(&NerAction::Begin("ORG".to_owned())));
        assert!(!state.is_valid(&NerAction::Unit("ORG".to_owned())));
        assert!(state.is_valid(&NerAction::Out));
    }

    #[test]
    fn biluo_state_closes_before_sentence_boundaries() {
        let mut doc = Doc::new(vec![
            TokenData::new("山田", true, 0),
            TokenData::new("太郎", false, 3),
            TokenData::new("受託者", false, 5),
        ]);
        doc.tokens_mut()[2].sent_start = 1;
        let mut state = NerState::new(&doc);

        state.apply(&NerAction::Begin("PERSON".to_owned())).unwrap();
        assert!(!state.is_valid(&NerAction::In("PERSON".to_owned())));
        assert!(state.is_valid(&NerAction::Last("PERSON".to_owned())));
        state.apply(&NerAction::Last("PERSON".to_owned())).unwrap();
        assert!(state.is_valid(&NerAction::Unit("ORG".to_owned())));
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
        tokens[0].ent_id = StringStore::id("jane-smith");
        tokens[1].ent_iob = 1;
        tokens[1].ent_type = person;
        tokens[1].ent_id = StringStore::id("jane-smith");
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
        let entity_ids = vec![(StringStore::id("jane-smith"), "jane-smith".to_owned())];
        let selected = collect_entities(&doc, &labels, &entity_ids, |entity_type| {
            entity_type == StringStore::id("PERSON")
        });
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].text, "Jane Smith");
        assert_eq!(selected[0].start_char, 0);
        assert_eq!(selected[0].end_char, 10);
        assert_eq!(selected[0].ent_id.as_deref(), Some("jane-smith"));
        assert!(collect_entities(&doc, &labels, &[], |_| false).is_empty());
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
