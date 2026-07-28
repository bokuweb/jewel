use spacy_core::{Doc, StringStore};
use spacy_model::Bundle;
use thiserror::Error;

use crate::{Matrix, TransitionScorer, TransitionScorerError};

const ROOT_LABEL: &str = "ROOT";
const SUBTOKEN_LABEL: &str = "subtok";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParserAction {
    Shift,
    Reduce,
    Left(String),
    Right(String),
    Break(String),
}

impl ParserAction {
    /// Parse one ordered class name from spaCy's transition system.
    ///
    /// # Errors
    ///
    /// Returns an error if the class name does not identify an arc-eager move.
    pub fn parse(name: &str) -> Result<Self, DependencyParserError> {
        if name == "S" {
            return Ok(Self::Shift);
        }
        if name == "D" {
            return Ok(Self::Reduce);
        }
        let (prefix, label) = name.split_once('-').ok_or_else(|| {
            DependencyParserError::InvalidMove(format!("invalid move name {name:?}"))
        })?;
        match prefix {
            "L" => Ok(Self::Left(label.to_owned())),
            "R" => Ok(Self::Right(label.to_owned())),
            "B" => Ok(Self::Break(label.to_owned())),
            _ => Err(DependencyParserError::InvalidMove(format!(
                "unsupported move prefix {prefix:?}"
            ))),
        }
    }
}

#[derive(Debug, Error)]
pub enum DependencyParserError {
    #[error(transparent)]
    Scorer(#[from] TransitionScorerError),
    #[error("dependency parser component {0:?} is missing")]
    MissingComponent(String),
    #[error("dependency parser model is invalid: {0}")]
    InvalidModel(String),
    #[error("invalid dependency parser move: {0}")]
    InvalidMove(String),
    #[error("no valid parser transition at step {step}")]
    NoValidMove { step: usize },
    #[error("parser exceeded its {limit}-step safety limit")]
    StepLimit { limit: usize },
    #[error("token index {index} cannot be represented as a relative i32 head")]
    HeadOverflow { index: usize },
}

/// Mutable arc-eager transition state matching spaCy's eight parser features.
#[derive(Clone, Debug)]
pub struct ArcEagerState {
    length: usize,
    buffer_index: usize,
    stack: Vec<usize>,
    rebuffer: Vec<usize>,
    heads: Vec<Option<usize>>,
    labels: Vec<Option<String>>,
    left_arcs: Vec<Vec<usize>>,
    right_arcs: Vec<Vec<usize>>,
    unshiftable: Vec<bool>,
    sent_starts: Vec<bool>,
    cannot_sent_start: Vec<bool>,
}

impl ArcEagerState {
    #[must_use]
    pub fn new(doc: &Doc) -> Self {
        let length = doc.len();
        let mut sent_starts = doc
            .tokens()
            .iter()
            .map(|token| token.sent_start == 1)
            .collect::<Vec<_>>();
        if let Some(first) = sent_starts.first_mut() {
            *first = true;
        }
        Self {
            length,
            buffer_index: 0,
            stack: Vec::new(),
            rebuffer: Vec::new(),
            heads: vec![None; length],
            labels: vec![None; length],
            left_arcs: vec![Vec::new(); length],
            right_arcs: vec![Vec::new(); length],
            unshiftable: vec![false; length],
            sent_starts,
            cannot_sent_start: doc
                .tokens()
                .iter()
                .map(|token| token.sent_start == -1)
                .collect(),
        }
    }

    #[must_use]
    pub fn is_final(&self) -> bool {
        self.stack.is_empty() && self.buffer_length() == 0
    }

    #[must_use]
    pub fn features(&self) -> [i32; 8] {
        [
            self.b(0),
            self.b(1),
            self.s(0),
            self.s(1),
            self.s(2),
            self.left(self.b(0), 1),
            self.left(self.s(0), 1),
            self.right(self.s(0), 1),
        ]
    }

    #[must_use]
    pub fn is_valid(&self, action: &ParserAction) -> bool {
        match action {
            ParserAction::Shift => {
                if self.stack.is_empty() {
                    self.buffer_length() > 0
                } else {
                    self.buffer_length() >= 2
                        && !self.is_sent_start(self.b(0))
                        && !self.is_unshiftable(self.b(0))
                }
            }
            ParserAction::Reduce => {
                if self.stack.is_empty() {
                    false
                } else if self.buffer_length() == 0 {
                    true
                } else {
                    self.stack.len() != 1 || !self.cannot_start(self.b(0))
                }
            }
            ParserAction::Left(label) | ParserAction::Right(label) => {
                !self.stack.is_empty()
                    && self.buffer_length() > 0
                    && !self.is_sent_start(self.b(0))
                    && (label != SUBTOKEN_LABEL || self.s(0) == self.b(0) - 1)
            }
            ParserAction::Break(_) => {
                self.buffer_length() >= 2
                    && self.b(1) == self.b(0) + 1
                    && !self.is_sent_start(self.b(1))
                    && !self.cannot_start(self.b(1))
            }
        }
    }

    /// Apply one valid transition.
    ///
    /// # Errors
    ///
    /// Returns an error if the transition is invalid in the current state.
    pub fn apply(&mut self, action: &ParserAction) -> Result<(), DependencyParserError> {
        if !self.is_valid(action) {
            return Err(DependencyParserError::InvalidMove(format!(
                "move {action:?} is not valid for features {:?}",
                self.features()
            )));
        }
        match action {
            ParserAction::Shift => self.push(),
            ParserAction::Reduce => {
                let Some(top) = self.stack.last().copied() else {
                    return Err(DependencyParserError::InvalidMove(
                        "reduce requires a non-empty stack".to_owned(),
                    ));
                };
                if self.heads[top].is_some() || self.stack.len() == 1 {
                    self.stack.pop();
                } else {
                    self.unshift();
                }
            }
            ParserAction::Left(label) => {
                let head = usize::try_from(self.b(0)).map_err(|_| {
                    DependencyParserError::InvalidMove(
                        "left arc requires a non-empty buffer".to_owned(),
                    )
                })?;
                let child = usize::try_from(self.s(0)).map_err(|_| {
                    DependencyParserError::InvalidMove(
                        "left arc requires a non-empty stack".to_owned(),
                    )
                })?;
                self.add_arc(head, child, label);
                self.unshiftable[head] = false;
                self.stack.pop();
            }
            ParserAction::Right(label) => {
                let head = usize::try_from(self.s(0)).map_err(|_| {
                    DependencyParserError::InvalidMove(
                        "right arc requires a non-empty stack".to_owned(),
                    )
                })?;
                let child = usize::try_from(self.b(0)).map_err(|_| {
                    DependencyParserError::InvalidMove(
                        "right arc requires a non-empty buffer".to_owned(),
                    )
                })?;
                self.add_arc(head, child, label);
                self.push();
            }
            ParserAction::Break(_) => {
                let next = usize::try_from(self.b(1)).map_err(|_| {
                    DependencyParserError::InvalidMove(
                        "break requires two buffered tokens".to_owned(),
                    )
                })?;
                self.sent_starts[next] = true;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn heads(&self) -> &[Option<usize>] {
        &self.heads
    }

    #[must_use]
    pub fn labels(&self) -> &[Option<String>] {
        &self.labels
    }

    fn buffer_length(&self) -> usize {
        (self.length - self.buffer_index) + self.rebuffer.len()
    }

    fn s(&self, depth: usize) -> i32 {
        self.stack
            .len()
            .checked_sub(depth + 1)
            .and_then(|index| self.stack.get(index))
            .and_then(|index| i32::try_from(*index).ok())
            .unwrap_or(-1)
    }

    fn b(&self, depth: usize) -> i32 {
        if depth < self.rebuffer.len() {
            return self
                .rebuffer
                .get(self.rebuffer.len() - depth - 1)
                .and_then(|index| i32::try_from(*index).ok())
                .unwrap_or(-1);
        }
        let index = self.buffer_index + depth - self.rebuffer.len();
        if index >= self.length {
            -1
        } else {
            i32::try_from(index).unwrap_or(-1)
        }
    }

    fn left(&self, head: i32, child_index: usize) -> i32 {
        Self::nth_child(&self.left_arcs, head, child_index)
    }

    fn right(&self, head: i32, child_index: usize) -> i32 {
        Self::nth_child(&self.right_arcs, head, child_index)
    }

    fn nth_child(arcs: &[Vec<usize>], head: i32, child_index: usize) -> i32 {
        if head < 0 || child_index == 0 {
            return -1;
        }
        arcs.get(usize::try_from(head).unwrap_or(usize::MAX))
            .and_then(|children| children.iter().rev().nth(child_index - 1))
            .and_then(|child| i32::try_from(*child).ok())
            .unwrap_or(-1)
    }

    fn is_sent_start(&self, token: i32) -> bool {
        usize::try_from(token)
            .ok()
            .and_then(|token| self.sent_starts.get(token))
            .copied()
            .unwrap_or(false)
    }

    fn cannot_start(&self, token: i32) -> bool {
        usize::try_from(token)
            .ok()
            .and_then(|token| self.cannot_sent_start.get(token))
            .copied()
            .unwrap_or(false)
    }

    fn is_unshiftable(&self, token: i32) -> bool {
        usize::try_from(token)
            .ok()
            .and_then(|token| self.unshiftable.get(token))
            .copied()
            .unwrap_or(false)
    }

    fn push(&mut self) {
        if let Some(token) = self.rebuffer.pop() {
            self.stack.push(token);
        } else {
            self.stack.push(self.buffer_index);
            self.buffer_index += 1;
        }
    }

    fn unshift(&mut self) {
        let token = self.stack.pop().expect("validated stack");
        self.unshiftable[token] = true;
        self.rebuffer.push(token);
    }

    fn add_arc(&mut self, head: usize, child: usize, label: &str) {
        if let Some(old_head) = self.heads[child] {
            let old_arcs = if old_head > child {
                &mut self.left_arcs[old_head]
            } else {
                &mut self.right_arcs[old_head]
            };
            if let Some(index) = old_arcs.iter().position(|item| *item == child) {
                old_arcs.remove(index);
            }
        }
        let arcs = if head > child {
            &mut self.left_arcs[head]
        } else {
            &mut self.right_arcs[head]
        };
        arcs.push(child);
        self.heads[child] = Some(head);
        self.labels[child] = Some(label.to_owned());
    }
}

/// Greedy dependency parser using spaCy's exported arc-eager classes.
pub struct DependencyParser {
    scorer: TransitionScorer,
    actions: Vec<ParserAction>,
}

impl DependencyParser {
    /// Load the parser scorer and its exact ordered transition classes.
    ///
    /// # Errors
    ///
    /// Returns an error when the component has no transition metadata or its
    /// output dimension does not match the move count.
    pub fn load(bundle: &Bundle, component_name: &str) -> Result<Self, DependencyParserError> {
        let component = bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.name == component_name)
            .ok_or_else(|| DependencyParserError::MissingComponent(component_name.to_owned()))?;
        let scorer = TransitionScorer::load(bundle, component_name)?;
        if component.moves.len() != scorer.class_count() {
            return Err(DependencyParserError::InvalidModel(format!(
                "model has {} score classes but {} exported moves",
                scorer.class_count(),
                component.moves.len()
            )));
        }
        let actions = component
            .moves
            .iter()
            .map(|name| ParserAction::parse(name))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { scorer, actions })
    }

    /// Greedily decode and attach `HEAD` and `DEP` annotations.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible token vectors, an invalid transition
    /// state, or a head index that cannot fit spaCy's relative representation.
    pub fn annotate(
        &self,
        doc: &mut Doc,
        tok2vec: &Matrix,
    ) -> Result<Vec<usize>, DependencyParserError> {
        if doc.len() != tok2vec.rows() {
            return Err(DependencyParserError::InvalidModel(format!(
                "document has {} tokens but tok2vec has {} rows",
                doc.len(),
                tok2vec.rows()
            )));
        }
        let cache = self.scorer.precompute(tok2vec)?;
        let mut state = ArcEagerState::new(doc);
        let mut history = Vec::new();
        let limit = doc.len().saturating_mul(6).saturating_add(8);
        while !state.is_final() {
            if history.len() >= limit {
                return Err(DependencyParserError::StepLimit { limit });
            }
            let scores = self.scorer.score(&cache, &state.features())?;
            let mut best = None;
            for (index, action) in self.actions.iter().enumerate() {
                if state.is_valid(action)
                    && best.is_none_or(|previous| scores.row(0)[index] > scores.row(0)[previous])
                {
                    best = Some(index);
                }
            }
            let action_index = best.ok_or(DependencyParserError::NoValidMove {
                step: history.len(),
            })?;
            state.apply(&self.actions[action_index])?;
            history.push(action_index);
        }
        deprojectivize(
            &mut state.heads,
            &mut state.labels,
            doc.tokens()
                .iter()
                .map(|token| token.text.chars().all(char::is_whitespace))
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let sentence_starts = sentence_starts_from_heads(&state.heads);
        for (index, token) in doc.tokens_mut().iter_mut().enumerate() {
            // spaCy does not copy the parser transition state's BREAK markers
            // into the Doc. `set_children_from_heads` derives sentence starts
            // from the left edge of every dependency root instead.
            token.sent_start = if sentence_starts[index] { 1 } else { -1 };
            if let Some(head) = state.heads[index] {
                let head = i32::try_from(head)
                    .map_err(|_| DependencyParserError::HeadOverflow { index: head })?;
                let child = i32::try_from(index)
                    .map_err(|_| DependencyParserError::HeadOverflow { index })?;
                token.head = head - child;
                let label = state.labels[index]
                    .as_deref()
                    .unwrap_or(ROOT_LABEL)
                    .split("||")
                    .next()
                    .unwrap_or(ROOT_LABEL);
                token.dep = StringStore::id(label);
            } else {
                token.head = 0;
                token.dep = StringStore::id(ROOT_LABEL);
            }
        }
        Ok(history)
    }

    #[must_use]
    pub fn actions(&self) -> &[ParserAction] {
        &self.actions
    }
}

fn deprojectivize(heads: &mut [Option<usize>], labels: &mut [Option<String>], is_space: &[bool]) {
    for token in 0..heads.len() {
        let Some(label) = labels[token].clone() else {
            continue;
        };
        let Some((plain_label, head_label)) = label.split_once("||") else {
            continue;
        };
        if let Some(head) = heads[token] {
            if let Some(new_head) = find_new_head(token, head, head_label, heads, labels, is_space)
            {
                heads[token] = Some(new_head);
            }
        }
        labels[token] = Some(plain_label.to_owned());
    }
}

fn sentence_starts_from_heads(heads: &[Option<usize>]) -> Vec<bool> {
    let mut left_edges = (0..heads.len()).collect::<Vec<_>>();
    for token in 0..heads.len() {
        let mut root = token;
        for _ in 0..heads.len() {
            let Some(head) = heads[root] else {
                break;
            };
            if head == root || head >= heads.len() {
                break;
            }
            root = head;
        }
        left_edges[root] = left_edges[root].min(token);
    }

    let mut starts = vec![false; heads.len()];
    for (root, head) in heads.iter().enumerate() {
        if head.is_none() || *head == Some(root) {
            starts[left_edges[root]] = true;
        }
    }
    if !starts.is_empty() && !starts.iter().any(|is_start| *is_start) {
        starts[0] = true;
    }
    starts
}

fn find_new_head(
    token: usize,
    head: usize,
    head_label: &str,
    heads: &[Option<usize>],
    labels: &[Option<String>],
    is_space: &[bool],
) -> Option<usize> {
    let mut queue = vec![head];
    while !queue.is_empty() {
        let mut next = Vec::new();
        for parent in queue {
            for child in 0..heads.len() {
                if child == parent
                    || heads[child] != Some(parent)
                    || is_space.get(child).copied().unwrap_or(false)
                    || child == token
                {
                    continue;
                }
                if labels[child].as_deref() == Some(head_label) {
                    return Some(child);
                }
                next.push(child);
            }
        }
        queue = next;
    }
    None
}

#[cfg(test)]
mod tests {
    use spacy_core::Doc;

    use super::{deprojectivize, sentence_starts_from_heads, ArcEagerState, ParserAction};

    #[test]
    fn state_features_follow_arc_eager_stack_and_children() {
        let doc = Doc::from_words(&["A", "test", "works"], &[true, true, false]).unwrap();
        let mut state = ArcEagerState::new(&doc);
        assert_eq!(state.features(), [0, 1, -1, -1, -1, -1, -1, -1]);
        state.apply(&ParserAction::Shift).unwrap();
        state.apply(&ParserAction::Right("dep".to_owned())).unwrap();
        assert_eq!(state.features(), [2, -1, 1, 0, -1, -1, -1, -1]);
        state.apply(&ParserAction::Left("dep".to_owned())).unwrap();
        assert_eq!(state.features(), [2, -1, 0, -1, -1, 1, -1, -1]);
    }

    #[test]
    fn decorated_arc_is_reattached_and_undecorated() {
        let mut heads = [
            Some(1),
            Some(2),
            Some(2),
            Some(4),
            Some(5),
            Some(2),
            Some(7),
            Some(5),
            Some(2),
        ];
        let mut labels = [
            "det",
            "nsubj",
            "root",
            "det",
            "dobj",
            "aux",
            "nsubj",
            "acl||dobj",
            "punct",
        ]
        .map(|label| Some(label.to_owned()));
        deprojectivize(&mut heads, &mut labels, &[false; 9]);
        assert_eq!(
            heads,
            [
                Some(1),
                Some(2),
                Some(2),
                Some(4),
                Some(5),
                Some(2),
                Some(7),
                Some(4),
                Some(2)
            ]
        );
        assert_eq!(labels[7].as_deref(), Some("acl"));
    }

    #[test]
    fn sentence_starts_are_left_edges_of_dependency_roots() {
        let heads = [Some(2), Some(2), None, Some(4), None, None, Some(5)];
        assert_eq!(
            sentence_starts_from_heads(&heads),
            [true, false, false, true, false, true, false]
        );
    }

    #[test]
    fn sentence_starts_support_self_headed_roots() {
        let heads = [Some(1), Some(1), Some(3), Some(3)];
        assert_eq!(
            sentence_starts_from_heads(&heads),
            [true, false, true, false]
        );
    }
}
