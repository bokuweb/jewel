"""Generate the supported spaCy EntityRuler compatibility fixture."""

from __future__ import annotations

import json
from typing import Any

import spacy
from spacy.tokens import Doc, Span


CASES = [
    {
        "phrase_matcher_attr": "LOWER",
        "overwrite_ents": False,
        "words": ["ACME", "CORP", "and", "Acme", "Corp"],
        "spaces": [True, True, True, True, False],
        "patterns": [
            {"label": "ORG", "pattern": "Acme"},
            {"label": "ORG", "pattern": "Acme Corp"},
        ],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "ORTH",
        "overwrite_ents": False,
        "words": ["Acme", "Corp", "Japan"],
        "spaces": [True, True, False],
        "patterns": [{"label": "ORG", "pattern": "Acme Corp"}],
        "initial_entities": [{"start": 1, "end": 3, "label": "GPE"}],
    },
    {
        "phrase_matcher_attr": "ORTH",
        "overwrite_ents": True,
        "words": ["Acme", "Corp", "Japan"],
        "spaces": [True, True, False],
        "patterns": [{"label": "ORG", "pattern": "Acme Corp"}],
        "initial_entities": [{"start": 1, "end": 3, "label": "GPE"}],
    },
    {
        "phrase_matcher_attr": "NORM",
        "overwrite_ents": False,
        "words": ["ACME", "paid", "the", "Late", "Fee"],
        "spaces": [True, True, True, True, False],
        "patterns": [{"label": "CONTRACT_TERM", "pattern": "late fee"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "SHAPE",
        "overwrite_ents": False,
        "words": ["Invoice", "AB-1234", "matches", "XY-9876"],
        "spaces": [True, True, True, False],
        "patterns": [{"label": "REFERENCE", "pattern": "ZZ-0000"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "LENGTH",
        "overwrite_ents": False,
        "words": ["契約", "条件", "apply"],
        "spaces": [True, True, False],
        "patterns": [{"label": "TWO_CHARACTERS", "pattern": "ab"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "TEXT",
        "overwrite_ents": False,
        "words": ["ACME", "Acme"],
        "spaces": [True, False],
        "patterns": [{"label": "EXACT", "pattern": "Acme"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "IS_ALPHA",
        "overwrite_ents": False,
        "words": ["契約", "123", "Acme"],
        "spaces": [True, True, False],
        "patterns": [{"label": "ALPHA", "pattern": "word"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "LIKE_EMAIL",
        "overwrite_ents": False,
        "words": ["contact", "legal@example.com", "today"],
        "spaces": [True, True, False],
        "patterns": [{"label": "EMAIL", "pattern": "user@example.com"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "IS_STOP",
        "overwrite_ents": False,
        "words": ["The", "contract", "and", "terms"],
        "spaces": [True, True, True, False],
        "patterns": [{"label": "STOP", "pattern": "the"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "IS_SENT_START",
        "overwrite_ents": False,
        "words": ["First", "second"],
        "spaces": [True, False],
        "patterns": [{"label": "START", "pattern": "marker"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "SPACY",
        "overwrite_ents": False,
        "words": ["has-space", "final"],
        "spaces": [True, False],
        "patterns": [{"label": "NO_TRAILING_SPACE", "pattern": "marker"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "ENT_IOB",
        "overwrite_ents": False,
        "words": ["unannotated", "tokens"],
        "spaces": [True, False],
        "patterns": [{"label": "MISSING_IOB", "pattern": "marker"}],
        "initial_entities": [],
    },
    {
        "phrase_matcher_attr": "ENT_TYPE",
        "overwrite_ents": False,
        "words": ["Acme", "met", "Jane"],
        "spaces": [True, True, False],
        "patterns": [{"label": "NO_ENTITY", "pattern": "plain"}],
        "initial_entities": [{"start": 0, "end": 1, "label": "ORG"}],
    },
]


def attribute_id(token: Any, attribute: str) -> int:
    canonical = {
        "TEXT": "ORTH",
        "IS_SENT_START": "SENT_START",
    }.get(attribute, attribute)
    return int(token.doc.to_array([canonical])[token.i])


def generate_fixture() -> dict[str, Any]:
    cases = []
    for source in CASES:
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe(
            "entity_ruler",
            config={
                "phrase_matcher_attr": source["phrase_matcher_attr"],
                "overwrite_ents": source["overwrite_ents"],
            },
        )
        ruler.add_patterns(source["patterns"])
        doc = Doc(
            nlp.vocab,
            words=source["words"],
            spaces=source["spaces"],
        )
        doc.ents = [
            Span(
                doc,
                entity["start"],
                entity["end"],
                label=entity["label"],
            )
            for entity in source["initial_entities"]
        ]
        attribute = source["phrase_matcher_attr"]
        token_ids = [attribute_id(token, attribute) for token in doc]
        ruler(doc)
        patterns = []
        for pattern in source["patterns"]:
            pattern_doc = nlp.make_doc(pattern["pattern"])
            patterns.append(
                {
                    "label": pattern["label"],
                    "token_ids": [
                        attribute_id(token, attribute) for token in pattern_doc
                    ],
                }
            )
        cases.append(
            {
                "phrase_matcher_attr": attribute,
                "overwrite_ents": source["overwrite_ents"],
                "words": source["words"],
                "spaces": source["spaces"],
                "token_ids": token_ids,
                "patterns": patterns,
                "initial_entities": source["initial_entities"],
                "entities": [
                    {
                        "start": entity.start,
                        "end": entity.end,
                        "label": entity.label_,
                    }
                    for entity in doc.ents
                ],
            }
        )
    return {"spacy_version": spacy.__version__, "cases": cases}


def main() -> None:
    print(json.dumps(generate_fixture(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
