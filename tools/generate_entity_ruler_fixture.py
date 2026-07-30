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
]


def attribute_id(token: Any, attribute: str) -> int:
    return int(token.doc.to_array([attribute])[token.i])


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
        ruler(doc)
        attribute = source["phrase_matcher_attr"]
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
                "token_ids": [
                    attribute_id(token, attribute) for token in doc
                ],
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
