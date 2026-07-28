"""Generate the supported spaCy EntityRuler token-pattern fixture."""

from __future__ import annotations

import json
from typing import Any

import spacy
from spacy.tokens import Doc, Span

from export_spacy_model import normalize_entity_ruler_token_pattern


CASES = [
    {
        "words": ["Penalty", "is", "$", "1,000", "yen", "or", "2500", "円", "."],
        "spaces": [True, True, False, True, True, True, True, False, False],
        "patterns": [
            {
                "label": "MONEY",
                "pattern": [
                    {"IS_CURRENCY": True, "OP": "?"},
                    {"LIKE_NUM": True},
                    {"LOWER": {"IN": ["yen", "円"]}, "OP": "?"},
                ],
            }
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["Email", "legal@example.com", "or", "call", "03-1234-5678"],
        "spaces": [True, True, True, True, False],
        "patterns": [
            {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},
            {
                "label": "PHONE",
                "pattern": [
                    {"TEXT": {"REGEX": r"^\d{2,4}-\d{2,4}-\d{4}$"}}
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["東京都", "千代田区", "1", "2", "3", "丁目"],
        "spaces": [False, True, True, True, False, False],
        "patterns": [
            {
                "label": "ADDRESS",
                "pattern": [
                    {"LOWER": {"NOT_IN": ["東京都", "大阪府"]}},
                    {"IS_DIGIT": True, "OP": "+"},
                    {"TEXT": "丁目"},
                ],
            }
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["one", "21st", "1,000/2", "１", "²", "½", "一"],
        "spaces": [True, True, True, True, True, True, False],
        "patterns": [
            {"label": "NUMBER", "pattern": [{"LIKE_NUM": True}]},
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["Acme", ",", "-", "Corp", "and", "Test", "Corp"],
        "spaces": [False, True, True, True, True, True, False],
        "patterns": [
            {
                "label": "ORG",
                "pattern": [
                    {
                        "LOWER": {"NOT_IN": ["test"]},
                        "IS_ALPHA": True,
                        "IS_ASCII": True,
                    },
                    {"IS_PUNCT": True, "OP": "*"},
                    {"LOWER": "corp"},
                ],
            }
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["A", " ", "Late", "Fee"],
        "spaces": [False, False, True, False],
        "norms": ["a", " ", "late", "fee"],
        "patterns": [
            {"label": "SPACE", "pattern": [{"IS_SPACE": True}]},
            {
                "label": "TERM",
                "pattern": [{"NORM": "late"}, {"NORM": "fee"}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["Existing", "Acme", "Corp"],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "ORG",
                "pattern": [{"LOWER": "acme"}, {"LOWER": "corp"}],
            }
        ],
        "overwrite_ents": True,
        "initial_entities": [{"start": 1, "end": 3, "label": "PRODUCT"}],
    },
]


def generate_fixture() -> dict[str, Any]:
    cases = []
    for source in CASES:
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe(
            "entity_ruler",
            config={"overwrite_ents": source["overwrite_ents"]},
        )
        ruler.add_patterns(source["patterns"])
        doc = Doc(nlp.vocab, words=source["words"], spaces=source["spaces"])
        for token, norm in zip(doc, source.get("norms", [])):
            token.norm_ = norm
        doc.ents = [
            Span(doc, entity["start"], entity["end"], label=entity["label"])
            for entity in source["initial_entities"]
        ]
        ruler(doc)
        case = {
                "words": source["words"],
                "spaces": source["spaces"],
                "overwrite_ents": source["overwrite_ents"],
                "patterns": [
                    {
                        "label": pattern["label"],
                        "tokens": normalize_entity_ruler_token_pattern(
                            pattern["pattern"],
                            pattern=index,
                        ),
                    }
                    for index, pattern in enumerate(source["patterns"])
                ],
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
        if "norms" in source:
            case["norm_ids"] = [int(token.norm) for token in doc]
        cases.append(case)
    return {"spacy_version": spacy.__version__, "cases": cases}


def main() -> None:
    print(json.dumps(generate_fixture(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
