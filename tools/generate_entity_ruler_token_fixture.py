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
        "words": [
            "Visit",
            "https://example.com",
            "or",
            "example.jp",
            "8.8.8.8",
            "192.168.1.1",
            "ACME",
            "Alice",
            "lower",
        ],
        "spaces": [True, True, True, True, True, True, True, True, False],
        "patterns": [
            {"label": "URL", "pattern": [{"LIKE_URL": True}]},
            {
                "label": "UPPER",
                "pattern": [{"IS_UPPER": True, "IS_ASCII": True}],
            },
            {
                "label": "TITLE",
                "pattern": [
                    {
                        "IS_TITLE": True,
                        "LOWER": {"NOT_IN": ["visit"]},
                    }
                ],
            },
            {
                "label": "LOWER",
                "pattern": [
                    {
                        "IS_LOWER": True,
                        "LOWER": {"IN": ["lower"]},
                    }
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": [
            "〒",
            "123-4567",
            "account",
            "AB123456",
            "prod",
            "code",
            "test",
            "code",
        ],
        "spaces": [False, True, True, True, True, True, True, False],
        "patterns": [
            {
                "label": "POSTAL_CODE",
                "pattern": [
                    {
                        "SHAPE": "ddd-dddd",
                        "PREFIX": "1",
                        "SUFFIX": "567",
                        "LENGTH": {">=": 8},
                    }
                ],
            },
            {
                "label": "ACCOUNT_ID",
                "pattern": [
                    {
                        "SHAPE": "XXdddd",
                        "PREFIX": {"IN": ["A", "B"]},
                        "SUFFIX": "456",
                        "LENGTH": 8,
                    }
                ],
            },
            {
                "label": "NON_TEST_CODE",
                "pattern": [
                    {"LOWER": "test", "OP": "!"},
                    {"LOWER": "code"},
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["ACME", "ACNE", "AXME", "AXXE", "Other"],
        "spaces": [True, True, True, True, False],
        "patterns": [
            {
                "label": "PARTY",
                "pattern": [{"LOWER": {"FUZZY1": "acme"}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["Acme", "Acne", "Axxe", "Other"],
        "spaces": [True, True, True, False],
        "patterns": [
            {
                "label": "DEFAULT_FUZZY",
                "pattern": [{"TEXT": {"FUZZY": "Acme"}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["株式会社", "株式会杜", "有限会社"],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "COMPANY_SUFFIX",
                "pattern": [{"TEXT": {"FUZZY1": "株式会社"}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["AB123456", "A123456", "plain"],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "STRUCTURED_ID",
                "pattern": [{"SHAPE": {"FUZZY1": "XXdddd"}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["ACNE", "GLOBEX", "GLOBEK", "OTHER"],
        "spaces": [True, True, True, False],
        "patterns": [
            {
                "label": "KNOWN_PARTY",
                "pattern": [
                    {
                        "LOWER": {
                            "FUZZY1": {"IN": ["acme", "globex"]}
                        }
                    }
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["prod", "test", "tast", "live"],
        "spaces": [True, True, True, False],
        "patterns": [
            {
                "label": "NON_TEST",
                "pattern": [{"LOWER": {"FUZZY1": {"NOT_IN": ["test"]}}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["INV-123", "PO-999", "NOTE-1"],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "DOCUMENT_ID",
                "pattern": [
                    {
                        "TEXT": {
                            "REGEX": {
                                "IN": [r"^INV-\d+$", r"^PO-\d+$"]
                            }
                        }
                    }
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": [
            "public@example.com",
            "noreply@example.com",
            "admin@test.invalid",
        ],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "CONTACT_EMAIL",
                "pattern": [
                    {
                        "LIKE_EMAIL": True,
                        "TEXT": {
                            "REGEX": {
                                "NOT_IN": [r"^noreply@", r"\.invalid$"]
                            }
                        },
                    }
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["AB123456", "A123456", "plain"],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "STRUCTURED_ID_SET",
                "pattern": [
                    {
                        "SHAPE": {
                            "REGEX": {
                                "IN": [r"^XXdddd+$", r"^Xdddd+$"]
                            }
                        }
                    }
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["Acme", "Corp", "Ltd", "signed"],
        "spaces": [True, True, True, False],
        "patterns": [
            {
                "label": "PARTY",
                "pattern": [
                    {"ENT_TYPE": "ORG", "ENT_IOB": "B"},
                    {"ENT_TYPE": "ORG", "ENT_IOB": "I", "OP": "*"},
                    {"LOWER": {"IN": ["ltd", "inc"]}},
                ],
            },
        ],
        "overwrite_ents": True,
        "initial_entities": [{"start": 0, "end": 2, "label": "ORG"}],
    },
    {
        "words": ["Acme", "Corp", "met", "Jane", "Doe"],
        "spaces": [True, True, True, True, False],
        "patterns": [
            {
                "label": "KNOWN_ENTITY",
                "pattern": [
                    {
                        "ENT_TYPE": {"IN": ["ORG", "PERSON"]},
                        "ENT_IOB": "B",
                    },
                    {
                        "ENT_TYPE": {"IN": ["ORG", "PERSON"]},
                        "ENT_IOB": "I",
                        "OP": "*",
                    },
                ],
            },
        ],
        "overwrite_ents": True,
        "initial_entities": [
            {"start": 0, "end": 2, "label": "ORG"},
            {"start": 3, "end": 5, "label": "PERSON"},
        ],
    },
    {
        "words": ["Acme", "Corp", "met", "Jane", "Doe"],
        "spaces": [True, True, True, True, False],
        "patterns": [
            {
                "label": "NON_ORG_ENTITY",
                "pattern": [
                    {
                        "ENT_TYPE": {"NOT_IN": ["ORG"]},
                        "ENT_IOB": "B",
                    },
                    {
                        "ENT_TYPE": {"NOT_IN": ["ORG"]},
                        "ENT_IOB": {"NOT_IN": ["B", "O"]},
                        "OP": "*",
                    },
                ],
            },
        ],
        "overwrite_ents": True,
        "initial_entities": [
            {"start": 0, "end": 2, "label": "ORG"},
            {"start": 3, "end": 5, "label": "PERSON"},
        ],
    },
    {
        "words": ["signed", "by", "Jane", "Doe"],
        "spaces": [True, True, True, False],
        "patterns": [
            {
                "label": "SIGNATURE_CONTEXT",
                "pattern": [
                    {"LOWER": "signed"},
                    {"OP": "?"},
                    {"ENT_TYPE": "PERSON", "OP": "+"},
                ],
            },
        ],
        "overwrite_ents": True,
        "initial_entities": [{"start": 2, "end": 4, "label": "PERSON"}],
    },
    {
        "words": ["contact", "at", "legal@example.com"],
        "spaces": [True, True, False],
        "patterns": [
            {
                "label": "CONTACT",
                "pattern": [
                    {"LOWER": "contact"},
                    {},
                    {"LIKE_EMAIL": True},
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["12", "34", "Main", "and", "12", "Main"],
        "spaces": [True, True, True, True, True, False],
        "patterns": [
            {
                "label": "EXACT_ADDRESS",
                "pattern": [
                    {"IS_DIGIT": True, "OP": "{2}"},
                    {"LOWER": "main"},
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["A", "B", "C", "end", "A", "B", "C", "D", "end"],
        "spaces": [True, True, True, True, True, True, True, True, False],
        "patterns": [
            {
                "label": "BOUNDED_WORDS",
                "pattern": [
                    {
                        "IS_ALPHA": True,
                        "LOWER": {"NOT_IN": ["end"]},
                        "OP": "{1,3}",
                    },
                    {"LOWER": "end"},
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["1", "2", "丁目", "and", "1", "2", "3", "4", "丁目"],
        "spaces": [True, False, True, True, True, True, True, False, False],
        "patterns": [
            {
                "label": "OPEN_ADDRESS",
                "pattern": [
                    {"IS_DIGIT": True, "OP": "{2,}"},
                    {"TEXT": "丁目"},
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["丁目", "1", "丁目", "1", "2", "丁目"],
        "spaces": [True, False, True, True, False, False],
        "patterns": [
            {
                "label": "OPTIONAL_ADDRESS",
                "pattern": [
                    {"IS_DIGIT": True, "OP": "{,2}"},
                    {"TEXT": "丁目"},
                ],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["sing", "past", "both", "none"],
        "spaces": [True, True, True, False],
        "morphs": [
            "Number=Sing",
            "Tense=Past",
            "Number=Sing|Tense=Past",
            "",
        ],
        "patterns": [
            {
                "label": "MORPH_SUBSET",
                "pattern": [{"MORPH": {"IS_SUBSET": ["Number=Sing"]}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["sing", "past", "both", "none"],
        "spaces": [True, True, True, False],
        "morphs": [
            "Number=Sing",
            "Tense=Past",
            "Number=Sing|Tense=Past",
            "",
        ],
        "patterns": [
            {
                "label": "MORPH_SUPERSET",
                "pattern": [{"MORPH": {"IS_SUPERSET": ["Number=Sing"]}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["sing", "past", "both", "none"],
        "spaces": [True, True, True, False],
        "morphs": [
            "Number=Sing",
            "Tense=Past",
            "Number=Sing|Tense=Past",
            "",
        ],
        "patterns": [
            {
                "label": "MORPH_INTERSECTS",
                "pattern": [{"MORPH": {"INTERSECTS": ["Tense=Past"]}}],
            },
        ],
        "overwrite_ents": False,
        "initial_entities": [],
    },
    {
        "words": ["a", "ab", "abc", "abcd", "abcde"],
        "spaces": [True, True, True, True, False],
        "patterns": [
            {
                "label": "SELECTED_LENGTH",
                "pattern": [{"LENGTH": {"IN": [2, 4]}}],
            },
            {
                "label": "OTHER_LENGTH",
                "pattern": [{"LENGTH": {"NOT_IN": [2, 4]}}],
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
        for token, morph in zip(doc, source.get("morphs", [])):
            token.set_morph(morph)
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
        if "morphs" in source:
            case["morphs"] = source["morphs"]
        cases.append(case)
    return {"spacy_version": spacy.__version__, "cases": cases}


def main() -> None:
    print(json.dumps(generate_fixture(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
