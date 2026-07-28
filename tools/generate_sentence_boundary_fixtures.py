"""Generate spaCy sentence-boundary compatibility fixtures as JSON."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy
import spacy
from spacy.tokens import Doc


def encoded_sent_starts(doc: Doc) -> list[int]:
    return [
        0 if token.is_sent_start is None else 1 if token.is_sent_start else -1
        for token in doc
    ]


def make_doc(
    nlp: Any,
    words: list[str],
    spaces: list[bool],
    initial: list[int],
) -> Doc:
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    for token, value in zip(doc, initial, strict=True):
        token.is_sent_start = None if value == 0 else value == 1
    return doc


def sentencizer_fixture() -> dict[str, Any]:
    cases = [
        {
            "words": ["Alice", ".", ")", "Bob", "?", "Carol"],
            "spaces": [False, False, True, False, True, False],
            "punct_chars": [".", "?"],
            "overwrite": False,
            "initial": [0, 0, 0, 0, 0, 0],
        },
        {
            "words": ["甲", "。", "乙", "END", "丙"],
            "spaces": [False] * 5,
            "punct_chars": ["。", "END"],
            "overwrite": False,
            "initial": [0, 0, 0, 0, 0],
        },
        {
            "words": ["Alice", "‼", "”", "Bob"],
            "spaces": [False, False, True, False],
            "punct_chars": ["‼"],
            "overwrite": False,
            "initial": [0, 0, 0, 0],
        },
        {
            "words": ["Alice", ".", "Bob"],
            "spaces": [False, True, False],
            "punct_chars": ["."],
            "overwrite": False,
            "initial": [0, 1, 0],
        },
        {
            "words": ["Alice", ".", "Bob"],
            "spaces": [False, True, False],
            "punct_chars": ["."],
            "overwrite": True,
            "initial": [0, 1, 0],
        },
    ]
    for case in cases:
        nlp = spacy.blank("en")
        sentencizer = nlp.add_pipe(
            "sentencizer",
            config={
                "punct_chars": case["punct_chars"],
                "overwrite": case["overwrite"],
            },
        )
        doc = make_doc(nlp, case["words"], case["spaces"], case["initial"])
        sentencizer(doc)
        case["sent_starts"] = encoded_sent_starts(doc)
    return {"spacy_version": spacy.__version__, "cases": cases}


def senter_fixture() -> dict[str, Any]:
    cases = [
        {
            "words": ["Alice", "works", ".", "Bob", "signs", "."],
            "spaces": [True, False, True, True, False, False],
            "classes": [0, 0, 0, 1, 0, 0],
            "overwrite": False,
            "initial": [0, 0, 0, 0, 0, 0],
        },
        {
            "words": ["Alice", "works", ".", "Bob", "signs"],
            "spaces": [True, False, True, True, False],
            "classes": [0, 1, 1, 0, 1],
            "overwrite": False,
            "initial": [1, 0, 0, 1, 0],
        },
        {
            "words": ["Alice", "works", ".", "Bob", "signs"],
            "spaces": [True, False, True, True, False],
            "classes": [0, 1, 1, 0, 1],
            "overwrite": True,
            "initial": [1, 0, 0, 1, 0],
        },
        {
            "words": [],
            "spaces": [],
            "classes": [],
            "overwrite": False,
            "initial": [],
        },
    ]
    for case in cases:
        nlp = spacy.blank("en")
        senter = nlp.add_pipe(
            "senter",
            config={"overwrite": case["overwrite"]},
        )
        doc = make_doc(nlp, case["words"], case["spaces"], case["initial"])
        classes = numpy.asarray(case["classes"], dtype=numpy.int32)
        senter.set_annotations([doc], [classes])
        case["sent_starts"] = encoded_sent_starts(doc)
    return {"spacy_version": spacy.__version__, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "component",
        choices=("sentencizer", "senter"),
        help="sentence-boundary component to exercise",
    )
    args = parser.parse_args()
    fixture = (
        sentencizer_fixture()
        if args.component == "sentencizer"
        else senter_fixture()
    )
    print(json.dumps(fixture, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
