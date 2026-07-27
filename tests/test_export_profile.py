from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from export_profile import select_ner_components


class NerExportProfileTests(unittest.TestCase):
    def test_selects_only_a_self_contained_ner_component(self) -> None:
        self.assertEqual(
            select_ner_components(["ner"], uses_tok2vec_listener=False),
            frozenset(("ner",)),
        )

    def test_retains_tok2vec_for_a_listener_without_a_parser(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["tok2vec", "ner"],
                uses_tok2vec_listener=True,
            ),
            frozenset(("tok2vec", "ner")),
        )

    def test_retains_sentencizer_for_parserless_ner(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["sentencizer", "tok2vec", "ner"],
                uses_tok2vec_listener=True,
                sentencizer_names=("sentencizer",),
            ),
            frozenset(("sentencizer", "tok2vec", "ner")),
        )

    def test_retains_a_custom_named_sentencizer(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["sentences", "ner"],
                uses_tok2vec_listener=False,
                sentencizer_names=("sentences",),
            ),
            frozenset(("sentences", "ner")),
        )

    def test_retains_the_complete_parser_stage(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["tok2vec", "tagger", "parser", "ner"],
                uses_tok2vec_listener=True,
            ),
            frozenset(("tok2vec", "parser", "ner")),
        )

    def test_parser_supersedes_sentencizer_in_the_runtime_profile(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["tok2vec", "sentencizer", "parser", "ner"],
                uses_tok2vec_listener=True,
                sentencizer_names=("sentencizer",),
            ),
            frozenset(("tok2vec", "parser", "ner")),
        )

    def test_rejects_multiple_sentencizers(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one sentencizer"):
            select_ner_components(
                ["sentences_a", "sentences_b", "ner"],
                uses_tok2vec_listener=False,
                sentencizer_names=("sentences_a", "sentences_b"),
            )

    def test_rejects_missing_required_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 'ner'"):
            select_ner_components(["tok2vec"], uses_tok2vec_listener=False)
        with self.assertRaisesRegex(ValueError, "upstream tok2vec"):
            select_ner_components(["ner"], uses_tok2vec_listener=True)


if __name__ == "__main__":
    unittest.main()
