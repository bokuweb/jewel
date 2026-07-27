from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from export_profile import select_ner_components


class NerExportProfileTests(unittest.TestCase):
    def test_selects_only_a_self_contained_ner_component(self) -> None:
        self.assertEqual(
            select_ner_components(["ner"]),
            frozenset(("ner",)),
        )

    def test_retains_tok2vec_for_a_listener_without_a_parser(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["tok2vec", "ner"],
                tok2vec_upstreams={"ner": "tok2vec"},
            ),
            frozenset(("tok2vec", "ner")),
        )

    def test_retains_sentencizer_for_parserless_ner(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["sentencizer", "tok2vec", "ner"],
                sentencizer_names=("sentencizer",),
                tok2vec_upstreams={"ner": "tok2vec"},
            ),
            frozenset(("sentencizer", "tok2vec", "ner")),
        )

    def test_retains_a_custom_named_sentencizer(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["sentences", "ner"],
                sentencizer_names=("sentences",),
            ),
            frozenset(("sentences", "ner")),
        )

    def test_retains_custom_named_ner_and_tok2vec_components(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["encoder", "entities"],
                ner_names=("entities",),
                tok2vec_upstreams={"entities": "encoder"},
            ),
            frozenset(("encoder", "entities")),
        )

    def test_retains_the_complete_parser_stage(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["tok2vec", "tagger", "parser", "ner"],
                parser_names=("parser",),
                tok2vec_upstreams={
                    "parser": "tok2vec",
                    "ner": "tok2vec",
                },
            ),
            frozenset(("tok2vec", "parser", "ner")),
        )

    def test_parser_supersedes_sentencizer_in_the_runtime_profile(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["tok2vec", "sentencizer", "parser", "ner"],
                parser_names=("parser",),
                sentencizer_names=("sentencizer",),
                tok2vec_upstreams={
                    "parser": "tok2vec",
                    "ner": "tok2vec",
                },
            ),
            frozenset(("tok2vec", "parser", "ner")),
        )

    def test_retains_trainable_sentence_recognizer(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["sentence_model", "ner"],
                senter_names=("sentence_model",),
            ),
            frozenset(("sentence_model", "ner")),
        )

    def test_retains_upstream_tok2vec_for_sentence_recognizer_listener(self) -> None:
        self.assertEqual(
            select_ner_components(
                ["encoder", "sentence_model", "ner"],
                senter_names=("sentence_model",),
                tok2vec_upstreams={"sentence_model": "encoder"},
            ),
            frozenset(("encoder", "sentence_model", "ner")),
        )

    def test_rejects_multiple_sentence_boundary_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one sentence boundary"):
            select_ner_components(
                ["sentences_a", "sentences_b", "ner"],
                sentencizer_names=("sentences_a",),
                senter_names=("sentences_b",),
            )

    def test_rejects_missing_required_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one NER"):
            select_ner_components(["tok2vec"], ner_names=())
        with self.assertRaisesRegex(ValueError, "missing upstream tok2vec"):
            select_ner_components(
                ["ner"],
                tok2vec_upstreams={"ner": "encoder"},
            )

    def test_rejects_ambiguous_trainable_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one NER"):
            select_ner_components(
                ["ner_a", "ner_b"],
                ner_names=("ner_a", "ner_b"),
            )
        with self.assertRaisesRegex(ValueError, "at most one parser"):
            select_ner_components(
                ["parser_a", "parser_b", "ner"],
                parser_names=("parser_a", "parser_b"),
            )


if __name__ == "__main__":
    unittest.main()
