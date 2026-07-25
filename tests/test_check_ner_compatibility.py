from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from check_ner_compatibility import compare_results, load_cases


class CompatibilityHarnessTests(unittest.TestCase):
    def test_load_cases_accepts_jsonl_and_assigns_default_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                '{"id":"named","text":"Jane Smith"}\n{"text":"山田太郎"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                load_cases(path),
                [
                    {"id": "named", "text": "Jane Smith"},
                    {"id": "line-2", "text": "山田太郎"},
                ],
            )

    def test_load_cases_rejects_embedded_line_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                '{"id":"multiline","text":"first\\nsecond"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not contain line breaks"):
                load_cases(path)

    def test_compare_results_reports_exact_offset_drift(self) -> None:
        expected = [
            {
                "id": "person",
                "text": "Jane Smith",
                "language": "en",
                "entities": [
                    {
                        "text": "Jane Smith",
                        "label": "PERSON",
                        "start_token": 0,
                        "end_token": 2,
                        "start_char": 0,
                        "end_char": 10,
                    }
                ],
            }
        ]
        actual = json.loads(json.dumps(expected))
        actual[0]["entities"][0]["end_char"] = 9

        mismatches = compare_results(expected, actual)

        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["id"], "person")
        self.assertEqual(mismatches[0]["actual"]["entities"][0]["end_char"], 9)

    def test_compare_results_accepts_exact_parity(self) -> None:
        result = [
            {
                "id": "empty",
                "text": "No entities here.",
                "language": "en",
                "entities": [],
            }
        ]
        self.assertEqual(compare_results(result, result), [])


if __name__ == "__main__":
    unittest.main()
