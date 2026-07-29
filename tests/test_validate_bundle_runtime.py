from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from validate_bundle_runtime import (
    RuntimeValidationError,
    validate_bundle,
    validation_command,
)


class RuntimeValidationTests(unittest.TestCase):
    def test_selects_features_for_each_tokenizer_backend(self) -> None:
        manifest_path = Path("/workspace/Cargo.toml")
        bundle = Path("/models/model.spacy-rs")

        delarocha = validation_command(bundle, "delarocha", manifest_path)
        regex = validation_command(bundle, "regex", manifest_path)
        sudachi = validation_command(bundle, "sudachi", manifest_path)

        self.assertNotIn("--no-default-features", delarocha)
        self.assertIn("--no-default-features", regex)
        self.assertIn("--no-default-features", sudachi)
        self.assertNotIn("--features", regex)
        self.assertEqual(
            sudachi[sudachi.index("--features") + 1],
            "sudachi-tokenizer",
        )
        self.assertEqual(delarocha[-3:], ["--", "--json", str(bundle)])

    def test_rejects_an_unknown_tokenizer_backend(self) -> None:
        with self.assertRaisesRegex(
            RuntimeValidationError,
            "unsupported tokenizer kind",
        ):
            validation_command(Path("/model"), "unknown")

    def test_selects_the_ginza_transformer_validator(self) -> None:
        command = validation_command(
            Path("/model"),
            "sudachi",
            Path("/workspace/Cargo.toml"),
            transformer=True,
        )
        self.assertIn("jewel-ginza", command)
        self.assertIn("transformers", command)
        self.assertIn("inspect_transformer_bundle", command)

    def test_rejects_missing_bundle_metadata_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeValidationError,
                "cannot read bundle tokenizer metadata",
            ):
                validate_bundle(Path(directory))

    def test_returns_a_compatible_runtime_report(self) -> None:
        report = {
            "report_version": 1,
            "compatible": True,
            "bundle_path": "/model",
            "diagnostics": [],
        }
        with self.bundle("regex") as bundle:
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(report),
                stderr="",
            )
            with patch("validate_bundle_runtime.subprocess.run", return_value=completed):
                self.assertEqual(validate_bundle(bundle), report)

    def test_preserves_structured_incompatibility_details(self) -> None:
        report = {
            "report_version": 1,
            "compatible": False,
            "bundle_path": "/model",
            "diagnostics": [
                {
                    "code": "unsupported_graph_node",
                    "area": "graph_node",
                    "component": "ner",
                    "node": 4,
                    "message": "unsupported node",
                }
            ],
        }
        with self.bundle("regex") as bundle:
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout=json.dumps(report),
                stderr="",
            )
            with patch("validate_bundle_runtime.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(
                    RuntimeValidationError,
                    r"\[unsupported_graph_node\]",
                ) as context:
                    validate_bundle(bundle)

        self.assertEqual(context.exception.report, report)

    def test_rejects_non_json_validator_output(self) -> None:
        with self.bundle("regex") as bundle:
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=101,
                stdout="",
                stderr="cargo failed",
            )
            with patch("validate_bundle_runtime.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(
                    RuntimeValidationError,
                    "did not return a JSON report",
                ):
                    validate_bundle(bundle)

    def test_rejects_an_unknown_report_version(self) -> None:
        report = {
            "report_version": 2,
            "compatible": True,
            "bundle_path": "/model",
            "diagnostics": [],
        }
        with self.bundle("regex") as bundle:
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(report),
                stderr="",
            )
            with patch("validate_bundle_runtime.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(
                    RuntimeValidationError,
                    "unsupported report version",
                ):
                    validate_bundle(bundle)

    @staticmethod
    def bundle(tokenizer_kind: str, *, transformer: bool = False):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "tokenizer": {"kind": tokenizer_kind},
                    "pipeline": (
                        [{"factory": "transformer_custom"}] if transformer else []
                    ),
                }
            ),
            encoding="utf-8",
        )

        class BundleContext:
            def __enter__(self) -> Path:
                return root

            def __exit__(self, *_: object) -> None:
                temporary.cleanup()

        return BundleContext()


if __name__ == "__main__":
    unittest.main()
