"""Validate an exported model bundle with Jewel's Rust NER runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "Cargo.toml"
SUPPORTED_REPORT_VERSION = 1


class RuntimeValidationError(RuntimeError):
    """Raised when Jewel cannot load an exported bundle."""

    def __init__(
        self,
        message: str,
        *,
        report: dict[str, Any] | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.report = report
        self.stderr = stderr


def bundle_tokenizer_kind(bundle: Path) -> str:
    """Read the tokenizer backend declared by a bundle manifest."""
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeValidationError(
            f"cannot read bundle tokenizer metadata from {manifest_path}: {error}",
        ) from error
    kind = manifest.get("tokenizer", {}).get("kind")
    if not isinstance(kind, str) or not kind:
        raise RuntimeValidationError(
            f"{manifest_path}: missing tokenizer kind",
        )
    return kind


def validation_command(
    bundle: Path,
    tokenizer_kind: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> list[str]:
    """Build the cargo command for a bundle's tokenizer feature set."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(manifest_path),
    ]
    if tokenizer_kind == "sudachi":
        command.extend(
            ("--no-default-features", "--features", "sudachi-tokenizer")
        )
    elif tokenizer_kind == "regex":
        command.append("--no-default-features")
    elif tokenizer_kind != "delarocha":
        raise RuntimeValidationError(
            f"unsupported tokenizer kind {tokenizer_kind!r}",
        )
    command.extend(
        (
            "--example",
            "inspect_bundle",
            "--",
            "--json",
            str(bundle),
        )
    )
    return command


def validate_bundle(
    bundle: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load a bundle with Jewel and return its compatibility report."""
    tokenizer_kind = bundle_tokenizer_kind(bundle)
    completed = subprocess.run(
        validation_command(bundle, tokenizer_kind, manifest_path),
        check=False,
        text=True,
        capture_output=True,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeValidationError(
            "Jewel runtime validator did not return a JSON report"
            + (f": {detail}" if detail else ""),
            stderr=completed.stderr,
        ) from error
    if not isinstance(report, dict):
        raise RuntimeValidationError(
            "Jewel runtime validator returned a non-object JSON report",
            stderr=completed.stderr,
        )
    if report.get("report_version") != SUPPORTED_REPORT_VERSION:
        raise RuntimeValidationError(
            "Jewel runtime validator returned unsupported report version "
            f"{report.get('report_version')!r}; expected {SUPPORTED_REPORT_VERSION}",
            report=report,
            stderr=completed.stderr,
        )
    if completed.returncode != 0 or report.get("compatible") is not True:
        diagnostics = report.get("diagnostics")
        first = diagnostics[0] if isinstance(diagnostics, list) and diagnostics else {}
        code = first.get("code", "unknown") if isinstance(first, dict) else "unknown"
        message = (
            first.get("message", "bundle is incompatible")
            if isinstance(first, dict)
            else "bundle is incompatible"
        )
        raise RuntimeValidationError(
            f"Jewel rejected the exported bundle [{code}]: {message}",
            report=report,
            stderr=completed.stderr,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Cargo.toml for the Jewel runtime used to validate the bundle",
    )
    args = parser.parse_args()

    try:
        report = validate_bundle(args.bundle, args.manifest_path)
    except RuntimeValidationError as error:
        if error.report is not None:
            print(json.dumps(error.report, ensure_ascii=False, indent=2))
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
