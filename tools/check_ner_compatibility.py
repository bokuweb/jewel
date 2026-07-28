"""Compare spaCy NER output with an exported Jewel bundle.

The compatibility check runs spaCy and Jewel over the same JSONL corpus and
requires exact agreement for entity text, label, token range, and Unicode
code-point range. Python is used only by this development-time check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ENTITY_FIELDS = (
    "text",
    "label",
    "start_token",
    "end_token",
    "start_char",
    "end_char",
)
SEMANTIC_ENTITY_FIELDS = ("text", "label", "start_char", "end_char")
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def load_cases(path: Path) -> list[dict[str, str]]:
    cases = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        text = value.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
        case_id = value.get("id", f"line-{line_number}")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        cases.append({"id": case_id, "text": text})
    if not cases:
        raise ValueError(f"{path}: no compatibility cases found")
    return cases


def entity_payload(entity: Any) -> dict[str, Any]:
    return {
        "text": entity.text,
        "label": entity.label_,
        "start_token": entity.start,
        "end_token": entity.end,
        "start_char": entity.start_char,
        "end_char": entity.end_char,
    }


def spacy_results(model: str, cases: list[dict[str, str]]) -> tuple[Any, list[dict]]:
    import spacy

    nlp = spacy.load(model)
    documents = nlp.pipe(case["text"] for case in cases)
    results = []
    for case, document in zip(cases, documents, strict=True):
        results.append(
            {
                "id": case["id"],
                "text": case["text"],
                "language": nlp.lang,
                "entities": [entity_payload(entity) for entity in document.ents],
            }
        )
    return nlp, results


def export_bundle(
    model: str,
    output: Path,
    japanese_tokenizer: str,
    delarocha_dictionary: Path | None,
) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tools" / "export_spacy_model.py"),
        model,
        str(output),
        "--profile",
        "ner",
        "--japanese-tokenizer",
        japanese_tokenizer,
    ]
    if delarocha_dictionary is not None:
        command.extend(
            ("--delarocha-dictionary", str(delarocha_dictionary))
        )
    subprocess.run(command, check=True)


def jewel_results(
    bundle: Path,
    cases: list[dict[str, str]],
    tokenizer_kind: str,
) -> list[dict]:
    input_text = "".join(
        json.dumps(case["text"], ensure_ascii=False) + "\n"
        for case in cases
    )
    command = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(REPOSITORY_ROOT / "Cargo.toml"),
    ]
    if tokenizer_kind == "sudachi":
        command.extend(
            ("--no-default-features", "--features", "sudachi-tokenizer")
        )
    elif tokenizer_kind == "regex":
        command.append("--no-default-features")
    command.extend(
        (
            "--example",
            "entities_jsonl",
            "--",
            str(bundle),
            "--json-input",
        )
    )
    completed = subprocess.run(
        command,
        check=True,
        input=input_text,
        text=True,
        capture_output=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != len(cases):
        raise RuntimeError(
            f"Jewel returned {len(lines)} documents for {len(cases)} cases"
        )
    results = []
    for case, line in zip(cases, lines, strict=True):
        value = json.loads(line)
        value["id"] = case["id"]
        results.append(value)
    return results


def normalize_result(result: dict) -> dict:
    return {
        "id": result["id"],
        "text": result["text"],
        "language": result["language"],
        "entities": [
            {field: entity[field] for field in ENTITY_FIELDS}
            for entity in result["entities"]
        ],
    }


def bundle_tokenizer_kind(bundle: Path) -> str:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    kind = manifest.get("tokenizer", {}).get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{bundle}/manifest.json: missing tokenizer kind")
    return kind


def compare_results(expected: list[dict], actual: list[dict]) -> list[dict]:
    if len(expected) != len(actual):
        return [
            {
                "reason": "document_count",
                "expected": len(expected),
                "actual": len(actual),
            }
        ]
    mismatches = []
    for expected_document, actual_document in zip(expected, actual, strict=True):
        expected_document = normalize_result(expected_document)
        actual_document = normalize_result(actual_document)
        if expected_document != actual_document:
            mismatches.append(
                {
                    "id": expected_document["id"],
                    "expected": expected_document,
                    "actual": actual_document,
                }
            )
    return mismatches


def count_semantic_mismatches(mismatches: list[dict]) -> int:
    count = 0
    for mismatch in mismatches:
        if "expected" not in mismatch or "actual" not in mismatch:
            count += 1
            continue
        expected_entities = [
            {
                field: entity[field]
                for field in SEMANTIC_ENTITY_FIELDS
            }
            for entity in mismatch["expected"]["entities"]
        ]
        actual_entities = [
            {
                field: entity[field]
                for field in SEMANTIC_ENTITY_FIELDS
            }
            for entity in mismatch["actual"]["entities"]
        ]
        count += expected_entities != actual_entities
    return count


def write_report(
    path: Path | None,
    model: str,
    nlp: Any,
    expected: list[dict],
    mismatches: list[dict],
    tokenizer_kind: str,
) -> dict:
    import spacy

    semantic_mismatch_count = count_semantic_mismatches(mismatches)
    report = {
        "model": model,
        "model_version": str(nlp.meta.get("version", "")),
        "spacy_version": spacy.__version__,
        "language": nlp.lang,
        "tokenizer": tokenizer_kind,
        "case_count": len(expected),
        "entity_count": sum(len(result["entities"]) for result in expected),
        "mismatch_count": len(mismatches),
        "semantic_mismatch_count": semantic_mismatch_count,
        "token_only_mismatch_count": len(mismatches) - semantic_mismatch_count,
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def check(
    model: str,
    cases_path: Path,
    bundle: Path | None,
    report_path: Path | None,
    work_dir: Path | None,
    japanese_tokenizer: str,
    delarocha_dictionary: Path | None,
) -> int:
    cases = load_cases(cases_path)
    nlp, expected = spacy_results(model, cases)

    if bundle is not None:
        tokenizer_kind = bundle_tokenizer_kind(bundle)
        actual = jewel_results(
            bundle,
            cases,
            tokenizer_kind=tokenizer_kind,
        )
    else:
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="jewel-compat-",
            dir=work_dir,
        ) as directory:
            generated_bundle = Path(directory) / "model.spacy-rs"
            export_bundle(
                model,
                generated_bundle,
                japanese_tokenizer,
                delarocha_dictionary,
            )
            tokenizer_kind = bundle_tokenizer_kind(generated_bundle)
            actual = jewel_results(
                generated_bundle,
                cases,
                tokenizer_kind=tokenizer_kind,
            )

    mismatches = compare_results(expected, actual)
    report = write_report(
        report_path,
        model,
        nlp,
        expected,
        mismatches,
        tokenizer_kind,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="installed spaCy package name or model path")
    parser.add_argument("cases", type=Path, help="JSONL corpus with id and text fields")
    parser.add_argument(
        "--bundle",
        type=Path,
        help="use an existing Jewel bundle instead of exporting a temporary bundle",
    )
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="parent directory for temporary exported bundles",
    )
    parser.add_argument(
        "--japanese-tokenizer",
        choices=("sudachi", "delarocha"),
        default="delarocha",
        help="backend used for a generated Japanese bundle and Jewel execution",
    )
    parser.add_argument(
        "--delarocha-dictionary",
        type=Path,
        help="IPADIC-compatible Vibrato dictionary used by the delarocha backend",
    )
    args = parser.parse_args()
    raise SystemExit(
        check(
            args.model,
            args.cases,
            args.bundle,
            args.report,
            args.work_dir,
            args.japanese_tokenizer,
            args.delarocha_dictionary,
        )
    )


if __name__ == "__main__":
    main()
