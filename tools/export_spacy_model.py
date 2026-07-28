"""Export a spaCy pipeline into a Python-free runtime bundle.

The exporter runs in a Python build environment. The resulting bundle contains
only data and tensors; the Rust inference runtime does not embed or invoke
Python.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy
import spacy
from safetensors.numpy import save_file
from thinc.api import Model

from export_profile import select_ner_components
from validate_bundle_runtime import (
    DEFAULT_MANIFEST_PATH,
    RuntimeValidationError,
    validate_bundle,
)

FORMAT_VERSION = 1
MIN_RUNTIME_VERSION = "0.0.1"
DELAROCHA_MIN_RUNTIME_VERSION = "0.0.4"
DELAROCHA_COMPATIBILITY_TERMS = (
    "株式会社",
    "有限会社",
    "合同会社",
    "取締役",
    "代表取締役",
    "契約金額",
    "違約金",
    "損害賠償金",
    "遅延損害金",
    "保証金",
    "解約金",
    "初期費用",
    "成功報酬",
    "取引先",
    "所在地",
)

SAFETENSORS_DTYPES = {
    "bool": "BOOL",
    "int8": "I8",
    "uint8": "U8",
    "int16": "I16",
    "uint16": "U16",
    "int32": "I32",
    "uint32": "U32",
    "int64": "I64",
    "uint64": "U64",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
}


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, slice):
        return {
            "__type__": "slice",
            "start": json_value(value.start),
            "stop": json_value(value.stop),
            "step": json_value(value.step),
        }
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, set):
        return sorted((json_value(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    raise TypeError(type(value).__qualname__)


def regex_pattern(function: Any) -> str | None:
    if function is None:
        return None
    owner = getattr(function, "__self__", None)
    pattern = getattr(owner, "pattern", None)
    if not isinstance(pattern, str):
        raise TypeError(f"cannot extract regex from {function!r}")
    return pattern


def copy_sudachi_assets(output: Path) -> dict:
    try:
        import sudachidict_core
        import sudachipy
    except ImportError as error:
        raise RuntimeError(
            "Japanese export requires sudachipy and sudachidict_core"
        ) from error

    target = output / "tokenizer" / "sudachi"
    target.mkdir(parents=True)
    sudachipy_root = Path(sudachipy.__file__).parent
    dictionary_root = Path(sudachidict_core.__file__).parent
    resources = sudachipy_root / "resources"
    dictionary = dictionary_root / "resources" / "system.dic"

    for filename in ("sudachi.json", "char.def", "unk.def", "rewrite.def"):
        source = resources / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / filename)
    if not dictionary.is_file():
        raise FileNotFoundError(dictionary)
    shutil.copy2(dictionary, target / "system.dic")

    dictionary_dist = distribution("sudachidict-core")
    license_files = [
        file
        for file in dictionary_dist.files or ()
        if "LICENSE" in str(file).upper()
    ]
    for license_file in license_files:
        shutil.copy2(
            dictionary_dist.locate_file(license_file),
            target / f"LICENSE.{Path(str(license_file)).name}",
        )

    digest = hashlib.sha256()
    with dictionary.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "config_path": "tokenizer/sudachi/sudachi.json",
        "dictionary_path": "tokenizer/sudachi/system.dic",
        "sudachipy_version": distribution("sudachipy").version,
        "dictionary_version": dictionary_dist.version,
        "dictionary_sha256": digest.hexdigest(),
    }


def copy_delarocha_assets(output: Path, dictionary: Path) -> dict:
    dictionary = dictionary.resolve()
    if not dictionary.is_file():
        raise FileNotFoundError(dictionary)
    if dictionary.name.endswith(".dic.zst"):
        filename = "system.dic.zst"
    elif dictionary.suffix == ".dic":
        filename = "system.dic"
    else:
        raise ValueError(
            "delarocha requires a Vibrato system.dic or system.dic.zst dictionary"
        )

    target = output / "tokenizer" / "delarocha"
    target.mkdir(parents=True)
    bundled_dictionary = target / filename
    shutil.copy2(dictionary, bundled_dictionary)

    digest = hashlib.sha256()
    with dictionary.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "dictionary_path": f"tokenizer/delarocha/{filename}",
        "dictionary_sha256": digest.hexdigest(),
    }


def japanese_tag_payload() -> dict:
    from spacy.lang.ja import TAG_BIGRAM_MAP, TAG_MAP, TAG_ORTH_MAP
    from spacy.symbols import POS

    return {
        "tag_map": {
            tag: attrs[POS] for tag, attrs in sorted(TAG_MAP.items())
        },
        "tag_orth_map": {
            tag: dict(sorted(mapping.items()))
            for tag, mapping in sorted(TAG_ORTH_MAP.items())
        },
        "tag_bigram_map": [
            {
                "tag": tag,
                "next_tag": next_tag,
                "pos": pos,
                "next_pos": next_pos,
            }
            for (tag, next_tag), (pos, next_pos) in sorted(
                TAG_BIGRAM_MAP.items()
            )
        ],
    }


def delarocha_compatibility_rules(nlp: Any) -> list[dict]:
    rules = []
    for text in DELAROCHA_COMPATIBILITY_TERMS:
        document = nlp.make_doc(text)
        tokens = []
        for token in document:
            inflection = token.morph.get("Inflection")
            reading = token.morph.get("Reading")
            tokens.append(
                {
                    "surface": token.text,
                    "tag": token.tag_,
                    "inflection": ";".join(inflection),
                    "lemma": token.lemma_,
                    "norm": token.norm_,
                    "reading": reading[0] if reading else None,
                }
            )
        if len(tokens) > 1:
            rules.append({"text": text, "tokens": tokens})
    return rules


def export_tokenizer(
    nlp: Any,
    output: Path,
    japanese_tokenizer: str = "delarocha",
    delarocha_dictionary: Path | None = None,
) -> dict:
    tokenizer = nlp.tokenizer
    if all(
        hasattr(tokenizer, name)
        for name in ("rules", "prefix_search", "suffix_search", "infix_finditer")
    ):
        from spacy.lang.norm_exceptions import BASE_NORMS
        from spacy.strings import hash_string

        exceptions = {}
        for text, rules in sorted(tokenizer.rules.items()):
            exceptions[text] = [
                {
                    "orth": rule[65],
                    "norm": rule.get(67),
                }
                for rule in rules
            ]
        norm_overrides = {
            str(hash_string(text)): norm for text, norm in BASE_NORMS.items()
        }
        lexeme_norms = nlp.vocab.lookups.get_table("lexeme_norm", {})
        norm_overrides.update(
            {str(orth): norm for orth, norm in lexeme_norms.items()}
        )
        payload = {
            "format_version": 1,
            "language": nlp.lang,
            "prefix": regex_pattern(tokenizer.prefix_search),
            "suffix": regex_pattern(tokenizer.suffix_search),
            "infix": regex_pattern(tokenizer.infix_finditer),
            "token_match": regex_pattern(tokenizer.token_match),
            "url_match": regex_pattern(tokenizer.url_match),
            "exceptions": exceptions,
            "norm_overrides": dict(sorted(norm_overrides.items())),
        }
        (output / "tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return {"kind": "regex", "path": "tokenizer.json"}

    if nlp.lang == "ja":
        tag_payload = japanese_tag_payload()
        if japanese_tokenizer == "delarocha":
            if delarocha_dictionary is None:
                configured_dictionary = os.environ.get("DELAROCHA_SYSTEM_DIC")
                if configured_dictionary:
                    delarocha_dictionary = Path(configured_dictionary)
            if delarocha_dictionary is None:
                raise ValueError(
                    "set --delarocha-dictionary or DELAROCHA_SYSTEM_DIC "
                    "for Japanese model export"
                )
            assets = copy_delarocha_assets(output, delarocha_dictionary)
            payload = {
                "format_version": 1,
                "language": "ja",
                "dictionary_path": assets["dictionary_path"],
                "dictionary_sha256": assets["dictionary_sha256"],
                "feature_schema": "ipadic",
                "ignore_space": True,
                "max_grouping_len": 24,
                "merge_formatted_numbers": True,
                "merge_address_towns": True,
                "compatibility_rules": delarocha_compatibility_rules(nlp),
                **tag_payload,
            }
            (output / "tokenizer.json").write_text(
                json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
            return {"kind": "delarocha", "path": "tokenizer.json"}

        sudachi_assets = copy_sudachi_assets(output)
        payload = {
            "format_version": 1,
            "language": "ja",
            "split_mode": getattr(tokenizer, "split_mode", None),
            "config_path": sudachi_assets["config_path"],
            "dictionary_path": sudachi_assets["dictionary_path"],
            "sudachipy_version": sudachi_assets["sudachipy_version"],
            "dictionary_version": sudachi_assets["dictionary_version"],
            "dictionary_sha256": sudachi_assets["dictionary_sha256"],
            **tag_payload,
        }
        (output / "tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return {"kind": "sudachi", "path": "tokenizer.json"}

    raise TypeError(f"unsupported tokenizer: {type(tokenizer).__qualname__}")


def inspect_model(component_name: str, model: Model, tensors: dict) -> list[dict]:
    nodes = list(model.walk())
    node_indices = {node.id: index for index, node in enumerate(nodes)}
    manifests = []

    for index, node in enumerate(nodes):
        dims = {
            name: node.get_dim(name) if node.has_dim(name) else None
            for name in node.dim_names
        }
        refs = {
            name: (
                node_indices[node.get_ref(name).id]
                if node.has_ref(name)
                else None
            )
            for name in node.ref_names
        }

        params = {}
        for name in node.param_names:
            if not node.has_param(name):
                continue
            tensor = numpy.ascontiguousarray(node.ops.to_numpy(node.get_param(name)))
            key = f"components.{component_name}.nodes.{index}.{name}"
            tensors[key] = tensor
            params[name] = {
                "key": key,
                "dtype": SAFETENSORS_DTYPES[str(tensor.dtype)],
                "shape": list(tensor.shape),
            }

        attrs = {}
        omitted_attrs = []
        for name, value in node.attrs.items():
            try:
                attrs[name] = json_value(value)
            except TypeError:
                omitted_attrs.append(f"{name}:{type(value).__qualname__}")

        manifests.append(
            {
                "index": index,
                "name": node.name,
                "children": [node_indices[child.id] for child in node.layers],
                "dims": dims,
                "refs": refs,
                "params": params,
                "attrs": attrs,
                "omitted_attrs": sorted(omitted_attrs),
            }
        )

    return manifests


def tok2vec_listener_upstream(component: Any) -> str | None:
    model = getattr(component, "model", None)
    if not isinstance(model, Model):
        return None
    upstreams = {
        getattr(node, "upstream_name", None)
        for node in model.walk()
        if node.name == "tok2vec-listener"
    }
    if not upstreams:
        return None
    if None in upstreams or len(upstreams) != 1:
        raise ValueError(
            "Jewel requires every tok2vec listener in a component to name "
            "the same upstream component"
        )
    return upstreams.pop()


def component_settings(factory: str, component: Any) -> dict:
    settings = {}
    if factory == "sentencizer":
        settings.update({
            "punct_chars": sorted(component.punct_chars),
            "overwrite": bool(component.overwrite),
        })
    elif factory == "senter":
        settings["overwrite"] = bool(component.cfg["overwrite"])
    elif factory == "entity_ruler":
        phrase_matcher_attr = component.phrase_matcher_attr or "ORTH"
        if phrase_matcher_attr not in {"ORTH", "LOWER", "NORM"}:
            raise ValueError(
                "Jewel entity_ruler supports phrase_matcher_attr values "
                "ORTH, LOWER, and NORM"
            )
        if any(component.token_patterns.values()):
            raise ValueError(
                "Jewel entity_ruler supports exact phrase patterns only; "
                "the component contains token-based patterns"
            )
        patterns = []
        for internal_label, documents in component.phrase_patterns.items():
            label, _ = component._split_label(internal_label)
            for document in documents:
                token_ids = [
                    int(getattr(token, phrase_matcher_attr.lower()))
                    for token in document
                ]
                if not token_ids:
                    raise ValueError(
                        "Jewel entity_ruler phrase pattern tokenizes to no tokens"
                    )
                patterns.append(
                    {
                        "label": label,
                        "token_ids": token_ids,
                    }
                )
        settings.update(
            {
                "overwrite_ents": bool(component.overwrite),
                "phrase_matcher_attr": phrase_matcher_attr,
                "patterns": patterns,
            }
        )
    upstream = tok2vec_listener_upstream(component)
    if upstream is not None:
        settings["tok2vec_upstream"] = upstream
    return settings


def export_vectors(nlp: Any, tensors: dict) -> dict | None:
    vectors = nlp.vocab.vectors
    if not vectors.shape[0] or not vectors.shape[1]:
        return None
    data = numpy.ascontiguousarray(vectors.data, dtype=numpy.float32)
    keys = numpy.fromiter(vectors.key2row.keys(), dtype=numpy.uint64)
    rows = numpy.fromiter(vectors.key2row.values(), dtype=numpy.uint64)
    payload = {}
    for name, tensor in (("data", data), ("keys", keys), ("rows", rows)):
        key = f"vocab.vectors.{name}"
        tensors[key] = tensor
        payload[name] = {
            "key": key,
            "dtype": SAFETENSORS_DTYPES[str(tensor.dtype)],
            "shape": list(tensor.shape),
        }
    return payload


def export_model(
    model: str,
    output: Path,
    profile: str = "full",
    japanese_tokenizer: str = "delarocha",
    delarocha_dictionary: Path | None = None,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "components").mkdir()

    nlp = spacy.load(model)
    if profile == "ner":
        ner_names = tuple(
            name
            for name in nlp.pipe_names
            if nlp.get_pipe_meta(name).factory == "ner"
        )
        parser_names = tuple(
            name
            for name in nlp.pipe_names
            if nlp.get_pipe_meta(name).factory == "parser"
        )
        sentencizer_names = tuple(
            name
            for name in nlp.pipe_names
            if nlp.get_pipe_meta(name).factory == "sentencizer"
        )
        senter_names = tuple(
            name
            for name in nlp.pipe_names
            if nlp.get_pipe_meta(name).factory == "senter"
        )
        entity_ruler_names = tuple(
            name
            for name in nlp.pipe_names
            if nlp.get_pipe_meta(name).factory == "entity_ruler"
        )
        component_names = ner_names + parser_names + senter_names
        tok2vec_upstreams = {
            name: upstream
            for name in component_names
            if (upstream := tok2vec_listener_upstream(nlp.get_pipe(name)))
            is not None
        }
        try:
            selected_components = select_ner_components(
                nlp.pipe_names,
                ner_names=ner_names,
                parser_names=parser_names,
                sentencizer_names=sentencizer_names,
                senter_names=senter_names,
                entity_ruler_names=entity_ruler_names,
                tok2vec_upstreams=tok2vec_upstreams,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
    else:
        selected_components = frozenset(nlp.pipe_names)
    tokenizer_manifest = export_tokenizer(
        nlp,
        output,
        japanese_tokenizer=japanese_tokenizer,
        delarocha_dictionary=delarocha_dictionary,
    )
    tensors = {}
    vectors_manifest = export_vectors(nlp, tensors)
    pipeline = []

    for name, component in nlp.pipeline:
        if name not in selected_components:
            continue
        meta = nlp.get_pipe_meta(name)
        component_dir = output / "components" / name
        component_dir.mkdir()
        state_path = f"components/{name}/state.bin"
        (output / state_path).write_bytes(component.to_bytes(exclude=["vocab"]))

        thinc_model = getattr(component, "model", None)
        nodes = (
            inspect_model(name, thinc_model, tensors)
            if isinstance(thinc_model, Model)
            else []
        )
        transition_system = getattr(component, "moves", None)
        move_count = int(getattr(transition_system, "n_moves", 0))
        moves = [
            transition_system.get_class_name(index)
            for index in range(move_count)
        ]
        pipeline.append(
            {
                "name": name,
                "factory": meta.factory,
                "kind": "trainable" if nodes else "rule_based",
                "root_node": 0 if nodes else None,
                "settings": component_settings(meta.factory, component),
                "nodes": nodes,
                "state_path": state_path,
                "labels": list(getattr(component, "labels", ())),
                "moves": moves,
            }
        )

    model_name = str(nlp.meta.get("name", model))
    if nlp.meta.get("lang") and not model_name.startswith(f"{nlp.meta['lang']}_"):
        model_name = f"{nlp.meta['lang']}_{model_name}"
    manifest = {
        "format_version": FORMAT_VERSION,
        "source": {
            "spacy_version": spacy.__version__,
            "model_name": model_name,
            "model_version": str(nlp.meta.get("version", "")),
            "lang": str(nlp.lang),
        },
        "runtime": {
            "min_runtime_version": (
                DELAROCHA_MIN_RUNTIME_VERSION
                if tokenizer_manifest["kind"] == "delarocha"
                else MIN_RUNTIME_VERSION
            ),
            "requires_python": False,
        },
        "tokenizer": tokenizer_manifest,
        "vectors": vectors_manifest,
        "pipeline": pipeline,
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "config.cfg").write_text(nlp.config.to_str(), encoding="utf-8")
    (output / "meta.json").write_text(
        json.dumps(nlp.meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "strings.json").write_text(
        json.dumps(sorted(nlp.vocab.strings), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_file(tensors, output / "weights.safetensors")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="installed spaCy package name or model path")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--profile",
        choices=("full", "ner"),
        default="full",
        help=(
            "export all components or extraction-only NER, retaining "
            "tok2vec/parser, a parser-less sentence boundary component, and "
            "supported post-NER entity rulers"
        ),
    )
    parser.add_argument(
        "--japanese-tokenizer",
        choices=("sudachi", "delarocha"),
        default="delarocha",
        help="Japanese runtime backend; delarocha is the default",
    )
    parser.add_argument(
        "--delarocha-dictionary",
        type=Path,
        help=(
            "Vibrato system.dic or system.dic.zst built with an IPADIC feature "
            "schema; defaults to DELAROCHA_SYSTEM_DIC"
        ),
    )
    parser.add_argument(
        "--runtime-manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Cargo.toml for the Jewel runtime used to validate the export",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="skip Rust runtime loading; intended only for exporter debugging",
    )
    args = parser.parse_args()

    manifest = export_model(
        args.model,
        args.output,
        profile=args.profile,
        japanese_tokenizer=args.japanese_tokenizer,
        delarocha_dictionary=args.delarocha_dictionary,
    )
    runtime_validation: dict[str, Any]
    if args.skip_runtime_validation:
        runtime_validation = {"status": "skipped"}
    else:
        try:
            report = validate_bundle(args.output, args.runtime_manifest_path)
        except RuntimeValidationError as error:
            raise SystemExit(str(error)) from error
        runtime_validation = {
            "status": "passed",
            "report_version": report["report_version"],
        }
    node_count = sum(len(component["nodes"]) for component in manifest["pipeline"])
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile": args.profile,
                "tokenizer": manifest["tokenizer"]["kind"],
                "components": len(manifest["pipeline"]),
                "nodes": node_count,
                "requires_python": manifest["runtime"]["requires_python"],
                "runtime_validation": runtime_validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
