# Jewel

Jewel is a Python-free Rust inference runtime for selected spaCy pipelines.
It loads model bundles exported from spaCy and runs tokenization and neural
inference without embedding Python or starting a Python process.

The current implementation focuses on Japanese and English named-entity
recognition for applications that need people, organizations, locations, and
other model-defined entity spans in native Rust services.

> This project is an independent compatibility implementation. It is not
> affiliated with or endorsed by Explosion or the spaCy project.

## Status

Jewel is experimental and intentionally implements a focused inference subset,
not the complete spaCy platform.

Implemented:

- spaCy-compatible string hashing and core document/token representations
- regex-based tokenization for English model bundles
- Sudachi-based tokenization for Japanese model bundles
- selected Thinc-compatible neural operations
- `tok2vec`, fine-grained tagger, transition-based dependency parser, and NER
- extraction-only Japanese and English NER pipelines
- spaCy `DocBin` compatibility for the covered attributes
- model bundle validation and safetensors weight loading

Not implemented:

- model training or fine-tuning in Rust
- spaCy's Python pipeline, registry, extension, and plugin APIs
- every spaCy component, architecture, language, or third-party model
- automatic compatibility with model graphs that use unsupported Thinc nodes

Model packages and their dictionaries are not distributed in this repository.
Review the license and redistribution terms of every model and dictionary that
you export.

## Add the crate

The crate is currently consumed directly from Git:

```toml
[dependencies]
jewel_spacy = { git = "https://github.com/bokuweb/jewel.git", tag = "0.0.1" }
```

Pin a release tag or a tested commit with `rev` for reproducible builds.

## Export a model bundle

Python is required only in the build environment that exports the model. The
resulting bundle contains data and tensors and declares
`runtime.requires_python = false`.

Create an export environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install spacy numpy safetensors sudachipy sudachidict-core
python -m spacy download ja_core_news_sm
python -m spacy download en_core_web_sm
```

Export an extraction-only Japanese bundle:

```bash
python tools/export_spacy_model.py \
  ja_core_news_sm \
  /path/to/ja_core_news_sm.spacy-rs \
  --profile ner
```

Export English in the same way:

```bash
python tools/export_spacy_model.py \
  en_core_web_sm \
  /path/to/en_core_web_sm.spacy-rs \
  --profile ner
```

The `ner` profile keeps the `tok2vec`, `parser`, and `ner` components needed by
the extraction pipelines. The default `full` profile exports all source
components, but the Rust runtime can execute only the component types and
architectures documented above.

An exported bundle contains:

```text
model.spacy-rs/
├── manifest.json
├── weights.safetensors
├── config.cfg
├── meta.json
├── strings.json
├── tokenizer.json
├── components/
└── tokenizer/sudachi/    # Japanese bundles only
```

Keep the complete directory together when deploying a bundle.

## Examples

The repository includes examples for model validation, one-shot extraction,
batch processing, streaming JSONL output, and Unicode offset conversion.

| Example | Bundle required | Purpose |
| --- | --- | --- |
| `inspect_bundle` | any Jewel bundle | Validate and summarize a bundle |
| `extract_entities_ja` | Japanese | Extract every model-defined entity |
| `extract_people_ja` | Japanese | Extract only `PERSON` entities |
| `batch_entities_ja` | Japanese | Reuse one pipeline across many documents |
| `extract_entities_en` | English | Extract every model-defined entity |
| `entities_jsonl` | Japanese or English | Process stdin as a JSONL worker |
| `unicode_offsets` | none | Convert spaCy character offsets for Rust slicing |

Set paths once for the commands below:

```bash
export JEWEL_JA_BUNDLE=/path/to/ja_core_news_sm.spacy-rs
export JEWEL_EN_BUNDLE=/path/to/en_core_web_sm.spacy-rs
```

### Inspect a bundle

`inspect_bundle` calls the same validation used during pipeline loading. It
prints source model metadata, tokenizer type, component counts, and tensor
counts without running inference.

```bash
cargo run --example inspect_bundle -- "$JEWEL_JA_BUNDLE"
```

Example output:

```text
bundle: /path/to/ja_core_news_sm.spacy-rs
format version: 1
source: ja_core_news_sm 3.8.0 (spaCy 3.8.13, language ja)
runtime: minimum 0.1.0, requires Python: false
tokenizer: Sudachi (tokenizer.json)
components:
  tok2vec: factory=tok2vec, kind=Trainable, nodes=..., tensors=..., labels=0, moves=0
  parser: factory=parser, kind=Trainable, nodes=..., tensors=..., labels=..., moves=...
  ner: factory=ner, kind=Trainable, nodes=..., tensors=..., labels=..., moves=...
```

Exact model versions and counts depend on the exported package.

### Extract Japanese entities

Use `extract_entities_ja` when downstream code needs every entity label emitted
by the model:

```bash
cargo run --example extract_entities_ja -- \
  "$JEWEL_JA_BUNDLE" \
  "株式会社青空の山田太郎は東京都千代田区で契約を締結した。"
```

The tab-separated output contains the label, matched text, half-open token
range, and half-open Unicode character range:

```text
label   text             tokens  characters
ORG     株式会社         0..2    0..4
PERSON  山田太郎         4..6    7..11
GPE     東京都千代田区   7..11   12..19
```

Model output varies by model package and version. The ranges are the stable
runtime contract.

To keep only people:

```bash
cargo run --example extract_people_ja -- \
  "$JEWEL_JA_BUNDLE" \
  "受託者の山田太郎と担当者の佐藤花子が本業務を遂行する。"
```

### Process a Japanese batch

Load the Sudachi dictionary and neural weights once, then process multiple
documents with the same pipeline:

```bash
cargo run --example batch_entities_ja -- \
  "$JEWEL_JA_BUNDLE" \
  "甲株式会社の代表者は山田太郎です。" \
  "佐藤花子は大阪市の乙株式会社に所属します。"
```

The corresponding library API is `JapaneseNerPipeline::extract_entities_batch`.

### Extract English entities

```bash
cargo run --example extract_entities_en -- \
  "$JEWEL_EN_BUNDLE" \
  "Acme Corporation appointed Jane Smith in London on March 3, 2026."
```

### Run a JSONL extraction worker

`entities_jsonl` detects `ja` or `en` from the bundle manifest. It loads the
pipeline once, treats each non-empty stdin line as one document, and writes one
JSON object per line:

```bash
printf '%s\n' \
  "株式会社青空の山田太郎は東京都で勤務する。" \
  "佐藤花子は大阪支店を訪問した。" |
  cargo run --example entities_jsonl -- "$JEWEL_JA_BUNDLE"
```

Each output object has this shape:

```json
{
  "text": "株式会社青空の山田太郎は東京都で勤務する。",
  "language": "ja",
  "entities": [
    {
      "text": "山田太郎",
      "label": "PERSON",
      "start_token": 3,
      "end_token": 5,
      "start_char": 7,
      "end_char": 11
    }
  ]
}
```

This example is a minimal integration pattern for a long-running worker. A
production service should additionally define request size limits, timeouts,
concurrency, observability, and model/version reporting.

### Convert character offsets to Rust byte offsets

Jewel follows spaCy and reports Unicode code-point offsets. Rust string slices
use UTF-8 byte offsets. Do not pass `start_char` or `end_char` directly to
`&text[start..end]` for Japanese or other multibyte text.

```bash
cargo run --example unicode_offsets -- "契約者は山田太郎です" 4 8
```

```text
山田太郎
```

See [`examples/unicode_offsets.rs`](examples/unicode_offsets.rs) for the
conversion helper.

## Library usage

### Japanese NER

```rust
use jewel_spacy::{Bundle, JapaneseNerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = Bundle::load("/path/to/ja_core_news_sm.spacy-rs")?;
    let pipeline = JapaneseNerPipeline::load(&bundle)?;

    for entity in pipeline.extract_entities("受託者の山田太郎は東京で業務を行う。")? {
        println!(
            "{}\t{}\t{}..{}",
            entity.label, entity.text, entity.start_char, entity.end_char
        );
    }

    Ok(())
}
```

`start_char` and `end_char` are Unicode code-point offsets, matching Python
string indexing and spaCy's character-offset convention.

### English NER

```rust
use jewel_spacy::{Bundle, EnglishNerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = Bundle::load("/path/to/en_core_web_sm.spacy-rs")?;
    let pipeline = EnglishNerPipeline::load(&bundle)?;

    for entity in pipeline.extract_entities("Acme appointed Jane Smith in London.")? {
        println!(
            "{}\t{}\t{}..{}",
            entity.label, entity.text, entity.start_char, entity.end_char
        );
    }

    Ok(())
}
```

Load a pipeline once and reuse it across requests. Japanese bundles include a
Sudachi dictionary, and repeatedly loading that dictionary is unnecessary and
expensive.

### Batch Japanese NER

```rust
use jewel_spacy::{Bundle, JapaneseNerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = Bundle::load("/path/to/ja_core_news_sm.spacy-rs")?;
    let pipeline = JapaneseNerPipeline::load(&bundle)?;
    let texts = [
        "甲株式会社の代表者は山田太郎です。",
        "佐藤花子は大阪市の乙株式会社に所属します。",
    ];

    for (text, entities) in texts
        .iter()
        .zip(pipeline.extract_entities_batch(&texts)?)
    {
        println!("{text}");
        for entity in entities {
            println!(
                "  {}\t{}\t{}..{}",
                entity.label, entity.text, entity.start_char, entity.end_char
            );
        }
    }

    Ok(())
}
```

## Validation

Run the Rust compatibility suite:

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

The checked-in fixtures cover spaCy 3.8 string hashes, `DocBin` decoding, and
selected Thinc operations. End-to-end model tests live in downstream consumers
because exported model bundles are intentionally not checked into this
repository.

## Relationship to Ridley

Jewel was extracted from
[bokuweb/ridley](https://github.com/bokuweb/ridley). Ridley uses the NER
pipelines as one evidence source for reference-field extraction while keeping
contract-specific classification and review logic in Ridley.
