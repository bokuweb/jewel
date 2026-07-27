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
- delarocha/Vibrato tokenization as the default Japanese runtime
- optional Sudachi tokenization for compatibility investigations
- selected Thinc-compatible neural operations
- `tok2vec`, fine-grained tagger, transition-based dependency parser, and NER
- manifest-ordered tok2vec lexical columns and graph-derived convolution width,
  depth, and window size
- extraction-only Japanese and English NER pipelines
- language-aware NER loading, batch inference, and serializable entity output
- spaCy `DocBin` compatibility for the covered attributes
- model bundle validation and safetensors weight loading

Not implemented:

- model training or fine-tuning in Rust
- spaCy's Python pipeline, registry, extension, and plugin APIs
- every spaCy component, architecture, language, or third-party model
- automatic compatibility with model graphs that use unsupported Thinc nodes

See [ROADMAP.md](ROADMAP.md) for the extraction-focused priorities and explicit
non-goals.

Model packages and their dictionaries are not distributed in this repository.
Review the license and redistribution terms of every model and dictionary that
you export.

## Add the crate

The crate is currently consumed directly from Git:

```toml
[dependencies]
jewel = { git = "https://github.com/bokuweb/jewel.git", tag = "0.0.3" }
```

Pin a release tag or a tested commit with `rev` for reproducible builds.

Release `0.0.2` renamed the Rust package and library from `jewel_spacy` to
`jewel`. Update both the Cargo dependency key and Rust `use` paths when
upgrading from `0.0.1`.

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
mkdir -p target/vibrato-dic
curl -fsSL \
  https://github.com/daac-tools/vibrato/releases/download/v0.5.0/ipadic-mecab-2_7_0.tar.xz \
  | tar -xJ -C target/vibrato-dic
export DELAROCHA_SYSTEM_DIC="$PWD/target/vibrato-dic/ipadic-mecab-2_7_0/system.dic.zst"
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

The exporter validates every generated bundle by loading it with Jewel's Rust
NER runtime. The command fails with a structured compatibility diagnostic when
the tokenizer, component graph, attributes, or tensors cannot be loaded.
Rust and Cargo are therefore required in the build environment as well as
Python. Use `--runtime-manifest-path` when validating against another Jewel
checkout:

```bash
python tools/export_spacy_model.py \
  en_core_web_sm \
  /path/to/en_core_web_sm.spacy-rs \
  --profile ner \
  --runtime-manifest-path /path/to/jewel/Cargo.toml
```

`--skip-runtime-validation` is available only for diagnosing exporter output.
Do not deploy a bundle produced with validation skipped.

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
└── tokenizer/delarocha/  # Japanese bundles only
```

Keep the complete directory together when deploying a bundle.

### Default Japanese tokenizer

Jewel `0.0.4` defaults to a delarocha-compatible Vibrato dictionary whose
features use the IPADIC layout. Pass the dictionary explicitly or set
`DELAROCHA_SYSTEM_DIC`:

```bash
python tools/export_spacy_model.py \
  ja_core_news_sm \
  /path/to/ja_core_news_sm.spacy-rs \
  --profile ner \
  --delarocha-dictionary /path/to/system.dic.zst
```

The default crate features include only the delarocha Japanese backend. Enable
the larger Sudachi fallback explicitly when investigating compatibility:

```toml
[dependencies]
jewel = { git = "https://github.com/bokuweb/jewel.git", default-features = false, features = ["sudachi-tokenizer"] }
```

The exporter copies the dictionary into `tokenizer/delarocha/` and records its
SHA-256 checksum. Jewel validates the checksum before loading it. The source
dictionary must be a Vibrato `system.dic` or `system.dic.zst`; a Sudachi
`system.dic` is a different binary format and cannot be used directly.

delarocha changes token boundaries for compounds and formatted numbers compared
with Sudachi. The exporter records Sudachi-derived token attributes for a
focused contract vocabulary, including company types, officer titles, contract
amounts, penalties, deposits, damages, fees, and counterparties. The Rust
adapter also rejoins comma- and decimal-formatted ASCII numbers and the narrow
`〜町1丁目` address pattern. These rules restore exact output on the checked-in
Japanese smoke corpus with spaCy 3.8.7 and `ja_core_news_sm` 3.8.0, but they are
not a general replacement for Sudachi segmentation. Production users should
still run the application's full contract corpus. Review the dictionary's
license before redistributing an exported bundle.

The runtime consumes delarocha's borrowed worker token views, reads IPADIC
features without an intermediate field vector, performs borrowed compatibility
lookups, and moves unchanged tokens through the normalization passes. Use the
`benchmark_tokenizer` example below to measure the complete warm tokenization
path with an application-specific bundle and corpus.

## Examples

The repository includes examples for model validation, one-shot extraction,
batch processing, streaming JSONL output, and Unicode offset conversion.

| Example | Bundle required | Purpose |
| --- | --- | --- |
| `inspect_bundle` | any Jewel bundle | Validate and summarize a bundle |
| `tokenize` | any Jewel bundle | Inspect token text and Unicode offsets |
| `benchmark_tokenizer` | any Jewel bundle | Measure warm tokenizer throughput |
| `benchmark_ner` | Japanese or English | Measure cold loading and warm NER throughput |
| `extract_entities_ja` | Japanese | Extract every model-defined entity |
| `extract_people_ja` | Japanese | Extract only `PERSON` entities |
| `extract_signature_entities` | Japanese or English | Extract identity and location labels used by signature analysis |
| `batch_entities` | Japanese or English | Reuse an auto-selected pipeline |
| `batch_entities_ja` | Japanese | Reuse one pipeline across many documents |
| `extract_entities_en` | English | Extract every model-defined entity |
| `entities_jsonl` | Japanese or English | Process stdin as a JSONL worker |
| `unicode_offsets` | none | Convert spaCy character offsets for Rust slicing |

Inspect the token boundaries selected by a bundle:

```bash
cargo run --example tokenize -- \
  "$JEWEL_JA_BUNDLE" \
  "違約金1,200,000円を支払う。"
```

Measure a request-local tokenizer session after bundle loading and warm-up:

```bash
cargo run --release --example benchmark_tokenizer -- \
  "$JEWEL_JA_BUNDLE" \
  10000 \
  "違約金1,200,000円を支払う。"
```

Measure bundle loading, pipeline construction, and warm end-to-end NER
separately:

```bash
cargo run --release --example benchmark_ner -- \
  "$JEWEL_JA_BUNDLE" \
  1000 \
  "甲株式会社の代表取締役山田太郎は違約金1,200,000円を支払う。" \
  32
```

The optional final argument is the batch size. Batch measurements reuse one
request-local tokenizer session and report normalized microseconds per
document.

Set paths once for the commands below:

```bash
export JEWEL_JA_BUNDLE=/path/to/ja_core_news_sm.spacy-rs
export JEWEL_EN_BUNDLE=/path/to/en_core_web_sm.spacy-rs
```

### Inspect a bundle

`inspect_bundle` validates the bundle, initializes its tokenizer, and constructs
the language-aware NER pipeline without running inference. It prints source
model metadata, tokenizer type, component counts, and tensor counts.

```bash
cargo run --example inspect_bundle -- "$JEWEL_JA_BUNDLE"
```

Use `--json` in CI or deployment tooling. The versioned report contains stable
diagnostic codes and identifies the affected component, graph node, attribute,
tensor, tokenizer feature, or language when available. The command exits with
status 1 for an incompatible bundle.

```bash
cargo run --example inspect_bundle -- --json "$JEWEL_JA_BUNDLE"
```

```json
{
  "report_version": 1,
  "compatible": false,
  "bundle_path": "/path/to/model.spacy-rs",
  "source": {
    "spacy_version": "3.8.13",
    "model_name": "ja_core_news_sm",
    "model_version": "3.8.0",
    "lang": "ja"
  },
  "diagnostics": [
    {
      "code": "unsupported_graph_node",
      "area": "graph_node",
      "component": "tok2vec",
      "node": 12,
      "item": "maxout",
      "message": "node 12 is not a supported maxout node: missing dimension nO"
    }
  ]
}
```

Example output:

```text
bundle: /path/to/ja_core_news_sm.spacy-rs
NER compatible: yes
format version: 1
source: ja_core_news_sm 3.8.0 (spaCy 3.8.13, language ja)
runtime: minimum 0.0.1, requires Python: false
tokenizer: Delarocha (tokenizer.json)
components:
  tok2vec: factory=tok2vec, kind=Trainable, nodes=..., tensors=..., labels=0, moves=0
  parser: factory=parser, kind=Trainable, nodes=..., tensors=..., labels=..., moves=...
  ner: factory=ner, kind=Trainable, nodes=..., tensors=..., labels=..., moves=...
```

Exact model versions and counts depend on the exported package.

The same report is available as a library API:

```rust
use jewel::NerCompatibilityReport;

let report = NerCompatibilityReport::inspect("/path/to/model.spacy-rs");
if !report.compatible {
    for diagnostic in report.diagnostics {
        eprintln!(
            "{}: component={:?} node={:?}: {}",
            diagnostic.code,
            diagnostic.component,
            diagnostic.node,
            diagnostic.message
        );
    }
}
```

### Limit bundle resources

`Bundle::load` applies default limits before reading model files or allocating
from manifest collection sizes. The defaults cover manifest and tokenizer JSON,
weights, component state files, component and graph-node counts, tensor counts,
and tensor rank.

Applications accepting bundles from outside their deployment image can lower
the limits:

```rust
use jewel::{Bundle, BundleLimits};

let limits = BundleLimits {
    max_manifest_bytes: 2 * 1024 * 1024,
    max_weights_bytes: 256 * 1024 * 1024,
    max_components: 16,
    max_nodes_per_component: 4096,
    ..BundleLimits::default()
};
let bundle = Bundle::load_with_limits("/path/to/model.spacy-rs", limits)?;
```

Limit failures use the `bundle_limit_exceeded` compatibility diagnostic and
identify the guarded resource and component when available.

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

For signature analysis, select only identity, organization, title, and location
labels while keeping the same model inference and Unicode offsets:

```bash
cargo run --example extract_signature_entities -- \
  "$JEWEL_JA_BUNDLE" \
  $'【署名欄】\n乙：株式会社青空\n代表取締役：山田太郎\n所在地：東京都千代田区'
```

The corresponding API is
`NerPipeline::extract_entities_by_labels`. Its batch variant reuses one
tokenizer session. Deterministic email, telephone, and postal-code patterns
remain downstream application responsibilities.

For a fixed downstream schema, compile the labels once and reuse the numeric
spaCy string IDs across requests:

```rust
let filter = jewel::EntityLabelFilter::new(&[
    "PERSON", "ORG", "NORP", "GPE", "LOC", "FAC", "TITLE_AFFIX",
]);
let entities = pipeline.extract_entities_with_filter(signature, &filter)?;
```

`extract_entities_with_filter_batch` combines the same filter with one reused
tokenizer session. Duplicate labels are deduplicated and empty labels are
ignored. The label-list methods remain convenient wrappers.

Applications that accept interchangeable model bundles can inspect the
declared NER capabilities before compiling a downstream filter:

```rust
let available = pipeline.supported_entity_labels().collect::<Vec<_>>();
if pipeline.supports_entity_label("PERSON") {
    // Enable person-name enrichment for this model.
}
```

Capability inspection reads exported model metadata and does not run
tokenization or neural inference.

To compile a requested schema and retain unsupported-label diagnostics in one
value, use `select_entity_labels`:

```rust
let selection = pipeline.select_entity_labels(&[
    "PERSON", "ORG", "GPE", "LOC", "TITLE_AFFIX",
]);
let entities = pipeline.extract_entities_with_filter(
    signature,
    selection.filter(),
)?;
println!("enabled: {:?}", selection.selected_labels());
println!("missing: {:?}", selection.missing_labels());
```

The selection preserves first-request order, deduplicates labels, ignores empty
labels, and reports whether the loaded model supports the complete request.

### Process a Japanese batch

Load the delarocha dictionary and neural weights once, then process multiple
documents with the same pipeline:

```bash
cargo run --example batch_entities_ja -- \
  "$JEWEL_JA_BUNDLE" \
  "甲株式会社の代表者は山田太郎です。" \
  "佐藤花子は大阪市の乙株式会社に所属します。"
```

The corresponding library API is `JapaneseNerPipeline::extract_entities_batch`.
Extraction batches discard each intermediate `Doc` after copying its entity
spans, so peak document memory does not grow with the entire batch.

For language-aware batch processing, use `batch_entities` with either bundle:

```bash
cargo run --example batch_entities -- \
  "$JEWEL_EN_BUNDLE" \
  "Acme appointed Jane Smith in London." \
  "John Doe joined Example Incorporated in New York."
```

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

### Language-aware NER

Use `NerPipeline` when an application accepts either Japanese or English model
bundles. The implementation is selected from `manifest.source.lang`.

```rust
use jewel::{Bundle, NerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = Bundle::load("/path/to/model.spacy-rs")?;
    let pipeline = NerPipeline::load(&bundle)?;

    println!("language: {}", pipeline.language().code());
    for entity in pipeline.extract_entities("Acme appointed Jane Smith.")? {
        println!("{}", serde_json::to_string(&entity)?);
    }

    Ok(())
}
```

### Japanese NER

```rust
use jewel::{Bundle, JapaneseNerPipeline};

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
use jewel::{Bundle, EnglishNerPipeline};

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

Load a pipeline once and reuse it across requests. Repeated dictionary and
neural-weight loading is unnecessary and expensive.

### Inject a tokenizer from the application layer

Inference pipelines store a `SharedTokenizer` (`Arc<dyn Tokenizer>`). Their
regular `load` constructors use the tokenizer declared by the bundle.
`load_with_tokenizer` lets a higher layer own, wrap, instrument, or replace that
tokenizer without changing the neural pipeline:

```rust
use std::sync::Arc;

use jewel::{
    Bundle, Doc, JapaneseNerPipeline, RuntimeTokenizer, SharedTokenizer,
    TokenizeError, Tokenizer, TokenizerSession,
};

struct ObservedTokenizer {
    inner: RuntimeTokenizer,
}

impl Tokenizer for ObservedTokenizer {
    fn tokenize(&self, text: &str) -> Result<Doc, TokenizeError> {
        self.inner.tokenize(text).map_err(TokenizeError::new)
    }

    fn session(&self) -> Box<dyn TokenizerSession + '_> {
        Tokenizer::session(&self.inner)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bundle = Bundle::load("/path/to/ja_core_news_sm.spacy-rs")?;
    let tokenizer: SharedTokenizer = Arc::new(ObservedTokenizer {
        inner: bundle.load_tokenizer()?,
    });
    let pipeline =
        JapaneseNerPipeline::load_with_tokenizer(&bundle, tokenizer)?;

    for entity in pipeline.extract_entities("山田太郎が契約書に署名した。")? {
        println!("{}\t{}", entity.label, entity.text);
    }
    Ok(())
}
```

The same constructor is available on `NerPipeline`, `EnglishNerPipeline`,
`EnglishPipeline`, and `EnglishTaggerPipeline`. A replacement tokenizer must
produce the token boundaries, attributes, and Unicode code-point offsets
expected by the exported spaCy model; injection does not make incompatible
segmentation safe. `Tokenizer::session` has a delegating default implementation;
stateful backends can override it to retain request-local scratch buffers
without weakening the `Send + Sync` tokenizer contract.

### Batch Japanese NER

```rust
use jewel::{Bundle, JapaneseNerPipeline};

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

### Validate an existing bundle

Run the same Rust loading gate independently for a bundle created earlier:

```bash
python tools/validate_bundle_runtime.py "$JEWEL_JA_BUNDLE"
```

The command selects Cargo features from the tokenizer declared in
`manifest.json`, prints a versioned JSON compatibility report, and exits with
status 1 when the bundle is incompatible.

### Compare Jewel with spaCy

The compatibility harness runs spaCy and Jewel over the same JSONL corpus and
requires exact agreement for entity text, label, token range, and Unicode
code-point range.

To create and evaluate the default Japanese bundle in one command:

```bash
python tools/check_ner_compatibility.py \
  ja_core_news_sm \
  tests/fixtures/ner_compatibility_ja.jsonl \
  --delarocha-dictionary /path/to/system.dic.zst \
  --report /tmp/ja-delarocha-compatibility.json
```

An existing bundle is detected from `manifest.json`, so `--bundle` automatically
selects the Cargo features needed by its tokenizer kind.

The report separates strict mismatches from semantic entity mismatches.
`token_only_mismatch_count` means entity text, label, and character offsets
agree while token indexes differ; `semantic_mismatch_count` means the extracted
entity evidence itself differs.

Use an existing bundle:

```bash
python tools/check_ner_compatibility.py \
  ja_core_news_sm \
  tests/fixtures/ner_compatibility_ja.jsonl \
  --bundle "$JEWEL_JA_BUNDLE" \
  --report /tmp/ja-compatibility.json
```

Omit `--bundle` to export a temporary NER bundle before comparison:

```bash
python tools/check_ner_compatibility.py \
  en_core_web_sm \
  tests/fixtures/ner_compatibility_en.jsonl
```

Use `--work-dir` when the system temporary volume does not have enough free
space for model export:

```bash
python tools/check_ner_compatibility.py \
  ja_core_news_sm \
  tests/fixtures/ner_compatibility_ja.jsonl \
  --work-dir /path/to/large-temporary-volume
```

The manual `Model compatibility` GitHub Actions workflow runs the Japanese and
English matrix and uploads a JSON report for each model. Model packages are
downloaded during the workflow and are not committed or uploaded as artifacts.

### Limit DocBin decoding

`DocBin::from_bytes` limits compressed and decompressed payload sizes before
msgpack decoding and bounds decoded document, token, attribute, string, and
per-document metadata counts. Use `DocBin::from_bytes_with_limits` with a
custom `DocBinLimits` value when reading externally supplied corpora under a
smaller memory budget.

### Local checks

Run the Rust compatibility suite:

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets --no-default-features
cargo check --all-targets --no-default-features --features sudachi-tokenizer
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
python -m unittest discover -s tests -p "test_*.py"
```

The checked-in fixtures cover spaCy 3.8 string hashes, `DocBin` decoding, and
selected Thinc operations. The JSONL compatibility corpora exercise exact
end-to-end Japanese and English NER parity without checking model bundles into
the repository.

## Relationship to Ridley

Jewel was extracted from
[bokuweb/ridley](https://github.com/bokuweb/ridley). Ridley uses the NER
pipelines as one evidence source for reference-field extraction while keeping
contract-specific classification and review logic in Ridley.
