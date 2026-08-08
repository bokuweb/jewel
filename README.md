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
- rule-based sentence segmentation with exported spaCy `sentencizer` settings
- trainable sentence segmentation with spaCy `senter` models
- post-NER spaCy `entity_ruler` phrase matching across lexical, Boolean,
  sentence, whitespace, and upstream entity attributes
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

## Workspace crates

The repository is organized as three crates:

- `jewel-core`: Python-free spaCy bundle, tokenizer, Thinc, and NER runtime
- `jewel-transformers`: contextual encoder contracts and a native Candle
  Electra CPU backend
- `jewel-ginza`: GiNZA bundle validation and ENE label adaptation

Use `jewel-core` for the existing model runtime:

```toml
[dependencies]
jewel-core = { git = "https://github.com/bokuweb/jewel.git", tag = "0.0.5" }
```

Pin a release tag or a tested commit with `rev` for reproducible builds.

Applications retaining the previous `jewel::` import path can alias the new
package:

```toml
[dependencies]
jewel = { package = "jewel-core", git = "https://github.com/bokuweb/jewel.git", tag = "0.0.5" }
```

GiNZA applications can add the adapter independently. Enable `transformers`
only when the native Electra runtime is needed:

```toml
[dependencies]
jewel-ginza = { git = "https://github.com/bokuweb/jewel.git", tag = "0.0.5" }
# For ja_ginza_electra:
# jewel-ginza = { git = "https://github.com/bokuweb/jewel.git", rev = "<tested-commit>", features = ["transformers"] }
```

`jewel-ginza::GinzaPipeline` loads standard CNN GiNZA bundles exported with
their Sudachi tokenizer. It preserves the raw ENE label and adds an
extraction-oriented coarse label:

```rust
use jewel_core::Bundle;
use jewel_ginza::GinzaPipeline;

let bundle = Bundle::load("/path/to/ja_ginza.spacy-rs")?;
let pipeline = GinzaPipeline::load(&bundle)?;
for entity in pipeline.extract_entities("山田太郎は株式会社青空と契約した。")? {
    println!(
        "{}\t{:?}\t{}",
        entity.ene_label(),
        entity.coarse_label,
        entity.entity.text,
    );
}
```

GiNZA bundles also preserve the model package's complete ENE-to-OntoNotes
mapping. Use `extract_entities_ontonotes` for mapped spans or
`token_labels_ontonotes` for token-aligned `B-`, `I-`, and `O` labels:

```rust
for entity in pipeline.extract_entities_ontonotes(
    "山田太郎は株式会社青空と契約した。",
)? {
    println!("{}\t{}", entity.label, entity.text);
}
```

This exported mapping follows the installed GiNZA package, including its
`OTHERS` fallback. Raw `GinzaEntity` results also use the exported mapping to
fill meaningful coarse labels such as `PRODUCT` and `QUANTITY`. The separate
`coarse_label` helper remains the extraction-oriented override for labels such
as `ADDRESS` and `TITLE`; labels mapped to `OTHERS` keep a `None` coarse label.
Both standard and Electra pipelines expose
`extract_entities_ontonotes_batch` and
`extract_entities_ontonotes_batch_with_constraints`; mapped document order and
entity offsets are identical to the corresponding raw ENE batch.
`token_labels_ontonotes_batch` and its constrained counterpart return one
token-aligned `B-`/`I-`/`O` vector per input document.

The same standard-model flow is available as an example:

```bash
cargo run -p jewel-ginza --example extract_entities -- \
  "$JEWEL_GINZA_BUNDLE" \
  "山田太郎は株式会社青空と契約した。"
```

`jewel-transformers::CandleElectraEncoder` executes GiNZA 5.2 Electra without
Python or PyTorch. It reproduces SudachiTra split-mode-A tokenization,
`dictionary_and_surface` word forms, WordPiece alignment, strided transformer
windows, bounded batched inference, and mean pooling back to Jewel tokens:

```rust
use jewel_core::Bundle;
use jewel_ginza::{CandleElectraEncoder, GinzaElectraPipeline};

let bundle = Bundle::load("/path/to/ja_ginza_electra.spacy-rs")?;
let encoder = CandleElectraEncoder::load(&bundle)?;
let pipeline = GinzaElectraPipeline::load(&bundle, encoder)?;
for entity in pipeline.extract_entities(
    "株式会社リドリーの山田太郎です。違約金は金100万円とします。",
)? {
    println!(
        "{}\t{:?}\t{}",
        entity.ene_label(),
        entity.coarse_label,
        entity.entity.text,
    );
}
```

The encoder evaluates up to eight overlapping spans in each Candle forward
pass by default. Memory-constrained applications can select a smaller bounded
batch with
`CandleElectraEncoder::load_with_span_batch_size(&bundle, batch_size)`.
The Electra pipeline also executes exported post-NER `entity_ruler` components,
including their overwrite behavior and pattern IDs. Both GiNZA pipeline types
expose `has_entity_ruler`, `supported_entity_labels`, `supports_entity_label`,
and `select_entity_labels` for inspecting the combined statistical and ruler
label set. Parser-less Electra bundles run an exported trainable `senter`, an
exported rule-based `sentencizer`, or a document-start fallback with the same
precedence as the standard pipeline. The encoder is CPU-only in this initial
implementation. It is isolated in `jewel-transformers`, so applications using
`jewel-core` or standard GiNZA do not compile Candle.

### Export standard GiNZA

GiNZA 5.2.0 is exported with its spaCy 3.7 generation and Sudachi tokenizer:

```bash
uv run \
  --with "spacy==3.7.5" \
  --with "ginza==5.2.0" \
  --with "ja-ginza==5.2.0" \
  --with safetensors \
  --with click \
  python tools/export_spacy_model.py \
  ja_ginza \
  /path/to/ja_ginza.spacy-rs \
  --profile ner \
  --japanese-tokenizer sudachi
```

The extraction profile retains `tok2vec`, `parser`, and `ner`, resolves
GiNZA's wildcard `Tok2VecListener` to the concrete shared encoder, and rejects
ambiguous wildcard graphs. The checked-in GiNZA corpus covers 14 Japanese
contract and signature cases and 37 entities with exact spaCy/Jewel agreement
for ENE label, text, token span, and Unicode character offsets.

Jewel can also apply the preset entity annotations accepted by spaCy's NER
transition system:

```rust
use jewel_core::{CharSpanAlignment, EntityConstraint};

let entities = pipeline.extract_entities_with_constraints(
    "東京と大阪",
    &[
        EntityConstraint::Entity {
            start: 0,
            end: 1,
            label: "City".to_owned(),
        },
        EntityConstraint::Blocked { start: 1, end: 2 },
        EntityConstraint::OutsideChars {
            start: 3,
            end: 5,
            alignment: CharSpanAlignment::Strict,
        },
    ],
)?;
```

`Entity`, `Blocked`, `Missing`, and `Outside` ranges are token indexes after the
model's tokenizer. Their `*Chars` counterparts accept Unicode character
offsets with spaCy-compatible `Strict`, `Contract`, or `Expand`
`Doc.char_span` alignment.
Unaligned or empty character ranges return an error instead of silently
constraining the wrong tokens. `Entity` forces the matching BILUO path and
`Blocked` prevents an entity at that span, while `Missing` clears the preset
annotation. `Outside` reproduces spaCy's preset-O behavior: the statistical
NER component may replace it. `apply_entity_constraints_with_default` also
supports spaCy's `blocked`, `missing`, `outside`, and `unmodified` defaults for
uncovered tokens. The same explicit-constraint API is available on standard
GiNZA, GiNZA Electra, and the generic English/Japanese NER pipelines. The
standard GiNZA example accepts `START:END:LABEL`, using `-` for blocked, `?`
for missing, and `O` for outside:

```bash
cargo run -p jewel-ginza --example extract_entities -- \
  /path/to/ja_ginza.spacy-rs \
  "東京と大阪" \
  0:1:City 1:2:- 2:3:?
```

### Export GiNZA Electra

The Electra exporter retains `transformer`, `parser`, `ner`, and post-NER
`entity_ruler` components, resolves wildcard `TransformerListener` references,
exports Hugging Face config, WordPiece vocabulary, and safetensors, and omits
Python-specific serialized transformer state:

```bash
uv run \
  --python 3.11 \
  --with "spacy==3.7.5" \
  --with "ginza==5.2.0" \
  --with "ja-ginza-electra==5.2.0" \
  --with "numpy==1.26.4" \
  --with "click>=8.1,<8.2" \
  --with safetensors \
  python tools/export_spacy_model.py \
  ja_ginza_electra \
  /path/to/ja_ginza_electra.spacy-rs \
  --profile ner \
  --japanese-tokenizer sudachi
```

The exported model is large: the Electra weights are approximately 414 MiB
and the bundled Sudachi dictionary is approximately 207 MiB. Export-time
validation loads the tokenizer, transformer contract, parser or sentence
boundary component, NER scorer, and every post-NER entity ruler through the
same component loader used for inference, without running Python.
Each parser, NER, or externally encoded `senter` must contain a transformer
listener whose exported upstream name matches the bundle's single transformer
component, including pipelines that use custom component names.

Run native extraction and the checked-in contract parity corpus with:

```bash
cargo run -p jewel-ginza --features transformers \
  --example extract_electra -- \
  /path/to/ja_ginza_electra.spacy-rs \
  "株式会社リドリーの山田太郎です。違約金は金100万円とします。"

cargo run -p jewel-ginza --features transformers \
  --example electra_parity -- \
  /path/to/ja_ginza_electra.spacy-rs \
  tests/fixtures/ja_ginza_electra_ner_parity.json
```

The current corpus covers 10 contract, contact, address, and multiline
signature cases with 21 entities. Native Candle inference exactly matches
`ja_ginza_electra` 5.2.0 for token text, ENE label, entity text, token span,
and Unicode character offsets.

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

The `ner` profile keeps the single component whose factory is `ner`, regardless
of its instance name. It also keeps the upstream `tok2vec` and `parser`
components when the source pipeline has a dependency parser. A self-contained
NER component can therefore be exported and executed without a parser; Jewel
treats the complete document as one sentence in that case.
For efficient spaCy configs backed by `Tok2VecListener`, the profile retains
the listener's named upstream component and Jewel shares its output directly
with NER and `senter`. Custom names such as `encoder`, `sentence_model`, and
`entities` are preserved in the bundle instead of being normalized to spaCy's
factory names.
When a parser-less source pipeline has a rule-based `sentencizer`, the profile
also retains its serialized terminal characters and overwrite policy. Jewel
then reproduces those sentence boundaries before NER instead of treating the
complete document as one sentence.
Parser-less trainable `senter` components are also retained. Jewel executes
their private `HashEmbedCNN` encoder or a shared upstream `Tok2VecListener`,
applies the two-class `I`/`S` classifier, and preserves the exported overwrite
policy.
Post-NER `entity_ruler` components are retained for supported phrase and token
patterns. Phrase patterns support `ORTH`/`TEXT`, `LOWER`, `NORM`, `SHAPE`,
`LENGTH`, lexical Boolean attributes, `SENT_START`/`IS_SENT_START`, `SPACY`,
and upstream `ENT_IOB`, `ENT_TYPE`, `ENT_ID`, and `ENT_KB_ID` annotations.
Token patterns support:

- `TEXT`/`ORTH`, `LOWER`, `NORM`, `PREFIX`, `SUFFIX`, `SHAPE`, `LEMMA`, `POS`,
  `TAG`, `DEP`, `MORPH`, `ENT_TYPE`, and `ENT_ID` equality
- `ENT_KB_ID` equality for entities annotated by an upstream knowledge base
- `IN` and `NOT_IN` comparisons for those string attributes
- `IS_SUBSET`, `IS_SUPERSET`, and `INTERSECTS` set relations for scalar string
  attributes
- `MORPH` `IS_SUBSET`, `IS_SUPERSET`, and `INTERSECTS` feature-set comparisons
- `ENT_IOB` equality and `IN`/`NOT_IN` comparisons using `B`, `I`, and `O`
- direct `REGEX` comparisons and nested `IN`/`NOT_IN` regex sets for
  `TEXT`/`ORTH`, `LOWER`, `PREFIX`, `SUFFIX`, and `SHAPE`
- direct `FUZZY` and `FUZZY1` through `FUZZY9` comparisons, including nested
  `IN`/`NOT_IN` candidate sets, for `TEXT`/`ORTH`, `LOWER`, `PREFIX`, `SUFFIX`,
  and `SHAPE`
- `LENGTH` equality, `IN`/`NOT_IN`, and `==`, `!=`, `>=`, `<=`, `>`, and `<`
  comparisons
- `IS_ALPHA`, `IS_ASCII`, `IS_BRACKET`, `IS_CURRENCY`, `IS_DIGIT`,
  `IS_LEFT_PUNCT`, `IS_LOWER`, `IS_PUNCT`, `IS_QUOTE`, `IS_RIGHT_PUNCT`,
  `IS_SPACE`, `IS_STOP`, `IS_TITLE`, `IS_UPPER`, `LIKE_EMAIL`, `LIKE_NUM`, `LIKE_URL`,
  `IS_SENT_START`/`SENT_START`, and `SPACY`
- the default single-token match, `!`, `?`, `*`, and `+`, plus bounded
  repetition with `{n}`, `{n,m}`, `{n,}`, and `{,m}`
- wildcard token objects such as `{}` and `{"OP": "?"}`

Constraints in the same token object are combined with AND. Jewel preserves
spaCy's longest-first overlap resolution and `overwrite_ents` behavior.
Optional EntityRuler pattern `id` values are exported for phrase and token
patterns, attached to every matched token as `ENT_ID`, and returned as
`NamedEntity::ent_id`; entities from patterns without an ID return `None`. These
rules can add known counterparties and people or recognize structured evidence
such as amounts, addresses, email addresses, and phone numbers after
statistical NER. Entity rulers placed before NER and unsupported attributes,
comparisons, or quantifiers are rejected during export instead of being
silently approximated. In particular, extraction profiles do not retain the
components required to produce reliable `LEMMA`, `POS`, or `TAG` constraints.

For example, a source pipeline can add structured extraction evidence before
it is exported:

```python
ruler = nlp.add_pipe("entity_ruler", after="ner")
ruler.add_patterns(
    [
        {
            "label": "MONEY",
            "pattern": [
                {"IS_CURRENCY": True, "OP": "?"},
                {"LIKE_NUM": True},
                {"LOWER": {"IN": ["yen", "円"]}, "OP": "?"},
            ],
        },
        {
            "label": "EMAIL",
            "pattern": [{"LIKE_EMAIL": True}],
        },
        {
            "label": "PHONE",
            "pattern": [
                {"TEXT": {"REGEX": r"^\d{2,4}-\d{2,4}-\d{4}$"}},
            ],
        },
        {
            "label": "POSTAL_CODE",
            "pattern": [
                {
                    "SHAPE": "ddd-dddd",
                    "PREFIX": "1",
                    "LENGTH": {">=": 8},
                }
            ],
        },
        {
            "label": "URL",
            "pattern": [{"LIKE_URL": True}],
        },
        {
            "label": "KNOWN_PARTY",
            "pattern": [
                {
                    "LOWER": {
                        "FUZZY1": {
                            "IN": ["acme", "globex", "株式会社サンプル"]
                        }
                    }
                }
            ],
        },
        {
            "label": "ADDRESS_NUMBER",
            "pattern": [
                {"IS_DIGIT": True, "OP": "{1,3}"},
                {"TEXT": "丁目"},
            ],
        },
        {
            "label": "PARTY_WITH_SUFFIX",
            "pattern": [
                {"ENT_TYPE": "ORG", "ENT_IOB": "B"},
                {"ENT_TYPE": "ORG", "ENT_IOB": "I", "OP": "*"},
                {"LOWER": {"IN": ["ltd", "inc", "株式会社"]}},
            ],
        },
        {
            "label": "SIGNATURE_CONTEXT",
            "pattern": [
                {"LOWER": {"IN": ["signed", "署名"]}},
                {"OP": "?"},
                {"ENT_TYPE": "PERSON", "OP": "+"},
            ],
        },
    ]
)
```

`FUZZY` uses spaCy's default threshold of at least two edits or 30% of the
pattern length. For short personal names, company suffixes, and identifiers,
prefer an explicit threshold such as `FUZZY1` to avoid overly broad matches.
Nested fuzzy and regex sets use spaCy's predicate-first form, for example
`{"LOWER": {"FUZZY1": {"IN": ["acme", "globex"]}}}` and
`{"TEXT": {"REGEX": {"NOT_IN": ["^test-", "^dummy-"]}}}`.

The default `full` profile exports all source components, but the Rust runtime
can execute only the component types and architectures documented above.

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
jewel-core = { git = "https://github.com/bokuweb/jewel.git", default-features = false, features = ["sudachi-tokenizer"] }
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
| `sentence_boundaries` | Japanese or English | Inspect parser, senter, or sentencizer sentence starts |
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

Inspect the sentence annotations consumed by NER:

```bash
cargo run --example sentence_boundaries -- \
  "$JEWEL_EN_BUNDLE" \
  "Alice works at Acme. Bob works at Example Corp."
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
use jewel_core::NerCompatibilityReport;

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
use jewel_core::{Bundle, BundleLimits};

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
let filter = jewel_core::EntityLabelFilter::new(&[
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

Each document can carry different token-index or Unicode character constraints:

```rust
use jewel_core::{
    CharSpanAlignment, EntityConstraint, EntityConstraintDefault, NerBatchInput,
};

let first = [
    EntityConstraint::Default(EntityConstraintDefault::Missing),
    EntityConstraint::EntityChars {
        start: 0,
        end: 6,
        label: "Company".to_owned(),
        alignment: CharSpanAlignment::Strict,
    },
];
let second = [EntityConstraint::BlockedChars {
    start: 0,
    end: 4,
    alignment: CharSpanAlignment::Expand,
}];
let inputs = [
    NerBatchInput::new("株式会社青空と契約した。", &first),
    NerBatchInput::new("山田太郎が署名した。", &second),
];
let batches = pipeline.extract_entities_batch_with_constraints(&inputs)?;
```

The same constrained batch API is exposed by `NerPipeline`, standard
`GinzaPipeline`, and `GinzaElectraPipeline`. Core and standard GiNZA batches
reuse one tokenizer session. Electra batches call the backend-neutral
`TransformerEncoder::encode_batch` hook; its default preserves compatibility
by encoding documents in order, while accelerated backends may override it.

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

Use `--json-input` to send one JSON string per input line. This framing
preserves embedded newlines in contract and email signature blocks.

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
use jewel_core::{Bundle, NerPipeline};

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
use jewel_core::{Bundle, JapaneseNerPipeline};

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
use jewel_core::{Bundle, EnglishNerPipeline};

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

use jewel_core::{
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
use jewel_core::{Bundle, JapaneseNerPipeline};

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

The manual `Model compatibility` GitHub Actions workflow runs English,
`ja_core_news_sm`, and standard GiNZA matrices and uploads a JSON report for
each model. Model packages are downloaded during the workflow and are not
committed or uploaded as artifacts.

For transition-level diagnosis, `tok2vec_json` emits the shared encoder matrix
and `pipeline_json` emits token, dependency, sentence-boundary, and NER
annotations. Standard GiNZA diagnostics must select the Sudachi feature:

```bash
cargo run --no-default-features --features sudachi-tokenizer \
  --example tok2vec_json -- "$JEWEL_GINZA_BUNDLE" tok2vec < input.txt
cargo run --no-default-features --features sudachi-tokenizer \
  --example pipeline_json -- "$JEWEL_GINZA_BUNDLE" < input.txt
```

### Regenerate sentence-boundary fixtures

The checked-in `sentencizer` and `senter` fixtures record spaCy's exact
annotation behavior, including unset boundaries, overwrite behavior, and empty
documents. Generate either fixture with the matching spaCy environment:

```bash
python tools/generate_sentence_boundary_fixtures.py sentencizer
python tools/generate_sentence_boundary_fixtures.py senter
```

The phrase and token ruler fixtures record spaCy's overlap, attribute,
comparison, fuzzy matching, bounded repetition, lexical-attribute, shape,
length, URL, and overwrite behavior:

```bash
python tools/generate_entity_ruler_fixture.py
python tools/generate_entity_ruler_token_fixture.py
```

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
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p jewel-core --all-targets --no-default-features
cargo check -p jewel-core --all-targets --no-default-features --features sudachi-tokenizer
cargo test --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
python -m unittest discover -s tests -p "test_*.py"
```

The checked-in fixtures cover spaCy 3.8 string hashes, `DocBin` decoding, and
selected Thinc operations. The JSONL compatibility corpora exercise exact
end-to-end Japanese and English NER parity without checking model bundles into
the repository.
