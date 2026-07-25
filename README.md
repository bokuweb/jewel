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
jewel_spacy = { git = "https://github.com/bokuweb/jewel.git" }
```

For reproducible builds, pin a tested commit with `rev`.

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

## Run Japanese NER

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

The repository also includes a command-line example:

```bash
cargo run --example extract_people_ja -- \
  /path/to/ja_core_news_sm.spacy-rs \
  "受託者の山田太郎は本業務を遂行する。"
```

## Run English NER

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
