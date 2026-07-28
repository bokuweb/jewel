# Roadmap

Jewel is an extraction-focused, Python-free compatibility runtime. The roadmap
prioritizes dependable Japanese and English entity inference for native Rust
applications rather than broad spaCy API coverage.

The workspace separates the generic runtime (`jewel-core`), backend-neutral
contextual encoder contracts (`jewel-transformers`), and GiNZA-specific model
and label adaptation (`jewel-ginza`). Heavy transformer engines remain outside
the core dependency graph.

## Completed foundation

- spaCy-compatible document offsets, string hashing, and covered `DocBin`
  attributes
- English regex tokenization and optional Japanese Sudachi tokenization
- default delarocha/Vibrato Japanese tokenization with explicit IPADIC feature
  adaptation, exported contract-term boundaries, formatted-number merging, and
  compatibility measurement
- the Thinc operations required by the tested small model pipelines
- `tok2vec`, tagger, dependency parser, and NER inference
- validated safetensors-based model bundles with no Python runtime dependency
- Japanese and English extraction-only pipelines
- parser-optional extraction pipelines for both self-contained NER components
  and upstream `Tok2VecListener` configurations
- standard GiNZA 5.2 CNN export and inference through Sudachi, including
  wildcard listener resolution and exact ENE parity on the contract and
  multiline signature corpus
- spaCy-compatible parser-derived sentence boundaries and BILUO whitespace
  transitions for multiline entity inference
- workspace isolation for the core runtime, transformer contracts, and GiNZA
  label/model adaptation
- rule-based `sentencizer` execution with exported custom terminal characters
  and overwrite behavior
- trainable `senter` execution with private or upstream tok2vec encoders
- factory-based extraction component discovery with preserved custom instance
  names and exported `Tok2VecListener` upstream relationships
- post-NER `entity_ruler` phrase matching for `ORTH`, `LOWER`, and `NORM`, plus
  extraction-oriented token rules with string comparisons, regular
  expressions and regex sets, bounded Unicode fuzzy matching against direct
  values or candidate sets, lexical Boolean attributes, upstream entity
  attributes, wildcard tokens, shape and length constraints, and simple or
  bounded repetition operators, with spaCy-compatible overlap and overwrite
  behavior
- language-aware pipeline loading, symmetric batch inference, and serializable
  entity spans
- repeatable Japanese and English spaCy-to-Jewel NER parity checks with
  versioned input corpora and machine-readable reports
- versioned compatibility reports with stable diagnostics for bundle,
  tokenizer, tensor, component, and graph-node failures
- export-time Rust runtime validation enabled by default, with structured
  incompatibility reporting
- configurable pre-allocation limits for manifest, weights, tokenizer
  configuration, component state, graph metadata, and tensor metadata
- bounded `DocBin` compressed and decompressed payloads, decoded collection
  counts, and per-document metadata
- tok2vec lexical layouts with `ORTH`, `LOWER`, `NORM`, `PREFIX`, `SUFFIX`,
  `SHAPE`, `LENGTH`, `SPACY`, and `IS_SPACE`, plus graph-derived CNN width,
  depth, and window size

## Priority 0: extraction reliability

- Fuzz manifest, tokenizer, `DocBin`, and tensor metadata parsing.
- Persist reviewed golden reports for supported spaCy and model-package
  versions, including model source, license, and checksum metadata.
- Expand the Japanese contract corpus and keep a reviewed acceptance gate for
  the default delarocha backend.

## Priority 1: production operation

- Benchmark cold bundle loading, warm single-document inference, batch
  inference, memory use, and Japanese dictionary startup cost.
- Define and test the concurrency contract for sharing pipelines across worker
  threads or using a bounded pipeline pool.
- Add runtime and bundle compatibility negotiation with actionable upgrade
  errors.
- Add optional tracing hooks for load time, tokenization, neural inference, and
  entity counts without recording document text.
- Document deployment packaging, model provenance, checksum verification, and
  rollback practices.

## Priority 2: selective compatibility

Add components or Thinc graph operations only when a supported extraction model
requires them. Each addition should include Python-generated golden fixtures
and an end-to-end model test.

Potential candidates include:

- additional transition-system configurations
- static vector configurations used by medium-sized models
- narrowly scoped rule-based components needed to preserve extraction output

## Downstream responsibilities

Jewel returns model-defined entity evidence. Application-specific semantics
belong in downstream projects such as Ridley:

- contract amount, penalty, deposit, fee, and damages classification
- party-role and counterparty assignment
- deterministic email, phone, postal-code, and account-number patterns
- conflict resolution, confidence policy, review workflow, and audit evidence
- domain corpus annotation and precision/recall acceptance gates

## Non-goals

- model training or fine-tuning in Rust
- compatibility with spaCy's Python registry, plugin, extension, or callback
  APIs
- automatic support for every spaCy model or third-party pipeline
- silently approximating unsupported graph operations
