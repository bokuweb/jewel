# Roadmap

Jewel is an extraction-focused, Python-free compatibility runtime. The roadmap
prioritizes dependable Japanese and English entity inference for native Rust
applications rather than broad spaCy API coverage.

The workspace separates the generic runtime (`jewel-core`), contextual encoder
contracts and the optional native Candle engine (`jewel-transformers`), and
GiNZA-specific model and label adaptation (`jewel-ginza`). Transformer
dependencies remain outside the core dependency graph.

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
- GiNZA 5.2 Electra export with Hugging Face safetensors and WordPiece assets,
  wildcard transformer listener resolution, native Candle CPU inference,
  SudachiTra-compatible alignment and pooling, bounded batched execution of
  overlapping spans, and exact ENE parity on the initial contract, contact,
  address, and signature corpus
- transition scorer support for spaCy parser models whose precomputable affine
  layer is followed by `noop`
- spaCy-compatible preset entity, blocked-span, missing-span, and outside-span
  annotations for standard GiNZA, GiNZA Electra, and generic English/Japanese
  NER, including all `Doc.set_ents` defaults and Unicode character ranges with
  strict, contract, or expand `Doc.char_span` alignment
- exported GiNZA ENE-to-OntoNotes label mappings for span and token-aligned
  output, including standard and Electra batch span and token-label extraction,
  with post-NER labels falling back to `OTHERS`
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
  attributes including linguistic `LEMMA`/`POS`/`TAG`/`DEP`/`MORPH` values and
  `ENT_ID`/`ENT_KB_ID`, sentence-start and trailing-space conditions, wildcard
  tokens, bracket/quote direction flags, shape and length constraints, and
  simple or bounded repetition operators, with spaCy-compatible overlap and
  overwrite behavior, including phrase/token pattern IDs exposed through token
  `ENT_ID` and extracted entity metadata
- language-aware pipeline loading, symmetric batch inference with
  document-specific token or character constraints, standard GiNZA batch
  adaptation, an overridable GiNZA Electra encoder batch boundary, and
  serializable entity spans
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

- Benchmark and optimize native Electra cold loading, CPU inference, strided
  long-document execution, and memory use; evaluate optional Metal and CUDA
  backends without adding them to `jewel-core`.
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
