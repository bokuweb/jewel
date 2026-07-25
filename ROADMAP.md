# Roadmap

Jewel is an extraction-focused, Python-free compatibility runtime. The roadmap
prioritizes dependable Japanese and English entity inference for native Rust
applications rather than broad spaCy API coverage.

## Completed foundation

- spaCy-compatible document offsets, string hashing, and covered `DocBin`
  attributes
- English regex tokenization and Japanese Sudachi tokenization
- the Thinc operations required by the tested small model pipelines
- `tok2vec`, tagger, dependency parser, and NER inference
- validated safetensors-based model bundles with no Python runtime dependency
- Japanese and English extraction-only pipelines
- language-aware pipeline loading, symmetric batch inference, and serializable
  entity spans
- repeatable Japanese and English spaCy-to-Jewel NER parity checks with
  versioned input corpora and machine-readable reports

## Priority 0: extraction reliability

- Produce structured compatibility diagnostics that identify the unsupported
  component, graph node, attribute, tensor, or tokenizer feature.
- Add export-time validation that loads the generated bundle with the Rust
  runtime before the bundle is accepted for deployment.
- Fuzz manifest, tokenizer, `DocBin`, and tensor metadata parsing, with explicit
  allocation and input-size limits for untrusted bundles.
- Persist reviewed golden reports for supported spaCy and model-package
  versions, including model source, license, and checksum metadata.

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

- additional tok2vec feature layouts
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
