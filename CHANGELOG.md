# Changelog

## v2.0.0 — Serviceable Inference

Released 2026-11-16 as tag `v2.0.0`.

This release turns the V2 foundation into a reusable local inference service
while preserving the V1 benchmark path:

- versioned inference requests, responses, stream events, and error envelopes;
- one resident local engine behind a bounded FIFO admission queue;
- context/token budgets, idempotency, cancellation, and request deadlines;
- durable SQLite job state with explicit restart recovery;
- correlated observability events, phase metrics, and redaction;
- offline LLMOps evaluation and provenance gates;
- `/api/v2` Demo/API integration with resident lifecycle and terminal-state UI;
- explicit V1 cold versus V2 resident protocol boundaries.

Validation for the release:

- V2 foundation: 38/38 targeted tests;
- Demo/API integration: 7/7 tests;
- full regression: 182/182 tests;
- Python compilation, JavaScript syntax, and diff checks passed.

GPU inference was not run as part of this release validation. Runtime GPU
compatibility remains an environment-specific deployment check.

## v1.0.0 — Local Inference Lab V1

See the [V1 release](https://github.com/DS-Jerry-in-Taiwan/modular-llm-serving/releases/tag/v1.0.0)
for the reproducible Qwen benchmark and preserved Modal/Gemma deployment path.
