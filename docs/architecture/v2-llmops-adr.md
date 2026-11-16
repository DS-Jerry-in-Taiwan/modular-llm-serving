# V2 LLMOps foundation decision

The LLMOps module owns immutable model/config/dataset/evaluator/run lineage and the evaluation decision. It consumes versioned manifests and reports through data contracts and does not import the serving implementation. The first registry is a local append-only JSONL index; raw evidence remains in its run directory and is never rewritten by the index.

A report can pass only when its schema, run identity, manifest digest, quality score, latency ratio, GPU threshold, and provenance status satisfy the configured gate. Missing or tampered evidence fails closed.

This slice does not provide a remote registry, dashboard, promotion API, or distributed locking. Concurrent writers and durable cross-machine storage are follow-up work.
