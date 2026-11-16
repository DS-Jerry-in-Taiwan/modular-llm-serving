# ADR: V2 local web/API integration

## Decision

The loopback demo exposes V2 through explicit `/api/v2/*` routes and a
resident execution mode.  V1 remains the default cold-process comparison
protocol under `/api/*`; the two wire contracts cannot be selected by the same
request.  Operators select the mode with `--execution-mode cold|resident`.

Resident requests are mapped to the versioned `InferenceRequest` contract,
admitted by the process-local bounded `AdmissionQueue`, and executed through a
single `ResidentSupervisor` engine.  The engine is started once and reused for
subsequent requests; shutdown occurs only after queued and active work has
reached terminal state.

## State and failure boundaries

`DurableJobStore` records a request before admission.  A request is published
as terminal only after its runner and cleanup callback return.  Successful
responses write immutable local evidence and include its digest in the durable
terminal record.  Restart recovery fails queued/running records under the
store's `fail_incomplete_v1` policy.

The API returns bounded error envelopes and never serializes prompts, raw
provider errors, credentials, or runtime paths.  Queue-full, cancellation, and
deadline outcomes remain distinct.  Streaming uses the V2 `StreamEvent`
sequence and terminal-event rules; disconnects request cancellation through the
same admission handle.

## Test seam

`ResidentDemoManager` accepts an `engine_factory`, so contract mapping,
resident reuse, durable transitions, cancellation, deadlines, and streaming
can be tested with a deterministic fake engine without a GPU.  The production
factory adapts the existing local `llama-server` process and does not modify
the preserved Modal/provider or llama.cpp submodule paths.
