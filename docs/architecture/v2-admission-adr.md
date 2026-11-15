# ADR: V2 bounded admission and idempotency

## Decision

V2 serving uses a process-local, FIFO admission queue in front of the resident
`ResidentSupervisor`.  Exactly one request may be active at a time.  The queue
has an explicit maximum waiting depth; once that bound is reached, new work is
rejected with a stable `queue_full` admission error rather than silently
growing memory or spawning another engine.

Admission is checked synchronously before the supervisor is started.  A
deterministic token counter (four UTF-8 bytes per estimated token by default)
or a pinned tokenizer supplied by the caller measures prompt usage, and
`prompt_tokens + max_new_tokens` must fit the configured context size.

## Idempotency

An `InferenceRequest.idempotency_key` identifies one execution for the life of
the queue.  Repeating the key returns a new handle to the original job,
including after that job reaches a terminal state, so retries cannot execute a
generation twice.  Reusing a key with a different request payload is rejected
as `idempotency_conflict`.  Requests without a key retain ordinary at-most-one
execution per submitted handle.

## Cancellation and deadlines

Queued cancellation removes work logically and publishes `cancelled`
immediately.  Active cancellation sets a cooperative event; the engine runner
must observe it.  A deadline uses the same event and publishes
`deadline_exceeded`.  Cancellation is idempotent, and active jobs do not become
terminal until the runner and the optional cleanup hook return.  This ordering
ensures clients never observe a completed request while its owned cleanup is
still in progress.

The queue invokes the ready engine obtained from the supervisor and never owns
engine construction or shutdown.  `shutdown(stop_engine=True)` is the explicit
boundary for stopping the resident engine.  The implementation is intentionally
single-process and deterministic: Redis, Kafka, remote provider calls, and
GPU-specific behavior are out of scope for this milestone.

## Consequences

The local seam is straightforward to test with a fake engine and gives bounded
backpressure, deterministic FIFO ordering, and retry-safe request identity.
An HTTP adapter must map `AdmissionRejected.code` and terminal
`AdmissionResult` values into the versioned V2 error/response contracts.  A
non-cooperative engine cannot be forcibly interrupted by a Python event; a
future process-isolated runner would be required for hard kill semantics.
