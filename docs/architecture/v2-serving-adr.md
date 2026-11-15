# ADR: V2 resident inference boundary

## Decision

V2 Inference Serving owns one resident local engine behind a versioned contract. The supervisor owns startup, readiness, shutdown, and the engine reference. It publishes `ready` only after the engine reports readiness and publishes `stopped` only after engine shutdown returns successfully.

## Boundaries

Serving and LLMOps communicate with versioned DTOs and IDs. Serving does not import evaluation code. LLMOps does not control engine lifecycle. The legacy Modal/Gemma provider path remains unchanged.

## Scope

The first resident milestone is one Qwen model, one active generation, and a local fake-engine seam for deterministic tests. Queueing, streaming transport, authentication, and remote deployment are later tickets.

## Failure rules

Startup failure stops and releases the new engine, then enters `failed`. Stop failure enters `failed` and retains ownership for inspection. Stop is idempotent after `stopped`. No request path accepts a model filename or runtime argument.
