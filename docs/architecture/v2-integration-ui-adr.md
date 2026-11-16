# ADR: V2 resident mode in the local Demo

## Decision

The existing Demo keeps its V1 cold benchmark as the default path.  A visible
mode selector adds a separate **Warm resident** path for the V2 serving
contract.  Cold mode remains the A/B comparison at `/api/runs`; resident mode
uses the versioned `/api/v2/runs` adapter and submits an `InferenceRequest`
shape (`request_id`, `model_id`, `config_id`, `max_new_tokens`, `stream`, and
an idempotency key).

The UI never silently presents the two protocols as equivalent.  It labels the
selected protocol, keeps V1 configuration cards in cold mode, and displays a
resident lifecycle panel with request/run identity, queue state, stream state,
terminal state, phase timings, and metrics.  Stream chunks are appended only
from V2 payload events; the terminal event is the completion boundary.

## Consequences

The Demo can explain cold-load benchmark numbers separately from warm resident
latency and queue behavior without changing the preserved V1 workflow.  The
browser uses DOM `textContent` builders for server-owned values and sends no
credentials or third-party requests.  Until a V2 HTTP adapter is available,
selecting resident mode reports its adapter error and does not fall back to V1,
so a cold result cannot be mistaken for resident evidence.

The resident UI is intentionally browser-light: polling is used for the
versioned run handle, while the optional `StreamEvent` list is rendered from a
transport adapter response.  A future native SSE/WebSocket transport can feed
the same identity and terminal fields without changing the mode boundary.

