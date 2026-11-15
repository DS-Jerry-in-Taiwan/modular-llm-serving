# ADR: V2 local streaming contract

## Decision

V2 streaming uses an in-process `StreamAdapter` as the transport seam. The adapter
turns an iterable of text chunks into versioned `StreamEvent` values with contiguous
zero-based sequence numbers. Every normally consumed stream ends with exactly one
`terminal` event. Source failures and client disconnects produce a controlled
`error` event followed by that terminal event.

`validate_stream` is the fail-closed boundary for a complete stream: it rejects
missing or duplicate terminals, gaps, identity changes, malformed events, and
events after termination. `StreamEvent.to_dict()` exposes only the fixed wire
fields; prompts, credentials, exception details, and provider-specific fields are
never serialized.

## Cancellation

The local seam accepts a thread-safe `CancellationToken` or a cancellation probe,
plus an optional cleanup callback. Disconnect cleanup is invoked at most once.
Closing an iterator before it can emit its terminal marker invokes the callback;
the transport cannot append an event to an iterator that the client has already
closed.

## Scope and follow-ups

This is intentionally a fake/local transport. It does not own sockets, retries,
authentication, provider calls, or remote deployment. A future transport may adapt
its source into this seam while preserving the event and cancellation invariants.
