"""RuntimeProvider contract — runtime-neutral, pure Python, no modal import. (V2 Ticket 1)

Defines the synchronous, runtime-neutral contract consumed by provider adapters (e.g.
LlamaCppProvider, Ticket 2) and verified by the provider conformance suite. It has no
dependency on V1 deployment internals and never imports modal.

Design constraints (from TaskPlan / DeveloperPrompt):
- Synchronous API only (locked): V1 uses `urllib` (sync); for consistency and
  runtime-neutrality the contract is sync and does NOT accept async alternatives.
- Capability scope (phase-one): the only declared, conformance-enforced capability set
  is VERIFIED_CAPABILITIES = {"chat", "stream"}. `tool_calls` is FUTURE-ONLY: V1's
  validators do not verify tool-call behavior, so it is NOT declared here and is NOT
  conformance-enforced (see TaskPlan §2 "Capability 契約").
- Streaming: `stream()` returns a sync `Iterator[StreamEvent]`; termination is the
  provider's responsibility (provider yields a terminal `StreamEvent(done=True)`).

This module intentionally mirrors the fail-closed semantics of V1's validators
(`validate_models_response`, `validate_chat_response`, `parse_sse`) as reusable,
runtime-neutral validation — but does not import them, keeping the contract decoupled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import FrozenSet, Iterator, Protocol, Sequence, runtime_checkable

# The verified phase-one declared capability set. `tool_calls` is future-only: it is
# intentionally NOT in this set and is NOT conformance-enforced (V1 validator does not
# verify tool-call behavior). Do not add it here without an approved capability story.
VERIFIED_CAPABILITIES: FrozenSet[str] = frozenset({"chat", "stream"})


class ProviderError(RuntimeError):
    """Controlled, runtime-neutral provider error.

    Raised for HTTP 4xx (auth/unknown model), timeout/deadline exhaustion, and invalid
    responses. Callers may also treat it as a plain RuntimeError (base error hierarchy).
    """


@dataclass(frozen=True)
class HealthStatus:
    """Provider readiness observation for `RuntimeProvider.health()`."""

    ready: bool
    detail: str | None = None


@dataclass(frozen=True)
class Message:
    """A single chat message in the runtime-neutral wire shape (role + content)."""

    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """Non-stream chat result with non-empty content (V1 validate_chat_response semantics).

    `model_id` records which model produced the response for traceability.
    """

    content: str
    model_id: str


@dataclass(frozen=True)
class StreamEvent:
    """A single incremental stream chunk.

    `done=True` is the terminal marker the provider MUST emit exactly once as the final
    event (stream termination is handled by the provider). Content may be empty for the
    terminal marker.
    """

    content: str
    done: bool = False


@dataclass(frozen=True)
class ModelDescriptor:
    """Runtime-neutral description of a deployable model.

    Fields:
        model_id:   stable identifier (e.g. V1 MODEL_ALIAS "gemma-4-e2b-it-q4_0").
        base_url:   the provider's OpenAI-compatible base URL (from modal.Server.get_url(),
                    never hand-built).
        format:     model file format, e.g. "gguf".
        runtime:    runtime family, e.g. "llama-cpp".
        capabilities: declared capability subset of VERIFIED_CAPABILITIES.
        context_length: finite positive int (e.g. V1 ctx-size=4096).
    """

    model_id: str
    base_url: str
    format: str
    runtime: str
    capabilities: FrozenSet[str] = field(default_factory=frozenset)
    context_length: int = 0

    def __post_init__(self) -> None:
        # Normalize the user-supplied set to a frozenset (hashable, set-like).
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))

    def validate(self) -> None:
        """Fail-closed validation of this descriptor's invariants.

        Raises ProviderError on: non-finite/non-positive context_length, empty model_id,
        empty format/runtime/base_url, or capabilities not being a subset of
        VERIFIED_CAPABILITIES (excludes future-only tool_calls in phase one).
        """
        if not self.model_id or not self.model_id.strip():
            raise ProviderError("model_id must be non-empty")
        if not self.format or not self.format.strip():
            raise ProviderError("format must be non-empty")
        if not self.runtime or not self.runtime.strip():
            raise ProviderError("runtime must be non-empty")
        if not self.base_url or not self.base_url.strip():
            raise ProviderError("base_url must be non-empty")
        if (not isinstance(self.context_length, int)
                or isinstance(self.context_length, bool)
                or self.context_length <= 0
                or math.isinf(float(self.context_length))):
            raise ProviderError(
                "context_length must be a finite positive integer"
            )
        if not self.capabilities <= VERIFIED_CAPABILITIES:
            raise ProviderError(
                "capabilities must be a subset of the verified set "
                f"{sorted(VERIFIED_CAPABILITIES)}; future-only tool_calls is not declared"
            )


@runtime_checkable
class RuntimeProvider(Protocol):
    """Synchronous, runtime-neutral contract every provider adapter must satisfy.

    Conformance tests operate ONLY through this Protocol — a runtime-neutral seam that
    never references provider implementation details. All methods are sync (locked); no
    async equivalents are accepted.
    """

    def health(self) -> HealthStatus: ...  # 200 = ready; 503 = not ready; 401 -> ProviderError.

    def models(self) -> list[ModelDescriptor]: ...
    # list[ModelDescriptor] — each descriptor survives validate().

    def chat(self, messages: Sequence[Message]) -> ChatResponse: ...
    # Non-stream chat; content must be non-empty (V1 validate_chat_response semantics);
    # 4xx -> ProviderError.

    def stream(self, messages: Sequence[Message]) -> Iterator[StreamEvent]: ...
    # Sync generator over SSE. Provider handles termination: requires data event(s) AND
    # a terminal marker, else ProviderError (V1 parse_sse semantics).

    def capabilities(self) -> ModelDescriptor: ...
    # The registered descriptor for this provider/runtime.
