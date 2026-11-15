"""Deterministic local V2 streaming transport.

This module deliberately stops at an in-process iterable.  A provider or HTTP
transport can adapt its output to this contract later without changing callers.
The adapter owns sequence numbers and always emits one terminal event, including
for controlled source failures and client cancellation.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import threading
from typing import Any

from inference_lab.v2_contracts import ErrorEnvelope, StreamEvent


class StreamProtocolError(ValueError):
    """The source violated the V2 stream event protocol."""


class CancellationToken:
    """Thread-safe cancellation hook suitable for a local fake transport."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def _controlled_error(code: str, request_id: str, *, retryable: bool = False) -> ErrorEnvelope:
    # Do not include source exception text: it may contain prompts or credentials.
    messages = {
        "CLIENT_DISCONNECTED": "client disconnected",
        "STREAM_CANCELLED": "stream cancelled",
        "STREAM_MALFORMED": "stream source produced a malformed event",
        "STREAM_FAILED": "stream source failed",
    }
    return ErrorEnvelope(
        code=code,
        message=messages[code],
        request_id=request_id,
        retryable=retryable,
    )


def validate_stream(events: Iterable[StreamEvent | dict[str, Any]]) -> tuple[StreamEvent, ...]:
    """Validate a complete event sequence and return its immutable canonical form.

    Sequences begin at zero, increase by one, and contain exactly one terminal
    event as their final event.  Strict parsing rejects unknown/private fields.
    """
    parsed: list[StreamEvent] = []
    request_id: str | None = None
    run_id: str | None = None
    terminal_seen = False
    for item in events:
        try:
            event = item if isinstance(item, StreamEvent) else StreamEvent.from_dict(item)
        except (TypeError, ValueError) as exc:
            raise StreamProtocolError("malformed stream event") from exc
        if request_id is None:
            request_id, run_id = event.request_id, event.run_id
        if event.request_id != request_id or event.run_id != run_id:
            raise StreamProtocolError("stream event identity changed")
        if event.sequence != len(parsed):
            raise StreamProtocolError("stream event sequence is not contiguous")
        if terminal_seen:
            raise StreamProtocolError("stream event follows terminal")
        if event.kind == "terminal":
            terminal_seen = True
        parsed.append(event)
    if not terminal_seen:
        raise StreamProtocolError("stream is missing terminal event")
    return tuple(parsed)


class StreamAdapter:
    """Adapt local text chunks into the V2 stream contract.

    ``source`` is an iterable of text chunks.  Source exceptions and malformed
    chunks become controlled error + terminal events; exception details are never
    serialized.  ``cancel`` is safe to call from a disconnect callback while the
    consumer is iterating.
    """

    def __init__(
        self,
        source: Iterable[str],
        request_id: str,
        run_id: str,
        *,
        cancellation: CancellationToken | Callable[[], bool] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        # Validate IDs through the first event construction, before source access.
        self._source = iter(source)
        self._request_id = request_id
        self._run_id = run_id
        self._cancellation = cancellation
        self._on_cancel = on_cancel
        self._cancelled = False
        self._cancel_hook_called = False
        self._terminal = False
        self._sequence = 0

    def _is_cancelled(self) -> bool:
        if self._cancelled:
            return True
        if isinstance(self._cancellation, CancellationToken):
            return self._cancellation.is_cancelled()
        if self._cancellation is not None:
            try:
                return bool(self._cancellation())
            except Exception:
                # A disconnect probe is untrusted; treat probe failure as cancel.
                return True
        return False

    def cancel(self) -> None:
        self._cancelled = True
        self._invoke_cancel_hook()

    disconnect = cancel

    def _invoke_cancel_hook(self) -> None:
        if not self._cancel_hook_called:
            self._cancel_hook_called = True
            if self._on_cancel is not None:
                self._on_cancel()

    def _event(self, kind: str, *, text: str = "", error: ErrorEnvelope | None = None) -> StreamEvent:
        event = StreamEvent(self._request_id, self._run_id, self._sequence, kind, text, error)
        self._sequence += 1
        return event

    def _finish(self) -> StreamEvent:
        self._terminal = True
        return self._event("terminal")

    def __iter__(self) -> Iterator[StreamEvent]:
        if self._terminal:
            return
        try:
            while True:
                if self._is_cancelled():
                    self._invoke_cancel_hook()
                    yield self._event(
                        "error",
                        error=_controlled_error("CLIENT_DISCONNECTED", self._request_id),
                    )
                    yield self._finish()
                    return
                try:
                    chunk = next(self._source)
                except StopIteration:
                    yield self._finish()
                    return
                except Exception:
                    yield self._event(
                        "error",
                        error=_controlled_error("STREAM_FAILED", self._request_id, retryable=True),
                    )
                    yield self._finish()
                    return
                if not isinstance(chunk, str) or not chunk:
                    yield self._event(
                        "error",
                        error=_controlled_error("STREAM_MALFORMED", self._request_id),
                    )
                    yield self._finish()
                    return
                yield self._event("chunk", text=chunk)
        except GeneratorExit:
            # A consumer closing the iterator is a disconnect even though no
            # further event can be yielded.  The hook still releases local work.
            self.cancel()
            raise

    def events(self) -> Iterator[StreamEvent]:
        return iter(self)


def stream_events(
    source: Iterable[str],
    request_id: str,
    run_id: str,
    *,
    cancellation: CancellationToken | Callable[[], bool] | None = None,
    on_cancel: Callable[[], None] | None = None,
) -> Iterator[StreamEvent]:
    """Small functional entry point for local/fake transports."""
    return StreamAdapter(
        source,
        request_id,
        run_id,
        cancellation=cancellation,
        on_cancel=on_cancel,
    ).events()
