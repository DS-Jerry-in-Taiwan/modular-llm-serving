"""Versioned, local-only observability for V2 inference serving.

The serving contract deliberately records metadata and measurements, never the
prompt or generated text.  Events are immutable JSON-safe values and every
event carries the same request/job/run/model/config identity, which makes a
stream, disconnect, and terminal outcome joinable without a tracing service.

This module is a small seam rather than an exporter.  ``InMemoryEventSink`` is
useful for tests and a local process; ``JsonlEventSink`` is suitable for
inspection after a run.  Both are synchronous and bounded by the caller's
event rate.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Protocol
import uuid


OBSERVABILITY_VERSION = 1
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SECRET = re.compile(
    r"(?i)(?:bearer\s+|(?:api[_-]?key|token|secret|password|credential)\s*[:=]\s*)"
    r"[^\s,;]+|\b(?:sk|wk|ws|ak|as)-[A-Za-z0-9._~-]+"
)
_STATUS = frozenset({"queued", "running", "completed", "failed", "cancelled", "deadline_exceeded"})
_TERMINAL_STATUS = frozenset({"completed", "failed", "cancelled", "deadline_exceeded"})


class ObservabilityError(ValueError):
    """Base error for malformed observability input."""


class MalformedMetrics(ObservabilityError):
    """Raised when a metric is unknown, non-finite, or out of bounds."""


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ObservabilityError(f"{name} must be a bounded identifier")
    return value


def redact_text(value: str, *, limit: int = 500) -> str:
    """Return a bounded message with common credential forms removed.

    Prompts are not accepted by any event API.  This helper is only for safe,
    short operator messages and is intentionally conservative about secrets.
    """
    if not isinstance(value, str):
        raise ObservabilityError("message must be a string")
    return _SECRET.sub("[REDACTED]", value)[:limit]


def _number(value: Any, name: str, *, integer: bool = False, maximum: float | None = None) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedMetrics(f"{name} must be a number")
    if integer and type(value) is not int:
        raise MalformedMetrics(f"{name} must be an integer")
    if not math.isfinite(value) or value < 0:
        raise MalformedMetrics(f"{name} must be finite and non-negative")
    if maximum is not None and value > maximum:
        raise MalformedMetrics(f"{name} exceeds its bound")
    return value


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Strict measurements for one inference request.

    Missing values mean that a phase or probe was unavailable; ``0`` means the
    phase was observed to take no measurable time.  This distinction avoids
    silently inventing TTFT or GPU values for non-stream requests.
    """

    queue_wait_seconds: float | None = None
    engine_load_seconds: float | None = None
    ttft_seconds: float | None = None
    tpot_seconds: float | None = None
    itl_seconds: float | None = None
    generation_seconds: float | None = None
    total_seconds: float | None = None
    cleanup_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    gpu_utilization_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None
    ram_utilization_percent: float | None = None

    def __post_init__(self) -> None:
        duration_names = {
            "queue_wait_seconds", "engine_load_seconds", "ttft_seconds",
            "tpot_seconds", "itl_seconds", "generation_seconds",
            "total_seconds", "cleanup_seconds",
        }
        count_names = {
            "prompt_tokens", "completion_tokens", "total_tokens",
            "gpu_memory_used_bytes", "gpu_memory_total_bytes",
            "ram_used_bytes", "ram_total_bytes",
        }
        percent_names = {"gpu_utilization_percent", "ram_utilization_percent"}
        for name in duration_names:
            value = getattr(self, name)
            if value is not None:
                _number(value, name)
        for name in count_names:
            value = getattr(self, name)
            if value is not None:
                _number(value, name, integer=True)
        for name in percent_names:
            value = getattr(self, name)
            if value is not None:
                _number(value, name, maximum=100)
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            expected = self.prompt_tokens + self.completion_tokens
            if self.total_tokens is not None and self.total_tokens != expected:
                raise MalformedMetrics("total_tokens does not equal prompt + completion tokens")
        for used, total in ((self.gpu_memory_used_bytes, self.gpu_memory_total_bytes),
                            (self.ram_used_bytes, self.ram_total_bytes)):
            if used is not None and total is not None and used > total:
                raise MalformedMetrics("used memory exceeds total memory")

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetricsSnapshot":
        if not isinstance(value, Mapping):
            raise MalformedMetrics("metrics must be an object")
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise MalformedMetrics("unknown metric fields")
        return cls(**dict(value))


InferenceMetrics = MetricsSnapshot


def validate_metrics(value: MetricsSnapshot | Mapping[str, Any] | None) -> MetricsSnapshot | None:
    """Canonicalize metrics and fail closed on malformed input."""
    if value is None:
        return None
    if isinstance(value, MetricsSnapshot):
        return value
    return MetricsSnapshot.from_dict(value)


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    """One versioned event with a stable five-part correlation identity."""

    event_type: str
    request_id: str
    job_id: str
    run_id: str
    model_id: str
    config_id: str
    timestamp_ns: int
    terminal: bool = False
    status: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    sequence: int | None = None
    token_count: int | None = None
    metrics: MetricsSnapshot | None = None
    event_id: str = ""
    schema_version: int = OBSERVABILITY_VERSION

    def __post_init__(self) -> None:
        _identifier(self.event_type, "event_type")
        for name in ("request_id", "job_id", "run_id", "model_id", "config_id"):
            _identifier(getattr(self, name), name)
        if type(self.timestamp_ns) is not int or self.timestamp_ns <= 0:
            raise ObservabilityError("timestamp_ns must be a positive integer")
        if type(self.schema_version) is not int or self.schema_version != OBSERVABILITY_VERSION:
            raise ObservabilityError("unsupported observability schema")
        if type(self.terminal) is not bool:
            raise ObservabilityError("terminal must be a boolean")
        if self.status is not None and self.status not in _STATUS:
            raise ObservabilityError("invalid request status")
        if self.terminal and self.status not in _TERMINAL_STATUS:
            raise ObservabilityError("terminal event needs a terminal status")
        if self.error_code is not None:
            if not isinstance(self.error_code, str) or not _CODE.fullmatch(self.error_code):
                raise ObservabilityError("invalid error code")
        if self.error_message is not None:
            object.__setattr__(self, "error_message", redact_text(self.error_message))
        if self.sequence is not None:
            _number(self.sequence, "sequence", integer=True)
        if self.token_count is not None:
            _number(self.token_count, "token_count", integer=True)
        object.__setattr__(self, "metrics", validate_metrics(self.metrics))
        if not self.event_id:
            object.__setattr__(self, "event_id", uuid.uuid4().hex)
        _identifier(self.event_id, "event_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp_ns": self.timestamp_ns,
            "request_id": self.request_id,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "terminal": self.terminal,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "sequence": self.sequence,
            "token_count": self.token_count,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
        }


ObservabilityEvent = InferenceEvent


class EventSink(Protocol):
    def emit(self, event: InferenceEvent) -> None: ...


class InMemoryEventSink:
    """Thread-safe local sink returning immutable event snapshots."""

    def __init__(self) -> None:
        self._events: list[InferenceEvent] = []
        self._lock = threading.Lock()

    def emit(self, event: InferenceEvent) -> None:
        if not isinstance(event, InferenceEvent):
            raise TypeError("sink accepts InferenceEvent values")
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[InferenceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def events(self) -> tuple[InferenceEvent, ...]:
        return self.snapshot()


class JsonlEventSink:
    """Synchronous JSONL sink for local evidence; it has no network behavior."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: InferenceEvent) -> None:
        if not isinstance(event, InferenceEvent):
            raise TypeError("sink accepts InferenceEvent values")
        line = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class RequestObservation:
    """Lifecycle helper that derives phase timings and emits correlated events."""

    def __init__(
        self,
        recorder: "ObservationRecorder",
        *,
        request_id: str,
        job_id: str,
        run_id: str,
        model_id: str,
        config_id: str,
    ) -> None:
        for name, value in (("request_id", request_id), ("job_id", job_id), ("run_id", run_id),
                            ("model_id", model_id), ("config_id", config_id)):
            _identifier(value, name)
        self.recorder = recorder
        self.identity = {"request_id": request_id, "job_id": job_id, "run_id": run_id,
                         "model_id": model_id, "config_id": config_id}
        self._started = recorder.clock()
        self._queue_end: float | None = None
        self._load_start: float | None = None
        self._load_end: float | None = None
        self._generation_start: float | None = None
        self._first_token: float | None = None
        self._generation_end: float | None = None
        self._cleanup_start: float | None = None
        self._cleanup_end: float | None = None
        self._token_times: list[float] = []
        self._completion_tokens = 0
        self._terminal: InferenceEvent | None = None
        self.emit("request.started", status="queued")

    @property
    def terminal_event(self) -> InferenceEvent | None:
        return self._terminal

    def emit(self, event_type: str, **kwargs: Any) -> InferenceEvent:
        event = InferenceEvent(timestamp_ns=self.recorder.wall_clock_ns(), event_type=event_type,
                                **self.identity, **kwargs)
        self.recorder.sink.emit(event)
        return event

    def queue_complete(self) -> None:
        self._queue_end = self.recorder.clock()
        self.emit("queue.completed", status="running")

    def engine_load_start(self) -> None:
        self._load_start = self.recorder.clock()
        self.emit("engine.load.started", status="running")

    def engine_load_complete(self) -> None:
        self._load_end = self.recorder.clock()
        self.emit("engine.load.completed", status="running")

    def generation_start(self) -> None:
        self._generation_start = self.recorder.clock()
        self.emit("generation.started", status="running")

    def token(self, count: int = 1) -> None:
        if type(count) is not int or count <= 0:
            raise ObservabilityError("token count must be a positive integer")
        now = self.recorder.clock()
        self._token_times.extend([now] * count)
        self._completion_tokens += count
        if self._first_token is None:
            self._first_token = now
        self.emit("generation.token", status="running", token_count=count)

    def stream_chunk(self, sequence: int, *, token_count: int = 1) -> None:
        if type(sequence) is not int or sequence < 0:
            raise ObservabilityError("stream sequence must be a non-negative integer")
        self.token(token_count)
        self.emit("stream.chunk", status="running", sequence=sequence, token_count=token_count)

    def generation_complete(self) -> None:
        self._generation_end = self.recorder.clock()
        self.emit("generation.completed", status="running")

    def cleanup_start(self) -> None:
        self._cleanup_start = self.recorder.clock()
        self.emit("cleanup.started", status="running")

    def cleanup_complete(self) -> None:
        self._cleanup_end = self.recorder.clock()
        self.emit("cleanup.completed", status="running")

    def disconnect(self, *, sequence: int | None = None) -> InferenceEvent:
        """Record a correlated disconnect and close the request as cancelled."""
        self.emit("stream.disconnect", status="cancelled", sequence=sequence)
        return self.terminal(status="cancelled", error_code="client_disconnected")

    def _derived_metrics(self, supplied: MetricsSnapshot | Mapping[str, Any] | None) -> MetricsSnapshot:
        extra = validate_metrics(supplied) or MetricsSnapshot()
        now = self.recorder.clock()
        values = extra.to_dict()
        values.update({
            "queue_wait_seconds": self._duration(self._started, self._queue_end, values["queue_wait_seconds"]),
            "engine_load_seconds": self._duration(self._load_start, self._load_end, values["engine_load_seconds"]),
            "ttft_seconds": self._duration(self._generation_start, self._first_token, values["ttft_seconds"]),
            "generation_seconds": self._duration(self._generation_start, self._generation_end or now, values["generation_seconds"]),
            "total_seconds": self._duration(self._started, now, values["total_seconds"]),
            "cleanup_seconds": self._duration(self._cleanup_start, self._cleanup_end, values["cleanup_seconds"]),
            "completion_tokens": self._completion_tokens or values["completion_tokens"],
        })
        if values["prompt_tokens"] is not None and values["completion_tokens"] is not None:
            values["total_tokens"] = values["prompt_tokens"] + values["completion_tokens"]
        if len(self._token_times) > 1:
            mean_itl = sum(b - a for a, b in zip(self._token_times, self._token_times[1:])) / (len(self._token_times) - 1)
            values["itl_seconds"] = mean_itl
            values["tpot_seconds"] = mean_itl
        return MetricsSnapshot(**values)

    @staticmethod
    def _duration(start: float | None, end: float | None, supplied: float | None) -> float | None:
        if start is None or end is None:
            return supplied
        value = end - start
        if not math.isfinite(value) or value < 0:
            raise MalformedMetrics("clock produced an invalid duration")
        return value

    def terminal(
        self,
        *,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
        metrics: MetricsSnapshot | Mapping[str, Any] | None = None,
    ) -> InferenceEvent:
        if self._terminal is not None:
            return self._terminal
        if status not in _TERMINAL_STATUS:
            raise ObservabilityError("terminal status required")
        event = self.emit("request.terminal", terminal=True, status=status, error_code=error_code,
                          error_message=error_message, metrics=self._derived_metrics(metrics))
        self._terminal = event
        return event


class ObservationRecorder:
    """Factory for correlated observations using injectable clocks and sinks."""

    def __init__(self, sink: EventSink, *, clock: Callable[[], float] = time.monotonic,
                 wall_clock_ns: Callable[[], int] = time.time_ns) -> None:
        if not hasattr(sink, "emit") or not callable(sink.emit):
            raise TypeError("sink must provide emit(event)")
        self.sink = sink
        self.clock = clock
        self.wall_clock_ns = wall_clock_ns

    def start(self, *, request_id: str, job_id: str, run_id: str, model_id: str, config_id: str) -> RequestObservation:
        return RequestObservation(self, request_id=request_id, job_id=job_id, run_id=run_id,
                                  model_id=model_id, config_id=config_id)


__all__ = [
    "OBSERVABILITY_VERSION", "EventSink", "InferenceEvent", "ObservabilityEvent",
    "InferenceMetrics", "MetricsSnapshot", "MalformedMetrics", "ObservabilityError",
    "InMemoryEventSink", "JsonlEventSink", "ObservationRecorder", "RequestObservation",
    "redact_text", "validate_metrics",
]
