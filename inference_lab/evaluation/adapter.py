"""Shared evaluation adapter for V1 cold and V2 resident inference.

The adapter is deliberately runtime neutral.  A backend is a small fakeable seam with
``infer(request)``; the real V1/V2 transports can be plugged in without changing the run
record.  Every terminal record carries the same case, prompt, model, config and dataset
identity, and a run id is first-write-wins.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import threading
from typing import Any, Literal, Mapping, Protocol

ExecutionMode = Literal["cold", "resident"]
TerminalStatus = Literal["completed", "failed"]
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AdapterError(ValueError):
    """A backend or run payload violated the shared adapter contract."""


class RunIdentityConflict(AdapterError):
    """A run id was reused for a different request identity."""


class InferenceBackend(Protocol):
    def infer(self, request: "RunRequest") -> Mapping[str, Any]: ...


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise AdapterError(f"{name} must be a bounded identifier")
    return value


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise AdapterError(f"{name} must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or (positive and value <= 0) or (not positive and value < 0):
        raise AdapterError(f"{name} must be a finite {('positive ' if positive else '')}number")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunRequest:
    """Input identity shared by cold and resident execution paths."""

    run_id: str
    case_id: str
    prompt: str
    model_id: str
    config_id: str
    dataset_id: str
    execution_mode: ExecutionMode
    max_new_tokens: int = 192
    prompt_hash: str = ""
    model_config_dataset_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("run_id", "case_id", "model_id", "config_id", "dataset_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 3000:
            raise AdapterError("prompt must contain 1..3000 characters")
        if self.execution_mode not in ("cold", "resident"):
            raise AdapterError("execution_mode must be cold or resident")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 4096:
            raise AdapterError("max_new_tokens must be an integer from 1..4096")
        expected_prompt = _sha256(self.prompt, "prompt")
        if self.prompt_hash and (not _SHA256.fullmatch(self.prompt_hash) or self.prompt_hash != expected_prompt):
            raise AdapterError("prompt_hash does not match prompt")
        object.__setattr__(self, "prompt_hash", expected_prompt)
        expected_identity = _canonical_digest({
            "model_id": self.model_id,
            "config_id": self.config_id,
            "dataset_id": self.dataset_id,
        })
        if self.model_config_dataset_hash and (
            not _SHA256.fullmatch(self.model_config_dataset_hash)
            or self.model_config_dataset_hash != expected_identity
        ):
            raise AdapterError("model_config_dataset_hash does not match identity")
        object.__setattr__(self, "model_config_dataset_hash", expected_identity)

    @property
    def identity(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "config_id": self.config_id,
            "dataset_id": self.dataset_id,
            "model_config_dataset_hash": self.model_config_dataset_hash,
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True)
class RunMetrics:
    """Required successful-run metrics. Missing values are never fabricated."""

    request_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        _finite(self.request_seconds, "request_seconds", positive=True)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise AdapterError(f"{name} must be a non-negative integer")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise AdapterError("total_tokens does not equal prompt + completion tokens")

    @classmethod
    def from_value(cls, value: Any) -> "RunMetrics":
        if not isinstance(value, Mapping):
            raise AdapterError("completed run requires metrics")
        expected = {"request_seconds", "prompt_tokens", "completion_tokens", "total_tokens"}
        if set(value) != expected:
            raise AdapterError("completed run metrics are incomplete or contain unknown fields")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunRecord:
    """Immutable terminal evidence record returned by :class:`InferenceAdapter`."""

    identity: dict[str, str]
    status: TerminalStatus
    text: str = ""
    truncated: bool = False
    metrics: RunMetrics | None = None
    error: dict[str, str] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.status not in ("completed", "failed"):
            raise AdapterError("invalid run record schema")
        required = {"run_id", "case_id", "prompt_hash", "model_id", "config_id", "dataset_id", "model_config_dataset_hash", "execution_mode"}
        if set(self.identity) != required:
            raise AdapterError("run identity fields are incomplete")
        if self.status == "completed":
            if not isinstance(self.text, str) or not self.text.strip() or self.truncated or self.metrics is None or self.error is not None:
                raise AdapterError("completed run must have non-empty, non-truncated text and metrics")
        else:
            if self.metrics is not None:
                # A failed result may carry diagnostics, but malformed metrics are never accepted.
                RunMetrics.from_value(self.metrics.to_dict())
            if self.error is None:
                raise AdapterError("failed run requires an error")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metrics"] = self.metrics.to_dict() if self.metrics is not None else None
        return value


class InferenceAdapter:
    """Run one request through a cold or resident backend with first-write-wins state."""

    def __init__(self, *, cold_backend: InferenceBackend, resident_backend: InferenceBackend):
        self._backends = {"cold": cold_backend, "resident": resident_backend}
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    @property
    def records(self) -> tuple[RunRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def run(self, request: RunRequest) -> RunRecord:
        # Serialize the check-and-execute window: concurrent retries must not invoke a
        # backend twice for one run id.
        with self._lock:
            existing = self._records.get(request.run_id)
            if existing is not None:
                if existing.identity != request.identity:
                    raise RunIdentityConflict("run_id is already bound to another request identity")
                return existing
            backend = self._backends[request.execution_mode]
            try:
                payload = backend.infer(request)
                record = self._record_from_payload(request, payload)
            except AdapterError as exc:
                record = self._failed(request, "contract_violation", str(exc))
            except Exception:
                # Provider exception details may contain prompts or credentials; never publish them.
                record = self._failed(request, "backend_failure", "inference backend failed")
            self._records[request.run_id] = record
            return record

    @staticmethod
    def _failed(request: RunRequest, code: str, message: str, *, truncated: bool = False) -> RunRecord:
        return RunRecord(identity=request.identity, status="failed", truncated=truncated,
                         error={"code": code, "message": message[:200]})

    @classmethod
    def _record_from_payload(cls, request: RunRequest, payload: Any) -> RunRecord:
        if not isinstance(payload, Mapping):
            raise AdapterError("backend result must be an object")
        status = payload.get("status", "completed")
        if status != "completed":
            code = payload.get("error_code", "backend_failed")
            if not isinstance(code, str) or not _ID.fullmatch(code):
                raise AdapterError("invalid failure error_code")
            return cls._failed(request, code, "inference failed")
        truncated = payload.get("truncated")
        if type(truncated) is not bool:
            raise AdapterError("backend result must declare truncated boolean")
        if truncated:
            return cls._failed(request, "truncated", "inference output was truncated", truncated=True)
        text = payload.get("text", payload.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise AdapterError("completed run requires non-empty text")
        metrics = RunMetrics.from_value(payload.get("metrics"))
        return RunRecord(identity=request.identity, status="completed", text=text, metrics=metrics)

    def terminal(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)


__all__ = ["AdapterError", "InferenceAdapter", "InferenceBackend", "RunIdentityConflict", "RunMetrics", "RunRecord", "RunRequest"]
