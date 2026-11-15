"""Small, JSON-safe V2 contracts.  Keep this module free of serving/evaluation imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Literal

CONTRACT_VERSION = 1
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a bounded identifier")
    return value


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    prompt: str
    model_id: str
    config_id: str
    max_new_tokens: int = 192
    stream: bool = False
    idempotency_key: str | None = None
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _id(self.request_id, "request_id"); _id(self.model_id, "model_id"); _id(self.config_id, "config_id")
        if self.idempotency_key is not None: _id(self.idempotency_key, "idempotency_key")
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 3000:
            raise ValueError("prompt must contain 1..3000 characters")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be an integer from 1..4096")
        if type(self.stream) is not bool or self.contract_version != CONTRACT_VERSION:
            raise ValueError("invalid request contract")

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    run_id: str
    model_id: str
    text: str
    finish_reason: str
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _id(self.request_id, "request_id"); _id(self.run_id, "run_id"); _id(self.model_id, "model_id")
        if not isinstance(self.text, str) or self.contract_version != CONTRACT_VERSION:
            raise ValueError("invalid response contract")

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class StreamEvent:
    request_id: str
    run_id: str
    sequence: int
    kind: Literal["chunk", "terminal", "error"]
    text: str = ""
    error: "ErrorEnvelope | None" = None
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _id(self.request_id, "request_id"); _id(self.run_id, "run_id")
        if type(self.sequence) is not int or self.sequence < 0 or self.kind not in {"chunk", "terminal", "error"}:
            raise ValueError("invalid stream event")
        if self.kind == "terminal" and self.error is not None: raise ValueError("terminal event cannot contain error")
        if self.kind == "error" and self.error is None: raise ValueError("error event needs error envelope")
        if self.contract_version != CONTRACT_VERSION: raise ValueError("invalid stream contract")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self); return value


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    message: str
    request_id: str | None = None
    retryable: bool = False
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _id(self.code, "error code")
        if self.request_id is not None: _id(self.request_id, "request_id")
        if not self.message or len(self.message) > 500 or self.contract_version != CONTRACT_VERSION:
            raise ValueError("invalid error envelope")

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    sha256: str
    size_bytes: int
    quantization: str

    def __post_init__(self) -> None:
        _id(self.model_id, "model_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("invalid model identity")


@dataclass(frozen=True)
class InferenceConfig:
    config_id: str
    runtime_revision: str
    context_size: int
    gpu_layers: int
    threads: int
    execution_mode: Literal["cold", "resident"]

    def __post_init__(self) -> None:
        _id(self.config_id, "config_id")
        if any(type(v) is not int or v <= 0 for v in (self.context_size, self.gpu_layers, self.threads)):
            raise ValueError("runtime config values must be positive integers")
        if self.execution_mode not in {"cold", "resident"}: raise ValueError("invalid execution mode")


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    request_id: str
    model_id: str
    config_id: str
    source_commit: str
    dataset_id: str | None = None
    evaluator_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("run_id", self.run_id), ("request_id", self.request_id), ("model_id", self.model_id), ("config_id", self.config_id)):
            _id(value, name)
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_commit): raise ValueError("source_commit must be a 40-char SHA")


@dataclass(frozen=True)
class EvidenceManifest:
    run: RunIdentity
    contract_version: int = CONTRACT_VERSION
    evidence_sha256: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.contract_version != CONTRACT_VERSION: raise ValueError("invalid evidence contract")
        if self.evidence_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.evidence_sha256):
            raise ValueError("evidence_sha256 must be a 64-char SHA")

    def to_dict(self) -> dict[str, Any]: return asdict(self)
