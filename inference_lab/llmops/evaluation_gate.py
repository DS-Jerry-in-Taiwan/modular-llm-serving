"""Fail-closed quality, performance, and provenance gate."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .registry import Registry, RegistryError


class EvaluationGateError(ValueError):
    pass


@dataclass(frozen=True)
class GateDecision:
    status: str  # pass or fail
    reasons: tuple[str, ...]
    run_id: str
    manifest_digest: str


class EvaluationGate:
    """Evaluate a report against a registered run without importing serving code."""
    def __init__(self, registry: Registry, *, min_quality: float = 1.0,
                 max_latency_ratio: float = 1.0, max_gpu_bytes: int | None = None):
        if min_quality < 0 or min_quality > 1 or max_latency_ratio <= 0:
            raise EvaluationGateError("invalid thresholds")
        self.registry = registry
        self.min_quality = min_quality
        self.max_latency_ratio = max_latency_ratio
        self.max_gpu_bytes = max_gpu_bytes

    def evaluate(self, run_id: str, report: dict[str, Any]) -> GateDecision:
        registered = self.registry.get(run_id)
        if registered is None:
            raise EvaluationGateError("run is not registered")
        reasons: list[str] = []
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            reasons.append("invalid report schema")
            return GateDecision("fail", tuple(reasons), run_id, registered.digest)
        if report.get("run_id") != run_id or report.get("manifest_digest") != registered.digest:
            reasons.append("report identity does not match registered manifest")
        quality = report.get("quality_score")
        if not isinstance(quality, (int, float)) or isinstance(quality, bool) or not 0 <= quality <= 1:
            reasons.append("invalid quality score")
        elif quality < self.min_quality:
            reasons.append("quality threshold failed")
        ratio = report.get("latency_ratio", 1.0)
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or ratio <= 0:
            reasons.append("invalid latency ratio")
        elif ratio > self.max_latency_ratio:
            reasons.append("latency threshold failed")
        if self.max_gpu_bytes is not None:
            gpu = report.get("gpu_peak_bytes")
            if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu <= 0:
                reasons.append("invalid GPU peak")
            elif gpu > self.max_gpu_bytes:
                reasons.append("GPU threshold failed")
        if report.get("provenance_status") != "verified":
            reasons.append("provenance not verified")
        return GateDecision("pass" if not reasons else "fail", tuple(reasons), run_id, registered.digest)
