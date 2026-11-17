"""Wire immutable evaluation comparisons into the LLMOps gate.

The comparison writer and the registry intentionally have different schemas.  This
module is the small, offline seam between them: it verifies the comparison's
raw evidence, derives the gate's canonical metrics, and enforces that candidates
are descendants of an explicitly accepted baseline.  It never starts an engine
or assumes that a fixture represents real GPU evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from ..evaluation.comparison import ComparisonError, load_evidence
from .evaluation_gate import EvaluationGate, EvaluationGateError, GateDecision
from .registry import Registry, RegistryError


class EvaluationAdapterError(ValueError):
    """Malformed comparison, threshold configuration, or lineage."""


def _number(value: Any, *, name: str, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationAdapterError(f"{name} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationAdapterError(f"{name} must be a finite number")
    if integer and (not isinstance(value, int) or value <= 0):
        raise EvaluationAdapterError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class GateThresholds:
    """Configurable thresholds in the metric units consumed by ``EvaluationGate``."""

    min_quality: float = 1.0
    max_latency_ratio: float = 1.0
    max_gpu_bytes: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.min_quality, bool) or not isinstance(self.min_quality, (int, float)) or not math.isfinite(self.min_quality) or not 0 <= self.min_quality <= 1:
            raise EvaluationAdapterError("min_quality must be finite and in [0, 1]")
        if isinstance(self.max_latency_ratio, bool) or not isinstance(self.max_latency_ratio, (int, float)) or not math.isfinite(self.max_latency_ratio) or self.max_latency_ratio <= 0:
            raise EvaluationAdapterError("max_latency_ratio must be finite and positive")
        if self.max_gpu_bytes is not None and (isinstance(self.max_gpu_bytes, bool) or not isinstance(self.max_gpu_bytes, int) or self.max_gpu_bytes <= 0):
            raise EvaluationAdapterError("max_gpu_bytes must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GateThresholds":
        """Read a strict, small threshold mapping.

        Both flat keys and the documented ``quality``/``latency``/``gpu``
        groups are accepted so a deployment config can be mapped without
        duplicating policy in callers.
        """
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise EvaluationAdapterError("gate thresholds must be an object")
        root: Mapping[str, Any] = value.get("evaluation_gate", value.get("gate", value))
        if not isinstance(root, Mapping):
            raise EvaluationAdapterError("gate thresholds must be an object")

        quality = root.get("quality", {})
        latency = root.get("latency", root.get("performance", {}))
        gpu = root.get("gpu", root.get("resources", {}))
        if quality is not None and not isinstance(quality, Mapping):
            raise EvaluationAdapterError("quality thresholds must be an object")
        if latency is not None and not isinstance(latency, Mapping):
            raise EvaluationAdapterError("latency thresholds must be an object")
        if gpu is not None and not isinstance(gpu, Mapping):
            raise EvaluationAdapterError("GPU thresholds must be an object")

        min_quality = root.get("min_quality", quality.get("min", quality.get("min_score", 1.0)))
        max_latency = root.get("max_latency_ratio", latency.get("max_ratio", latency.get("max", 1.0)))
        max_gpu = root.get("max_gpu_bytes", gpu.get("max_bytes", gpu.get("max", None)))
        return cls(min_quality=min_quality, max_latency_ratio=max_latency, max_gpu_bytes=max_gpu)

    def to_dict(self) -> dict[str, Any]:
        return {"min_quality": self.min_quality, "max_latency_ratio": self.max_latency_ratio, "max_gpu_bytes": self.max_gpu_bytes}


@dataclass(frozen=True)
class GateInput:
    """Canonical report and provenance identifiers derived from a comparison."""

    run_id: str
    report: dict[str, Any]
    baseline_run_id: str
    evidence: dict[str, dict[str, str]]


_REQUIRED_EVIDENCE = ("run_sha256", "review_sha256", "manifest_sha256")


def _fail(registry: Registry, run_id: str, reason: str) -> GateDecision:
    registered = registry.get(run_id) if isinstance(run_id, str) else None
    return GateDecision("fail", (reason,), run_id if isinstance(run_id, str) else "", registered.digest if registered else "")


def _evidence_ids(comparison: Mapping[str, Any], key: str) -> dict[str, str]:
    value = comparison.get(key)
    if not isinstance(value, Mapping) or set(value) != set(_REQUIRED_EVIDENCE):
        raise EvaluationAdapterError(f"{key} is missing or incomplete")
    result: dict[str, str] = {}
    for field in _REQUIRED_EVIDENCE:
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise EvaluationAdapterError(f"{key}.{field} is not a SHA-256 digest")
        result[field] = digest
    return result


def _verify_evidence(comparison: Mapping[str, Any], evidence: dict[str, dict[str, str]]) -> None:
    locations = comparison.get("evidence_locations")
    if not isinstance(locations, Mapping):
        raise EvaluationAdapterError("evidence locations are missing")
    for role, result_key in (("baseline", "baseline_evidence"), ("candidate", "candidate_evidence")):
        location = locations.get(role)
        if not isinstance(location, str) or not location:
            raise EvaluationAdapterError(f"{role} evidence location is missing")
        try:
            _, observed = load_evidence(Path(location))
        except (ComparisonError, OSError, KeyError, TypeError, ValueError) as exc:
            raise EvaluationAdapterError(f"{role} evidence is missing or tampered") from exc
        if observed != evidence[result_key]:
            raise EvaluationAdapterError(f"{role} evidence identity mismatch")


def comparison_gate_input(comparison: Mapping[str, Any]) -> GateInput:
    """Map a comparison result to the report consumed by ``EvaluationGate``.

    The comparison writer stores ``baseline / candidate`` latency.  The gate
    consumes ``candidate / baseline`` so values below one are improvements.
    """
    if not isinstance(comparison, Mapping) or comparison.get("schema_version") != 1:
        raise EvaluationAdapterError("unsupported comparison schema")
    baseline_id = comparison.get("baseline_run_id")
    candidate_id = comparison.get("candidate_run_id")
    if not isinstance(baseline_id, str) or not isinstance(candidate_id, str) or not baseline_id or not candidate_id:
        raise EvaluationAdapterError("comparison run identities are required")

    evidence = {"baseline_evidence": _evidence_ids(comparison, "baseline_evidence"), "candidate_evidence": _evidence_ids(comparison, "candidate_evidence")}
    _verify_evidence(comparison, evidence)

    quality = comparison.get("quality")
    if not isinstance(quality, Mapping) or quality.get("candidate") not in {"pass", "fail"}:
        quality_score: float | None = None
    else:
        quality_score = 1.0 if quality["candidate"] == "pass" else 0.0

    measured = comparison.get("measured_request_totals_seconds")
    latency_ratio: float | None = None
    if isinstance(measured, Mapping):
        baseline_seconds = measured.get("baseline")
        candidate_seconds = measured.get("candidate")
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in (baseline_seconds, candidate_seconds)):
            latency_ratio = candidate_seconds / baseline_seconds
    if latency_ratio is None:
        # A hand-authored fixture may already provide the gate-oriented ratio.
        value = comparison.get("candidate_latency_ratio", comparison.get("latency_ratio"))
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            latency_ratio = value

    resources = comparison.get("resources")
    gpu_peak: Any = None
    if isinstance(resources, Mapping):
        candidate_resources = resources.get("candidate")
        if isinstance(candidate_resources, Mapping):
            gpu_peak = candidate_resources.get("gpu_device_used_bytes", candidate_resources.get("gpu_peak_bytes"))
    if gpu_peak is None:
        gpu_peak = comparison.get("candidate_gpu_peak_bytes")

    report = {
        "schema_version": 1,
        "run_id": candidate_id,
        "quality_score": quality_score,
        "latency_ratio": latency_ratio,
        "gpu_peak_bytes": gpu_peak,
        "provenance_status": "verified",
    }
    return GateInput(candidate_id, report, baseline_id, {"baseline": evidence["baseline_evidence"], "candidate": evidence["candidate_evidence"]})


class EvaluationGateAdapter:
    """Stateful accepted-baseline and candidate gate wiring."""

    def __init__(self, registry: Registry, thresholds: GateThresholds | Mapping[str, Any] | None = None):
        self.registry = registry
        self.thresholds = thresholds if isinstance(thresholds, GateThresholds) else GateThresholds.from_mapping(thresholds)
        self._gate = EvaluationGate(registry, **self.thresholds.to_dict())
        self._accepted_baseline_run_id: str | None = None

    @property
    def accepted_baseline_run_id(self) -> str | None:
        return self._accepted_baseline_run_id

    def accept_baseline(self, run_id: str, report: Mapping[str, Any]) -> GateDecision:
        """Accept exactly one registered, fully verified baseline report."""
        decision = self._gate.evaluate(run_id, dict(report))
        if decision.status == "pass":
            self._accepted_baseline_run_id = run_id
        return decision

    def evaluate_comparison(self, comparison: Mapping[str, Any], *, accepted_baseline_run_id: str | None = None) -> GateDecision:
        try:
            mapped = comparison_gate_input(comparison)
        except EvaluationAdapterError as exc:
            candidate = comparison.get("candidate_run_id", "") if isinstance(comparison, Mapping) else ""
            return _fail(self.registry, candidate, str(exc))

        accepted = accepted_baseline_run_id or self._accepted_baseline_run_id
        if not accepted:
            return _fail(self.registry, mapped.run_id, "no accepted baseline")
        if accepted != mapped.baseline_run_id:
            return _fail(self.registry, mapped.run_id, "comparison baseline is not the accepted baseline")
        try:
            baseline = self.registry.get(mapped.baseline_run_id)
            candidate = self.registry.get(mapped.run_id)
            if baseline is None or candidate is None:
                return _fail(self.registry, mapped.run_id, "comparison run is not registered")
            if candidate.manifest.get("parent_run_id") != mapped.baseline_run_id:
                return _fail(self.registry, mapped.run_id, "candidate parent_run_id is not the accepted baseline")
            # The call also rejects a corrupted registry and missing parents.
            self.registry.lineage(mapped.run_id)
            return self._gate.evaluate(mapped.run_id, mapped.report)
        except (RegistryError, EvaluationGateError) as exc:
            return _fail(self.registry, mapped.run_id, str(exc))

    # Short alias for callers that treat the adapter as a gate.
    evaluate = evaluate_comparison


ComparisonGateAdapter = EvaluationGateAdapter
map_comparison_to_gate_input = comparison_gate_input

__all__ = [
    "ComparisonGateAdapter", "EvaluationAdapterError", "EvaluationGateAdapter",
    "GateInput", "GateThresholds", "comparison_gate_input", "map_comparison_to_gate_input",
]
