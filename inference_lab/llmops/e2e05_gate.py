"""Small, offline E2E-05 LLMOps gate runner.

E2E-05 intentionally uses sanitized fixtures.  This module only registers the
three manifest identities and verifies their local evidence references; it does
not start an inference server or claim that a fixture contains real GPU data.
The registry remains the source of immutable manifest identity while this
runner owns the promotion rule: an accepted baseline can only be produced by a
passing gate decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .evaluation_gate import EvaluationGate, EvaluationGateError, GateDecision
from .registry import Registry, RegistryError, Registration


class E2E05GateError(ValueError):
    """Malformed fixture or an unavailable/tampered evidence reference."""


_SHA256 = frozenset("0123456789abcdef")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _SHA256 for c in value):
        raise E2E05GateError(f"{name} must be a SHA-256 digest")
    return value


def _evidence_ref(manifest: Mapping[str, Any]) -> tuple[Path, str]:
    """Read the intentionally small evidence reference from a fixture manifest."""
    value = manifest.get("evidence")
    if not isinstance(value, Mapping):
        # ``evidence_path``/``evidence_sha256`` are accepted for hand-authored
        # sanitized fixtures, but never as a digest-only claim.
        path = manifest.get("evidence_path")
        digest = manifest.get("evidence_sha256")
    else:
        path = value.get("path")
        digest = value.get("sha256", value.get("digest"))
    if not isinstance(path, str) or not path:
        raise E2E05GateError("evidence path is required")
    return Path(path), _sha(digest, "evidence digest")


def verify_fixture_evidence(manifest: Mapping[str, Any]) -> str:
    """Return the observed digest, failing closed on missing or changed bytes."""
    path, expected = _evidence_ref(manifest)
    try:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise E2E05GateError("evidence is missing or unreadable") from exc
    if observed != expected:
        raise E2E05GateError("evidence digest mismatch")
    return observed


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (
        isinstance(value, int) or math.isfinite(value)
    )


def _manifest_with_parent(manifest: Mapping[str, Any], parent: str | None) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise E2E05GateError("manifest must be an object")
    result = dict(manifest)
    if parent is not None and result.get("parent_run_id") not in (None, parent):
        raise E2E05GateError("manifest lineage does not match E2E-05 order")
    if parent is not None:
        result["parent_run_id"] = parent
    # Verify bytes before registration so an invalid provenance cannot become
    # an apparently usable baseline identity.
    verify_fixture_evidence(result)
    return result


@dataclass(frozen=True)
class E2E05Registration:
    """The three fixture registrations in lineage order."""

    v1_24: Registration
    v1_40: Registration
    v2_packaged: Registration

    @property
    def run_ids(self) -> tuple[str, str, str]:
        return (self.v1_24.manifest["run_id"], self.v1_40.manifest["run_id"], self.v2_packaged.manifest["run_id"])


class E2E05GateRunner:
    """Register E2E-05 fixtures and gate a packaged baseline/candidate."""

    def __init__(self, registry: Registry, *, min_quality: float = 1.0,
                 max_latency_ratio: float = 1.0, max_gpu_bytes: int | None = None):
        self.registry = registry
        self._gate = EvaluationGate(
            registry,
            min_quality=min_quality,
            max_latency_ratio=max_latency_ratio,
            max_gpu_bytes=max_gpu_bytes,
        )
        self._accepted_baseline_run_id: str | None = None

    @property
    def accepted_baseline_run_id(self) -> str | None:
        return self._accepted_baseline_run_id

    def register_manifests(self, v1_24: Mapping[str, Any], v1_40: Mapping[str, Any],
                           v2_packaged: Mapping[str, Any]) -> E2E05Registration:
        """Register V1 24 -> V1 40 -> packaged V2 and validate the lineage."""
        # Validate every fixture before mutating the append-only index.  A bad
        # packaged record must not leave behind a partially registered chain.
        first = _manifest_with_parent(v1_24, None)
        if first.get("parent_run_id") is not None:
            raise E2E05GateError("V1 24 manifest must be the lineage root")
        second = _manifest_with_parent(v1_40, first["run_id"])
        third = _manifest_with_parent(v2_packaged, second["run_id"])
        r24 = self.registry.register(first)
        r40 = self.registry.register(second)
        r2 = self.registry.register(third)
        # This call catches a conflicting/missing parent and also proves the
        # append-only index has not been tampered with.
        chain = self.registry.lineage(r2.manifest["run_id"])
        if [item.manifest["run_id"] for item in chain] != [r2.manifest["run_id"], r40.manifest["run_id"], r24.manifest["run_id"]]:
            raise E2E05GateError("E2E-05 manifest lineage is incomplete")
        return E2E05Registration(r24, r40, r2)

    # Explicit alias for callers that prefer the E2E terminology.
    register_e2e05_manifests = register_manifests

    def _report_with_evidence(self, run_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
        registration = self.registry.get(run_id)
        if registration is None:
            raise E2E05GateError("run is not registered")
        observed = verify_fixture_evidence(registration.manifest)
        result = dict(report)
        result.setdefault("run_id", run_id)
        # The report must bind the immutable manifest and the bytes checked at
        # gate time.  A caller cannot replace either with a hand-authored claim.
        if result.get("manifest_digest") != registration.digest:
            raise E2E05GateError("report manifest digest is missing or mismatched")
        if result.get("evidence_sha256") != observed:
            raise E2E05GateError("report evidence digest is missing or mismatched")
        if result.get("provenance_status") != "verified":
            raise E2E05GateError("provenance is not verified")
        gpu = result.get("gpu_peak_bytes")
        if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu <= 0:
            raise E2E05GateError("invalid GPU peak")
        return result

    def accept_baseline(self, run_id: str, report: Mapping[str, Any]) -> GateDecision:
        """Return the gate decision; only a pass mutates accepted baseline state."""
        try:
            checked = self._report_with_evidence(run_id, report)
            decision = self._gate.evaluate(run_id, checked)
        except (E2E05GateError, RegistryError, EvaluationGateError) as exc:
            registered = self.registry.get(run_id) if isinstance(run_id, str) else None
            return GateDecision("fail", (str(exc),), run_id if isinstance(run_id, str) else "", registered.digest if registered else "")
        if decision.status == "pass":
            self._accepted_baseline_run_id = run_id
        return decision

    def evaluate_packaged_baseline(self, run_id: str, report: Mapping[str, Any]) -> GateDecision:
        """Alias documenting that the V2 packaged record is the accepted baseline."""
        return self.accept_baseline(run_id, report)

    def register_candidate(self, manifest: Mapping[str, Any]) -> Registration:
        """Register one child fixture under the already accepted baseline."""
        if not self._accepted_baseline_run_id:
            raise E2E05GateError("no accepted baseline")
        value = _manifest_with_parent(manifest, self._accepted_baseline_run_id)
        return self.registry.register(value)

    def evaluate_candidate(self, run_id: str, report: Mapping[str, Any], *,
                           baseline_run_id: str | None = None) -> GateDecision:
        """Gate a child run only after the packaged baseline passed."""
        accepted = baseline_run_id or self._accepted_baseline_run_id
        registered = self.registry.get(run_id) if isinstance(run_id, str) else None
        digest = registered.digest if registered else ""
        if not accepted:
            return GateDecision("fail", ("no accepted baseline",), run_id, digest)
        try:
            baseline = self.registry.get(accepted)
            if baseline is None:
                return GateDecision("fail", ("accepted baseline is not registered",), run_id, digest)
            if registered is None:
                return GateDecision("fail", ("run is not registered",), run_id, digest)
            if registered.manifest.get("parent_run_id") != accepted:
                return GateDecision("fail", ("candidate parent_run_id is not the accepted baseline",), run_id, digest)
            self.registry.lineage(run_id)
            checked = self._report_with_evidence(run_id, report)
            return self._gate.evaluate(run_id, checked)
        except (E2E05GateError, RegistryError, EvaluationGateError) as exc:
            return GateDecision("fail", (str(exc),), run_id, digest)


# Short aliases make the seam easy to discover without coupling callers to a
# particular E2E naming convention.
E2E05Gate = E2E05GateRunner


def register_e2e05_manifests(registry: Registry, v1_24: Mapping[str, Any],
                             v1_40: Mapping[str, Any], v2_packaged: Mapping[str, Any],
                             **thresholds: Any) -> E2E05Registration:
    """Convenience wrapper for one-shot fixture registration."""
    return E2E05GateRunner(registry, **thresholds).register_manifests(v1_24, v1_40, v2_packaged)


__all__ = [
    "E2E05Gate", "E2E05GateError", "E2E05GateRunner", "E2E05Registration",
    "register_e2e05_manifests", "verify_fixture_evidence",
]
