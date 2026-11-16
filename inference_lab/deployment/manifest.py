"""Versioned, JSON-safe deployment identity manifest.

The manifest is an offline boundary.  It contains observations and pinned
configuration identities only; it never reads credentials or starts a model.
Creation is fail-closed: missing or contradictory GPU/CUDA observations result
in ``status=blocked`` and a non-empty blocker list.  A ready manifest is not
inference evidence; it only authorizes the next runtime-validation step.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from typing import Any

DEPLOYMENT_SCHEMA_VERSION = 1
MANIFEST_KIND = "local_gpu_deployment_manifest"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STATUSES = frozenset({"ready", "blocked"})
_IDENTITY_FIELDS = frozenset({"schema_version", "kind", "status", "created_at_utc", "experiment_id", "gpu", "cuda", "model", "runtime", "deployment", "blockers", "limitations", "source"})


class ManifestError(ValueError):
    """Malformed or unsafe deployment manifest."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ManifestError("manifest is not JSON-safe") from exc


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return a stable digest after validating the public manifest shape."""
    value = validate_manifest(manifest)
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ManifestError(f"{name} must be a positive integer")


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ManifestError(f"{name} must be a 64-character lowercase SHA-256")


def _commit(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ManifestError(f"{name} must be a 40-character lowercase commit SHA")


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Validate and copy a manifest, rejecting unknown top-level fields."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be an object")
    if set(manifest) != _IDENTITY_FIELDS:
        missing = sorted(_IDENTITY_FIELDS - set(manifest))
        extra = sorted(set(manifest) - _IDENTITY_FIELDS)
        raise ManifestError(f"manifest fields mismatch; missing={missing}, extra={extra}")
    if manifest["schema_version"] != DEPLOYMENT_SCHEMA_VERSION or manifest["kind"] != MANIFEST_KIND:
        raise ManifestError("unsupported deployment manifest schema")
    if manifest["status"] not in _STATUSES:
        raise ManifestError("status must be ready or blocked")
    for key in ("experiment_id", "created_at_utc"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ManifestError(f"{key} must be a non-empty string")
    if not isinstance(manifest["blockers"], list) or any(not isinstance(v, str) or not v for v in manifest["blockers"]):
        raise ManifestError("blockers must be a list of non-empty strings")
    if not isinstance(manifest["limitations"], list) or any(not isinstance(v, str) or not v for v in manifest["limitations"]):
        raise ManifestError("limitations must be a list of non-empty strings")
    gpu = manifest["gpu"]
    if not isinstance(gpu, dict):
        raise ManifestError("gpu must be an object")
    if gpu.get("status") not in {"ok", "unavailable"}:
        raise ManifestError("gpu.status must be ok or unavailable")
    if gpu["status"] == "ok":
        for key in ("name",):
            if not isinstance(gpu.get(key), str) or not gpu[key]:
                raise ManifestError(f"gpu.{key} is required")
        _positive_int(gpu.get("count"), "gpu.count")
        _positive_int(gpu.get("memory_total_bytes"), "gpu.memory_total_bytes")
    elif "reason" in gpu and not isinstance(gpu["reason"], str):
        raise ManifestError("gpu.reason must be a string")
    cuda = manifest["cuda"]
    if not isinstance(cuda, dict) or cuda.get("status") not in {"ok", "unavailable"}:
        raise ManifestError("cuda.status must be ok or unavailable")
    if cuda["status"] == "ok":
        _positive_int(cuda.get("device_count"), "cuda.device_count")
    elif "reason" in cuda and not isinstance(cuda["reason"], str):
        raise ManifestError("cuda.reason must be a string")
    model = manifest["model"]
    if not isinstance(model, dict):
        raise ManifestError("model must be an object")
    for key in ("repository", "revision", "filename", "quantization"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ManifestError(f"model.{key} is required")
    _commit(model["revision"], "model.revision")
    _sha(model["sha256"], "model.sha256")
    _positive_int(model.get("size_bytes"), "model.size_bytes")
    runtime = manifest["runtime"]
    if not isinstance(runtime, dict):
        raise ManifestError("runtime must be an object")
    for key in ("engine", "revision"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ManifestError(f"runtime.{key} is required")
    _commit(runtime["revision"], "runtime.revision")
    deployment = manifest["deployment"]
    if not isinstance(deployment, dict) or deployment.get("mode") != "loopback" or deployment.get("host") != "127.0.0.1":
        raise ManifestError("deployment must pin loopback mode and host")
    source = manifest["source"]
    if not isinstance(source, dict) or not isinstance(source.get("commit"), str):
        raise ManifestError("source.commit is required")
    if source["commit"] != "unknown":
        _commit(source["commit"], "source.commit")
    if len(_canonical(manifest)) > 128 * 1024:
        raise ManifestError("manifest exceeds 128 KiB")
    if manifest["status"] == "ready" and manifest["blockers"]:
        raise ManifestError("ready manifest cannot contain blockers")
    if manifest["status"] == "blocked" and not manifest["blockers"]:
        raise ManifestError("blocked manifest must contain blockers")
    return json.loads(json.dumps(manifest, ensure_ascii=False))


def _gpu_matches_target(name: str, target: str) -> bool:
    """Match stable model family tokens while tolerating vendor suffixes."""
    target_tokens = {token for token in re.findall(r"[a-z0-9]+", target.lower()) if len(token) >= 3}
    observed = name.lower()
    family = [token for token in target_tokens if token.startswith(("rtx", "gtx", "tesla", "a", "h"))]
    return not family or any(token in observed for token in family)


def build_manifest(config: dict[str, Any], *, gpu: dict[str, Any], cuda: dict[str, Any], source_commit: str = "unknown", now: datetime | None = None) -> dict[str, Any]:
    """Build an immutable deployment manifest from observations and config."""
    if not isinstance(config, dict):
        raise ManifestError("config must be an object")
    try:
        model = config["model"]
        runtime = config["runtime"]
        resources = config["resources"]
        experiment_id = config["experiment_id"]
    except KeyError as exc:
        raise ManifestError(f"config missing {exc.args[0]}") from exc
    blockers: list[str] = []
    if not isinstance(gpu, dict) or gpu.get("status") != "ok":
        blockers.append("gpu_telemetry_unavailable")
    if not isinstance(cuda, dict) or cuda.get("status") != "ok":
        blockers.append("cuda_unavailable")
    if isinstance(gpu, dict) and gpu.get("status") == "ok":
        if gpu.get("count") != 1:
            blockers.append("gpu_count_must_equal_one")
        if not _gpu_matches_target(str(gpu.get("name", "")), str(resources.get("target_gpu", ""))):
            blockers.append("gpu_model_mismatch")
    if isinstance(cuda, dict) and cuda.get("status") == "ok" and cuda.get("device_count", 0) < 1:
        blockers.append("cuda_device_unavailable")
    if source_commit != "unknown":
        _commit(source_commit, "source_commit")
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    manifest = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "status": "ready" if not blockers else "blocked",
        "created_at_utc": stamp,
        "experiment_id": experiment_id,
        "gpu": {key: gpu.get(key) for key in ("status", "name", "count", "memory_total_bytes", "reason") if isinstance(gpu, dict) and key in gpu},
        "cuda": {key: cuda.get(key) for key in ("status", "device_count", "reason") if isinstance(cuda, dict) and key in cuda},
        "model": {key: model.get(key) for key in ("repository", "revision", "filename", "sha256", "size_bytes", "quantization")},
        "runtime": {key: runtime.get(key) for key in ("engine", "revision", "tag", "context_size", "gpu_layers", "threads", "parallel")},
        "deployment": {"mode": "loopback", "host": "127.0.0.1", "port": 0, "target_gpu": resources.get("target_gpu")},
        "blockers": sorted(set(blockers)),
        "limitations": ["Manifest readiness does not establish model load or CUDA inference compatibility.", "GPU memory values are device-wide observations and may include other workloads."],
        "source": {"commit": source_commit, "hostname": platform.node(), "platform": platform.platform(), "cpu_count": os.cpu_count()},
    }
    return validate_manifest(manifest)


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest: {path}") from exc
    return validate_manifest(value)


def write_manifest_new(path: str | Path, manifest: dict[str, Any]) -> str:
    """Validate then create exactly one manifest, never overwriting evidence."""
    value = validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest_digest(value)
