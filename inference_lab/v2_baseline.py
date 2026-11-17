"""Sanitized V2 resident baseline packaging.

The resident service keeps its complete request evidence locally.  E2E-04
needs a small, portable record that can be reviewed or registered without
publishing prompts, model replies, host paths, or host secrets.  This module
uses an allow-list for every value copied from a run record and hashes the
resource evidence rather than embedding it in the package.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from .v2_metrics import resident_response_metrics

SCHEMA_VERSION = 1
EXPECTED_SAMPLE_COUNT = 12
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class BaselinePackagingError(ValueError):
    """Input evidence cannot be represented by the V2 package contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise BaselinePackagingError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _copy_scalars(value: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy known JSON scalar fields only; unknown fields are intentionally dropped."""
    result: dict[str, Any] = {}
    for field in fields:
        item = value.get(field)
        if isinstance(item, (str, int, float, bool)):
            result[field] = item
    return result


def _model_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    model = run.get("model")
    if not isinstance(model, Mapping):
        raise BaselinePackagingError("model identity is required")
    result = _copy_scalars(model, (
        "repository", "revision", "filename", "sha256", "size_bytes",
        "quantization", "upstream_repository", "upstream_revision", "license",
    ))
    if "sha256" in result:
        _sha(result["sha256"], "model.sha256")
    elif isinstance(run.get("model_sha256"), str):
        result["sha256"] = _sha(run["model_sha256"], "model_sha256")
    else:
        raise BaselinePackagingError("model SHA-256 identity is required")
    # A model filename is an identity label, not a path.  Reject separators so
    # an accidentally supplied local path can never cross this boundary.
    filename = result.get("filename")
    if filename is not None and (not isinstance(filename, str) or "/" in filename or "\\" in filename):
        raise BaselinePackagingError("model filename must not be a local path")
    return result


def _config_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    config_sha = _sha(run.get("config_sha256"), "config_sha256")
    runtime = run.get("runtime")
    if not isinstance(runtime, Mapping):
        raise BaselinePackagingError("runtime/config identity is required")
    # Runtime values are useful for reproducing the contract, while free-form
    # notes and command lines are deliberately excluded.
    safe_runtime = _copy_scalars(runtime, (
        "engine", "tag", "revision", "context_size", "gpu_layers", "threads", "parallel",
    ))
    return {"sha256": config_sha, "runtime": safe_runtime}


def _dataset_identity(run: Mapping[str, Any], samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    dataset = run.get("dataset")
    digest = run.get("dataset_sha256")
    if isinstance(dataset, Mapping):
        digest = dataset.get("sha256", digest)
    digest = _sha(digest, "dataset_sha256")
    protocol = run.get("measurement_protocol")
    repeats = protocol.get("repeats") if isinstance(protocol, Mapping) else None
    if type(repeats) is not int or repeats <= 0:
        raise BaselinePackagingError("measurement_protocol.repeats is required")
    return {"sha256": digest, "repeats": repeats, "sample_count": len(samples)}


def _source_identity(run: Mapping[str, Any], source: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = source if source is not None else run.get("source")
    if not isinstance(supplied, Mapping):
        # Keep the package useful for old resident records while making the
        # missing provenance explicit and machine-checkable.
        commit = run.get("source_commit")
        if isinstance(commit, str) and _COMMIT.fullmatch(commit):
            return {"status": "available", "commit": commit}
        return {"status": "unavailable", "reason": "source identity was not recorded"}
    result: dict[str, Any] = {}
    commit = supplied.get("commit", supplied.get("source_commit"))
    if commit is not None:
        if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            raise BaselinePackagingError("source commit must be a 40-character hexadecimal value")
        result["commit"] = commit
    digest = supplied.get("sha256", supplied.get("source_sha256"))
    if digest is not None:
        result["sha256"] = _sha(digest, "source.sha256")
    dirty = supplied.get("dirty")
    if dirty is not None:
        if type(dirty) is not bool:
            raise BaselinePackagingError("source.dirty must be boolean")
        result["dirty"] = dirty
    return {"status": "available" if result else "unavailable",
            **result} if result else {"status": "unavailable", "reason": "source identity was empty"}


def _sample_metrics(sample: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = sample.get("sample_id")
    case_id = sample.get("case_id")
    if not isinstance(case_id, str) or not _ID.fullmatch(case_id):
        raise BaselinePackagingError("sample case IDs must be safe identifiers")
    status = sample.get("status")
    if status not in {"completed", "failed", "cancelled", "pending"}:
        raise BaselinePackagingError("sample status is invalid")
    result = sample.get("result")
    if not isinstance(result, Mapping):
        normalized: dict[str, Any] = {}
    else:
        # ``resident_response_metrics`` consumes the same shape emitted by the
        # V2 engine.  Older E2E records retain the response fields inline, so
        # reconstruct only the timing envelope and never copy its content.
        normalized = {
            "elapsed_seconds": result.get("elapsed_seconds"),
            "response": {"timings": result.get("timings", {})},
        }
    metrics = resident_response_metrics(normalized)
    output_hash = sample.get("output_sha256")
    output_hash = _sha(output_hash, "sample.output_sha256")
    repeat = sample.get("repeat")
    if type(repeat) is not int or repeat <= 0:
        raise BaselinePackagingError("sample repeat must be a positive integer")
    return {
        "sample_id": sample_id,
        "case_id": case_id,
        "repeat": repeat,
        "status": sample.get("status"),
        "output_sha256": output_hash,
        "metrics": metrics,
    }


def _resource_evidence(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for sample in samples:
        # These are telemetry values only.  Do not copy arbitrary nested keys
        # (which could contain command output, paths, or credentials).
        item: dict[str, Any] = {"sample_id": sample.get("sample_id")}
        for name in ("gpu_before", "gpu_after"):
            value = sample.get(name)
            if isinstance(value, Mapping):
                item[name] = _copy_scalars(value, (
                    "status", "name", "count", "memory_total_bytes", "memory_free_bytes", "memory_used_bytes",
                ))
        memory = sample.get("memory")
        if isinstance(memory, Mapping):
            item["memory"] = _copy_scalars(memory, (
                "status", "MemTotal_bytes", "MemAvailable_bytes", "SwapTotal_bytes", "SwapFree_bytes",
            ))
        resources = sample.get("resources")
        if isinstance(resources, Mapping):
            item["resources"] = _copy_scalars(resources, ("sample_count", "interval_seconds"))
        evidence.append(item)
    return {
        "digest": _digest(evidence),
        "sample_count": len(evidence),
        "scope": "sanitized device-wide GPU and host-memory samples; digest only",
    }


def package_baseline(run: Mapping[str, Any], *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the portable E2E-04 package for one resident baseline run."""
    if not isinstance(run, Mapping):
        raise BaselinePackagingError("run evidence must be an object")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise BaselinePackagingError("run_id is invalid")
    if run.get("kind", "baseline") != "baseline" or run.get("execution_mode") != "resident":
        raise BaselinePackagingError("E2E-04 V2 packaging requires a resident baseline run")
    status = run.get("status")
    if status != "completed":
        raise BaselinePackagingError("run must be completed")
    samples = run.get("samples")
    if not isinstance(samples, list) or any(not isinstance(s, Mapping) for s in samples):
        raise BaselinePackagingError("run samples must be an array of objects")
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise BaselinePackagingError(f"sample cardinality must be exactly {EXPECTED_SAMPLE_COUNT}")
    sample_ids = [s.get("sample_id") for s in samples]
    if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise BaselinePackagingError("sample IDs must be unique safe identifiers")
    if any(s.get("status") != "completed" for s in samples):
        raise BaselinePackagingError("all baseline samples must be completed")
    sample_packages = [_sample_metrics(s) for s in samples]
    metric_values = [m["metrics"]["request_seconds"].get("value")
                     for m in sample_packages if m["metrics"]["request_seconds"].get("status") == "ok"]
    response_metrics = {
        "samples": sample_packages,
        "request_seconds": {
            "status": "ok" if metric_values else "unavailable",
            **({"sample_count": len(metric_values), "mean": mean(metric_values),
                "min": min(metric_values), "max": max(metric_values)} if metric_values else
               {"reason": "no valid request timings"}),
        },
    }
    cleanup_confirmed = run.get("cleanup_confirmed") is True
    limitations = list(run.get("limitations", [])) if isinstance(run.get("limitations"), list) else []
    # Limitations are explanatory text, never an evidence transport.  Drop
    # values that look like paths or secret-bearing diagnostics from old runs.
    limitations = [item for item in limitations
                   if isinstance(item, str) and len(item) <= 500
                   and "/" not in item and "\\" not in item
                   and not any(token in item.lower() for token in (
                       "secret", "token=", "password", "credential", "api_key", "authorization", "bearer",
                   ))]
    source_commit = run.get("source_commit")
    has_legacy_commit = isinstance(source_commit, str) and _COMMIT.fullmatch(source_commit)
    if source is None and not isinstance(run.get("source"), Mapping) and not has_legacy_commit:
        limitations.append("Source commit identity was not recorded in the resident run.")
    if "cleanup_confirmed" not in run:
        limitations.append("Cleanup confirmation was not recorded in the resident run.")
    cleanup = {"status": "confirmed" if cleanup_confirmed else "unavailable", "confirmed": cleanup_confirmed}
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "kind": "baseline",
        "execution_mode": "resident",
        "status": status,
        "model": _model_identity(run),
        "config": _config_identity(run),
        "dataset": _dataset_identity(run, samples),
        "source": _source_identity(run, source),
        "sample_cardinality": {"expected": EXPECTED_SAMPLE_COUNT, "observed": len(samples),
                               "completed": sum(s.get("status") == "completed" for s in samples)},
        "response_metrics": response_metrics,
        "resource_evidence": _resource_evidence(samples),
        "cleanup": cleanup,
        "limitations": limitations,
    }


def package_run(run_json: str | Path, *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read one run JSON file and return its sanitized package."""
    path = Path(run_json)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselinePackagingError(f"unable to read run evidence: {type(exc).__name__}") from exc
    return package_baseline(value, source=source)


def write_package_new(output: str | Path, package: Mapping[str, Any]) -> None:
    """Write a package once, preserving append-only evidence semantics."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(package, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")

