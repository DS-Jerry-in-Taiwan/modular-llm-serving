"""Append-only local lineage registry for V2 LLMOps.

The registry indexes immutable manifests. Raw experiment evidence remains owned by
its run directory; this index never rewrites or deletes it.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = ("run_id", "model_id", "config_id", "dataset_id", "evaluator_id", "source_commit")


class RegistryError(ValueError):
    """Fail-closed manifest or registry error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise RegistryError("manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise RegistryError("unsupported manifest schema_version")
    for field in _REQUIRED:
        value = manifest.get(field)
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise RegistryError(f"invalid {field}")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]):
        raise RegistryError("invalid source_commit")
    for field in ("model_sha256", "config_sha256", "dataset_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or not _SHA.fullmatch(value):
            raise RegistryError(f"invalid {field}")
    parent = manifest.get("parent_run_id")
    if parent is not None and (not isinstance(parent, str) or not _ID.fullmatch(parent)):
        raise RegistryError("invalid parent_run_id")
    # Keep registered content bounded and JSON-safe; raw outputs stay outside the index.
    if len(_canonical(manifest)) > 128 * 1024:
        raise RegistryError("manifest exceeds 128 KiB")
    return json.loads(json.dumps(manifest, ensure_ascii=False))


@dataclass(frozen=True)
class Registration:
    manifest: dict[str, Any]
    digest: str


class Registry:
    """A deterministic append-only JSONL index keyed by run_id."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Registration]:
        if not self.path.exists():
            return {}
        result: dict[str, Registration] = {}
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    manifest = _validate_manifest(record["manifest"])
                    digest = record["digest"]
                    if digest != manifest_digest(manifest):
                        raise RegistryError("digest mismatch")
                    run_id = manifest["run_id"]
                    previous = result.get(run_id)
                    if previous and previous.digest != digest:
                        raise RegistryError(f"conflicting manifest for {run_id}")
                    result[run_id] = Registration(manifest, digest)
                except (KeyError, TypeError, json.JSONDecodeError, RegistryError) as exc:
                    raise RegistryError(f"invalid registry record at line {number}: {exc}") from exc
        return result

    def register(self, manifest: dict[str, Any]) -> Registration:
        value = _validate_manifest(manifest)
        digest = manifest_digest(value)
        existing = self._read().get(value["run_id"])
        if existing:
            if existing.digest != digest:
                raise RegistryError(f"run_id already registered with different manifest: {value['run_id']}")
            return existing
        record = {"schema_version": 1, "digest": digest, "manifest": value}
        # append only; exclusive lock is intentionally out of scope for the first local process.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return Registration(value, digest)

    def get(self, run_id: str) -> Registration | None:
        if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
            raise RegistryError("invalid run_id")
        return self._read().get(run_id)

    def list(self) -> list[Registration]:
        return list(self._read().values())

    def lineage(self, run_id: str) -> list[Registration]:
        """Return the run and its parent chain, rejecting cycles and missing parents."""
        current = self.get(run_id)
        if current is None:
            return []
        chain: list[Registration] = []
        seen: set[str] = set()
        while current:
            rid = current.manifest["run_id"]
            if rid in seen:
                raise RegistryError("lineage cycle")
            seen.add(rid)
            chain.append(current)
            parent = current.manifest.get("parent_run_id")
            current = self.get(parent) if parent else None
            if parent and current is None:
                raise RegistryError(f"missing parent run: {parent}")
        return chain
