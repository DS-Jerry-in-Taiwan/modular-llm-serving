"""Immutable local evidence, deterministic hashes, and unique run allocation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from .config import SCHEMA_VERSION


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def output_hash(result):
    return hashlib.sha256(result["content"].encode("utf-8")).hexdigest()


def write_json_new(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def new_run(output_root, kind):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    directory = Path(output_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory, {"schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": kind,
                       "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}


