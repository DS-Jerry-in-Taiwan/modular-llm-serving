"""Offline deployment-manifest preflight.

This command performs bounded read-only probes. It writes a new manifest even
when blocked, so an unavailable GPU is visible as evidence rather than being
silently treated as success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from ..config import load_config
from ..telemetry import cuda_probe, gpu_probe
from .manifest import build_manifest, write_manifest_new


def _source_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else "unknown"


def deployment_preflight(config_path: str | Path, output_path: str | Path, *, cuda_library: str | None = None, nvml_library: str | None = None) -> dict:
    """Probe local identity and write one immutable deployment manifest."""
    config = load_config(config_path)
    manifest = build_manifest(config, gpu=gpu_probe(nvml_library), cuda=cuda_probe(cuda_library), source_commit=_source_commit())
    digest = write_manifest_new(output_path, manifest)
    return {"status": manifest["status"], "blockers": manifest["blockers"], "manifest": str(output_path), "manifest_sha256": digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed local GPU deployment manifest")
    parser.add_argument("command", choices=("preflight",))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda-library")
    parser.add_argument("--nvml-library")
    args = parser.parse_args(argv)
    result = deployment_preflight(args.config, args.output, cuda_library=args.cuda_library, nvml_library=args.nvml_library)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
