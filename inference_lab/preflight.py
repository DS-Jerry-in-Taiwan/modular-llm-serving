"""Read-only, bounded host/model preflight. No model downloads or services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import urllib.error
import urllib.request

from .config import SCHEMA_VERSION, load_config
from .telemetry import command, gpu_probe, cuda_probe, memory_probe
from .artifacts import write_json_new


def model_probe(model):
    url = f'https://huggingface.co/api/models/{model["repository"]}/revision/{model["revision"]}?blobs=true'
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            metadata = json.load(response)
        item = next((x for x in metadata.get("siblings", []) if x.get("rfilename") == model["filename"]), None)
        if item is None:
            return {"status": "mismatch", "reason": "pinned file absent"}
        lfs = item.get("lfs", {})
        actual = {"revision": metadata.get("sha"), "size_bytes": item.get("size"), "sha256": lfs.get("sha256")}
        matches = all(actual[key] == model[key] for key in actual)
        return {"status": "ok" if matches else "mismatch", "url": url, **actual}
    except urllib.error.HTTPError as exc:
        return {"status": "unavailable", "url": url, "http_status": exc.code}
    except (OSError, ValueError, TimeoutError) as exc:
        return {"status": "unavailable", "url": url, "reason": type(exc).__name__}


def preflight(config, disk_path, check_network=False, cuda_library=None, nvml_library=None):
    gpu = gpu_probe(nvml_library)
    cuda = cuda_probe(cuda_library)
    disk = shutil.disk_usage(disk_path)
    needed = config["model"]["size_bytes"] + config["resources"]["disk_reserve_bytes"]
    blockers = []
    if gpu["status"] != "ok":
        blockers.append("gpu_telemetry_unavailable")
    if cuda["status"] != "ok":
        blockers.append("cuda_unavailable")
    if disk.free < needed:
        blockers.append("insufficient_disk")
    model = model_probe(config["model"]) if check_network else {"status": "not_checked"}
    if model["status"] != "ok":
        blockers.append("model_identity_unverified")
    return {"schema_version": SCHEMA_VERSION, "kind": "preflight", "experiment_id": config["experiment_id"],
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ready_for_runtime_validation" if not blockers else "blocked", "blockers": blockers,
            "environment": {"hostname": platform.node(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                            "cpu": command(["lscpu"]), "memory": memory_probe(), "gpu": gpu, "cuda": cuda,
                            "gpu_pci": command(["lspci", "-d", "10de:"]),
                            "disk": {"free_bytes": disk.free, "required_bytes": needed},
                            "tools": {name: shutil.which(name) for name in ("cmake", "nvcc", "docker", "llama-server", "ollama")}},
            "model_verification": model, "config": config,
            "limitations": ["Preflight does not establish CUDA inference or model load compatibility.",
                            "No downloads, builds, services, or cloud mutations performed."]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument("--cuda-library", help="Explicit isolated matching driver library path")
    parser.add_argument("--nvml-library", help="Explicit isolated NVML library path")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = preflight(config, output.parent, args.check_network, args.cuda_library, args.nvml_library)
    # Do not overwrite prior evidence.
    write_json_new(output, report)
    print(json.dumps({"status": report["status"], "blockers": report["blockers"], "report": str(output)}))
    return 0 if report["status"] == "ready_for_runtime_validation" else 2
