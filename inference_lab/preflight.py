"""Read-only, bounded host/model preflight. No model downloads or services."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import urllib.error
import urllib.request

SCHEMA_VERSION = 1


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported configuration schema_version")
    for key in ("experiment_id", "model", "runtime", "resources", "budgets"):
        if key not in config:
            raise ValueError(f"missing {key}")
    model = config["model"]
    for key in ("repository", "revision", "filename", "sha256", "size_bytes"):
        if key not in model:
            raise ValueError(f"missing model.{key}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repository"]):
        raise ValueError("model repository must be owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("model revision must be immutable 40-character commit")
    if not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
        raise ValueError("model sha256 must be 64 hexadecimal characters")
    if not isinstance(model["filename"], str) or Path(model["filename"]).name != model["filename"]:
        raise ValueError("model filename must be a basename")
    for group, key in ((model, "size_bytes"), (config["resources"], "disk_reserve_bytes"),
                       (config["budgets"], "download_bytes"), (config["budgets"], "smoke_seconds"),
                       (config["budgets"], "experiment_seconds")):
        if type(group.get(key)) is not int or group[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if model["size_bytes"] > config["budgets"]["download_bytes"]:
        raise ValueError("model exceeds download budget")
    if not re.fullmatch(r"[0-9a-f]{40}", config["runtime"].get("revision", "")):
        raise ValueError("runtime revision must be immutable")
    for key in ("context_size", "gpu_layers", "threads", "parallel"):
        if type(config["runtime"].get(key)) is not int or config["runtime"][key] <= 0:
            raise ValueError(f"runtime.{key} must be a positive integer")
    if config["runtime"]["parallel"] != 1:
        raise ValueError("only one inference slot is supported")
    return config


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return {"status": "ok" if result.returncode == 0 else "unavailable",
                "returncode": result.returncode, "output": result.stdout.strip()[:4096]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}


def gpu_probe(library_path=None):
    """NVML success is required; device nodes or empty nvidia-smi are insufficient."""
    try:
        library = ctypes.CDLL(library_path or ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1")
        code = library.nvmlInit_v2()
        if code:
            return {"status": "unavailable", "reason": f"nvmlInit_v2={code}"}
        try:
            count = ctypes.c_uint()
            code = library.nvmlDeviceGetCount_v2(ctypes.byref(count))
            if code or not count.value:
                return {"status": "unavailable", "reason": f"device_count={count.value}, rc={code}"}
            handle = ctypes.c_void_p()
            code = library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
            if code:
                return {"status": "unavailable", "reason": f"device_handle={code}"}
            name = ctypes.create_string_buffer(96)
            code = library.nvmlDeviceGetName(handle, name, 96)
            if code:
                return {"status": "unavailable", "reason": f"device_name={code}"}
            class Memory(ctypes.Structure):
                _fields_ = [(k, ctypes.c_ulonglong) for k in ("total", "free", "used")]
            memory = Memory()
            code = library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            if code:
                return {"status": "unavailable", "reason": f"memory_info={code}"}
            return {"status": "ok", "name": name.value.decode(), "count": count.value,
                    "memory_total_bytes": memory.total, "memory_free_bytes": memory.free,
                    "memory_used_bytes": memory.used}
        finally:
            library.nvmlShutdown()
    except (OSError, AttributeError) as exc:
        return {"status": "unavailable", "reason": str(exc)[:300]}


def cuda_probe(library_path=None):
    """Initialize the driver API; NVML telemetry alone does not establish CUDA."""
    try:
        library = ctypes.CDLL(library_path or ctypes.util.find_library("cuda") or "libcuda.so.1")
        code = library.cuInit(0)
        if code:
            return {"status": "unavailable", "reason": f"cuInit={code}"}
        count = ctypes.c_int()
        code = library.cuDeviceGetCount(ctypes.byref(count))
        if code or count.value < 1:
            return {"status": "unavailable", "reason": f"cuda_device_count={count.value}, rc={code}"}
        return {"status": "ok", "device_count": count.value}
    except (OSError, AttributeError) as exc:
        return {"status": "unavailable", "reason": str(exc)[:300]}


def memory_probe():
    result = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                result[key + "_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        return {"status": "unavailable"}
    return {"status": "ok", **result}


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
    with output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "blockers": report["blockers"], "report": str(output)}))
    return 0 if report["status"] == "ready_for_runtime_validation" else 2
