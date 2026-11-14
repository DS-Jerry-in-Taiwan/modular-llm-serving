"""Bounded host probes, scoped resource sampling, and response metrics."""
import ctypes
import ctypes.util
import math
from pathlib import Path
import subprocess
import threading
import time


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


def process_rss(pid):
    if callable(pid):
        pid = pid()
    if pid is None:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class ResourceSampler:
    """Sampled maxima, not continuous peaks; device and system scopes explicit."""
    def __init__(self, interval=0.25):
        self.interval = interval
        self.pid = None
        self.samples = []
        self.stop_event = threading.Event()
        self.worker = None
        self.start_time = None
        self.lock = threading.Lock()

    def sample(self):
        gpu = gpu_probe()
        memory = memory_probe()
        values = {"monotonic_offset_seconds": time.monotonic() - self.start_time,
                  "gpu_device_used_bytes": gpu.get("memory_used_bytes"),
                  "system_memory_used_bytes": memory.get("MemTotal_bytes", 0) - memory.get("MemAvailable_bytes", 0) if memory.get("status") == "ok" else None,
                  "server_rss_bytes": process_rss(self.pid)}
        with self.lock:
            self.samples.append(values)

    def start(self):
        self.start_time = time.monotonic()
        self.sample()
        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.worker = threading.Thread(target=loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=3)
            if self.worker.is_alive():
                raise RuntimeError("resource sampler did not stop")
        self.sample()

    def summary(self, start_offset=None, end_offset=None):
        with self.lock:
            selected = [s for s in self.samples if (start_offset is None or s["monotonic_offset_seconds"] >= start_offset) and (end_offset is None or s["monotonic_offset_seconds"] <= end_offset)]
        result = {"interval_seconds": self.interval, "sample_count": len(selected), "scope": {
            "gpu_device_used_bytes": "whole GPU including other processes; sampled maximum, not continuous peak",
            "system_memory_used_bytes": "host MemTotal-MemAvailable; includes competing workloads and reclaimable-memory estimate",
            "server_rss_bytes": "owned server process VmRSS only; not all host RAM or child RSS"}}
        for metric in ("gpu_device_used_bytes", "system_memory_used_bytes", "server_rss_bytes"):
            values = [s[metric] for s in selected if s[metric] is not None]
            result[metric] = {"status": "ok", "sampled_peak_bytes": max(values)} if values else {"status": "unavailable", "reason": "no valid samples"}
        return result


def response_metrics(result):
    response = result["response"]
    timings = response.get("timings", {})
    def number(value):
        return {"status": "ok", "value": value} if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else {"status": "unavailable", "reason": "not returned by engine"}
    return {"request_seconds": number(result["elapsed_seconds"]),
            "prompt_tokens_per_second": number(timings.get("prompt_per_second")),
            "generated_tokens_per_second": number(timings.get("predicted_per_second")),
            "ttft_seconds": {"status": "unavailable", "reason": "non-stream HTTP response; first token arrival not observed"},
            "cached_prompt_tokens": number(response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")),
            "prompt_tokens": number(response.get("usage", {}).get("prompt_tokens")),
            "generated_tokens": number(response.get("usage", {}).get("completion_tokens"))}


