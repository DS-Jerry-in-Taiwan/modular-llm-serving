"""Loopback-only, single-GPU live comparison; no cloud or model acquisition."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import html
from pathlib import Path
import re
import signal
import threading
import time
from urllib.parse import urlsplit
import uuid

from .acquisition import verify_model
from .baseline import ResourceSampler, response_metrics
from .preflight import gpu_probe, load_config
from .runtime import LocalServer, RuntimeFailure, RunCancelled, runtime_identity
from .tasks import write_json_new

ROOT = Path(__file__).resolve().parent.parent
LIMITS = {"max_prompt_chars": 3000, "min_tokens": 16, "max_tokens": 256}
EXAMPLES = [
    {"id": "summary", "label": "摘要", "prompt": "請用一句繁體中文摘要：公司邀請12人試用新系統，9人完成任務，3人因連線逾時中止，團隊預定11月20日重測。"},
    {"id": "code", "label": "程式碼", "prompt": "請寫一個 Python 函式，移除串列中的重複項目並保留原始順序，附上一個簡短範例。"},
    {"id": "json", "label": "JSON 格式", "prompt": "請只輸出合法 JSON，列出三種水果，每個項目包含 name 和 color 欄位。"},
]


def loopback_authority(value):
    """Validate the browser-facing authority, including a forwarded local port."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?", value, re.IGNORECASE)
    if not match:
        return None
    port = int(match.group(2)) if match.group(2) is not None else 80
    if not 1 <= port <= 65535:
        return None
    return match.group(1).lower(), port


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_error(exc):
    # Paths, raw model replies and runtime stderr belong in local evidence only.
    if isinstance(exc, RunCancelled):
        return "Run cancelled."
    if isinstance(exc, TimeoutError):
        return "Inference deadline exceeded."
    if isinstance(exc, (ValueError, RuntimeFailure)):
        return "Runtime or pinned identity validation failed; inspect local evidence."
    return "Local inference failed; inspect local evidence."


class DemoManager:
    def __init__(self, binary, model, output_root, baseline_config, optimized_config, *, verify_async=True):
        self.binary, self.model, self.output_root = Path(binary), Path(model), Path(output_root)
        self.configs = {"baseline": baseline_config, "optimized": optimized_config}
        if baseline_config["model"] != optimized_config["model"]:
            raise ValueError("variants must use the same pinned model")
        baseline_runtime, optimized_runtime = copy.deepcopy(baseline_config["runtime"]), copy.deepcopy(optimized_config["runtime"])
        if baseline_runtime.pop("gpu_layers") != 24 or optimized_runtime.pop("gpu_layers") != 40 or baseline_runtime != optimized_runtime:
            raise ValueError("variants must differ only in 24 versus 40 GPU layers")
        self.lock = threading.RLock()
        self.jobs, self.events = {}, {}
        self.active = self.server = self.worker = None
        self.closing = False
        self.status, self.error, self.identity = "checking", None, None
        self.identity_worker = None
        if verify_async:
            self.identity_worker = threading.Thread(target=self.verify, daemon=True)
            self.identity_worker.start()

    def verify(self):
        try:
            identity = {"model": verify_model(self.model, self.configs["baseline"]["model"]),
                        "runtime": runtime_identity(self.binary, self.configs["baseline"]["runtime"]["revision"]),
                        "gpu": gpu_probe()}
            if identity["gpu"].get("status") != "ok":
                raise RuntimeFailure("GPU telemetry unavailable")
            with self.lock:
                self.identity, self.status = identity, "ready"
        except Exception as exc:
            with self.lock:
                self.status, self.error = "blocked", safe_error(exc)

    def variant(self, key):
        return {"id": key, "label": "Baseline · 24 GPU layers" if key == "baseline" else "Optimized · 40 GPU layers",
                "gpu_layers": self.configs[key]["runtime"]["gpu_layers"]}

    def info(self):
        with self.lock:
            result = {"model": self.configs["baseline"]["model"]["filename"], "ready": self.status == "ready" and not self.closing,
                      "status": self.status, "variants": [self.variant(k) for k in self.configs],
                      "examples": EXAMPLES, "limits": LIMITS, "busy": self.active is not None}
            if self.error:
                result["error"] = self.error
            return copy.deepcopy(result)

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.jobs.get(job_id))

    def history(self):
        with self.lock:
            return {"runs": copy.deepcopy(list(self.jobs.values())[::-1])}

    def submit(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"prompt", "max_tokens", "variants"}:
            raise ValueError("Expected prompt, max_tokens and variants only.")
        prompt, tokens = payload.get("prompt"), payload.get("max_tokens", 192)
        variants = payload.get("variants", ["baseline", "optimized"])
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 3000:
            raise ValueError("Prompt must contain 1–3000 characters.")
        if type(tokens) is not int or not 16 <= tokens <= 256:
            raise ValueError("max_tokens must be an integer from 16 to 256.")
        if not isinstance(variants, list) or not variants or any(type(v) is not str or v not in self.configs for v in variants) or len(set(variants)) != len(variants):
            raise ValueError("Select baseline and/or optimized once each.")
        with self.lock:
            if self.status != "ready" or self.closing:
                raise RuntimeError("Model verification is not ready.")
            if self.active is not None:
                raise BlockingIOError("GPU is busy with another run.")
            job_id = uuid.uuid4().hex
            directory = self.output_root / job_id
            directory.mkdir(parents=True, exist_ok=False)
            job = {"id": job_id, "status": "queued", "phase": "queued", "current_variant": None,
                   "created_at": now(), "prompt": prompt, "max_tokens": tokens, "variants": [{**self.variant(v), "status": "queued"} for v in variants]}
            write_json_new(directory / "manifest.json", {"job": job, "prompt": prompt, "max_tokens": tokens,
                           "configs": {v: self.configs[v] for v in variants}, "identity": self.identity,
                           "generation": {"temperature": 0, "seed": 42, "cache_prompt": False},
                           "measurement": "Non-stream request; TTFT unavailable. No warmup. Filesystem cache uncontrolled. User prompt quality is not automatically assessed."})
            while len(self.jobs) >= 20:
                oldest = next(iter(self.jobs))
                del self.jobs[oldest]
                self.events.pop(oldest, None)
            self.jobs[job_id], self.events[job_id], self.active = job, threading.Event(), job_id
            self.worker = threading.Thread(target=self._run, args=(job_id, prompt, tokens, directory), daemon=True)
            self.worker.start()
            return job_id

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return False
            self.events[job_id].set()
            return True

    def _update(self, target, **values):
        with self.lock:
            target.update(values)

    def _run(self, job_id, prompt, tokens, directory):
        job, cancel = self.jobs[job_id], self.events[job_id]
        deadline = time.monotonic() + 1200
        self._update(job, status="running", phase="loading")
        try:
            for item in job["variants"]:
                if cancel.is_set():
                    raise RunCancelled()
                key = item["id"]
                variant_dir = directory / key
                variant_dir.mkdir()
                server = LocalServer(self.binary, self.model, self.configs[key]["runtime"], variant_dir,
                                     min(deadline, time.monotonic() + 600))
                sampler = ResourceSampler()
                monitor_done = threading.Event()
                def interrupt():
                    while not monitor_done.wait(0.05):
                        if cancel.is_set():
                            # Only this owned process group; stop/reap remains worker-owned.
                            server._expire()
                monitor = threading.Thread(target=interrupt, daemon=True)
                evidence = {"argv": server.argv, "started_at": now()}
                failure = None
                self._update(job, phase="loading", current_variant=key)
                self._update(item, status="loading")
                with self.lock:
                    self.server = server
                try:
                    if gpu_probe().get("status") != "ok":
                        raise RuntimeFailure("GPU telemetry required")
                    sampler.start()
                    sampler.pid = lambda: server.process.pid if server.process is not None else None
                    monitor.start()
                    if cancel.is_set():
                        raise RunCancelled()
                    server.start()
                    if cancel.is_set():
                        raise RunCancelled()
                    offload = server.offload_evidence()
                    evidence["offload"] = offload
                    if offload["gpu_layers_observed"] < item["gpu_layers"]:
                        raise RuntimeFailure("configured GPU offload not demonstrated")
                    self._update(item, status="running", load_seconds=server.load_seconds)
                    self._update(job, phase="generating")
                    result = server.chat(prompt, max_tokens=tokens, temperature=0, seed=42)
                    if cancel.is_set():
                        raise RunCancelled()
                    if gpu_probe().get("status") != "ok":
                        raise RuntimeFailure("GPU telemetry lost")
                    evidence["result"] = result
                    metrics = response_metrics(result)
                    values = {"text": result["content"], "request_seconds": result["elapsed_seconds"],
                              "truncated": result["truncated"]}
                    for public, metric in (("tokens_per_second", "generated_tokens_per_second"), ("output_tokens", "generated_tokens")):
                        if metrics[metric]["status"] == "ok":
                            values[public] = metrics[metric]["value"]
                    self._update(item, **values)
                except Exception as exc:
                    failure = RunCancelled() if cancel.is_set() else exc
                finally:
                    self._update(job, phase="cleaning_up")
                    # Keep cancellation monitoring until stop finishes, avoiding a creation race.
                    try:
                        server.stop()
                        if not server.cleanup_confirmed:
                            raise RuntimeFailure("cleanup unconfirmed")
                        evidence["cleanup_confirmed"] = True
                    except Exception as exc:
                        evidence["cleanup_confirmed"] = False
                        failure = exc
                        with self.lock:
                            self.status, self.error = "blocked", "Owned GPU process cleanup unconfirmed; restart after inspecting local evidence."
                    monitor_done.set()
                    if monitor.is_alive():
                        monitor.join(timeout=1)
                    try:
                        if sampler.start_time is not None:
                            sampler.stop()
                            evidence["resources"] = sampler.summary()
                            write_json_new(variant_dir / "resource-samples.json", sampler.samples)
                            peak = evidence["resources"]["gpu_device_used_bytes"]
                            if peak["status"] == "ok":
                                self._update(item, gpu_peak_bytes=peak["sampled_peak_bytes"], gpu_peak_scope="Whole device including other processes; sampled maximum")
                    except Exception as exc:
                        failure = exc
                    # Evidence failures must not bypass terminal state or ownership cleanup.
                    if server.last_response is not None:
                        try:
                            write_json_new(variant_dir / "raw-response.json", server.last_response)
                        except Exception as exc:
                            failure = exc
                    with self.lock:
                        if evidence["cleanup_confirmed"]:
                            self.server = None
                    self._update(item, cleanup_confirmed=evidence["cleanup_confirmed"])
                    if failure:
                        self._update(item, status="cancelled" if isinstance(failure, RunCancelled) else "failed", error=safe_error(failure))
                        evidence["error"] = {"type": type(failure).__name__, "message": safe_error(failure)}
                    else:
                        self._update(item, status="completed")
                    evidence["variant"] = copy.deepcopy(item)
                    try:
                        write_json_new(variant_dir / "run.json", evidence)
                    except Exception as exc:
                        failure = exc
                        self._update(item, status="failed", error=safe_error(exc))
                if failure:
                    raise failure
            self._update(job, status="completed", phase="completed")
        except Exception as exc:
            self._update(job, status="cancelled" if isinstance(exc, RunCancelled) else "failed",
                         phase="cancelled" if isinstance(exc, RunCancelled) else "failed", error=safe_error(exc))
            for item in job["variants"]:
                if item["status"] == "queued":
                    self._update(item, status="cancelled", error="Not started because the run stopped.")
        finally:
            # A failed writer must never leave a published variant in an active phase.
            for item in job["variants"]:
                if item["status"] in {"loading", "running"}:
                    self._update(item, status="failed", error=job.get("error", "Local inference failed; inspect local evidence."))
            self._update(job, current_variant=None, finished_at=now(), cleanup_confirmed=self.server is None)
            try:
                write_json_new(directory / "run.json", self.get(job_id))
            except Exception:
                self._update(job, status="failed", phase="failed", error="Failed to persist local run evidence.")
            with self.lock:
                # Retain the owned server and active reservation when stop was not confirmed.
                if self.server is None:
                    self.active = None

    def shutdown(self):
        with self.lock:
            self.closing = True
            if self.active:
                self.events[self.active].set()
            worker = self.worker
        if worker:
            worker.join(timeout=15)
            if worker.is_alive():
                raise RuntimeFailure("inference worker did not stop")


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def handler_for(manager, static_root=None):
    static_root = Path(static_root or ROOT / "demo_web").resolve()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *args):
            pass  # Do not echo prompts or arbitrary request paths.

        def send_data(self, status, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            hosts = self.headers.get_all("Host", [])
            return len(hosts) == 1 and loopback_authority(hosts[0]) is not None

        def valid_origin(self):
            origins = self.headers.get_all("Origin", [])
            if not origins:  # Local command-line clients need not supply Origin.
                return True
            if len(origins) != 1 or not origins[0].startswith("http://"):
                return False
            authority = loopback_authority(origins[0][len("http://"):])
            return authority is not None and authority == loopback_authority(self.headers.get("Host"))

        def do_GET(self):
            if not self.valid_host():
                return self.send_data(403, {"error": "Loopback Host required."})
            path = urlsplit(self.path).path
            if path == "/api/info":
                return self.send_data(200, manager.info())
            if path == "/api/history":
                return self.send_data(200, manager.history())
            if path.startswith("/api/runs/"):
                job = manager.get(path.removeprefix("/api/runs/"))
                return self.send_data(200 if job else 404, job or {"error": "Run not found."})
            if path == "/reference":
                report = ROOT / "results/qwen38-rtx3080-20261114-v2/report.md"
                if report.is_file():
                    document = "<!doctype html><meta charset=utf-8><title>Recorded inference comparison</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Recorded comparison</h1><p>Historical controlled benchmark; live user prompts are not automatically quality assessed.</p><pre>" + html.escape(report.read_text()) + "</pre>"
                    return self.send_data(200, document.encode(), "text/html; charset=utf-8")
            allowed = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
                       "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in allowed:
                name, mime = allowed[path]
                file = (static_root / name).resolve()
                if file.parent == static_root and file.is_file():
                    return self.send_data(200, file.read_bytes(), mime)
            self.send_data(404, {"error": "Not found."})

        def do_POST(self):
            if not self.valid_host() or not self.valid_origin():
                return self.send_data(403, {"error": "Same-origin requests required."})
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                return self.send_data(415, {"error": "JSON request required."})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if not 0 <= length <= 65536:
                return self.send_data(413, {"error": "Request body must be at most 64 KB."})
            self.connection.settimeout(5)
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, OSError):
                return self.send_data(400, {"error": "Invalid JSON body."})
            path = urlsplit(self.path).path
            if path == "/api/runs":
                try:
                    job_id = manager.submit(payload)
                    return self.send_data(202, {"id": job_id})
                except ValueError as exc:
                    return self.send_data(400, {"error": str(exc)})
                except BlockingIOError:
                    return self.send_data(409, {"error": "GPU is busy with another run."})
                except RuntimeError:
                    return self.send_data(503, {"error": "Model verification is not ready."})
                except OSError:
                    return self.send_data(500, {"error": "Could not create local evidence."})
            if path.startswith("/api/runs/") and path.endswith("/cancel"):
                if payload != {}:
                    return self.send_data(400, {"error": "Cancellation expects an empty JSON object."})
                found = manager.cancel(path[len("/api/runs/"):-len("/cancel")])
                return self.send_data(202 if found else 404, {"status": "cancellation_requested"} if found else {"error": "Run not found."})
            self.send_data(404, {"error": "Not found."})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/inference-lab"
    parser.add_argument("--binary", type=Path, default=cache / "runtime/llama-fa676981/build/bin/llama-server")
    parser.add_argument("--model", type=Path, default=cache / "models/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--output-root", type=Path, default=ROOT / "docs/local-inference/demo-runs")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "experiments/qwen38-27b-q4.json")
    parser.add_argument("--optimized-config", type=Path, default=ROOT / "experiments/qwen38-27b-q4-gpu40.json")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0–65535")
    manager = DemoManager(args.binary, args.model, args.output_root, load_config(args.baseline_config), load_config(args.optimized_config))
    server = DemoHTTPServer(("127.0.0.1", args.port), handler_for(manager))
    def shutdown_signal(signum, frame):
        # shutdown() must run outside serve_forever's thread.
        threading.Thread(target=server.shutdown, daemon=True).start()
    previous = {sig: signal.signal(sig, shutdown_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    print(f"Local demo: http://127.0.0.1:{server.server_port} (verifying model identity)", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        manager.shutdown()
        server.server_close()
        for sig, old in previous.items():
            signal.signal(sig, old)


if __name__ == "__main__":
    main()
