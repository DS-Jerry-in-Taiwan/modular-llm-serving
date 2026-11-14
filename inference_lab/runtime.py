"""Ticket-owned local llama-server lifecycle with bounded requests and cleanup."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import threading
import urllib.error
import urllib.request
import uuid

from .acquisition import sha256_file, verify_model
from .preflight import command, gpu_probe, memory_probe


class RuntimeFailure(RuntimeError):
    pass


def free_loopback_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class HttpResponseFailure(RuntimeFailure):
    def __init__(self, reason, raw_payload, status=None):
        super().__init__(reason)
        self.raw_payload = raw_payload
        self.status = status


def http_json(url, body=None, timeout=30):
    """Enforce wall time, not merely urllib's socket-inactivity timeout."""
    outcome = []
    done = threading.Event()
    def fetch():
        try:
            data = json.dumps(body).encode() if body is not None else None
            request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {}, method="POST" if data else "GET")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
            except urllib.error.HTTPError as exc:
                raw = exc.read(16 * 1024 * 1024)
                raise HttpResponseFailure(f"HTTP status {exc.code}", raw.decode(errors="replace"), exc.code) from exc
            if len(raw) > 16 * 1024 * 1024:
                raise HttpResponseFailure("HTTP response exceeds size limit", raw[:4096].decode(errors="replace"))
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise HttpResponseFailure("malformed HTTP JSON response", raw.decode(errors="replace")) from exc
            outcome.append((True, payload))
        except BaseException as exc:
            outcome.append((False, exc))
        finally:
            done.set()
    worker = threading.Thread(target=fetch, daemon=True)
    worker.start()
    if not done.wait(timeout):
        raise TimeoutError("HTTP wall deadline exceeded")
    success, result = outcome[0]
    if not success:
        raise result
    return result


class RunCancelled(Exception):
    pass


def cancellation_signal(signum, frame):
    raise RunCancelled(f"received signal {signum}")


class LocalServer:
    def __init__(self, binary, model_path, runtime_config, run_dir, deadline, port=None):
        self.binary = str(Path(binary).resolve())
        self.model_path = str(Path(model_path).resolve())
        self.config = runtime_config
        self.run_dir = Path(run_dir)
        self.deadline = deadline
        self.port = port or free_loopback_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = None
        self.log = None
        self.cleanup_confirmed = False
        self.started = None
        self.load_seconds = None
        self.last_response = None
        self.watchdog = None
        self.argv = [self.binary, "--model", self.model_path, "--host", "127.0.0.1", "--port", str(self.port),
                     "--ctx-size", str(self.config["context_size"]), "--n-gpu-layers", str(self.config["gpu_layers"]),
                     "--threads", str(self.config["threads"]), "--parallel", "1", "--alias", "local-qwen", "--reasoning", "off", "--log-verbosity", "4"]

    def remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("experiment deadline exceeded")
        return left

    def start(self):
        if self.process is not None:
            raise RuntimeFailure("server already started")
        self.log = (self.run_dir / "server.log").open("x")
        self.started = time.monotonic()
        try:
            self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            self.watchdog = threading.Timer(self.remaining(), self._expire)
            self.watchdog.daemon = True
            self.watchdog.start()
            while True:
                left = self.remaining()
                if self.process.poll() is not None:
                    raise RuntimeFailure(f"server exited before readiness: {self.process.returncode}")
                try:
                    result = http_json(self.url + "/health", timeout=min(2, left))
                    if result.get("status") == "ok":
                        self.load_seconds = time.monotonic() - self.started
                        return self
                except HttpResponseFailure as exc:
                    if exc.status != 503:
                        self.last_response = {"raw_payload": exc.raw_payload}
                        raise
                except (urllib.error.URLError, OSError, ValueError):
                    pass
                time.sleep(min(0.2, self.remaining()))
        except BaseException:
            self.stop()
            raise

    def _expire(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    def group_has_live_members(pgid):
        """Ignore exited zombie members; they no longer run or hold resources."""
        try:
            entries = Path("/proc").iterdir()
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    stat = (entry / "stat").read_text()
                    fields = stat[stat.rfind(")") + 2:].split()
                    if int(fields[2]) == pgid and fields[0] != "Z":
                        return True
                except (OSError, ValueError, IndexError):
                    continue
            return False
        except OSError:
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False

    def stop(self):
        if self.watchdog is not None:
            self.watchdog.cancel()
        process = self.process
        try:
            if process is None:
                self.cleanup_confirmed = True
                return
            for sig, wait in ((signal.SIGTERM, 3), (signal.SIGKILL, 2)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    process.poll()  # reap owned leader if exited
                    if not self.group_has_live_members(process.pid):
                        break
                    time.sleep(0.02)
                if not self.group_has_live_members(process.pid):
                    break
            process.wait(timeout=2)
            self.cleanup_confirmed = not self.group_has_live_members(process.pid)
            if not self.cleanup_confirmed:
                raise RuntimeFailure("owned process group cleanup not confirmed")
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None

    def chat(self, prompt, max_tokens=64, seed=42, temperature=0):
        self.last_response = None
        start = time.monotonic()
        try:
            response = http_json(self.url + "/v1/chat/completions", {
            "model": "local-qwen", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=min(120, self.remaining()))
        except HttpResponseFailure as exc:
            self.last_response = {"raw_payload": exc.raw_payload}
            raise
        self.last_response = response
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeFailure("malformed chat response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeFailure("empty chat content")
        return {"prompt": prompt, "response": response, "content": content,
                "elapsed_seconds": time.monotonic() - start,
                "truncated": response["choices"][0].get("finish_reason") == "length",
                "generation": {"max_tokens": max_tokens, "temperature": temperature, "seed": seed, "cache_prompt": False}}

    def offload_evidence(self):
        text = (self.run_dir / "server.log").read_text(errors="replace")
        lines = [line for line in text.splitlines() if re.search(r"offload|CUDA.*buffer|CUDA.*device", line, re.I)]
        layers = [int(n) for n in re.findall(r"offloaded\s+(\d+)/\d+\s+layers", text)]
        return {"gpu_layers_observed": max(layers, default=0), "log_lines": lines[-30:]}


def new_run(output_root, kind):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    directory = Path(output_root) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory, {"schema_version": 1, "run_id": run_id, "kind": kind,
                       "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}


def runtime_identity(binary, configured_revision):
    binary = str(Path(binary).resolve())
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
    text = (version.stdout + "\n" + version.stderr).strip()
    if version.returncode != 0 or configured_revision[:7] not in text:
        raise RuntimeFailure("binary version does not demonstrate configured runtime revision")
    return {"binary": binary, "sha256": sha256_file(binary), "version": text[:4096],
            "configured_revision": configured_revision, "pin_verified": True}


def smoke(config, binary, model_path, output_root, prompt="Reply with exactly: inference ok"):
    run_dir, record = new_run(output_root, "smoke")
    server = None
    previous_signals = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, cancellation_signal)
    start = time.monotonic()
    record.update(config=config, requested_binary=str(binary), requested_model=str(model_path))
    try:
        identity = verify_model(model_path, config["model"])
        record.update({"config": config, "model_identity": identity,
                       "runtime_identity": runtime_identity(binary, config["runtime"]["revision"]),
                       "project": command(["git", "rev-parse", "HEAD"]),
                       "host": {"gpu_before": gpu_probe(), "memory_before": memory_probe()}})
        if record["host"]["gpu_before"]["status"] != "ok":
            raise RuntimeFailure("usable NVML GPU evidence is required")
        server = LocalServer(binary, model_path, config["runtime"], run_dir,
                             time.monotonic() + config["budgets"]["smoke_seconds"])
        record["argv"] = server.argv
        server.start()
        record["load_seconds"] = server.load_seconds
        record["offload"] = server.offload_evidence()
        if record["offload"]["gpu_layers_observed"] < 1:
            raise RuntimeFailure("GPU offload not demonstrated; CPU-only is not accepted")
        record["result"] = server.chat(prompt)
        if record["result"]["truncated"]:
            raise RuntimeFailure("smoke response truncated at token limit")
        record["host"]["gpu_during"] = gpu_probe()
        if record["host"]["gpu_during"]["status"] != "ok":
            raise RuntimeFailure("GPU telemetry lost during inference")
        record["host"]["memory_during"] = memory_probe()
        record["status"] = "success"
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="cancelled" if isinstance(exc, (RunCancelled, KeyboardInterrupt)) else "failed", failure={"type": type(exc).__name__, "reason": str(exc)[:500]})
    finally:
        for sig in previous_signals:
            signal.signal(sig, signal.SIG_IGN)
        if server is not None:
            if server.last_response is not None:
                record["raw_response"] = server.last_response
            try:
                server.stop()
                record["cleanup_confirmed"] = server.cleanup_confirmed
            except Exception as exc:
                record.update(status="failed", cleanup_confirmed=False,
                              cleanup_failure={"type": type(exc).__name__, "reason": str(exc)[:500]})
        else:
            record["cleanup_confirmed"] = True
        record["elapsed_seconds"] = time.monotonic() - start
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with (run_dir / "run.json").open("x") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        for sig, old_handler in previous_signals.items():
            signal.signal(sig, old_handler)
    return record, run_dir
