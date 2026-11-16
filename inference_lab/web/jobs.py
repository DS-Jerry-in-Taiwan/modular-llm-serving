"""Serialized GPU jobs with owned cancellation, cleanup, and immutable evidence."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

from ..acquisition import verify_model
from ..telemetry import ResourceSampler, response_metrics, gpu_probe
from ..runtime import LocalServer, RuntimeFailure, RunCancelled, runtime_identity
from ..artifacts import write_json_new
from ..llmops.durable import DurableJobStore, EvidenceRef, DurableJobError
from ..serving import InMemoryEventSink, ObservationRecorder, ResidentSupervisor, stream_events
from ..serving.admission import AdmissionQueue, AdmissionRejected, AdmissionStatus
from ..v2_contracts import ErrorEnvelope, InferenceRequest, InferenceResponse

LIMITS = {"max_prompt_chars": 3000, "min_tokens": 16, "max_tokens": 256}
EXAMPLES = [
    {"id": "summary", "label": "摘要", "prompt": "請用一句繁體中文摘要：公司邀請12人試用新系統，9人完成任務，3人因連線逾時中止，團隊預定11月20日重測。"},
    {"id": "code", "label": "程式碼", "prompt": "請寫一個 Python 函式，移除串列中的重複項目並保留原始順序，附上一個簡短範例。"},
    {"id": "json", "label": "JSON 格式", "prompt": "請只輸出合法 JSON，列出三種水果，每個項目包含 name 和 color 欄位。"},
]


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
    mode = "cold"
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


class _ResidentLocalEngine:
    """One owned llama-server process adapted to the resident engine seam."""

    def __init__(self, binary, model, runtime_config, directory, deadline):
        self.server = LocalServer(binary, model, runtime_config, directory, deadline)

    def start(self):
        self.server.start()

    def ready(self):
        return self.server.process is not None and self.server.process.poll() is None

    def stop(self):
        self.server.stop()

    def generate(self, request, cancellation):
        if cancellation.is_set():
            raise RunCancelled()
        result = self.server.chat(request.prompt, max_tokens=request.max_new_tokens, temperature=0, seed=42)
        if cancellation.is_set():
            raise RunCancelled()
        return result


class ResidentDemoManager:
    """Local V2 HTTP manager backed by one resident engine and durable state.

    This class is deliberately separate from :class:`DemoManager`: V1 requests
    use the old cold-process/variant protocol, while V2 requests are mapped to
    ``InferenceRequest`` and cannot accidentally select V1 fields or paths.
    ``engine_factory`` is an explicit fake-engine seam for tests and local
    service development; production construction supplies the owned llama
    server adapter above.
    """

    mode = "resident"
    contract_version = 1

    def __init__(self, binary, model, output_root, config, *, engine_factory=None,
                 state_path=None, max_queue=16, request_timeout=1200,
                 source_commit=None, sink=None, verify_async=True):
        self.binary, self.model, self.output_root = Path(binary), Path(model), Path(output_root)
        self.config = copy.deepcopy(config)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_id = self.config.get("model", {}).get("filename", "qwen-model").replace(".gguf", "")
        self.config_id = self.config.get("runtime", {}).get("revision", "resident")
        self.context_size = int(self.config.get("runtime", {}).get("context_size", 4096))
        self.request_timeout = request_timeout
        self.source_commit = source_commit or ("0" * 40)
        if not isinstance(self.source_commit, str) or len(self.source_commit) != 40:
            raise ValueError("source_commit must be a 40-character commit")
        self.sink = sink or InMemoryEventSink()
        self.recorder = ObservationRecorder(self.sink)
        self.store = DurableJobStore(state_path or self.output_root / "v2-jobs.sqlite3")
        self.db_lock = threading.RLock()
        # A restart never resumes an in-flight generation.
        with self.db_lock:
            self.store.recover()
        self.lock = threading.RLock()
        self.handles, self.observations, self.deadlines = {}, {}, {}
        # A stream runner can fail before it has a value for the admission
        # result.  Keep its already-sanitized wire events until cleanup has
        # published the durable terminal state; never retain the raw error.
        self.stream_failures = {}
        self.closing = False
        self.status, self.error, self.identity = ("checking" if verify_async else "ready"), None, None
        self.engine_factory = engine_factory or self._default_factory
        self.supervisor = ResidentSupervisor(self.engine_factory)
        self.queue = AdmissionQueue(self.supervisor, context_size=self.context_size,
                                    max_queue=max_queue, runner=self._run_request,
                                    cleanup=self._finish_request)
        if verify_async:
            self.identity_worker = threading.Thread(target=self.verify, daemon=True)
            self.identity_worker.start()
        else:
            self.identity_worker = None

    def _default_factory(self):
        # Each resident engine owns a fresh evidence directory.  LocalServer
        # writes server.log exclusively, so reusing one directory would make a
        # later service restart fail before the engine can become ready.
        directory = self.output_root / "resident-engine" / f"instance-{uuid.uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=True)
        return _ResidentLocalEngine(self.binary, self.model, self.config["runtime"],
                                    directory, time.monotonic() + self.request_timeout)

    def verify(self):
        """Perform the same fail-closed identity checks as V1 when possible."""
        try:
            identity = {
                "model": verify_model(self.model, self.config["model"]),
                "runtime": runtime_identity(self.binary, self.config["runtime"]["revision"]),
            }
            with self.lock:
                self.identity, self.status, self.error = identity, "ready", None
        except Exception as exc:
            with self.lock:
                self.status, self.error = "blocked", safe_error(exc)

    def info(self):
        with self.lock:
            return {
                "mode": self.mode, "contract_version": self.contract_version,
                "model": self.model_id, "config_id": self.config_id,
                "ready": self.status == "ready" and not self.closing,
                "status": self.status, "busy": self.queue.active is not None,
                "queued": self.queue.queued, "limits": {
                    "max_prompt_chars": 3000, "min_tokens": 1,
                    "max_tokens": 4096, "context_size": self.context_size,
                },
            }

    @staticmethod
    def _error(code, message, request_id=None, retryable=False):
        return ErrorEnvelope(code=code, message=message, request_id=request_id, retryable=retryable).to_dict()

    def _public_record(self, record):
        result = {
            "id": record.job_id, "request_id": record.request_id, "run_id": record.run_id,
            "status": record.state, "phase": record.state, "created_at": record.created_at,
            "updated_at": record.updated_at, "contract_version": self.contract_version,
        }
        metadata = record.metadata
        if "response" in metadata:
            result["response"] = copy.deepcopy(metadata["response"])
        if "events" in metadata:
            result["events"] = copy.deepcopy(metadata["events"])
        if record.terminal_record and record.terminal_record.get("error"):
            result["error"] = copy.deepcopy(record.terminal_record["error"])
        if record.terminal_record:
            provenance = record.terminal_record.get("provenance", {})
            if provenance.get("error"):
                result["error"] = copy.deepcopy(provenance["error"])
        return result

    def get(self, job_id):
        try:
            with self.db_lock:
                record = self.store.get(job_id)
        except DurableJobError:
            return None
        return self._public_record(record) if record else None

    def history(self):
        # SQLite ordering is stable and records remain available across restart.
        with self.db_lock, self.store._connection as connection:
            rows = connection.execute("SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT 100").fetchall()
        return {"runs": [self.get(row[0]) for row in rows]}

    def _request(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("V2 request must be a JSON object")
        forbidden = {"max_tokens", "variants", "binary", "model", "runtime"}
        if forbidden.intersection(payload):
            raise ValueError("V1 request fields cannot be used in resident mode")
        allowed = {"request_id", "prompt", "model_id", "config_id", "max_new_tokens",
                   "stream", "idempotency_key", "contract_version", "timeout_seconds"}
        if set(payload) - allowed:
            raise ValueError("unknown V2 request field")
        if payload.get("contract_version", self.contract_version) != self.contract_version:
            raise ValueError("unsupported request contract version")
        request_id = payload.get("request_id") or uuid.uuid4().hex
        model_id = payload.get("model_id", self.model_id)
        config_id = payload.get("config_id", self.config_id)
        if model_id != self.model_id or config_id != self.config_id:
            raise ValueError("model_id and config_id are fixed by the local service")
        timeout = payload.get("timeout_seconds", self.request_timeout)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > self.request_timeout:
            raise ValueError("timeout_seconds must be positive and bounded")
        request = InferenceRequest(
            request_id=request_id, prompt=payload.get("prompt", ""), model_id=model_id,
            config_id=config_id, max_new_tokens=payload.get("max_new_tokens", 192),
            stream=payload.get("stream", False), idempotency_key=payload.get("idempotency_key"),
            contract_version=self.contract_version,
        )
        return request, float(timeout)

    def submit(self, payload):
        if self.status != "ready" or self.closing:
            raise RuntimeError("resident service is not ready")
        request, timeout = self._request(payload)
        # Resolve retries before creating a second durable row.  The queue also
        # validates the request fingerprint once an admitted handle exists.
        if request.idempotency_key is not None:
            with self.db_lock:
                rows = self.store._connection.execute("SELECT job_id, metadata_json FROM jobs").fetchall()
            for row in rows:
                try:
                    metadata = json.loads(row[1])
                except (TypeError, ValueError):
                    continue
                if metadata.get("idempotency_key") == request.idempotency_key:
                    with self.db_lock:
                        existing = self.store.get(row[0])
                    if existing is None:
                        continue
                    if metadata.get("request_fingerprint") != self._fingerprint(request):
                        raise ValueError("idempotency key is bound to another request")
                    return existing.job_id
        # Durable queued state is committed before the admission worker can run.
        try:
            with self.db_lock:
                self.store.create_job(request.request_id, request.request_id, request.request_id,
                                      metadata={"model_id": request.model_id, "config_id": request.config_id,
                                                "stream": request.stream,
                                                "idempotency_key": request.idempotency_key,
                                                "request_fingerprint": self._fingerprint(request)})
        except DurableJobError as exc:
            # Idempotent retries return the existing request only when the payload
            # itself is identical; the queue enforces that same rule.
            with self.db_lock:
                existing = self.store.get(request.request_id)
            if existing and existing.metadata.get("model_id") == request.model_id:
                return request.request_id
            raise ValueError("request_id is already in use") from exc
        deadline = time.monotonic() + timeout
        try:
            handle = self.queue.submit(request, deadline=deadline)
        except AdmissionRejected as exc:
            try:
                with self.db_lock:
                    self.store.transition(request.request_id, "failed", reason=exc.code,
                                          provenance={"source_commit": self.source_commit})
            except DurableJobError:
                pass
            raise exc
        with self.lock:
            self.handles[request.request_id] = handle
            self.deadlines[request.request_id] = deadline
        return request.request_id

    @staticmethod
    def _fingerprint(request):
        return [request.prompt, request.model_id, request.config_id,
                request.max_new_tokens, request.stream, request.contract_version]

    def cancel(self, request_id):
        with self.lock:
            handle = self.handles.get(request_id)
        if handle is None:
            return False
        changed = handle.cancel()
        if changed and handle.status is AdmissionStatus.CANCELLED:
            try:
                with self.db_lock:
                    self.store.transition(request_id, "cancelled", reason="client cancellation")
            except DurableJobError:
                pass
        return changed

    def _run_request(self, engine, request, cancellation):
        observation = self.recorder.start(request_id=request.request_id, job_id=request.request_id,
                                           run_id=request.request_id, model_id=request.model_id,
                                           config_id=request.config_id)
        with self.lock:
            self.observations[request.request_id] = observation
        with self.db_lock:
            self.store.transition(request.request_id, "running", reason="admitted")
        observation.queue_complete()
        observation.generation_start()
        try:
            result = engine.generate(request, cancellation)
            text_value, chunks = self._result_chunks(result)
        except BaseException as exc:
            if request.stream:
                # Convert a runner failure (including cooperative cancellation)
                # into the same bounded stream protocol as a source failure.
                # The exception itself remains private to the admission result.
                def failed_source(failure=exc):
                    if cancellation.is_set():
                        return
                    yield from ()
                    raise failure

                events = list(stream_events(
                    failed_source(), request.request_id, request.request_id,
                    cancellation=cancellation.is_set,
                    on_cancel=cancellation.set,
                ))
                with self.lock:
                    self.stream_failures[request.request_id] = events
            raise
        if request.stream:
            events = list(stream_events(chunks, request.request_id, request.request_id,
                                        cancellation=cancellation.is_set,
                                        on_cancel=cancellation.set))
            for event in events:
                if event.kind == "chunk":
                    observation.stream_chunk(event.sequence)
            return {"text": text_value, "events": [event.to_dict() for event in events]}
        return {"text": text_value}

    @staticmethod
    def _result_chunks(result):
        if isinstance(result, dict):
            text_value = result.get("content", "")
        else:
            text_value = result
        if isinstance(text_value, str):
            return text_value, ([text_value] if text_value else [])
        try:
            chunks = list(text_value)
        except TypeError as exc:
            raise RuntimeFailure("engine returned a malformed response") from exc
        if any(not isinstance(chunk, str) or not chunk for chunk in chunks):
            raise RuntimeFailure("engine returned a malformed response")
        return "".join(chunks), chunks

    def _finish_request(self, request, value, error, cancellation):
        observation = self.observations.get(request.request_id)
        if observation is not None:
            observation.generation_complete()
            observation.cleanup_start()
        deadline = self.deadlines.get(request.request_id)
        expired = deadline is not None and time.monotonic() >= deadline
        if expired:
            status, reason, code = "deadline_exceeded", "request deadline exceeded", "deadline_exceeded"
        elif cancellation.is_set():
            status, reason, code = "cancelled", "client cancellation", "cancelled"
        elif error is not None:
            status, reason, code = "failed", "inference failed", "inference_failed"
        else:
            status, reason, code = "completed", "inference completed", None
        with self.db_lock:
            existing = self.store.get(request.request_id)
        metadata = dict(existing.metadata) if existing else {}
        with self.lock:
            failed_events = self.stream_failures.pop(request.request_id, None)
        if value is not None and isinstance(value, dict):
            text_value = value.get("text", "")
            if "events" in value:
                metadata["events"] = value["events"]
            if status == "completed":
                response = InferenceResponse(request.request_id, request.request_id, request.model_id,
                                             text_value, "stop").to_dict()
                metadata["response"] = response
        if failed_events is not None:
            metadata["events"] = [event.to_dict() for event in failed_events]
        if observation is not None:
            observation.cleanup_complete()
            error_envelope = self._error(code, reason, request.request_id, retryable=(status == "failed")) if code else None
            observation.terminal(status=status, error_code=code, error_message=reason)
        try:
            if status == "completed":
                directory = self.output_root / request.request_id
                directory.mkdir(parents=True, exist_ok=True)
                evidence_path = directory / "response.json"
                evidence_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True), encoding="utf-8")
                digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                evidence = EvidenceRef(str(evidence_path), digest, self.source_commit)
                with self.db_lock:
                    self.store.transition(request.request_id, "completed", reason=reason, evidence=evidence,
                                          provenance={"source_commit": self.source_commit, "request_id": request.request_id})
                    self.store._connection.execute("UPDATE jobs SET metadata_json=? WHERE job_id=?",
                                                   (json.dumps(metadata, ensure_ascii=False, sort_keys=True), request.request_id))
                    self.store._connection.commit()
            else:
                terminal = {"error": self._error(code, reason, request.request_id, retryable=(status == "failed"))}
                with self.db_lock:
                    self.store.transition(request.request_id, status, reason=reason,
                                          provenance={"source_commit": self.source_commit,
                                                      "error": terminal["error"]})
                    # Stream error/terminal events are part of the public
                    # terminal record even when the runner failed or was
                    # cancelled.  Persist them after the state transition so
                    # readers never observe an active row with terminal data.
                    self.store._connection.execute(
                        "UPDATE jobs SET metadata_json=? WHERE job_id=?",
                        (json.dumps(metadata, ensure_ascii=False, sort_keys=True), request.request_id),
                    )
                    self.store._connection.commit()
                # Keep the public error in the manager's in-memory sidecar;
                # durable transition remains the authoritative state machine.
                with self.lock:
                    self._errors = getattr(self, "_errors", {})
                    self._errors[request.request_id] = terminal["error"]
        except DurableJobError:
            with self.lock:
                self.status, self.error = "blocked", "durable job state could not be finalized"
        finally:
            with self.lock:
                if observation is not None:
                    self.observations.pop(request.request_id, None)

    def get(self, job_id):
        try:
            with self.db_lock:
                record = self.store.get(job_id)
        except DurableJobError:
            return None
        if record is None:
            return None
        result = self._public_record(record)
        with self.lock:
            error = getattr(self, "_errors", {}).get(job_id)
        if error:
            result["error"] = copy.deepcopy(error)
        return result

    def shutdown(self):
        with self.lock:
            self.closing = True
        self.queue.shutdown(stop_engine=True, timeout=15)
        self.store.close()


