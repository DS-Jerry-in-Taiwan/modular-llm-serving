"""Repeated baseline measurements with explicit cache/metric scope and provenance."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import threading
import time

from .acquisition import verify_model
from .telemetry import command, gpu_probe, memory_probe
from .runtime import LocalServer, RuntimeFailure, RunCancelled, cancellation_signal, runtime_identity
from .tasks import load_dataset, evaluate
from .telemetry import ResourceSampler, response_metrics
from .artifacts import digest_json, output_hash, write_json_new, new_run


def source_identity():
    root = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("baseline.py", "runtime.py", "acquisition.py", "tasks.py", "config.py", "telemetry.py", "artifacts.py")}
    return {"files": hashes, "sha256": digest_json(hashes), "git_commit": command(["git", "rev-parse", "HEAD"]),
            "git_status": command(["git", "status", "--short"])}


def aggregate_samples(samples):
    result = {}
    for case_id in sorted({s["case_id"] for s in samples}):
        group = [s for s in samples if s["case_id"] == case_id and s["status"] == "completed"]
        metrics = {}
        for key in ("request_seconds", "prompt_tokens_per_second", "generated_tokens_per_second"):
            values = [s["metrics"][key]["value"] for s in group if s["metrics"][key]["status"] == "ok"]
            metrics[key] = {"status": "ok", "count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)} if values else {"status": "unavailable"}
        result[case_id] = metrics
    return result


def baseline(config, dataset_path, binary, model_path, output_root, repeats=2, baseline_run_id=None):
    if type(repeats) is not int or not 1 <= repeats <= 2:
        raise ValueError("initial measurement budget supports 1..2 repeats")
    dataset, dataset_sha = load_dataset(dataset_path)
    if len(dataset["cases"]) > config.get("budgets", {}).get("suite_prompts", 6):
        raise ValueError("dataset exceeds initial prompt budget")
    directory, record = new_run(output_root, "optimization" if baseline_run_id else "baseline")
    record.update(config=config, config_sha256=digest_json(config), dataset=dataset, dataset_sha256=dataset_sha,
                  source=source_identity(), samples=[], quality_status="pending_review",
                  baseline_run_id=baseline_run_id or record["run_id"],
                  measurement_protocol={"repeats": repeats, "warmup_requests": 1, "order": "repeat-major, dataset order", "cache_prompt": False,
                                        "process_state": "new server process; model filesystem cache state not controlled", "measurement_state": "after one warmup; no prefix caching requested"},
                  baseline_eligibility={"measurement_complete": False, "quality_status": "pending_review", "eligible_for_quality_comparison": False})
    server = None
    sampler = ResourceSampler()
    previous = {}
    start = time.monotonic()
    progress = {"run_id": record["run_id"], "stage": "initializing", "completed_samples": 0}
    def update_progress(stage):
        progress.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start)
        (directory / "progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
    update_progress("verifying_model")
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM): previous[sig] = signal.signal(sig, cancellation_signal)
    try:
        record["model_identity"] = verify_model(model_path, config["model"])
        record["runtime_identity"] = runtime_identity(binary, config["runtime"]["revision"])
        record["environment"] = {"platform": platform.platform(), "hostname": platform.node(), "cpu": command(["lscpu"]),
                                 "load_average": list(os.getloadavg()), "gpu_before": gpu_probe(), "memory_before": memory_probe()}
        record["environment_sha256"] = digest_json(record["environment"])
        if record["environment"]["gpu_before"]["status"] != "ok": raise RuntimeFailure("usable GPU telemetry required")
        write_json_new(directory / "manifest.json", {key: record[key] for key in ("schema_version", "run_id", "kind", "config", "config_sha256", "dataset", "dataset_sha256", "source", "model_identity", "runtime_identity", "environment", "environment_sha256", "measurement_protocol", "baseline_run_id")})
        record["manifest_sha256"] = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        sampler.start()
        server = LocalServer(binary, model_path, config["runtime"], directory, time.monotonic() + min(1200, config["budgets"]["experiment_seconds"]))
        sampler.pid = lambda: server.process.pid if server.process is not None else None
        record["argv"] = server.argv
        update_progress("loading_model")
        server.start()
        record["load_seconds"] = server.load_seconds
        record["offload"] = server.offload_evidence()
        if record["offload"]["gpu_layers_observed"] < 1: raise RuntimeFailure("GPU offload must be observed")
        update_progress("warmup")
        record["warmup"] = server.chat("請只回答：暖機完成", max_tokens=16)
        if record["warmup"]["truncated"]: raise RuntimeFailure("warmup truncated")
        for repetition in range(1, repeats + 1):
            for case in dataset["cases"]:
                sample_id = f'{case["id"]}-r{repetition}'
                update_progress(sample_id)
                item = {"sample_id": sample_id, "case_id": case["id"], "repeat": repetition, "status": "running"}
                sample_start = time.monotonic() - sampler.start_time
                try:
                    response = server.chat(case["prompt"], case["max_tokens"])
                    item.update(status="completed", result=response, output_sha256=output_hash(response), evaluation=evaluate(case, response), metrics=response_metrics(response))
                except (Exception, KeyboardInterrupt) as exc:
                    item.update(status="cancelled" if isinstance(exc, (RunCancelled, KeyboardInterrupt)) else "failed", failure={"type": type(exc).__name__, "reason": str(exc)[:500]}, raw_response=server.last_response)
                    if isinstance(exc, (RunCancelled, KeyboardInterrupt)):
                        record["samples"].append(item)
                        write_json_new(directory / (sample_id + ".json"), item)
                        raise
                item["resources"] = sampler.summary(sample_start, time.monotonic() - sampler.start_time)
                record["samples"].append(item)
                write_json_new(directory / (sample_id + ".json"), item)
                progress["completed_samples"] += 1
        record["status"] = "completed" if all(s["status"] == "completed" for s in record["samples"]) else "failed"
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="cancelled" if isinstance(exc, (RunCancelled, KeyboardInterrupt)) else "failed", failure={"type": type(exc).__name__, "reason": str(exc)[:500]})
    finally:
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        try:
            if server is not None:
                server.stop()
                record["cleanup_confirmed"] = server.cleanup_confirmed
            else: record["cleanup_confirmed"] = True
            if sampler.start_time is not None:
                sampler.stop()
                record["resources"] = sampler.summary()
                write_json_new(directory / "resource-samples.json", sampler.samples)
        except Exception as exc:
            record.update(status="failed", cleanup_confirmed=False, cleanup_failure=str(exc)[:500])
        record["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.json") if p.name not in {"progress.json", "run.json"}}
        record["aggregates"] = aggregate_samples(record["samples"])
        expected_count = len(dataset["cases"]) * repeats
        complete = record["status"] == "completed" and len(record["samples"]) == expected_count and record["cleanup_confirmed"] and all(not s["result"].get("truncated") for s in record["samples"])
        record["baseline_eligibility"]["measurement_complete"] = complete
        record["elapsed_seconds"] = time.monotonic() - start
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_new(directory / "run.json", record)
        update_progress(record["status"])
        for sig, handler in previous.items(): signal.signal(sig, handler)
    render_baseline_report(directory)
    return record, directory


def apply_baseline_review(run_dir, review_path):
    directory = Path(run_dir)
    raw = (directory / "run.json").read_bytes()
    record = json.loads(raw)
    review = json.loads(Path(review_path).read_text())
    if review.get("run_id") != record["run_id"] or review.get("run_sha256") != hashlib.sha256(raw).hexdigest() or review.get("dataset_sha256") != record["dataset_sha256"]:
        raise ValueError("review must bind exact run bytes and dataset")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip(): raise ValueError("reviewer required")
    cases = {c["id"]: c for c in record["dataset"]["cases"]}
    samples = {s["sample_id"]: s for s in record["samples"] if cases[s["case_id"]]["manual_required"]}
    if set(review.get("samples", {})) != set(samples): raise ValueError("every manual sample must be reviewed")
    for sample_id, item in samples.items():
        result = item.get("result", {})
        if item["status"] != "completed" or result.get("truncated") or not result.get("content", "").strip(): raise ValueError("failed/empty/truncated sample cannot pass review")
        entry = review["samples"][sample_id]
        if entry.get("output_sha256") != output_hash(result): raise ValueError("actual output hash required")
        if set(entry.get("rubric", {})) != set(cases[item["case_id"]]["rubric"]) or not all(type(x) is bool for x in entry["rubric"].values()): raise ValueError("each rubric criterion required")
        if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip(): raise ValueError("review rationale required")
        entry["verdict"] = "pass" if all(entry["rubric"].values()) else "fail"
    review["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_new(directory / "review.json", review)
    render_baseline_report(directory)
    return review


def quality_results(record, review=None):
    result = {}
    cases = {c["id"]: c for c in record["dataset"]["cases"]}
    for item in record["samples"]:
        sid = item["sample_id"]
        if item["status"] != "completed" or item.get("result", {}).get("truncated"):
            result[sid] = "fail"
        elif cases[item["case_id"]]["manual_required"]:
            result[sid] = review["samples"][sid]["verdict"] if review else "pending_review"
        else: result[sid] = item["evaluation"]["quality_status"]
    return result


def render_baseline_report(run_dir):
    directory = Path(run_dir)
    record = json.loads((directory / "run.json").read_text())
    review = json.loads((directory / "review.json").read_text()) if (directory / "review.json").exists() else None
    quality = quality_results(record, review)
    eligible = record["baseline_eligibility"]["measurement_complete"] and quality and all(v == "pass" for v in quality.values())
    text = f'# {record["kind"]} {record["run_id"]}\n\nExecution: {record["status"]}; measurement complete: {record["baseline_eligibility"]["measurement_complete"]}; quality-reviewed eligible: {bool(eligible)}.\n\n'
    text += '| Sample | Execution | Quality | Request seconds |\n|---|---|---|---|\n'
    for item in record["samples"]:
        text += f'| {item["sample_id"]} | {item["status"]} | {quality[item["sample_id"]]} | {item.get("result", {}).get("elapsed_seconds", "unavailable")} |\n'
    text += '\nTTFT unavailable: non-stream response. Cache requested off; actual cached tokens retained per sample. New server process does not imply cold filesystem cache. Resource maxima are sampled; GPU is device-wide, system RAM host-wide, RSS owned server only. Host contention may affect timing.\n'
    text += '\nTwo repeats describe variation but do not support statistical-significance claims.\n'
    if review: text += '\nIndependent reviewer: ' + review["reviewer"] + '\n'
    (directory / "report.md").write_text(text)
    return text
