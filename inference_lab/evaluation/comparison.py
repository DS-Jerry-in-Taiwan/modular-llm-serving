"""Validate and compare immutable, quality-reviewed single-factor experiments."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..artifacts import digest_json
from .tasks import evaluate
from ..artifacts import output_hash, write_json_new


class ComparisonError(ValueError):
    pass


def config_differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result[name] = {"baseline": left.get(key), "candidate": right.get(key)}
            else: result.update(config_differences(left[key], right[key], name))
        return result
    return {} if left == right else {prefix: {"baseline": left, "candidate": right}}


def hardware_identity(run):
    env = run["environment"]
    gpu = env["gpu_before"]
    cpu_lines = [s.strip() for s in env.get("cpu", {}).get("output", "").splitlines() if s.split(":", 1)[0].strip() in {"Architecture", "CPU(s)", "Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core"}]
    return {"hostname": env["hostname"], "platform": env["platform"], "gpu_name": gpu.get("name"),
            "gpu_total_bytes": gpu.get("memory_total_bytes"), "gpu_count": gpu.get("count"), "cpu": cpu_lines}


def load_evidence(directory):
    directory = Path(directory)
    raw = (directory / "run.json").read_bytes()
    run = json.loads(raw)
    if run.get("status") != "completed" or not run.get("cleanup_confirmed") or not run.get("baseline_eligibility", {}).get("measurement_complete"):
        raise ComparisonError("incomplete execution, measurements or cleanup")
    for name, sha in run.get("artifact_sha256", {}).items():
        if Path(name).name != name or hashlib.sha256((directory / name).read_bytes()).hexdigest() != sha:
            raise ComparisonError("artifact identity mismatch")
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != run.get("manifest_sha256"):
        raise ComparisonError("manifest identity mismatch")
    manifest = json.loads(manifest_raw)
    for key in ("run_id", "kind", "config", "config_sha256", "dataset", "dataset_sha256", "source", "model_identity", "runtime_identity", "environment", "environment_sha256", "measurement_protocol", "baseline_run_id"):
        if manifest.get(key) != run.get(key):
            raise ComparisonError("pre-start manifest does not match final run")
    if digest_json(run["environment"]) != run.get("environment_sha256"):
        raise ComparisonError("environment identity mismatch")
    if digest_json(run["config"]) != run.get("config_sha256") or digest_json(run["source"]["files"]) != run["source"].get("sha256"):
        raise ComparisonError("config/source identity mismatch")
    review_path = directory / "review.json"
    if not review_path.exists(): raise ComparisonError("quality review pending")
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    if review.get("run_id") != run["run_id"] or review.get("run_sha256") != hashlib.sha256(raw).hexdigest() or review.get("dataset_sha256") != run["dataset_sha256"]:
        raise ComparisonError("review not bound to exact run/dataset")
    if not review.get("reviewer", "").strip(): raise ComparisonError("reviewer absent")
    cases = {c["id"]: c for c in run["dataset"]["cases"]}
    expected = {(case_id, repeat) for case_id in cases for repeat in range(1, run["measurement_protocol"]["repeats"] + 1)}
    observed = [(s["case_id"], s["repeat"]) for s in run["samples"]]
    if set(observed) != expected or len(observed) != len(expected): raise ComparisonError("samples missing or duplicated")
    manual_ids = {s["sample_id"] for s in run["samples"] if cases[s["case_id"]]["manual_required"]}
    if set(review.get("samples", {})) != manual_ids: raise ComparisonError("review sample coverage mismatch")
    for sample in run["samples"]:
        result = sample.get("result", {})
        if sample["status"] != "completed" or result.get("truncated") or not result.get("content", "").strip(): raise ComparisonError("failed/missing/truncated sample")
        case = cases[sample["case_id"]]
        if result.get("prompt") != case["prompt"]:
            raise ComparisonError("actual prompt differs from fixed dataset")
        generation = result.get("generation", {})
        if generation != {"max_tokens": case["max_tokens"], "temperature": 0, "seed": 42, "cache_prompt": False}:
            raise ComparisonError("generation settings differ from fixed experiment protocol")
        if case["manual_required"]:
            entry = review["samples"][sample["sample_id"]]
            if entry.get("output_sha256") != output_hash(result): raise ComparisonError("review output identity mismatch")
            if set(entry.get("rubric", {})) != set(case["rubric"]) or not all(v is True for v in entry["rubric"].values()) or entry.get("verdict") != "pass" or not entry.get("rationale", "").strip():
                raise ComparisonError("quality review did not pass")
        elif evaluate(case, result)["quality_status"] != "pass": raise ComparisonError("strict automatic quality failed")
        elapsed = result.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0: raise ComparisonError("invalid elapsed metric")
    return run, {"run_sha256": hashlib.sha256(raw).hexdigest(), "review_sha256": hashlib.sha256(review_raw).hexdigest(), "manifest_sha256": run["manifest_sha256"]}


def paired_metrics(baseline, candidate):
    rows = {}
    for case in baseline["dataset"]["cases"]:
        groups = [[s for s in run["samples"] if s["case_id"] == case["id"]] for run in (baseline, candidate)]
        means = [statistics.mean(s["result"]["elapsed_seconds"] for s in group) for group in groups]
        tokens = [[s["result"]["response"].get("usage", {}).get("completion_tokens") for s in group] for group in groups]
        outputs = [[output_hash(s["result"]) for s in group] for group in groups]
        throughputs = [[s["metrics"]["generated_tokens_per_second"].get("value") for s in group if s["metrics"]["generated_tokens_per_second"]["status"] == "ok"] for group in groups]
        rows[case["id"]] = {"baseline_request_seconds": means[0], "candidate_request_seconds": means[1],
                             "latency_ratio": means[0] / means[1], "latency_reduction_percent": (1 - means[1] / means[0]) * 100,
                             "baseline_range_seconds": [min(s["result"]["elapsed_seconds"] for s in groups[0]), max(s["result"]["elapsed_seconds"] for s in groups[0])],
                             "candidate_range_seconds": [min(s["result"]["elapsed_seconds"] for s in groups[1]), max(s["result"]["elapsed_seconds"] for s in groups[1])],
                             "baseline_output_tokens": tokens[0], "candidate_output_tokens": tokens[1], "outputs_identical_by_repeat": outputs[0] == outputs[1],
                             "baseline_generated_tokens_per_second": statistics.mean(throughputs[0]) if throughputs[0] else None,
                             "candidate_generated_tokens_per_second": statistics.mean(throughputs[1]) if throughputs[1] else None}
    return rows


def compare(baseline_dir, candidate_dir, output_dir):
    before, before_ids = load_evidence(baseline_dir)
    after, after_ids = load_evidence(candidate_dir)
    if before.get("kind") != "baseline" or after.get("kind") != "optimization" or after.get("baseline_run_id") != before["run_id"]:
        raise ComparisonError("candidate must link exact baseline")
    for label, left, right in (
        ("model", (before["model_identity"]["sha256"], before["model_identity"]["size_bytes"]), (after["model_identity"]["sha256"], after["model_identity"]["size_bytes"])),
        ("runtime", (before["runtime_identity"]["sha256"], before["runtime_identity"]["configured_revision"]), (after["runtime_identity"]["sha256"], after["runtime_identity"]["configured_revision"])),
        ("dataset", (before["dataset_sha256"], before["dataset"]), (after["dataset_sha256"], after["dataset"])),
        ("source", before["source"]["sha256"], after["source"]["sha256"]),
        ("hardware", hardware_identity(before), hardware_identity(after)),
        ("protocol", before["measurement_protocol"], after["measurement_protocol"])):
        if left != right: raise ComparisonError(f"unmatched {label}: not same-condition comparison")
    differences = config_differences(before["config"], after["config"])
    if set(differences) != {"runtime.gpu_layers"}: raise ComparisonError("only declared single-factor GPU layers change is allowed")
    before_samples = {(s["case_id"], s["repeat"]): s for s in before["samples"]}
    for sample in after["samples"]:
        previous = before_samples[(sample["case_id"], sample["repeat"])]
        if previous["result"].get("generation") != sample["result"].get("generation"):
            raise ComparisonError("generation settings differ")
    for run in (before, after):
        if any(s["metrics"]["cached_prompt_tokens"].get("value") != 0 for s in run["samples"]):
            raise ComparisonError("actual prompt cache reuse differs from uncached comparison protocol")
    per_case = paired_metrics(before, after)
    totals = [sum(s["result"]["elapsed_seconds"] for s in run["samples"]) for run in (before, after)]
    result = {"schema_version": 1, "kind": "single_factor_comparison", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_run_id": before["run_id"], "candidate_run_id": after["run_id"], "baseline_evidence": before_ids, "candidate_evidence": after_ids,
              "evidence_locations": {"baseline": str(baseline_dir), "candidate": str(candidate_dir)},
              "experiment_identity": {"model": before["model_identity"], "runtime": before["runtime_identity"], "source": before["source"], "dataset_id": before["dataset"].get("dataset_id"), "dataset_sha256": before["dataset_sha256"], "hardware": hardware_identity(before)},
              "hypothesis": "More GPU layers reduce CPU offload work at the cost of higher VRAM consumption.",
              "declared_changes": differences, "quality": {"baseline": "pass", "candidate": "pass", "scope": "six fixed fixtures, two repeats, independent AI rubric reviews plus strict JSON checks"},
              "per_case": per_case,
              "measured_request_totals_seconds": {"baseline": totals[0], "candidate": totals[1], "latency_ratio": totals[0] / totals[1]},
              "resources": {"baseline": before["resources"], "candidate": after["resources"]},
              "environment": {"baseline_load_average": before["environment"]["load_average"], "candidate_load_average": after["environment"]["load_average"]},
              "limitations": ["Two repeats only; no statistical significance or production SLA claim.", "Host load may differ; sequential runs are not randomized.", "Output lengths/content may differ despite fixed prompts and generation settings; inspect per-case token counts.", "GPU/system peaks include competing processes; RSS is owned server only; all peaks sampled.", "New process, warm filesystem cache not controlled; TTFT unavailable from non-stream requests."]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json_new(output / "comparison.json", result)
    lines = [f'# Qwen3.8 single-factor comparison\n', f'Baseline: `{before["run_id"]}`; candidate: `{after["run_id"]}`.\n', f'GPU layers: {differences["runtime.gpu_layers"]["baseline"]} → {differences["runtime.gpu_layers"]["candidate"]}. Quality passed on the fixed fixtures in both runs.\n', '| Task | Baseline mean s | Candidate mean s | Observed latency ratio | Output tokens baseline → candidate |', '|---|---:|---:|---:|---|']
    for name, values in per_case.items(): lines.append(f'| {name} | {values["baseline_request_seconds"]:.2f} | {values["candidate_request_seconds"]:.2f} | {values["latency_ratio"]:.2f}× | {values["baseline_output_tokens"]} → {values["candidate_output_tokens"]} |')
    lines += ['', f'Measured request totals: {totals[0]:.2f}s → {totals[1]:.2f}s ({totals[0]/totals[1]:.2f}× observed latency ratio).', '', 'Resource trade-offs (sampled peaks, bytes):']
    for key in ("gpu_device_used_bytes", "system_memory_used_bytes", "server_rss_bytes"):
        lines.append(f'- {key}: {before["resources"][key]} → {after["resources"][key]}')
    lines += ['', 'Interpretation: these short Chinese summary, structured extraction and Python explanation/repair fixtures passed independent quality review. They support these limited uses on this configuration, not full-agent coding, long documents, image/video/audio generation or a general capability guarantee.', '', 'Limits:'] + ['- ' + x for x in result["limitations"]]
    with (output / "report.md").open("x") as handle: handle.write('\n'.join(lines) + '\n')
    return result, output
