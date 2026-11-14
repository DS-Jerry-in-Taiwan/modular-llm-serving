"""Versioned usefulness suite; automatic filters are not human quality judgments."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

from .acquisition import verify_model
from .artifacts import output_hash, write_json_new, new_run
from .telemetry import command, gpu_probe, memory_probe
from .runtime import LocalServer, RuntimeFailure, RunCancelled, cancellation_signal, runtime_identity


def load_dataset(path):
    raw = Path(path).read_bytes()
    dataset = json.loads(raw)
    if dataset.get("schema_version") != 1 or not isinstance(dataset.get("cases"), list) or not dataset["cases"]:
        raise ValueError("invalid dataset schema")
    seen = set()
    for case in dataset["cases"]:
        if not isinstance(case.get("id"), str) or case["id"] in seen or not case["id"].replace("-", "").isalnum():
            raise ValueError("case ids must be unique safe names")
        seen.add(case["id"])
        if not isinstance(case.get("prompt"), str) or not case["prompt"]:
            raise ValueError("case prompt required")
        if type(case.get("max_tokens")) is not int or not 1 <= case["max_tokens"] <= 256:
            raise ValueError("case token cap must be 1..256")
        if type(case.get("manual_required")) is not bool or not isinstance(case.get("rubric"), list):
            raise ValueError("explicit manual review contract required")
        if case["manual_required"] and not case["rubric"]:
            raise ValueError("manual cases require rubric")
        if "expected_json" not in case and "checks" not in case:
            raise ValueError("case needs automatic check contract")
    return dataset, hashlib.sha256(raw).hexdigest()


def strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(strict_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def evaluate(case, result):
    content = result["content"]
    checks = {"not_truncated": not result.get("truncated", False)}
    if "expected_json" in case:
        try:
            checks["exact_typed_json"] = strict_equal(json.loads(content, object_pairs_hook=unique_object), case["expected_json"])
        except ValueError:
            checks["exact_typed_json"] = False
    else:
        contract = case["checks"]
        checks["required_facts_filter"] = all(any(word in content for word in group) for group in contract.get("required_groups", []))
        checks["no_forbidden_claims_filter"] = not any(word in content for word in contract.get("forbidden", []))
        checks["length_filter"] = len(content) <= contract["max_characters"]
    passed = all(checks.values())
    return {"automatic_checks": checks, "automatic_status": "pass" if passed else "fail",
            "manual_required": case["manual_required"], "manual_status": "pending" if case["manual_required"] else "not_required",
            "quality_status": "pending_review" if case["manual_required"] else ("pass" if passed else "fail"),
            "rubric": case["rubric"],
            "note": "Fact/keyword filters flag candidates only; manual rubric review determines summary/code quality." if case["manual_required"] else "Exact typed JSON equality, no extra fields or text."}


def suite(config, dataset_path, binary, model_path, output_root):
    dataset, dataset_sha = load_dataset(dataset_path)
    directory, record = new_run(output_root, "task_suite")
    record.update(config=config, dataset=dataset, dataset_sha256=dataset_sha, cases=[], quality_status="pending_review")
    server = None
    previous = {}
    start = time.monotonic()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, cancellation_signal)
    try:
        record["model_identity"] = verify_model(model_path, config["model"])
        record["runtime_identity"] = runtime_identity(binary, config["runtime"]["revision"])
        record["project"] = command(["git", "rev-parse", "HEAD"])
        record["host"] = {"gpu_before": gpu_probe(), "memory_before": memory_probe()}
        if record["host"]["gpu_before"]["status"] != "ok":
            raise RuntimeFailure("usable GPU telemetry required")
        server = LocalServer(binary, model_path, config["runtime"], directory, time.monotonic() + min(1200, config["budgets"]["experiment_seconds"]))
        record["argv"] = server.argv
        server.start()
        record["load_seconds"] = server.load_seconds
        record["offload"] = server.offload_evidence()
        if record["offload"]["gpu_layers_observed"] < 1:
            raise RuntimeFailure("GPU offload must be demonstrated")
        for case in dataset["cases"]:
            item = {"case_id": case["id"], "category": case["category"], "status": "running"}
            try:
                result = server.chat(case["prompt"], case["max_tokens"])
                item.update(status="completed", result=result, output_sha256=output_hash(result), evaluation=evaluate(case, result))
            except (Exception, KeyboardInterrupt) as exc:
                item.update(status="cancelled" if isinstance(exc, (RunCancelled, KeyboardInterrupt)) else "failed", failure={"type": type(exc).__name__, "reason": str(exc)[:500]}, raw_response=server.last_response)
                if isinstance(exc, (RunCancelled, KeyboardInterrupt)):
                    record["cases"].append(item)
                    write_json_new(directory / (case["id"] + ".json"), item)
                    raise
            record["cases"].append(item)
            write_json_new(directory / (case["id"] + ".json"), item)
        record["status"] = "completed" if all(x["status"] == "completed" for x in record["cases"]) else "failed"
        record["host"]["gpu_during"] = gpu_probe()
        if record["host"]["gpu_during"]["status"] != "ok":
            raise RuntimeFailure("GPU telemetry lost during suite")
        record["host"]["memory_during"] = memory_probe()
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="cancelled" if isinstance(exc, (RunCancelled, KeyboardInterrupt)) else "failed", failure={"type": type(exc).__name__, "reason": str(exc)[:500]})
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if server is not None:
                server.stop()
                record["cleanup_confirmed"] = server.cleanup_confirmed
            else:
                record["cleanup_confirmed"] = True
        except Exception as exc:
            record.update(status="failed", cleanup_confirmed=False, cleanup_failure=str(exc)[:500])
        record["elapsed_seconds"] = time.monotonic() - start
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_new(directory / "run.json", record)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    render_report(directory)
    return record, directory


def apply_review(run_dir, review_path):
    directory = Path(run_dir)
    run_bytes = (directory / "run.json").read_bytes()
    run = json.loads(run_bytes)
    review = json.loads(Path(review_path).read_text())
    if review.get("run_sha256") != hashlib.sha256(run_bytes).hexdigest():
        raise ValueError("review must bind immutable run content hash")
    if review.get("run_id") != run["run_id"] or review.get("dataset_sha256") != run["dataset_sha256"]:
        raise ValueError("review must bind exact run and dataset")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("reviewer identity required")
    expected = {c["id"]: c for c in run["dataset"]["cases"] if c["manual_required"]}
    entries = review.get("cases", {})
    if set(entries) != set(expected):
        raise ValueError("review must cover every manual case exactly once")
    executed = {item["case_id"]: item for item in run["cases"]}
    for key, case in expected.items():
        if executed.get(key, {}).get("status") != "completed":
            raise ValueError("cannot review a case without a completed response")
        result = executed[key].get("result", {})
        if not isinstance(result.get("content"), str) or not result["content"].strip() or result.get("truncated"):
            raise ValueError("cannot review malformed, empty or truncated response")
        entry = entries[key]
        if entry.get("output_sha256") != output_hash(result):
            raise ValueError("review must bind actual output hash")
        if set(entry.get("rubric", {})) != set(case["rubric"]) or not all(type(v) is bool for v in entry["rubric"].values()):
            raise ValueError("review must decide each rubric criterion")
        if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip():
            raise ValueError("review rationale required")
        entry["verdict"] = "pass" if all(entry["rubric"].values()) else "fail"
    review["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_new(directory / "review.json", review)
    render_report(directory)
    return review


def render_report(run_dir):
    directory = Path(run_dir)
    run = json.loads((directory / "run.json").read_text())
    review = json.loads((directory / "review.json").read_text()) if (directory / "review.json").exists() else None
    rows = []
    for item in run["cases"]:
        ev = item.get("evaluation", {})
        reviewed = review["cases"].get(item["case_id"]) if review else None
        quality = reviewed["verdict"] if reviewed else ev.get("quality_status", "unavailable")
        if item.get("result", {}).get("truncated"):
            quality = "fail (truncated)"
        rows.append(f'| {item["case_id"]} | {item["status"]} | {ev.get("automatic_status", "unavailable")} | {quality} |')
    text = f'# Task suite {run["run_id"]}\n\nExecution: {run["status"]}; cleanup: {run["cleanup_confirmed"]}.\n\n'
    text += '| Case | Execution | Automatic checks | Quality |\n|---|---|---|---|\n' + '\n'.join(rows)
    text += '\n\nAutomatic summary/code checks are candidate filters. Independent rubric review is required; pending is not pass.\n'
    if review:
        text += f'\nReviewer: {review["reviewer"]}\n'
        for key, entry in review["cases"].items():
            text += f'\n- {key}: {entry["verdict"]} — {entry["rationale"]}\n'
    # The report is a derived view; immutable run/case/review evidence is never overwritten.
    (directory / "report.md").write_text(text)
    return text
