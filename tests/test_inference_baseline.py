import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from inference_lab.baseline import ResourceSampler, aggregate_samples, response_metrics, source_identity, baseline, apply_baseline_review, quality_results
from inference_lab.tasks import load_dataset, output_hash

DATASET = Path(__file__).resolve().parents[1] / "experiments" / "tasks-v1.json"


class BaselineTests(unittest.TestCase):
    def test_ttft_is_unavailable_and_cache_counts_are_observed(self):
        result = {"elapsed_seconds": 2, "response": {"timings": {"predicted_per_second": 5}, "usage": {"prompt_tokens_details": {"cached_tokens": 3}}}}
        metrics = response_metrics(result)
        self.assertEqual(metrics["ttft_seconds"]["status"], "unavailable")
        self.assertEqual(metrics["cached_prompt_tokens"]["value"], 3)
        self.assertEqual(metrics["prompt_tokens_per_second"]["status"], "unavailable")

    def test_nonfinite_metrics_are_unavailable(self):
        for value in (float("nan"), float("inf"), -1):
            metrics = response_metrics({"elapsed_seconds": value, "response": {}})
            self.assertEqual(metrics["request_seconds"]["status"], "unavailable")

    def test_sampled_peaks_scope_and_unavailable_not_zero(self):
        sampler = ResourceSampler(0.1)
        sampler.start_time = time.monotonic()
        with patch("inference_lab.baseline.gpu_probe", return_value={"status": "ok", "memory_used_bytes": 456}), patch("inference_lab.baseline.memory_probe", return_value={"status": "ok", "MemTotal_bytes": 1000, "MemAvailable_bytes": 300}):
            sampler.sample()
        summary = sampler.summary()
        self.assertEqual(summary["gpu_device_used_bytes"]["sampled_peak_bytes"], 456)
        self.assertEqual(summary["system_memory_used_bytes"]["sampled_peak_bytes"], 700)
        self.assertEqual(summary["server_rss_bytes"]["status"], "unavailable")
        self.assertIn("whole GPU", summary["scope"]["gpu_device_used_bytes"])

    def test_repeat_aggregates_keep_variation(self):
        samples = [{"case_id": "one", "status": "completed", "metrics": {key: {"status": "ok", "value": v} for key in ("request_seconds", "prompt_tokens_per_second", "generated_tokens_per_second")}} for v in (2, 4)]
        result = aggregate_samples(samples)["one"]["request_seconds"]
        self.assertEqual((result["count"], result["mean"], result["min"], result["max"]), (2, 3, 2, 4))

    def test_source_hashes_include_actual_execution_code_not_future_comparison(self):
        identity = source_identity()
        self.assertIn("baseline.py", identity["files"])
        self.assertIn("runtime.py", identity["files"])
        self.assertNotIn("comparison.py", identity["files"])
        self.assertEqual(len(identity["sha256"]), 64)

    def test_review_quality_cannot_pass_failed_or_truncated_samples(self):
        dataset, _ = load_dataset(DATASET)
        record = {"dataset": dataset, "samples": [
            {"sample_id": "summary-pilot-r1", "case_id": "summary-pilot", "status": "completed", "result": {"content": "partial", "truncated": True}},
            {"sample_id": "summary-pilot-r2", "case_id": "summary-pilot", "status": "failed"}]}
        forged = {"samples": {"summary-pilot-r1": {"verdict": "pass"}, "summary-pilot-r2": {"verdict": "pass"}}}
        self.assertEqual(set(quality_results(record, forged).values()), {"fail"})

    def test_invalid_repetitions_rejected_before_run(self):
        for n in (0, 3, True):
            with self.assertRaises(ValueError): baseline({}, DATASET, "/unused", "/unused", "/tmp", n)

    def test_failed_preload_cannot_be_measurement_complete(self):
        with tempfile.TemporaryDirectory() as d, patch("inference_lab.baseline.verify_model", side_effect=ValueError("bad hash")):
            record, directory = baseline({"model": {}}, DATASET, "/unused", "/unused", d)
            self.assertEqual(record["status"], "failed")
            self.assertFalse(record["baseline_eligibility"]["measurement_complete"])
            self.assertTrue(record["cleanup_confirmed"])
            self.assertTrue((directory / "progress.json").exists())

    def test_review_binds_each_repeat_not_previous_suite(self):
        dataset, dataset_hash = load_dataset(DATASET)
        case = dataset["cases"][0]
        samples = [{"sample_id": f'{case["id"]}-r{n}', "case_id": case["id"], "status": "completed", "result": {"content": "actual output", "truncated": False}, "evaluation": {"quality_status": "pending_review"}} for n in (1, 2)]
        run = {"run_id": "baseline", "kind": "baseline", "dataset": dataset, "dataset_sha256": dataset_hash, "samples": samples, "baseline_eligibility": {"measurement_complete": True}, "status": "completed"}
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            raw = json.dumps(run).encode()
            (directory / "run.json").write_bytes(raw)
            review = {"run_id": "baseline", "run_sha256": hashlib.sha256(raw).hexdigest(), "dataset_sha256": dataset_hash, "reviewer": "independent AI test reviewer", "samples": {s["sample_id"]: {"output_sha256": output_hash(s["result"]), "rubric": {r: True for r in case["rubric"]}, "rationale": "Compared this exact repeated output against source."} for s in samples}}
            path = directory / "input.json"
            review["samples"].pop(samples[1]["sample_id"])
            path.write_text(json.dumps(review))
            with self.assertRaises(ValueError): apply_baseline_review(directory, path)
            review["samples"][samples[1]["sample_id"]] = review["samples"][samples[0]["sample_id"]].copy()
            path.write_text(json.dumps(review))
            accepted = apply_baseline_review(directory, path)
            self.assertEqual(set(quality_results(run, accepted).values()), {"pass"})
            self.assertEqual((directory / "run.json").read_bytes(), raw)
            with self.assertRaises(FileExistsError): apply_baseline_review(directory, path)


if __name__ == "__main__": unittest.main()
