import hashlib
import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from inference_lab.evaluation.baseline import aggregate_samples, source_identity, baseline, apply_baseline_review, quality_results
from inference_lab.telemetry import ResourceSampler, response_metrics
from inference_lab.evaluation.tasks import load_dataset
from inference_lab.artifacts import output_hash

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
        with patch("inference_lab.telemetry.gpu_probe", return_value={"status": "ok", "memory_used_bytes": 456}), patch("inference_lab.telemetry.memory_probe", return_value={"status": "ok", "MemTotal_bytes": 1000, "MemAvailable_bytes": 300}):
            sampler.sample()
        summary = sampler.summary()
        self.assertEqual(summary["gpu_device_used_bytes"]["sampled_peak_bytes"], 456)
        self.assertEqual(summary["system_memory_used_bytes"]["sampled_peak_bytes"], 700)
        self.assertEqual(summary["server_rss_bytes"]["status"], "unavailable")
        self.assertIn("whole GPU", summary["scope"]["gpu_device_used_bytes"])

    def test_sampler_stop_failure_is_reported(self):
        sampler = ResourceSampler()
        class StuckWorker:
            def join(self, timeout):
                pass
            def is_alive(self):
                return True
        sampler.worker = StuckWorker()
        with self.assertRaisesRegex(RuntimeError, 'resource sampler did not stop'):
            sampler.stop()
        self.assertTrue(sampler.stop_event.is_set())

    def test_repeat_aggregates_keep_variation(self):
        samples = [{"case_id": "one", "status": "completed", "metrics": {key: {"status": "ok", "value": v} for key in ("request_seconds", "prompt_tokens_per_second", "generated_tokens_per_second")}} for v in (2, 4)]
        result = aggregate_samples(samples)["one"]["request_seconds"]
        self.assertEqual((result["count"], result["mean"], result["min"], result["max"]), (2, 3, 2, 4))

    def test_source_hashes_include_actual_execution_code_not_future_comparison(self):
        identity = source_identity()
        self.assertIn("evaluation/baseline.py", identity["files"])
        self.assertIn("runtime.py", identity["files"])
        for name in ('config.py', 'telemetry.py', 'artifacts.py'):
            self.assertIn(name, identity['files'])
        self.assertNotIn("evaluation/comparison.py", identity["files"])
        self.assertEqual(len(identity["sha256"]), 64)

    def test_source_identity_covers_transitive_execution_and_detects_changes(self):
        identity = source_identity()
        expected = {'__init__.py', 'evaluation/__init__.py', 'evaluation/baseline.py', 'evaluation/tasks.py',
                    'runtime.py', 'acquisition.py', 'artifacts.py', 'config.py', 'telemetry.py'}
        self.assertEqual(set(identity['files']), expected)
        root = DATASET.parent.parent / 'inference_lab'
        original_read = Path.read_bytes
        for name in expected:
            changed = root / name
            def changed_bytes(path):
                raw = original_read(path)
                return raw + b'\n# changed execution source\n' if path == changed else raw
            with self.subTest(name=name), patch.object(Path, 'read_bytes', changed_bytes):
                self.assertNotEqual(source_identity()['sha256'], identity['sha256'])

    def test_evaluation_has_no_legacy_module_aliases_or_reverse_dependencies(self):
        for name in ('tasks', 'baseline', 'comparison'):
            self.assertIsNone(importlib.util.find_spec('inference_lab.' + name))
            self.assertIsNotNone(importlib.util.find_spec('inference_lab.evaluation.' + name))
        root = DATASET.parent.parent / 'inference_lab'
        for name in ('config', 'artifacts', 'telemetry', 'runtime', 'acquisition', 'demo', 'web/jobs', 'web/server'):
            tree = ast.parse((root / (name + '.py')).read_text())
            imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any(module.startswith(('evaluation', 'inference_lab.evaluation', '..evaluation')) for module in imports), name)

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
        with tempfile.TemporaryDirectory() as d, patch("inference_lab.evaluation.baseline.verify_model", side_effect=ValueError("bad hash")):
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
