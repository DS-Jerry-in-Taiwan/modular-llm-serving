import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference_lab.artifacts import digest_json
from inference_lab.comparison import compare, ComparisonError, load_evidence, config_differences


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.left = self.root / "baseline"
        self.right = self.root / "candidate"
        self.make_run(self.left, "base", "baseline", 24, 2)
        self.make_run(self.right, "candidate", "optimization", 40, 1)

    def tearDown(self): self.temp.cleanup()

    def make_run(self, directory, run_id, kind, layers, elapsed):
        directory.mkdir()
        config = {"runtime": {"gpu_layers": layers}}
        source_files = {"runtime.py": "a" * 64}
        dataset = {"cases": [{"id": "one", "prompt": "extract x", "max_tokens": 10, "expected_json": {"x": 1}, "manual_required": False, "rubric": []}]}
        run = {"run_id": run_id, "kind": kind, "baseline_run_id": "base", "status": "completed", "cleanup_confirmed": True,
               "baseline_eligibility": {"measurement_complete": True}, "config": config, "config_sha256": digest_json(config),
               "source": {"files": source_files, "sha256": digest_json(source_files)}, "dataset": dataset, "dataset_sha256": "b" * 64,
               "model_identity": {"sha256": "c" * 64, "size_bytes": 4}, "runtime_identity": {"sha256": "d" * 64, "configured_revision": "e" * 40},
               "environment": {"hostname": "test", "platform": "Linux", "gpu_before": {"name": "testGPU", "memory_total_bytes": 12000, "count": 1}, "cpu": {"output": "Model name: testCPU"}, "load_average": [1, 1, 1]},
               "measurement_protocol": {"repeats": 2, "cache_prompt": False}, "samples": [],
               "resources": {k: {"status": "ok", "sampled_peak_bytes": 1000} for k in ("gpu_device_used_bytes", "system_memory_used_bytes", "server_rss_bytes")}}
        run["environment_sha256"] = digest_json(run["environment"])
        for repeat in (1, 2):
            run["samples"].append({"sample_id": f"one-r{repeat}", "case_id": "one", "repeat": repeat, "status": "completed", "result": {"content": '{"x":1}', "prompt": "extract x", "truncated": False, "elapsed_seconds": elapsed, "response": {"usage": {"completion_tokens": 5}}, "generation": {"max_tokens": 10, "temperature": 0, "seed": 42, "cache_prompt": False}}, "metrics": {"cached_prompt_tokens": {"status": "ok", "value": 0}, "generated_tokens_per_second": {"status": "ok", "value": 5 / elapsed}}})
        self.save_run(directory, run)

    def save_run(self, directory, run):
        keys = ("run_id", "kind", "config", "config_sha256", "dataset", "dataset_sha256", "source", "model_identity", "runtime_identity", "environment", "environment_sha256", "measurement_protocol", "baseline_run_id")
        manifest_raw = json.dumps({k: run.get(k) for k in keys}).encode()
        (directory / "manifest.json").write_bytes(manifest_raw)
        run["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
        run["artifact_sha256"] = {"manifest.json": run["manifest_sha256"]}
        raw = json.dumps(run).encode()
        (directory / "run.json").write_bytes(raw)
        review = {"run_id": run["run_id"], "run_sha256": hashlib.sha256(raw).hexdigest(), "dataset_sha256": run["dataset_sha256"], "reviewer": "independent AI test reviewer", "samples": {}}
        (directory / "review.json").write_text(json.dumps(review))

    def mutate(self, key, value):
        run = json.loads((self.right / "run.json").read_text())
        run[key] = value
        self.save_run(self.right, run)

    def test_single_factor_comparison_is_immutable_and_reports_token_counts(self):
        result, output = compare(self.left, self.right, self.root / "comparison")
        self.assertEqual(result["measured_request_totals_seconds"]["latency_ratio"], 2)
        self.assertEqual(result["per_case"]["one"]["candidate_output_tokens"], [5, 5])
        self.assertTrue((output / "report.md").exists())
        with self.assertRaises(FileExistsError): compare(self.left, self.right, output)

    def test_rejects_unmatched_model_dataset_runtime_source_hardware_protocol(self):
        mutations = {"model_identity": {"sha256": "f" * 64, "size_bytes": 4}, "dataset_sha256": "f" * 64,
                     "runtime_identity": {"sha256": "f" * 64, "configured_revision": "e" * 40}, "measurement_protocol": {"repeats": 2, "cache_prompt": True}}
        for key, value in mutations.items():
            with self.subTest(key=key):
                original = json.loads((self.right / "run.json").read_text())
                self.mutate(key, value)
                with self.assertRaises(ComparisonError): compare(self.left, self.right, self.root / key)
                self.save_run(self.right, original)
        run = json.loads((self.right / "run.json").read_text())
        run["source"]["files"]["runtime.py"] = "f" * 64
        run["source"]["sha256"] = digest_json(run["source"]["files"])
        self.save_run(self.right, run)
        with self.assertRaises(ComparisonError): compare(self.left, self.right, self.root / "source")

    def test_rejects_quality_pending_truncation_missing_or_duplicated_samples(self):
        original = json.loads((self.right / "run.json").read_text())
        (self.right / "review.json").unlink()
        with self.assertRaises(ComparisonError): load_evidence(self.right)
        self.save_run(self.right, original)
        for mutation in ("truncated", "missing", "duplicated"):
            run = copy.deepcopy(original)
            if mutation == "truncated": run["samples"][0]["result"]["truncated"] = True
            elif mutation == "missing": run["samples"].pop()
            else: run["samples"].append(copy.deepcopy(run["samples"][0]))
            self.save_run(self.right, run)
            with self.assertRaises(ComparisonError): compare(self.left, self.right, self.root / mutation)

    def test_rejects_undeclared_config_change_actual_cache_or_generation_change(self):
        original = json.loads((self.right / "run.json").read_text())
        for change in ("config", "cache", "generation", "prompt"):
            run = copy.deepcopy(original)
            if change == "config":
                run["config"]["threads"] = 99
                run["config_sha256"] = digest_json(run["config"])
            elif change == "cache": run["samples"][0]["metrics"]["cached_prompt_tokens"]["value"] = 1
            elif change == "generation": run["samples"][0]["result"]["generation"]["max_tokens"] = 20
            else: run["samples"][0]["result"]["prompt"] = "different task"
            self.save_run(self.right, run)
            with self.assertRaises(ComparisonError): compare(self.left, self.right, self.root / change)

    def test_tampered_artifact_or_review_hash_rejected(self):
        (self.right / "manifest.json").write_text("tampered")
        with self.assertRaises(ComparisonError): load_evidence(self.right)

    def test_hardware_mismatch_rejected(self):
        run = json.loads((self.right / "run.json").read_text())
        run["environment"]["gpu_before"]["name"] = "differentGPU"
        self.save_run(self.right, run)
        with self.assertRaises(ComparisonError): compare(self.left, self.right, self.root / "hardware")


if __name__ == "__main__": unittest.main()
