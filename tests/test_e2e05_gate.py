import hashlib
import tempfile
import unittest
from pathlib import Path

from inference_lab.llmops import E2E05GateRunner, Registry


class E2E05GateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evidence = self.root / "sanitized.json"
        self.evidence.write_text('{"fixture":true}\n', encoding="utf-8")
        self.sha = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        self.registry = Registry(self.root / "registry.jsonl")
        self.runner = E2E05GateRunner(self.registry)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, run_id, parent=None):
        return {
            "schema_version": 1, "run_id": run_id, "parent_run_id": parent,
            "model_id": "fixture-model", "config_id": "fixture-config",
            "dataset_id": "fixture-dataset", "evaluator_id": "fixture-evaluator",
            "source_commit": "a" * 40, "model_sha256": "b" * 64,
            "config_sha256": "c" * 64, "dataset_sha256": "d" * 64,
            "evidence": {"path": str(self.evidence), "sha256": self.sha},
        }

    def test_registers_three_manifests_in_lineage_and_accepts_only_gate_pass(self):
        records = self.runner.register_manifests(self.manifest("v1-24"), self.manifest("v1-40"), self.manifest("v2-packaged"))
        self.assertEqual(records.run_ids, ("v1-24", "v1-40", "v2-packaged"))
        self.assertEqual([x.manifest["run_id"] for x in self.registry.lineage("v2-packaged")], ["v2-packaged", "v1-40", "v1-24"])
        reg = self.registry.get("v2-packaged")
        report = {"schema_version": 1, "run_id": "v2-packaged", "manifest_digest": reg.digest,
                  "quality_score": 1.0, "latency_ratio": .8, "gpu_peak_bytes": 123,
                  "provenance_status": "verified", "evidence_sha256": self.sha}
        self.assertEqual(self.runner.accept_baseline("v2-packaged", {**report, "quality_score": .5}).status, "fail")
        self.assertIsNone(self.runner.accepted_baseline_run_id)
        self.assertEqual(self.runner.accept_baseline("v2-packaged", report).status, "pass")
        self.assertEqual(self.runner.accepted_baseline_run_id, "v2-packaged")

    def test_tamper_and_missing_digest_fail_closed(self):
        self.runner.register_manifests(self.manifest("v1-24"), self.manifest("v1-40"), self.manifest("v2-packaged"))
        reg = self.registry.get("v2-packaged")
        report = {"schema_version": 1, "run_id": "v2-packaged", "manifest_digest": reg.digest,
                  "quality_score": 1.0, "latency_ratio": .8, "gpu_peak_bytes": 123,
                  "provenance_status": "verified", "evidence_sha256": self.sha}
        self.evidence.write_text("tampered\n", encoding="utf-8")
        decision = self.runner.accept_baseline("v2-packaged", report)
        self.assertEqual(decision.status, "fail")
        self.assertIsNone(self.runner.accepted_baseline_run_id)


if __name__ == "__main__":
    unittest.main()
