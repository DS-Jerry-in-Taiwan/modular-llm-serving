import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference_lab.llmops import (
    DurableJobStore,
    EvidenceRef,
    InvalidTransitionError,
    JobConflictError,
    ProvenanceError,
    RECOVERY_POLICY,
    evaluate_fixture,
)


COMMIT = "a" * 40


class DurableStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evidence = self.root / "raw.json"
        self.evidence.write_text('{"answer":"fixture"}\n', encoding="utf-8")
        self.ref = EvidenceRef(str(self.evidence), hashlib.sha256(self.evidence.read_bytes()).hexdigest(), COMMIT)

    def tearDown(self):
        self.tmp.cleanup()

    def test_atomic_transitions_and_idempotent_terminal_record(self):
        db = self.root / "jobs.sqlite"
        with DurableJobStore(db) as store:
            queued = store.create("request-1", "job-1", "run-1", metadata={"fixture": True})
            self.assertEqual(queued.state, "queued")
            self.assertEqual(store.transition("job-1", "running").state, "running")
            terminal = store.record_terminal("job-1", "completed", reason="fixture complete", evidence=self.ref)
            self.assertEqual(store.record_terminal("job-1", "completed", reason="fixture complete", evidence=self.ref), terminal)
            self.assertEqual([event.to_state for event in store.history("job-1")], ["queued", "running", "completed"])
            with self.assertRaises(JobConflictError):
                store.record_terminal("job-1", "failed", reason="different terminal")

    def test_restart_recovery_is_explicit_and_idempotent(self):
        db = self.root / "jobs.sqlite"
        with DurableJobStore(db) as store:
            store.create("request-1", "job-1", "run-1")
            store.transition("job-1", "running")
        with DurableJobStore(db) as restarted:
            recovered = restarted.recover()
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0].state, "failed")
            self.assertIn(RECOVERY_POLICY, restarted.history("job-1")[-1].reason)
            self.assertEqual(restarted.recover(), [])

    def test_missing_or_tampered_evidence_blocks_success(self):
        db = self.root / "jobs.sqlite"
        with DurableJobStore(db) as store:
            store.create("request-1", "job-1", "run-1")
            store.transition("job-1", "running")
            self.evidence.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(ProvenanceError):
                store.transition("job-1", "completed", evidence=self.ref)
            with self.assertRaises(InvalidTransitionError):
                store.transition("job-1", "queued")


class OfflineGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "evidence.json"
        self.raw.write_text('{"quality":1}\n', encoding="utf-8")
        self.raw_sha = hashlib.sha256(self.raw.read_bytes()).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, **changes):
        value = {
            "schema_version": 1,
            "source_commit": COMMIT,
            "evidence": {"path": str(self.raw), "sha256": self.raw_sha, "source_commit": COMMIT},
            "cases": [{"id": "one", "expected": {"ok": True}, "actual": {"ok": True}}],
        }
        value.update(changes)
        path = self.root / "fixture.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_fixture_gate_is_gpu_free_and_report_is_versioned(self):
        result = evaluate_fixture(self.fixture())
        self.assertTrue(result.passed)
        self.assertEqual(result.report["schema_version"], 1)
        self.assertEqual(result.report["source_commit"], COMMIT)
        self.assertEqual(result.report["evidence_sha256"], self.raw_sha)

    def test_missing_and_tampered_provenance_fail_closed(self):
        missing = self.fixture(evidence=None)
        self.assertFalse(evaluate_fixture(missing).passed)
        self.raw.write_text("changed\n", encoding="utf-8")
        self.assertFalse(evaluate_fixture(self.fixture()).passed)

    def test_case_failure_is_reported_without_gpu(self):
        result = evaluate_fixture(self.fixture(cases=[{"id": "one", "expected": 1, "actual": 2}]))
        self.assertFalse(result.passed)
        self.assertIn("case failed: one", result.report["reasons"])


if __name__ == "__main__":
    unittest.main()

