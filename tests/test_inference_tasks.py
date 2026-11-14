import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_lab.evaluation.tasks import load_dataset, evaluate, strict_equal, apply_review, render_report, suite
from inference_lab.artifacts import output_hash

DATASET = Path(__file__).resolve().parents[1] / "experiments" / "tasks-v1.json"


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.sha = load_dataset(DATASET)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_six_cases_cover_three_categories(self):
        self.assertEqual(len(self.dataset["cases"]), 6)
        self.assertEqual({c["category"] for c in self.dataset["cases"]}, {"summary", "extraction", "code"})
        self.assertEqual(len(self.sha), 64)

    def test_exact_json_rejects_extra_keys_wrong_types_or_markdown(self):
        case = next(c for c in self.dataset["cases"] if c["id"] == "json-person")
        expected = case["expected_json"]
        for content, status in ((json.dumps(expected), "pass"), ('{"name":"小林","age":28,"age":29,"active":false}', "fail"), (json.dumps({**expected, "extra": 1}), "fail"), (json.dumps({**expected, "active": 0}), "fail"), (json.dumps({**expected, "age": 29.0}), "fail"), ("```json\n" + json.dumps(expected) + "\n```", "fail")):
            self.assertEqual(evaluate(case, {"content": content, "truncated": False})["quality_status"], status)

    def test_keyword_filter_cannot_claim_summary_quality_pass(self):
        case = self.dataset["cases"][0]
        result = evaluate(case, {"content": "12 9 3 逾時 11月20", "truncated": False})
        self.assertEqual(result["automatic_status"], "pass")
        self.assertEqual(result["quality_status"], "pending_review")

    def test_truncation_fails_automatic_check_even_with_correct_json(self):
        case = next(c for c in self.dataset["cases"] if c["id"] == "json-person")
        result = evaluate(case, {"content": json.dumps(case["expected_json"]), "truncated": True})
        self.assertEqual(result["quality_status"], "fail")

    def make_review_run(self):
        run = {"run_id": "test-run", "dataset_sha256": self.sha, "dataset": self.dataset, "status": "completed", "cleanup_confirmed": True, "cases": []}
        for case in self.dataset["cases"]:
            ev = evaluate(case, {"content": json.dumps(case.get("expected_json", "pending")), "truncated": False})
            run["cases"].append({"case_id": case["id"], "status": "completed", "result": {"content": "actual fixture output", "truncated": False}, "evaluation": ev})
        (self.root / "run.json").write_text(json.dumps(run))
        review = {"run_id": run["run_id"], "dataset_sha256": self.sha, "run_sha256": hashlib.sha256((self.root / "run.json").read_bytes()).hexdigest(), "reviewer": "independent-test-reviewer", "cases": {c["id"]: {"output_sha256": hashlib.sha256(b"actual fixture output").hexdigest(), "rubric": {r: True for r in c["rubric"]}, "rationale": "Reviewed the actual response against source and semantics."} for c in self.dataset["cases"] if c["manual_required"]}}
        path = self.root / "input-review.json"
        path.write_text(json.dumps(review))
        return run, review, path

    def test_review_binds_run_dataset_and_covers_rubric(self):
        run, review, path = self.make_review_run()
        review["dataset_sha256"] = "bad"
        path.write_text(json.dumps(review))
        with self.assertRaises(ValueError): apply_review(self.root, path)
        review["dataset_sha256"] = self.sha
        first = next(iter(review["cases"].values()))
        first["rubric"].pop(next(iter(first["rubric"])))
        path.write_text(json.dumps(review))
        with self.assertRaises(ValueError): apply_review(self.root, path)

    def test_review_appends_evidence_without_overwriting_run(self):
        _, _, path = self.make_review_run()
        original = (self.root / "run.json").read_bytes()
        apply_review(self.root, path)
        self.assertEqual((self.root / "run.json").read_bytes(), original)
        self.assertIn("independent-test-reviewer", (self.root / "report.md").read_text())
        with self.assertRaises(FileExistsError): apply_review(self.root, path)

    def test_unrun_case_cannot_receive_manual_pass(self):
        run, _, path = self.make_review_run()
        run["cases"][0]["status"] = "failed"
        (self.root / "run.json").write_text(json.dumps(run))
        with self.assertRaises(ValueError): apply_review(self.root, path)

    def test_review_rejects_changed_output_or_truncation(self):
        for mode in ("hash", "truncated"):
            run, review, path = self.make_review_run()
            if mode == "hash":
                review["cases"]["summary-pilot"]["output_sha256"] = "0" * 64
            else:
                run["cases"][0]["result"]["truncated"] = True
                (self.root / "run.json").write_text(json.dumps(run))
                review["run_sha256"] = hashlib.sha256((self.root / "run.json").read_bytes()).hexdigest()
            path.write_text(json.dumps(review))
            with self.assertRaises(ValueError): apply_review(self.root, path)

    def test_failed_model_suite_preserves_failed_run_and_cleanup(self):
        with patch("inference_lab.evaluation.tasks.verify_model", side_effect=ValueError("bad hash")):
            run, directory = suite({"model": {}}, DATASET, "/unused", "/unused", self.root)
        self.assertEqual(run["status"], "failed")
        self.assertTrue(run["cleanup_confirmed"])
        self.assertEqual(run["cases"], [])
        self.assertNotEqual(run["quality_status"], "pass")


if __name__ == "__main__": unittest.main()
