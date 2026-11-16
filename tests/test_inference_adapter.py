import threading
import time
import unittest

from inference_lab.evaluation.adapter import (
    AdapterError,
    InferenceAdapter,
    RunIdentityConflict,
    RunRequest,
)


class FakeBackend:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def infer(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        return self.result


def request(mode="cold", run_id="run-1", prompt="hello"):
    return RunRequest(run_id, "case-1", prompt, "model-1", "config-1", "dataset-1", mode)


def completed(text="answer"):
    return {
        "status": "completed",
        "text": text,
        "truncated": False,
        "metrics": {"request_seconds": 0.25, "prompt_tokens": 2,
                     "completion_tokens": 3, "total_tokens": 5},
    }


class InferenceAdapterTests(unittest.TestCase):
    def test_cold_and_resident_share_traceable_identity(self):
        cold = FakeBackend(completed("cold"))
        resident = FakeBackend(completed("resident"))
        adapter = InferenceAdapter(cold_backend=cold, resident_backend=resident)
        a = adapter.run(request("cold", "cold-run"))
        b = adapter.run(request("resident", "resident-run"))
        self.assertEqual(a.identity["case_id"], b.identity["case_id"])
        self.assertEqual(a.identity["prompt_hash"], b.identity["prompt_hash"])
        self.assertEqual(a.identity["model_config_dataset_hash"], b.identity["model_config_dataset_hash"])
        self.assertEqual(cold.calls[0].execution_mode, "cold")
        self.assertEqual(resident.calls[0].execution_mode, "resident")

    def test_success_requires_complete_metrics_and_nontruncated_output(self):
        backend = FakeBackend({"text": "ok", "truncated": False, "metrics": {"request_seconds": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        result = InferenceAdapter(cold_backend=backend, resident_backend=backend).run(request())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.metrics.total_tokens, 2)

        for malformed in (
            {"text": "ok", "truncated": False},
            {"text": "ok", "truncated": False, "metrics": {"request_seconds": 1}},
            {"text": "ok", "truncated": False, "metrics": {"request_seconds": float("nan"), "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        ):
            with self.subTest(malformed=malformed):
                fake = FakeBackend(malformed)
                record = InferenceAdapter(cold_backend=fake, resident_backend=fake).run(request(run_id="bad-" + str(len(fake.calls))))
                self.assertEqual(record.status, "failed")
                self.assertEqual(record.error["code"], "contract_violation")

    def test_truncation_is_failed_closed(self):
        fake = FakeBackend({"text": "partial", "truncated": True, "metrics": None})
        result = InferenceAdapter(cold_backend=fake, resident_backend=fake).run(request())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error["code"], "truncated")
        self.assertTrue(result.truncated)

    def test_backend_failure_is_safe_terminal_failure(self):
        fake = FakeBackend(error=RuntimeError("secret prompt should not leak"))
        result = InferenceAdapter(cold_backend=fake, resident_backend=fake).run(request())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, {"code": "backend_failure", "message": "inference backend failed"})

    def test_run_id_is_first_write_wins_and_idempotent(self):
        fake = FakeBackend(completed())
        adapter = InferenceAdapter(cold_backend=fake, resident_backend=fake)
        first = adapter.run(request())
        second = adapter.run(request())
        self.assertIs(first, second)
        self.assertEqual(len(fake.calls), 1)

    def test_concurrent_retries_execute_backend_once(self):
        class SlowBackend(FakeBackend):
            def infer(self, request):
                time.sleep(0.01)
                return super().infer(request)

        fake = SlowBackend(completed())
        adapter = InferenceAdapter(cold_backend=fake, resident_backend=fake)
        results = []
        threads = [threading.Thread(target=lambda: results.append(adapter.run(request()))) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len({id(result) for result in results}), 1)

    def test_run_id_cannot_be_reused_for_different_identity(self):
        fake = FakeBackend(completed())
        adapter = InferenceAdapter(cold_backend=fake, resident_backend=fake)
        adapter.run(request())
        with self.assertRaises(RunIdentityConflict):
            adapter.run(request(prompt="changed"))

    def test_request_hashes_are_derived_and_tamper_evident(self):
        req = request()
        self.assertEqual(len(req.prompt_hash), 64)
        self.assertEqual(len(req.model_config_dataset_hash), 64)
        with self.assertRaises(AdapterError):
            RunRequest(req.run_id, req.case_id, req.prompt, req.model_id, req.config_id, req.dataset_id, req.execution_mode, prompt_hash="0" * 64)


if __name__ == "__main__":
    unittest.main()
