"""No-GPU checks for the local V2 resident web/API integration."""
import tempfile
import threading
import time
import unittest

from inference_lab.web.jobs import ResidentDemoManager


class FakeResidentEngine:
    starts = 0
    entered = None
    release = None

    def start(self):
        type(self).starts += 1

    def ready(self):
        return True

    def stop(self):
        return None

    def generate(self, request, cancellation):
        if type(self).entered is not None:
            type(self).entered.set()
            while not type(self).release.wait(0.01):
                if cancellation.is_set():
                    raise RuntimeError("cancelled fake generation")
        if request.stream:
            return ["hello", " world"]
        return {"content": "hello"}


class ResidentAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        FakeResidentEngine.starts = 0
        FakeResidentEngine.entered = None
        FakeResidentEngine.release = threading.Event()
        config = {"model": {"filename": "qwen.gguf"},
                  "runtime": {"revision": "a" * 40, "context_size": 128}}
        self.manager = ResidentDemoManager(
            "unused", "unused", self.temp.name, config,
            engine_factory=FakeResidentEngine, verify_async=False,
            source_commit="b" * 40, max_queue=2,
        )
        self.addCleanup(self.manager.shutdown)

    def wait_terminal(self, request_id):
        for _ in range(300):
            record = self.manager.get(request_id)
            if record and record["status"] in {"completed", "failed", "cancelled", "deadline_exceeded"}:
                return record
            time.sleep(0.01)
        self.fail(f"request did not finish: {request_id}")

    def test_maps_contract_reuses_one_engine_and_persists_evidence(self):
        first = self.manager.submit({"request_id": "req-a", "prompt": "one", "max_new_tokens": 4})
        self.assertEqual(first, "req-a")
        result = self.wait_terminal(first)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["response"]["contract_version"], 1)
        second = self.manager.submit({"request_id": "req-b", "prompt": "two", "max_new_tokens": 4})
        self.assertEqual(self.wait_terminal(second)["status"], "completed")
        self.assertEqual(FakeResidentEngine.starts, 1)
        self.assertEqual(self.manager.store.get("req-a").state, "completed")
        self.assertTrue((self.manager.output_root / "req-a" / "response.json").is_file())

    def test_streaming_and_protocol_mixing_are_controlled(self):
        with self.assertRaises(ValueError):
            self.manager.submit({"prompt": "v1 field", "max_tokens": 16, "variants": ["baseline"]})
        request_id = self.manager.submit({"request_id": "stream-1", "prompt": "hi",
                                          "max_new_tokens": 4, "stream": True})
        result = self.wait_terminal(request_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([event["kind"] for event in result["events"]], ["chunk", "chunk", "terminal"])
        self.assertEqual(result["events"][-1]["sequence"], 2)

    def test_cancel_is_bounded_and_terminal_after_cleanup(self):
        FakeResidentEngine.entered = threading.Event()
        request_id = self.manager.submit({"request_id": "cancel-1", "prompt": "wait", "max_new_tokens": 4})
        self.assertTrue(FakeResidentEngine.entered.wait(1))
        self.assertTrue(self.manager.cancel(request_id))
        result = self.wait_terminal(request_id)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.manager.store.get(request_id).state, "cancelled")


if __name__ == "__main__":
    unittest.main()
