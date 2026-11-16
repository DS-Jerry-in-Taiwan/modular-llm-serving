"""No-GPU checks for the local V2 resident web/API integration."""
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

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


class FailingStreamEngine(FakeResidentEngine):
    def generate(self, request, cancellation):
        if request.stream:
            raise RuntimeError("prompt=private credential=secret")
        return super().generate(request, cancellation)


class ResidentAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        FakeResidentEngine.starts = 0
        FakeResidentEngine.entered = None
        FakeResidentEngine.release = threading.Event()
        config = {"model": {"filename": "qwen.gguf"},
                  "runtime": {"revision": "a" * 40, "context_size": 128,
                              "gpu_layers": 0, "threads": 1, "parallel": 1}}
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

    def test_resident_engine_instances_use_unique_evidence_directories(self):
        with patch("inference_lab.runtime.free_loopback_port", return_value=19001):
            first = self.manager._default_factory()
            second = self.manager._default_factory()
        self.assertNotEqual(first.server.run_dir, second.server.run_dir)
        self.assertEqual(first.server.run_dir.parent, second.server.run_dir.parent)
        self.assertEqual(first.server.run_dir.parent.name, "resident-engine")
        self.assertTrue(first.server.run_dir.name.startswith("instance-"))
        self.assertTrue(second.server.run_dir.name.startswith("instance-"))

    def test_stream_runner_failure_publishes_controlled_error_and_terminal(self):
        self.manager.shutdown()
        self.manager = ResidentDemoManager(
            "unused", "unused", self.temp.name, {
                "model": {"filename": "qwen.gguf"},
                "runtime": {"revision": "a" * 40, "context_size": 128},
            }, engine_factory=FailingStreamEngine, verify_async=False,
            source_commit="b" * 40, max_queue=2,
        )
        request_id = self.manager.submit({"request_id": "stream-failure", "prompt": "private", "max_new_tokens": 4, "stream": True})
        result = self.wait_terminal(request_id)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([event["kind"] for event in result["events"]], ["error", "terminal"])
        self.assertEqual(result["events"][0]["error"]["code"], "STREAM_FAILED")
        wire = str(result)
        self.assertNotIn("private", wire)
        self.assertNotIn("secret", wire)

    def test_stream_active_cancellation_publishes_terminal_after_cleanup(self):
        FakeResidentEngine.entered = threading.Event()
        request_id = self.manager.submit({"request_id": "stream-cancel", "prompt": "wait", "max_new_tokens": 4, "stream": True})
        self.assertTrue(FakeResidentEngine.entered.wait(1))
        self.assertTrue(self.manager.cancel(request_id))
        self.assertFalse(self.manager.cancel(request_id))
        result = self.wait_terminal(request_id)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual([event["kind"] for event in result["events"]], ["error", "terminal"])
        self.assertEqual(result["events"][0]["error"]["code"], "CLIENT_DISCONNECTED")


if __name__ == "__main__":
    unittest.main()
