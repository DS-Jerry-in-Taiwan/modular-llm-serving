import threading
import unittest

from inference_lab.serving import ResidentSupervisor
from inference_lab.serving.admission import (
    AdmissionQueue,
    AdmissionRejected,
    AdmissionStatus,
)
from inference_lab.v2_contracts import InferenceRequest


class FakeEngine:
    """Small cooperative engine used to exercise the supervisor seam."""

    def __init__(self, *, block=False):
        self.block = block
        self.started = threading.Event()
        self.calls = 0
        self.stop_calls = 0

    def start(self):
        self.started.set()

    def ready(self):
        return True

    def generate(self, request, cancel_event):
        self.calls += 1
        if self.block and self.calls == 1:
            while not cancel_event.wait(.005):
                pass
            return "cooperative cancellation"
        return f"answer:{request.request_id}"

    def stop(self):
        self.stop_calls += 1


def request(request_id, *, key=None, max_new_tokens=4, prompt="hello"):
    return InferenceRequest(
        request_id, prompt, "qwen", "resident", max_new_tokens=max_new_tokens,
        idempotency_key=key,
    )


class AdmissionTests(unittest.TestCase):
    def make_queue(self, engine, *, context_size=64, **kwargs):
        supervisor = ResidentSupervisor(lambda: engine)
        queue = AdmissionQueue(supervisor, context_size=context_size, **kwargs)
        self.addCleanup(queue.shutdown, stop_engine=True)
        return queue, supervisor

    def test_over_budget_rejected_before_engine_start(self):
        engine = FakeEngine()
        queue, supervisor = self.make_queue(engine, context_size=16, max_queue=2)

        with self.assertRaises(AdmissionRejected) as raised:
            queue.submit(request("too-large", max_new_tokens=9), prompt_tokens=8)

        self.assertEqual(raised.exception.code, "context_budget_exceeded")
        self.assertEqual(supervisor.state.value, "stopped")
        self.assertEqual(engine.calls, 0)
        self.assertEqual(queue.queued, 0)

    def test_nonfinite_deadline_rejected_before_admission(self):
        engine = FakeEngine()
        queue, supervisor = self.make_queue(engine, max_queue=2)

        for index, deadline in enumerate((float("nan"), float("inf"), float("-inf"))):
            with self.assertRaisesRegex(ValueError, "finite"):
                queue.submit(request(f"nonfinite-{index}"), deadline=deadline)

        self.assertEqual(supervisor.state.value, "stopped")
        self.assertEqual(queue.queued, 0)
        self.assertIsNone(queue.active)

    def test_queue_is_fifo_and_rejects_beyond_bound(self):
        engine = FakeEngine(block=True)
        queue, _ = self.make_queue(engine, max_queue=1)
        first = queue.submit(request("first"))
        self.assertTrue(engine.started.wait(1))
        second = queue.submit(request("second"))
        with self.assertRaises(AdmissionRejected) as raised:
            queue.submit(request("third"))
        self.assertEqual(raised.exception.code, "queue_full")
        self.assertEqual(queue.queued, 1)

        first.cancel()
        self.assertEqual(first.result(1).status, AdmissionStatus.CANCELLED)
        # The second item is FIFO and eventually runs after the active item cleans up.
        self.assertEqual(second.result(1).status, AdmissionStatus.COMPLETED)
        self.assertEqual(engine.calls, 2)

    def test_duplicate_idempotency_key_executes_once_and_survives_terminal_state(self):
        engine = FakeEngine()
        queue, _ = self.make_queue(engine, max_queue=2)
        original = queue.submit(request("original", key="same-key"))
        duplicate = queue.submit(request("different-request", key="same-key"))

        self.assertIsNot(original, duplicate)
        self.assertEqual(original.run_id, duplicate.run_id)
        self.assertEqual(original.result(1).status, AdmissionStatus.COMPLETED)
        self.assertEqual(duplicate.result(1).value, "answer:original")
        self.assertEqual(engine.calls, 1)

    def test_conflicting_idempotency_key_is_rejected(self):
        queue, _ = self.make_queue(FakeEngine(), max_queue=2)
        queue.submit(request("original", key="same-key"))
        with self.assertRaises(AdmissionRejected) as raised:
            queue.submit(request("conflict", key="same-key", prompt="different"))
        self.assertEqual(raised.exception.code, "idempotency_conflict")

    def test_queued_and_active_cancel_are_idempotent_and_terminal_after_cleanup(self):
        engine = FakeEngine(block=True)
        cleanup = []
        queue, _ = self.make_queue(engine, max_queue=2, cleanup=lambda req, value, error: cleanup.append(req.request_id))
        active = queue.submit(request("active"))
        self.assertTrue(engine.started.wait(1))
        queued = queue.submit(request("queued"))

        self.assertTrue(active.cancel())
        self.assertFalse(active.cancel())
        self.assertTrue(queued.cancel())
        self.assertFalse(queued.cancel())
        self.assertEqual(queued.result(1).status, AdmissionStatus.CANCELLED)
        self.assertEqual(active.result(1).status, AdmissionStatus.CANCELLED)
        self.assertEqual(cleanup, ["active"])
        self.assertTrue(active.done)
        self.assertTrue(queued.done)

    def test_active_deadline_requests_cooperative_cancel_and_terminal_cleanup(self):
        engine = FakeEngine(block=True)
        queue, _ = self.make_queue(engine, max_queue=1)
        handle = queue.submit(request("deadline"), timeout=.03)
        result = handle.result(1)
        self.assertEqual(result.status, AdmissionStatus.DEADLINE_EXCEEDED)
        self.assertTrue(result.terminal)


if __name__ == "__main__":
    unittest.main()
