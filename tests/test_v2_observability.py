import json
import tempfile
import unittest
from pathlib import Path

from inference_lab.serving.observability import (
    InMemoryEventSink,
    JsonlEventSink,
    MalformedMetrics,
    MetricsSnapshot,
    ObservationRecorder,
    ObservabilityError,
    redact_text,
)


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class V2ObservabilityTests(unittest.TestCase):
    def make_observation(self):
        self.clock = FakeClock()
        self.sink = InMemoryEventSink()
        recorder = ObservationRecorder(self.sink, clock=self.clock, wall_clock_ns=lambda: 1_700_000_000_000_000_000)
        return recorder.start(request_id="req-1", job_id="job-1", run_id="run-1", model_id="qwen", config_id="cfg")

    def test_terminal_event_has_correlated_identity_and_phase_metrics(self):
        observation = self.make_observation()
        self.clock.advance(0.25)
        observation.queue_complete()
        self.clock.advance(0.1)
        observation.engine_load_start()
        self.clock.advance(0.4)
        observation.engine_load_complete()
        observation.generation_start()
        self.clock.advance(0.2)
        observation.token()
        self.clock.advance(0.3)
        observation.token()
        observation.generation_complete()
        observation.cleanup_start()
        self.clock.advance(0.05)
        observation.cleanup_complete()
        terminal = observation.terminal(status="completed", metrics={"prompt_tokens": 3})
        self.assertTrue(terminal.terminal)
        self.assertEqual({terminal.request_id, terminal.job_id, terminal.run_id, terminal.model_id, terminal.config_id},
                         {"req-1", "job-1", "run-1", "qwen", "cfg"})
        self.assertEqual(terminal.status, "completed")
        self.assertEqual(terminal.metrics.prompt_tokens, 3)
        self.assertEqual(terminal.metrics.completion_tokens, 2)
        self.assertEqual(terminal.metrics.total_tokens, 5)
        self.assertAlmostEqual(terminal.metrics.queue_wait_seconds, 0.25)
        self.assertAlmostEqual(terminal.metrics.engine_load_seconds, 0.4)
        self.assertAlmostEqual(terminal.metrics.ttft_seconds, 0.2)
        self.assertAlmostEqual(terminal.metrics.generation_seconds, 0.5)
        self.assertAlmostEqual(terminal.metrics.cleanup_seconds, 0.05)
        self.assertEqual(len(self.sink.snapshot()), 11)

    def test_stream_and_disconnect_are_correlated_and_terminal(self):
        observation = self.make_observation()
        observation.generation_start()
        observation.stream_chunk(0, token_count=2)
        terminal = observation.disconnect(sequence=1)
        events = self.sink.snapshot()
        self.assertEqual(events[-2].event_type, "stream.disconnect")
        self.assertEqual(events[-2].sequence, 1)
        self.assertEqual(events[-1], terminal)
        for event in events:
            self.assertEqual((event.request_id, event.job_id, event.run_id), ("req-1", "job-1", "run-1"))
        self.assertEqual(terminal.status, "cancelled")
        self.assertEqual(terminal.error_code, "client_disconnected")

    def test_secrets_are_redacted_and_prompt_is_not_an_event_field(self):
        self.assertNotIn("wk-secret", redact_text("Bearer wk-secret token=ws-other password=hunter2"))
        observation = self.make_observation()
        terminal = observation.terminal(status="failed", error_code="provider_failed",
                                         error_message="Bearer wk-secret prompt text")
        payload = terminal.to_dict()
        self.assertNotIn("prompt", payload)
        self.assertNotIn("wk-secret", json.dumps(payload))
        self.assertIn("[REDACTED]", payload["error_message"])

    def test_malformed_metrics_fail_closed(self):
        with self.assertRaises(MalformedMetrics):
            MetricsSnapshot(total_seconds=float("nan"))
        with self.assertRaises(MalformedMetrics):
            MetricsSnapshot.from_dict({"total_seconds": 1, "uncontrolled_prompt": "secret"})
        with self.assertRaises(MalformedMetrics):
            MetricsSnapshot(prompt_tokens=2, completion_tokens=3, total_tokens=99)
        observation = self.make_observation()
        with self.assertRaises(MalformedMetrics):
            observation.terminal(status="completed", metrics={"gpu_utilization_percent": 101})
        self.assertIsNone(observation.terminal_event)

    def test_jsonl_sink_is_local_and_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            sink = JsonlEventSink(path)
            recorder = ObservationRecorder(sink, wall_clock_ns=lambda: 2)
            observation = recorder.start(request_id="r", job_id="j", run_id="run", model_id="m", config_id="c")
            observation.terminal(status="completed")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[-1])["schema_version"], 1)

    def test_identifiers_and_terminal_status_are_bounded(self):
        sink = InMemoryEventSink()
        recorder = ObservationRecorder(sink)
        with self.assertRaises(ObservabilityError):
            recorder.start(request_id="../secret", job_id="j", run_id="r", model_id="m", config_id="c")
        observation = self.make_observation()
        with self.assertRaises(ObservabilityError):
            observation.terminal(status="running")


if __name__ == "__main__":
    unittest.main()
