"""Focused V2 resident measurement contract tests."""
import unittest
from unittest.mock import patch

from inference_lab.v2_metrics import ResidentRequestMeasurement, engine_pid, resident_response_metrics


class FakeProcess:
    pid = 1234


class FakeEngine:
    process = FakeProcess()


class FakeSampler:
    def __init__(self):
        import threading
        self.pid = None
        self.lock = threading.Lock()
        self.samples = [{"gpu_device_used_bytes": 10, "system_memory_used_bytes": 20,
                         "server_rss_bytes": 30}]
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def summary(self):
        return {"gpu_device_used_bytes": {"status": "ok", "sampled_peak_bytes": 10},
                "system_memory_used_bytes": {"status": "ok", "sampled_peak_bytes": 20},
                "server_rss_bytes": {"status": "ok", "sampled_peak_bytes": 30}}


class V2MetricsTests(unittest.TestCase):
    def test_engine_pid_accepts_owned_server_and_missing_process(self):
        self.assertEqual(engine_pid(FakeEngine()), 1234)
        self.assertIsNone(engine_pid(object()))

    def test_response_metrics_preserve_v1_names_and_unavailable_timing(self):
        result = {"elapsed_seconds": 2.5, "response": {"timings": {"predicted_per_second": 4.0}}}
        metrics = resident_response_metrics(result)
        self.assertEqual(metrics["request_seconds"], {"status": "ok", "value": 2.5})
        self.assertEqual(metrics["generated_tokens_per_second"], {"status": "ok", "value": 4.0})
        missing = resident_response_metrics({"content": "fake"})
        self.assertEqual(missing["generated_tokens_per_second"]["status"], "unavailable")

    def test_measurement_captures_scoped_samples_and_sanitized_metrics(self):
        sampler = FakeSampler()
        measurement = ResidentRequestMeasurement(sampler_factory=lambda: sampler)
        measurement.start(FakeEngine())
        self.assertEqual(sampler.pid, 1234)
        evidence = measurement.finish({"elapsed_seconds": 1.5, "response": {"timings": {}}})
        self.assertTrue(sampler.started and sampler.stopped)
        self.assertEqual(evidence["schema_version"], 1)
        self.assertEqual(evidence["resources"]["server_rss_bytes"]["sampled_peak_bytes"], 30)
        self.assertEqual(evidence["response_metrics"]["request_seconds"]["value"], 1.5)
        samples = measurement.samples()
        samples[0]["gpu_device_used_bytes"] = 999
        self.assertEqual(sampler.samples[0]["gpu_device_used_bytes"], 10)

    def test_unstarted_measurement_fails_closed(self):
        evidence = ResidentRequestMeasurement(sampler_factory=FakeSampler).finish(None)
        self.assertEqual(evidence["resources"]["status"], "unavailable")
        self.assertEqual(evidence["response_metrics"]["request_seconds"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
