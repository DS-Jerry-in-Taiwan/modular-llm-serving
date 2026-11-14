import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from inference_lab.acquisition import verify_model, sha256_file, acquire_model
from inference_lab.runtime import LocalServer, RuntimeFailure, HttpResponseFailure, smoke, http_json, runtime_identity


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"context_size": 128, "gpu_layers": 1, "threads": 1}
        self.port_patch = patch("inference_lab.runtime.free_loopback_port", return_value=18441)
        self.port_patch.start()
        self.http_patch = patch("inference_lab.runtime.http_json", side_effect=OSError("not running"))
        self.http_patch.start()

    def tearDown(self):
        self.http_patch.stop()
        self.port_patch.stop()
        self.temp.cleanup()

    def executable(self, code):
        script = self.root / "fake-engine"
        script.write_text("#!/usr/bin/env python3\n" + code)
        script.chmod(0o700)
        return script

    def test_model_hash_and_size_both_required(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"GGUF-test")
        identity = {"size_bytes": model.stat().st_size, "sha256": sha256_file(model)}
        self.assertEqual(verify_model(model, identity)["sha256"], identity["sha256"])
        with self.assertRaises(ValueError):
            verify_model(model, {**identity, "sha256": "0" * 64})
        with self.assertRaises(ValueError):
            verify_model(model, {**identity, "size_bytes": 1})

    def test_startup_exit_cleans_owned_process(self):
        binary = self.executable("raise SystemExit(23)\n")
        server = LocalServer(binary, "/unused", self.config, self.root, time.monotonic() + 3)
        with self.assertRaises(RuntimeFailure):
            server.start()
        self.assertTrue(server.cleanup_confirmed)
        self.assertEqual(server.process.returncode, 23)

    def test_startup_deadline_kills_owned_group(self):
        binary = self.executable("import time,signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(30)\n")
        server = LocalServer(binary, "/unused", self.config, self.root, time.monotonic() + 0.25)
        start = time.monotonic()
        with self.assertRaises((TimeoutError, RuntimeFailure)):
            server.start()
        self.assertLess(time.monotonic() - start, 6)
        self.assertTrue(server.cleanup_confirmed)
        self.assertIsNotNone(server.process.poll())

    def test_successful_acquisition_publishes_without_replacing_existing_file(self):
        import hashlib
        data = b"GGUF"
        config = {"model": {"repository": "a/b", "revision": "a" * 40, "filename": "x.gguf", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, "budgets": {"download_bytes": 10}}
        destination = self.root / "download.gguf"
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, *args):
                if getattr(self, "done", False): return b""
                self.done = True
                return data
        with patch("inference_lab.acquisition.urllib.request.urlopen", return_value=Response()):
            acquired = acquire_model(config, destination)
        self.assertEqual(acquired["sha256"], config["model"]["sha256"])
        self.assertEqual(destination.read_bytes(), data)
        self.assertFalse(destination.with_name(destination.name + ".partial").exists())

    def test_endpoint_is_always_loopback(self):
        server = LocalServer("/unused", "/unused", self.config, self.root, time.monotonic() + 5)
        self.assertTrue(server.url.startswith("http://127.0.0.1:"))
        self.assertIn("127.0.0.1", server.argv)
        self.assertNotIn("0.0.0.0", server.argv)

    def test_partial_download_is_not_overwritten(self):
        destination = self.root / "model.gguf"
        destination.with_name("model.gguf.partial").write_bytes(b"prior")
        config = {"model": {"repository": "a/b", "revision": "a" * 40, "filename": "x", "size_bytes": 1}, "budgets": {"download_bytes": 2}}
        with self.assertRaises(FileExistsError):
            acquire_model(config, destination)

    def test_failed_model_verification_writes_failed_run_not_live_success(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"wrong")
        config = {"model": {"size_bytes": 123, "sha256": "0" * 64}}
        record, directory = smoke(config, "/unused", model, self.root / "runs")
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["cleanup_confirmed"])
        saved = json.loads((directory / "run.json").read_text())
        self.assertEqual(saved["failure"]["type"], "ValueError")
        self.assertNotIn("result", saved)

    def test_truncated_chat_is_explicit(self):
        server = LocalServer("/unused", "/unused", self.config, self.root, time.monotonic() + 5)
        response = {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
        with patch("inference_lab.runtime.http_json", return_value=response):
            self.assertTrue(server.chat("test")["truncated"])

    def test_offload_must_be_observed_not_requested(self):
        (self.root / "server.log").write_text("offloaded 0/65 layers to GPU\n")
        server = LocalServer("/unused", "/unused", self.config, self.root, time.monotonic() + 5)
        self.assertEqual(server.offload_evidence()["gpu_layers_observed"], 0)
        (self.root / "server.log").write_text("offloaded 24/65 layers to GPU\n")
        self.assertEqual(server.offload_evidence()["gpu_layers_observed"], 24)

    def test_term_ignoring_child_stops_before_terminal_success(self):
        binary = self.executable("import subprocess,time\nsubprocess.Popen(['python3','-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\ntime.sleep(30)\n")
        server = LocalServer(binary, "/unused", self.config, self.root, time.monotonic() + 10)
        with patch("inference_lab.runtime.http_json", return_value={"status": "ok"}):
            server.start()
        time.sleep(0.15)
        start = time.monotonic()
        server.stop()
        self.assertGreater(time.monotonic() - start, 2.5)
        self.assertTrue(server.cleanup_confirmed)
        self.assertFalse(server.group_has_live_members(server.process.pid))

    def test_runtime_version_must_match_pin(self):
        binary = self.executable("print('version: b10896 (fa676981)')\n")
        identity = runtime_identity(binary, "fa6769818708afd9807b22183ccda112fd563427")
        self.assertTrue(identity["pin_verified"])
        with self.assertRaises(RuntimeFailure):
            runtime_identity(binary, "a" * 40)

    def test_scoped_sigterm_cancels_real_owned_server_and_saves_receipt(self):
        import signal, threading
        binary = self.executable("import time\ntime.sleep(30)\n")
        config = {"model": {}, "runtime": {**self.config, "revision": "a" * 40}, "budgets": {"smoke_seconds": 10}}
        holder = {}
        original = LocalServer.start
        def start_and_signal(server):
            holder["server"] = server
            with patch("inference_lab.runtime.http_json", return_value={"status": "ok"}):
                original(server)
            os.kill(os.getpid(), signal.SIGTERM)
        import os
        with patch("inference_lab.runtime.verify_model", return_value={}), patch("inference_lab.runtime.runtime_identity", return_value={}), patch("inference_lab.runtime.gpu_probe", return_value={"status": "ok"}), patch.object(LocalServer, "start", start_and_signal):
            record, directory = smoke(config, binary, "/unused", self.root / "signal-runs")
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue(record["cleanup_confirmed"])
        self.assertIsNotNone(holder["server"].process.poll())
        self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "cancelled")

    def test_readiness_retries_only_loading503_and_stops_on_other_http_errors(self):
        for status in (503, 401):
            with self.subTest(status=status):
                case = self.root / str(status)
                case.mkdir()
                binary = self.executable("import time\ntime.sleep(30)\n")
                server = LocalServer(binary, "/unused", self.config, case, time.monotonic() + 3)
                with patch("inference_lab.runtime.http_json", side_effect=[HttpResponseFailure("status", "loading", status), {"status": "ok"}]):
                    if status == 503:
                        server.start()
                        server.stop()
                    else:
                        with self.assertRaises(HttpResponseFailure):
                            server.start()
                self.assertTrue(server.cleanup_confirmed)

    def test_http_wall_deadline_even_if_read_keeps_waiting(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, *args):
                time.sleep(0.3)
                return b'{}'
        with patch("inference_lab.runtime.urllib.request.urlopen", return_value=Response()):
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                http_json("http://127.0.0.1:18441", timeout=0.02)
            self.assertLess(time.monotonic() - start, 0.2)

    def test_cancellation_gets_terminal_receipt_and_restores_signals(self):
        import signal
        old = signal.getsignal(signal.SIGTERM)
        config = {"model": {"size_bytes": 1, "sha256": "0" * 64}}
        with patch("inference_lab.runtime.verify_model", side_effect=KeyboardInterrupt()):
            record, directory = smoke(config, "/unused", "/unused", self.root / "cancelled")
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue(record["cleanup_confirmed"])
        self.assertEqual(signal.getsignal(signal.SIGTERM), old)
        self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "cancelled")

    def test_malformed_received_payload_is_preserved(self):
        server = LocalServer("/unused", "/unused", self.config, self.root, time.monotonic() + 5)
        with patch("inference_lab.runtime.http_json", side_effect=HttpResponseFailure("bad JSON", "{bad")):
            with self.assertRaises(HttpResponseFailure):
                server.chat("test")
        self.assertEqual(server.last_response, {"raw_payload": "{bad"})


if __name__ == "__main__":
    unittest.main()
