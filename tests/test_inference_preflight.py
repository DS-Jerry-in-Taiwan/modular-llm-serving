import ctypes
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inference_lab import preflight as p

CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "qwen38-27b-q4.json"


class PreflightTests(unittest.TestCase):
    def config(self):
        return p.load_config(CONFIG)

    def test_pinned_config_validates(self):
        c = self.config()
        self.assertGreater(c["model"]["size_bytes"], 12 * 1024**3)
        self.assertTrue(c["resources"]["allow_cpu_offload"])

    def test_mutable_revisions_and_overbudget_models_rejected(self):
        for field, value in (("revision", "main"), ("sha256", "bad"), ("size_bytes", 26000000000), ("filename", "../model.gguf")):
            with self.subTest(field=field), TemporaryDirectory() as d:
                c = self.config()
                c["model"][field] = value
                file = Path(d) / "config.json"
                file.write_text(json.dumps(c))
                with self.assertRaises(ValueError):
                    p.load_config(file)

    def test_empty_gpu_stub_is_unavailable(self):
        with patch.object(ctypes, "CDLL", side_effect=OSError("file too short")):
            self.assertEqual(p.gpu_probe()["status"], "unavailable")
            self.assertEqual(p.cuda_probe()["status"], "unavailable")

    def test_offline_cannot_claim_verified_model(self):
        with patch.object(p, "cuda_probe", return_value={"status": "ok"}), patch.object(p, "gpu_probe", return_value={"status": "ok"}), patch.object(p, "command", return_value={}):
            report = p.preflight(self.config(), "/tmp", False)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("model_identity_unverified", report["blockers"])

    def test_gpu_failure_blocks_even_with_model_verified(self):
        with patch.object(p, "cuda_probe", return_value={"status": "ok"}), patch.object(p, "gpu_probe", return_value={"status": "unavailable"}), patch.object(p, "model_probe", return_value={"status": "ok"}), patch.object(p, "command", return_value={}):
            report = p.preflight(self.config(), "/tmp", True)
        self.assertEqual(report["blockers"], ["gpu_telemetry_unavailable"])

    def test_readiness_is_only_preflight_not_inference_claim(self):
        with patch.object(p, "cuda_probe", return_value={"status": "ok"}), patch.object(p, "gpu_probe", return_value={"status": "ok"}), patch.object(p, "model_probe", return_value={"status": "ok"}), patch.object(p, "command", return_value={}):
            report = p.preflight(self.config(), "/tmp", True)
        self.assertEqual(report["status"], "ready_for_runtime_validation")
        self.assertIn("does not establish CUDA inference", report["limitations"][0])

    def test_model_metadata_mismatch_rejected(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps({"sha": self_model["revision"], "siblings": [{"rfilename": self_model["filename"], "size": self_model["size_bytes"], "lfs": {"sha256": "0" * 64}}]}).encode()
        self_model = self.config()["model"]
        with patch.object(p.urllib.request, "urlopen", return_value=Response()):
            self.assertEqual(p.model_probe(self_model)["status"], "mismatch")

    def test_prior_report_never_overwritten(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "report.json"
            out.write_text("original")
            with patch.object(p, "preflight", return_value={"status": "blocked", "blockers": []}):
                with self.assertRaises(FileExistsError):
                    p.main(["preflight", "--config", str(CONFIG), "--output", str(out)])
            self.assertEqual(out.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
