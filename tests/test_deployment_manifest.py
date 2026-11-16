import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from inference_lab.deployment import ManifestError, build_manifest, load_manifest, manifest_digest, write_manifest_new
from inference_lab.deployment.preflight import deployment_preflight


CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "qwen38-27b-q4.json"
GPU = {"status": "ok", "name": "NVIDIA GeForce RTX 3080", "count": 1, "memory_total_bytes": 12 * 1024**3}
CUDA = {"status": "ok", "device_count": 1}


class DeploymentManifestTests(unittest.TestCase):
    def config(self):
        return json.loads(CONFIG.read_text())

    def test_ready_manifest_is_pinned_and_digest_is_stable(self):
        manifest = build_manifest(self.config(), gpu=GPU, cuda=CUDA, source_commit="a" * 40)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest_digest(manifest), manifest_digest(json.loads(json.dumps(manifest))))
        self.assertEqual(manifest["deployment"]["host"], "127.0.0.1")

    def test_missing_gpu_is_blocked_and_does_not_claim_readiness(self):
        manifest = build_manifest(self.config(), gpu={"status": "unavailable", "reason": "no driver"}, cuda={"status": "unavailable", "reason": "no driver"})
        self.assertEqual(manifest["status"], "blocked")
        self.assertEqual(set(manifest["blockers"]), {"gpu_telemetry_unavailable", "cuda_unavailable"})

    def test_mismatched_gpu_and_mutable_identity_are_rejected(self):
        mismatch = dict(GPU, name="NVIDIA A100")
        manifest = build_manifest(self.config(), gpu=mismatch, cuda=CUDA)
        self.assertIn("gpu_model_mismatch", manifest["blockers"])
        with self.assertRaises(ManifestError):
            build_manifest(self.config(), gpu=GPU, cuda=CUDA, source_commit="main")

    def test_manifest_is_exclusive_and_preflight_is_offline(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            manifest = build_manifest(self.config(), gpu=GPU, cuda=CUDA)
            write_manifest_new(output, manifest)
            with self.assertRaises(FileExistsError):
                write_manifest_new(output, manifest)
            self.assertEqual(load_manifest(output), manifest)

    def test_preflight_writes_blocked_report_without_gpu(self):
        with TemporaryDirectory() as directory, patch("inference_lab.deployment.preflight.gpu_probe", return_value={"status": "unavailable", "reason": "hidden"}), patch("inference_lab.deployment.preflight.cuda_probe", return_value={"status": "unavailable", "reason": "hidden"}), patch("inference_lab.deployment.preflight._source_commit", return_value="unknown"):
            output = Path(directory) / "manifest.json"
            result = deployment_preflight(CONFIG, output)
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(load_manifest(output)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
