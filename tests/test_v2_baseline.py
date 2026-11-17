import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inference_lab.v2_baseline import BaselinePackagingError, package_baseline, package_run, write_package_new


RUN = Path("/tmp/e2e04-v2-resident-metrics-20261116T141237Z/run.json")


class V2BaselinePackagingTests(unittest.TestCase):
    def test_packages_exact_twelve_samples_without_raw_content_or_paths(self):
        if not RUN.exists():
            self.skipTest("local E2E fixture is not available")
        package = package_run(RUN)
        self.assertEqual(package["sample_cardinality"], {"expected": 12, "observed": 12, "completed": 12})
        self.assertEqual(package["dataset"]["repeats"], 2)
        self.assertEqual(package["cleanup"]["status"], "unavailable")
        text = json.dumps(package, ensure_ascii=False)
        self.assertNotIn("content", text)
        self.assertNotIn("timings", text)  # timings are normalized into metric statuses
        self.assertNotIn("/tmp/e2e04", text)
        self.assertNotIn("server.log", text)
        self.assertEqual(package["resource_evidence"]["sample_count"], 12)

    def test_rejects_wrong_mode_cardinality_and_non_safe_ids(self):
        run = {"run_id": "r", "kind": "baseline", "execution_mode": "resident", "status": "completed",
               "model": {"filename": "m.gguf", "sha256": "a" * 64}, "config_sha256": "b" * 64,
               "dataset_sha256": "c" * 64, "runtime": {}, "measurement_protocol": {"repeats": 1}, "samples": []}
        for change in ({"execution_mode": "cold"}, {"samples": [{}]}, {"run_id": "../private"}):
            with self.subTest(change=change):
                with self.assertRaises(BaselinePackagingError):
                    package_baseline({**run, **change})

    def test_write_is_exclusive(self):
        if not RUN.exists():
            self.skipTest("local E2E fixture is not available")
        package = package_run(RUN)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "package.json"
            write_package_new(output, package)
            with self.assertRaises(FileExistsError):
                write_package_new(output, package)


if __name__ == "__main__":
    unittest.main()
