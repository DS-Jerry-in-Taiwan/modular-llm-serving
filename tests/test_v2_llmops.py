import hashlib, json, math, tempfile, unittest
from pathlib import Path
from inference_lab.llmops import EvaluationGate, Registry, RegistryError
from inference_lab.llmops.registry import manifest_digest


def manifest(run="run-1", parent=None):
    return {"schema_version":1,"run_id":run,"parent_run_id":parent,"model_id":"qwen-1","config_id":"cfg-1","dataset_id":"data-1","evaluator_id":"eval-1","source_commit":"a"*40,"model_sha256":"b"*64,"config_sha256":"c"*64,"dataset_sha256":"d"*64}

class RegistryTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.registry=Registry(Path(self.tmp.name)/"index.jsonl")
    def tearDown(self): self.tmp.cleanup()
    def test_register_is_idempotent_and_lineage_is_ordered(self):
        self.registry.register(manifest("base")); child=self.registry.register(manifest("candidate","base")); self.assertEqual(self.registry.register(manifest("candidate","base")),child); self.assertEqual([x.manifest["run_id"] for x in self.registry.lineage("candidate")],["candidate","base"]); self.assertEqual(len(self.registry.list()),2)
    def test_conflict_tamper_and_missing_parent_fail_closed(self):
        self.registry.register(manifest("base"))
        with self.assertRaises(RegistryError): self.registry.register({**manifest("base"),"config_id":"other"})
        self.registry.path.write_text(self.registry.path.read_text().replace('"digest": "','"digest": "0'))
        with self.assertRaises(RegistryError): self.registry.list()
        self.registry=Registry(Path(self.tmp.name)/"missing.jsonl"); self.registry.register(manifest("child","absent"))
        with self.assertRaises(RegistryError): self.registry.lineage("child")
    def test_invalid_identity_rejected(self):
        for key,value in (("run_id","../escape"),("model_sha256","x"),("source_commit","x")):
            with self.subTest(key=key):
                with self.assertRaises(RegistryError): self.registry.register({**manifest(),key:value})

class GateTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.registry=Registry(Path(self.tmp.name)/"index.jsonl"); self.reg=self.registry.register(manifest())
    def tearDown(self): self.tmp.cleanup()
    def report(self,**changes): return {"schema_version":1,"run_id":"run-1","manifest_digest":self.reg.digest,"quality_score":1.0,"latency_ratio":0.8,"gpu_peak_bytes":100,"provenance_status":"verified",**changes}
    def test_passes_only_verified_report(self): self.assertEqual(EvaluationGate(self.registry,max_gpu_bytes=200).evaluate("run-1",self.report()).status,"pass")
    def test_quality_latency_gpu_and_identity_failures_are_explicit(self):
        d=EvaluationGate(self.registry,min_quality=.9,max_latency_ratio=1,max_gpu_bytes=200).evaluate("run-1",self.report(quality_score=.5,latency_ratio=2,gpu_peak_bytes=300,provenance_status="pending",manifest_digest="bad")); self.assertEqual(d.status,"fail"); self.assertEqual(set(d.reasons),{"report identity does not match registered manifest","quality threshold failed","latency threshold failed","GPU threshold failed","provenance not verified"})

    def test_missing_and_non_finite_metrics_fail_closed(self):
        for field, value in (("latency_ratio", None), ("quality_score", math.nan),
                             ("quality_score", math.inf), ("latency_ratio", math.nan),
                             ("latency_ratio", math.inf), ("gpu_peak_bytes", math.nan),
                             ("gpu_peak_bytes", math.inf)):
            with self.subTest(field=field, value=value):
                changes = {field: value}
                if value is None:
                    changes = {"latency_ratio": None}
                decision = EvaluationGate(self.registry, max_gpu_bytes=200).evaluate("run-1", self.report(**changes))
                self.assertEqual(decision.status, "fail")
                expected = "invalid GPU peak" if field == "gpu_peak_bytes" else f"invalid {'quality score' if field == 'quality_score' else 'latency ratio'}"
                self.assertIn(expected, decision.reasons)

        report = self.report()
        del report["latency_ratio"]
        decision = EvaluationGate(self.registry).evaluate("run-1", report)
        self.assertEqual(decision.status, "fail")
        self.assertIn("invalid latency ratio", decision.reasons)

if __name__=='__main__': unittest.main()
