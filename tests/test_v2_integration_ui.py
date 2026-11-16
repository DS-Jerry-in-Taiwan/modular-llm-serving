"""Browser-free contract checks for the V1/V2 Demo integration."""
from html.parser import HTMLParser
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "inference_lab/web/static"


class _Ids(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.radio_values = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "input" and values.get("name") == "execution-mode":
            self.radio_values.add(values.get("value"))


class V2IntegrationUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (STATIC / "index.html").read_text(encoding="utf-8")
        cls.js = (STATIC / "app.js").read_text(encoding="utf-8")
        cls.css = (STATIC / "style.css").read_text(encoding="utf-8")
        cls.adr = (ROOT / "docs/architecture/v2-integration-ui-adr.md").read_text(encoding="utf-8")
        parser = _Ids()
        parser.feed(cls.html)
        cls.ids, cls.radio_values = parser.ids, parser.radio_values

    def test_mode_selector_and_lifecycle_fields_are_present(self):
        self.assertEqual(self.radio_values, {"cold", "resident"})
        for element_id in {
            "protocolNotice", "protocolBadge", "protocolDetail", "streamOutput",
            "v2Lifecycle", "requestId", "runId", "queueState", "streamState",
            "terminalState", "phaseTimings", "v2Metrics",
        }:
            self.assertIn(element_id, self.ids)
        self.assertIn("Cold benchmark", self.html)
        self.assertIn("Warm resident", self.html)

    def test_v1_and_v2_protocol_boundaries_are_explicit(self):
        self.assertIn("/api/runs", self.js)
        self.assertIn("/api/v2", self.js)
        self.assertIn("execution_mode", self.js)
        self.assertIn("contract_version", self.js)
        self.assertIn("request_id", self.js)
        self.assertIn("run_id", self.js)
        self.assertIn("phase_timings", self.js)
        self.assertIn("terminal_event", self.js)
        self.assertIn("StreamEvent", self.html)
        self.assertIn("v1-details", self.html)

    def test_server_values_use_text_content_and_no_remote_assets(self):
        self.assertNotIn("innerHTML", self.js)
        self.assertNotIn("document.write", self.js)
        self.assertNotIn("eval(", self.js)
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertIn("textContent", self.js)

    def test_lifecycle_styles_and_decision_record_exist(self):
        for marker in ("execution-mode", "protocol-strip", "v2-lifecycle", "identity-grid", "lifecycle-details"):
            self.assertIn(marker, self.css)
        for marker in ("V1 cold benchmark", "Warm resident", "request/run identity", "terminal event"):
            self.assertIn(marker, self.adr)


if __name__ == "__main__":
    unittest.main()
