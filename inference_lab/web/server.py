"""Loopback-only, single-GPU live comparison; no cloud or model acquisition."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import html
import math
from pathlib import Path
import re
import signal
import threading
from urllib.parse import urlsplit

from ..config import load_config
from .jobs import DemoManager

ROOT = Path(__file__).resolve().parents[2]


REFERENCE_DIRECTORY = Path("results/convergence-v1-20261114")
REFERENCE_MAX_BYTES = 2 * 1024 * 1024
REFERENCE_LABEL = "收斂後 V1 固定測試"
REFERENCE_SCOPE = "六個固定文字案例，每種設定各執行兩次；與即時自訂提示分開。"
REFERENCE_CASES = {
    "summary-pilot": "試用摘要", "summary-incident": "事件摘要",
    "json-person": "人物 JSON 擷取", "json-orders": "訂單 JSON 擷取",
    "code-default-list": "Python 預設串列", "code-missing-last": "Python 邊界修正",
}


def reference_text(filename):
    """Read only a fixed local result, with a hard size bound."""
    if filename not in {"report.md", "comparison.json"}:
        raise ValueError("unknown reference artifact")
    with (ROOT / REFERENCE_DIRECTORY / filename).open("rb") as handle:
        raw = handle.read(REFERENCE_MAX_BYTES + 1)
    if not raw or len(raw) > REFERENCE_MAX_BYTES:
        raise ValueError("reference artifact missing or oversized")
    return raw.decode("utf-8")


def reference_info():
    """Project public benchmark fields without exposing local evidence paths."""
    unavailable = {"status": "unavailable", "label": REFERENCE_LABEL,
                   "scope": REFERENCE_SCOPE, "reference_url": "/reference"}
    try:
        if not reference_text("report.md").strip():
            raise ValueError("empty reference report")
        comparison = json.loads(reference_text("comparison.json"))
        if comparison["schema_version"] != 1 or comparison["kind"] != "single_factor_comparison":
            raise ValueError("invalid reference schema")
        def positive(value):
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid reference metric")
            return value
        run_ids = {key: comparison[key] for key in ("baseline_run_id", "candidate_run_id")}
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) for value in run_ids.values()):
            raise ValueError("invalid reference identity")
        totals = {key: positive(comparison["measured_request_totals_seconds"][key]) for key in ("baseline", "candidate", "latency_ratio")}
        peaks = {key: positive(comparison["resources"][key]["gpu_device_used_bytes"]["sampled_peak_bytes"]) for key in ("baseline", "candidate")}
        if any(comparison["resources"][key]["gpu_device_used_bytes"]["status"] != "ok" for key in peaks):
            raise ValueError("unavailable resource metric")
        if any(comparison["quality"][key] != "pass" for key in ("baseline", "candidate")):
            raise ValueError("reference quality not accepted")
        if set(comparison["per_case"]) != set(REFERENCE_CASES):
            raise ValueError("incomplete reference cases")
        cases = []
        for case_id, label in REFERENCE_CASES.items():
            case = comparison["per_case"][case_id]
            for key in ("baseline_output_tokens", "candidate_output_tokens"):
                if len(case[key]) != 2 or any(type(value) is not int or value <= 0 for value in case[key]):
                    raise ValueError("invalid reference repeat coverage")
            cases.append({"id": case_id, "label": label,
                          "baseline_seconds": positive(case["baseline_request_seconds"]),
                          "candidate_seconds": positive(case["candidate_request_seconds"]),
                          "latency_ratio": positive(case["latency_ratio"])})
        return {**unavailable, "status": "available", **run_ids,
                "measured_request_totals_seconds": totals, "gpu_peak_bytes": peaks, "cases": cases,
                "quality": {"baseline": "pass", "candidate": "pass", "samples_per_configuration": len(cases) * 2,
                            "review": "獨立 AI 審閱＋嚴格 JSON 比對"},
                "limitations": ["僅六個短文字案例、各重複兩次，不能代表所有任務。",
                                "依序測試，主機負載與檔案快取可能不同。",
                                "GPU 記憶體為全裝置取樣峰值，可能包含其他程序。",
                                "非串流請求，未量測首字延遲；即時測試不自動判定品質。"]}
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return unavailable


def loopback_authority(value):
    """Validate the browser-facing authority, including a forwarded local port."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?", value, re.IGNORECASE)
    if not match:
        return None
    port = int(match.group(2)) if match.group(2) is not None else 80
    if not 1 <= port <= 65535:
        return None
    return match.group(1).lower(), port


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def handler_for(manager, static_root=None):
    static_root = Path(static_root or Path(__file__).resolve().parent / "static").resolve()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, *args):
            pass  # Do not echo prompts or arbitrary request paths.

        def send_data(self, status, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            hosts = self.headers.get_all("Host", [])
            return len(hosts) == 1 and loopback_authority(hosts[0]) is not None

        def valid_origin(self):
            origins = self.headers.get_all("Origin", [])
            if not origins:  # Local command-line clients need not supply Origin.
                return True
            if len(origins) != 1 or not origins[0].startswith("http://"):
                return False
            authority = loopback_authority(origins[0][len("http://"):])
            return authority is not None and authority == loopback_authority(self.headers.get("Host"))

        def do_GET(self):
            if not self.valid_host():
                return self.send_data(403, {"error": "Loopback Host required."})
            path = urlsplit(self.path).path
            if path == "/api/info":
                return self.send_data(200, manager.info())
            if path == "/api/reference":
                return self.send_data(200, reference_info())
            if path == "/api/history":
                return self.send_data(200, manager.history())
            if path.startswith("/api/runs/"):
                job = manager.get(path.removeprefix("/api/runs/"))
                return self.send_data(200 if job else 404, job or {"error": "Run not found."})
            if path == "/reference":
                try:
                    report = reference_text("report.md")
                    if not report.strip():
                        raise ValueError("empty report")
                except (OSError, ValueError):
                    report = "固定測試報告暫時無法讀取，請稍後再試。"
                document = "<!doctype html><html lang=zh-Hant><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>收斂後 V1 固定測試</title><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>收斂後 V1 固定測試</h1><p>六個固定文字案例、每種設定各測兩次；即時自訂提示不自動判定品質。</p><pre>" + html.escape(report) + "</pre></html>"
                return self.send_data(200, document.encode(), "text/html; charset=utf-8")
            allowed = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
                       "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in allowed:
                name, mime = allowed[path]
                file = (static_root / name).resolve()
                if file.parent == static_root and file.is_file():
                    return self.send_data(200, file.read_bytes(), mime)
            self.send_data(404, {"error": "Not found."})

        def do_POST(self):
            if not self.valid_host() or not self.valid_origin():
                return self.send_data(403, {"error": "Same-origin requests required."})
            if self.headers.get_content_type() != "application/json" or self.headers.get("Transfer-Encoding"):
                return self.send_data(415, {"error": "JSON request required."})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if not 0 <= length <= 65536:
                return self.send_data(413, {"error": "Request body must be at most 64 KB."})
            self.connection.settimeout(5)
            try:
                payload = json.loads(self.rfile.read(length))
            except (ValueError, OSError):
                return self.send_data(400, {"error": "Invalid JSON body."})
            path = urlsplit(self.path).path
            if path == "/api/runs":
                try:
                    job_id = manager.submit(payload)
                    return self.send_data(202, {"id": job_id})
                except ValueError as exc:
                    return self.send_data(400, {"error": str(exc)})
                except BlockingIOError:
                    return self.send_data(409, {"error": "GPU is busy with another run."})
                except RuntimeError:
                    return self.send_data(503, {"error": "Model verification is not ready."})
                except OSError:
                    return self.send_data(500, {"error": "Could not create local evidence."})
            if path.startswith("/api/runs/") and path.endswith("/cancel"):
                if payload != {}:
                    return self.send_data(400, {"error": "Cancellation expects an empty JSON object."})
                found = manager.cancel(path[len("/api/runs/"):-len("/cancel")])
                return self.send_data(202 if found else 404, {"status": "cancellation_requested"} if found else {"error": "Run not found."})
            self.send_data(404, {"error": "Not found."})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/inference-lab"
    parser.add_argument("--binary", type=Path, default=cache / "runtime/llama-fa676981/build/bin/llama-server")
    parser.add_argument("--model", type=Path, default=cache / "models/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument("--output-root", type=Path, default=ROOT / "docs/local-inference/demo-runs")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "experiments/qwen38-27b-q4.json")
    parser.add_argument("--optimized-config", type=Path, default=ROOT / "experiments/qwen38-27b-q4-gpu40.json")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0–65535")
    manager = DemoManager(args.binary, args.model, args.output_root, load_config(args.baseline_config), load_config(args.optimized_config))
    server = DemoHTTPServer(("127.0.0.1", args.port), handler_for(manager))
    def shutdown_signal(signum, frame):
        # shutdown() must run outside serve_forever's thread.
        threading.Thread(target=server.shutdown, daemon=True).start()
    previous = {sig: signal.signal(sig, shutdown_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    print(f"Local demo: http://127.0.0.1:{server.server_port} (verifying model identity)", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        manager.shutdown()
        server.server_close()
        for sig, old in previous.items():
            signal.signal(sig, old)


if __name__ == "__main__":
    main()
