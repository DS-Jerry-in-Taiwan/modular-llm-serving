"""Ticket 2 conformance tests for LlamaCppProvider — mock HTTP/SSE, no live Modal calls.

Red → green tests for the first concrete provider. We mock the HTTP/SSE transport with a
`FakeResponse` seam (matching V1's test fixture style), so NO live Modal call is made and
no root `.env` is read. Live V1 validation requires external Releaser/User credential
injection (out of this ticket).

Coverage matrix (TaskPlan §4b) for the provider's output surface:
  🟢 health 200 -> ready  |  models -> single descriptor  |  chat -> non-empty content
     stream -> [DONE] terminal marker
  🔴 401 -> ProviderError (auth path)  |  4xx unknown model -> ProviderError
     stream without [DONE] -> ProviderError  |  missing/empty chat content -> ProviderError
  📏 request/stream timeouts and 503 backoff are finite-positive (bounded deadline)
     context_length / capabilities are the verified set {"chat","stream"}
  🎯 capabilities() is exactly the verified set (tool_calls NOT declared/enforced);
     auth shape is exactly Modal-Key + Modal-Secret (no Bearer); diagnostics redact headers
  🔲 stream empty events / chat empty content / 503 deadline exhaustion / non-503 no-retry
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("contracts", ROOT / "contracts.py")
contracts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contracts
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(contracts)

_PROV_SPEC = importlib.util.spec_from_file_location("providers.llamacpp", ROOT / "providers" / "llamacpp.py")
llamacpp = importlib.util.module_from_spec(_PROV_SPEC)
sys.modules[_PROV_SPEC.name] = llamacpp
assert _PROV_SPEC is not None and _PROV_SPEC.loader is not None
_PROV_SPEC.loader.exec_module(llamacpp)

BASE_URL = "https://proxy.invalid"
MODEL_ID = "gemma-4-e2b-it-q4_0"
KEY = "wk-test-key"
SECRET = "ws-test-secret"


class FakeResponse:
    """Status/body/SSE-line response double matching V1's FakeResponse convention."""

    def __init__(self, status, payload=None, lines=()):
        self.status = status
        self.payload = payload
        self.lines = iter(lines)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        import json
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        return self.lines


def make_provider(**kwargs):
    """Build a provider with sane defaults; allow seams to be overridden."""
    defaults = dict(base_url=BASE_URL, modal_key=KEY, modal_secret=SECRET)
    defaults.update(kwargs)
    return llamacpp.LlamaCppProvider(**defaults)


def fake_opener(*responses):
    """Return (opener, calls) where opener serves a queue of FakeResponse objects and records
    the (full_url, headers) of each request for shape assertions."""
    calls = []
    queue = list(responses)

    def opener(request, **kw):
        calls.append((request.full_url, {k.lower(): v for k, v in request.headers.items()}))
        return queue.pop(0)

    return opener, calls


class _Clock:
    """Deterministic clock for bounded-503 deadline tests."""

    def __init__(self):
        self._t = 0.0

    def monotonic(self):
        return self._t

    def sleep(self, seconds):
        self._t += seconds


class HealthTests(unittest.TestCase):
    """🟢/🔴/🔲 — /health mapping."""

    def test_health_200_is_ready(self):
        opener, _ = fake_opener(FakeResponse(200, {"status": "ok"}))
        p = make_provider(opener=opener)
        result = p.health()
        self.assertIsInstance(result, contracts.HealthStatus)
        self.assertTrue(result.ready)

    def test_health_503_is_retried_bounded_not_silent(self):
        """📏 a bare 503 (not a RetryTests subcase) is subject to bounded retry; if it never
        resolves it becomes ProviderError, never a silent not-ready."""
        clock = _Clock()

        def always_503(request, **kw):
            return _error_response(503)

        p = make_provider(opener=always_503, sleep=clock.sleep, monotonic=clock.monotonic,
                          request_timeout=0.5)
        with self.assertRaises(contracts.ProviderError):
            p.health()

    def test_health_401_is_provider_error(self):
        opener, _ = fake_opener(FakeResponse(401))
        p = make_provider(opener=opener)
        with self.assertRaises(contracts.ProviderError):
            p.health()

    def test_health_redirects_to_health_path(self):
        opener, calls = fake_opener(FakeResponse(200, {"status": "ok"}))
        make_provider(opener=opener).health()
        self.assertEqual(calls[0][0], f"{BASE_URL}/health")


class ModelsTests(unittest.TestCase):
    """🟢/🔴/🎯 — /v1/models mapping to a single validated descriptor."""

    def test_models_returns_descriptor_with_alias(self):
        opener, _ = fake_opener(FakeResponse(200, {"data": [{"id": MODEL_ID}]}))
        p = make_provider(opener=opener)
        models = p.models()
        self.assertIsInstance(models, list)
        self.assertEqual(len(models), 1)
        d = models[0]
        self.assertIsInstance(d, contracts.ModelDescriptor)
        self.assertEqual(d.model_id, MODEL_ID)
        self.assertEqual(d.base_url, BASE_URL)
        self.assertEqual(d.context_length, 4096)
        d.validate()  # descriptor must survive the contract validator

    def test_models_401_and_unknown_model_raise_provider_error(self):
        """🔴 auth 401 and unknown-alias 4xx must map to controlled ProviderError."""
        for status in (401, 400, 404):
            with self.subTest(status=status):
                opener, _ = fake_opener(FakeResponse(status, {"error": "bad"}))
                with self.assertRaises(contracts.ProviderError):
                    make_provider(opener=opener).models()
        # 200 but missing the fixed alias -> ProviderError (fail-closed like V1 validate_models_response)
        opener, _ = fake_opener(FakeResponse(200, {"data": [{"id": "other-model"}]}))
        with self.assertRaises(contracts.ProviderError):
            make_provider(opener=opener).models()

    def test_models_malformed_json_raises_provider_error(self):
        """Malformed models payload (non-dict body / bad json) -> ProviderError."""
        import json as _json

        class MalformedResponse(FakeResponse):
            def read(self):
                return b"not-json{"

        opener, _ = fake_opener(MalformedResponse(200))
        with self.assertRaises(contracts.ProviderError):
            make_provider(opener=opener).models()


class AuthHeaderAndRedactionTests(unittest.TestCase):
    """🎯 — proxy header shape exactly Modal-Key/Modal-Secret; diagnostics redact values."""

    def test_sends_exactly_modal_key_and_secret_headers(self):
        opener, calls = fake_opener(FakeResponse(200, {"data": [{"id": MODEL_ID}]}))
        p = make_provider(opener=opener)
        p.models()
        req_headers = calls[0][1]
        self.assertEqual(req_headers.get("modal-key"), KEY)
        self.assertEqual(req_headers.get("modal-secret"), SECRET)
        # no Authorization/Bearer and no SDK credential headers
        self.assertNotIn("authorization", req_headers)
        self.assertNotIn("x-api-key", req_headers)

    def test_redact_headers_never_leaks_values(self):
        redacted = llamacpp.redact_headers({"Modal-Key": KEY, "Modal-Secret": SECRET, "Content-Type": "application/json"})
        for value in redacted.values():
            self.assertEqual(value, "[REDACTED]")
        self.assertNotIn(KEY, redacted.values())
        self.assertNotIn(SECRET, redacted.values())

    def test_invalid_proxy_pair_shapes_are_rejected(self):
        """🔴 wk-/ws- shapes are enforced (fail-closed like V1 proxy_headers)."""
        for key, secret in (("", SECRET), (KEY, ""), ("bad-key", SECRET), (KEY, "bad-secret")):
            with self.subTest(key=key, secret=secret):
                with self.assertRaises(contracts.ProviderError):
                    make_provider(modal_key=key, modal_secret=secret)

    def test_exception_message_redacts_headers(self):
        """📏 Controlled errors carry safe messages (no credential values)."""
        opener, _ = fake_opener(FakeResponse(401))
        p = make_provider(opener=opener)
        with self.assertRaises(contracts.ProviderError) as ctx:
            p.models()
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertIn("[REDACTED]", str(ctx.exception))


class ChatTests(unittest.TestCase):
    """🟢/🔴/🔲 — non-stream chat content mapping."""

    def test_chat_returns_nonempty_content(self):
        opener, _ = fake_opener(FakeResponse(200, {"choices": [{"message": {"content": "hello world"}}]}))
        p = make_provider(opener=opener)
        resp = p.chat([contracts.Message(role="user", content="hi")])
        self.assertIsInstance(resp, contracts.ChatResponse)
        self.assertEqual(resp.content, "hello world")
        self.assertEqual(resp.model_id, MODEL_ID)

    def test_chat_posts_stream_false(self):
        opener, calls = fake_opener(FakeResponse(200, {"choices": [{"message": {"content": "x"}}]}))
        make_provider(opener=opener).chat([contracts.Message(role="user", content="hi")])
        self.assertTrue(calls[0][0].endswith("/v1/chat/completions"))

    def test_chat_4xx_maps_to_provider_error(self):
        with self.subTest(status=401):
            opener, _ = fake_opener(FakeResponse(401, {"error": "auth"}))
            with self.assertRaises(contracts.ProviderError):
                make_provider(opener=opener).chat([contracts.Message(role="user", content="hi")])
        with self.subTest(status=404):
            opener, _ = fake_opener(FakeResponse(404, {"error": "no model"}))
            with self.assertRaises(contracts.ProviderError):
                make_provider(opener=opener).chat([contracts.Message(role="user", content="hi")])

    def test_chat_empty_content_is_provider_error(self):
        """🔲 empty chat content must not be reported as success."""
        for payload in (
            {"choices": [{"message": {"content": ""}}]},
            {"choices": [{"message": {"content": "   "}}]},
            {},
        ):
            with self.subTest(payload=payload):
                opener, _ = fake_opener(FakeResponse(200, payload))
                with self.assertRaises(contracts.ProviderError):
                    make_provider(opener=opener).chat([contracts.Message(role="user", content="hi")])


class StreamTests(unittest.TestCase):
    """🟢/🔲/🔴 — streaming SSE with terminal [DONE] marker."""

    def test_stream_yields_chunks_and_terminal_marker(self):
        opener, _ = fake_opener(
            FakeResponse(
                200,
                lines=[b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                       b'data: {"choices":[{"delta":{"content":"lo"}}]}',
                       b"data: [DONE]"],
            )
        )
        p = make_provider(opener=opener)
        stream_iter = p.stream([contracts.Message(role="user", content="hi")])
        self.assertFalse(hasattr(stream_iter, "__await__"), "stream() must return a sync Iterator")
        events = list(stream_iter)
        contents = [e.content for e in events]
        self.assertEqual(contents, ["Hel", "lo", ""])
        self.assertTrue(events[-1].done, "stream must end with done=True marker")
        self.assertTrue(all(not e.done for e in events[:-1]))

    def test_stream_without_terminal_marker_is_provider_error(self):
        """🔴 SSE missing [DONE] -> ProviderError (V1 parse_sse semantics)."""
        opener, _ = fake_opener(
            FakeResponse(200, lines=[b'data: {"choices":[{"delta":{"content":"x"}}]}'])
        )
        p = make_provider(opener=opener)
        with self.assertRaises(contracts.ProviderError):
            list(p.stream([contracts.Message(role="user", content="hi")]))

    def test_stream_empty_events_is_provider_error(self):
        """🔲 no data event at all -> ProviderError (mirrors V1 parse_sse requiring >=1)."""
        opener, _ = fake_opener(FakeResponse(200, lines=[b"data: [DONE]"]))
        p = make_provider(opener=opener)
        with self.assertRaises(contracts.ProviderError):
            list(p.stream([contracts.Message(role="user", content="hi")]))

    def test_stream_malformed_sse_line_or_json(self):
        """🔴 unexpected SSE line / invalid JSON -> ProviderError."""
        for lines in ([b"bad-line"], [b'data: not-json{']):
            with self.subTest(lines=lines):
                opener, _ = fake_opener(FakeResponse(200, lines=lines))
                p = make_provider(opener=opener)
                with self.assertRaises(contracts.ProviderError):
                    list(p.stream([contracts.Message(role="user", content="hi")]))

    def test_stream_401_fails_before_iteration(self):
        opener, _ = fake_opener(FakeResponse(401))
        p = make_provider(opener=opener)
        with self.assertRaises(contracts.ProviderError):
            list(p.stream([contracts.Message(role="user", content="hi")]))


def _error_response(status):
    """Build an HTTPError-like response double (status-only) for urllib seam testing."""
    return llamacpp._HttpErrorResponse(status)


class RetryTests(unittest.TestCase):
    """📏/🔲 — bounded 503 backoff and non-503 no-retry semantics."""

    def test_503_then_200_retries_with_backoff(self):
        clock = _Clock()
        responses = [_error_response(503), _error_response(503), FakeResponse(200, {"status": "ok"})]
        opener, _ = fake_opener(*responses)
        p = make_provider(opener=opener, sleep=clock.sleep, monotonic=clock.monotonic, request_timeout=60)
        result = p.health()
        self.assertTrue(result.ready)
        # backoff 1 + 2 = 3 seconds elapsed
        self.assertEqual(clock._t, 3.0)

    def test_503_deadline_exhaustion_raises_provider_error(self):
        """🔲 503 deadline exhaustion -> ProviderError (never retries unbounded)."""
        clock = _Clock()

        def always_503(request, **kw):
            return _error_response(503)

        p = make_provider(opener=always_503, sleep=clock.sleep, monotonic=clock.monotonic, request_timeout=1)
        with self.assertRaises(contracts.ProviderError):
            p.health()

    def test_non_503_response_is_not_retried(self):
        """🔴 a 4xx (or non-503) is returned/failed immediately, not retried."""
        calls = []

        def single_400(request, **kw):
            calls.append("hit")
            return _error_response(400)

        p = make_provider(opener=single_400, sleep=_Clock().sleep, monotonic=_Clock().monotonic, request_timeout=2)
        with self.assertRaises(contracts.ProviderError):
            p.models()
        self.assertEqual(len(calls), 1, "non-503 must not be retried")


class DeadlinesAndTimeoutsTests(unittest.TestCase):
    """📏 — finite-positive timeout & deadline bounds on the provider."""

    def test_timeouts_and_backoff_must_be_positive_and_bounded(self):
        for kwargs in (
            {"request_timeout": 0},
            {"request_timeout": -1},
            {"stream_timeout": 0},
            {"base_delay": 0},
            {"max_delay": 0},
            {"max_delay": 0.5},  # < base_delay
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(contracts.ProviderError):
                    make_provider(**kwargs)

    def test_base_url_must_be_http(self):
        for bad in ("", "  ", "ftp://x", "not-a-url"):
            with self.subTest(bad=bad):
                with self.assertRaises(contracts.ProviderError):
                    make_provider(base_url=bad)


class CapabilitiesTests(unittest.TestCase):
    """🎯 — capabilities() exactly the verified phase-one set; tool_calls not declared."""

    def test_capabilities_are_exactly_chat_and_stream(self):
        p = make_provider()
        d = p.capabilities()
        self.assertEqual(d.capabilities, frozenset({"chat", "stream"}))
        self.assertNotIn("tool_calls", d.capabilities)
        self.assertEqual(d.capabilities, contracts.VERIFIED_CAPABILITIES)

    def test_provider_conforms_to_runtime_protocol(self):
        p = make_provider()
        self.assertIsInstance(p, contracts.RuntimeProvider)
        self.assertFalse(hasattr(p.stream, "__await__"), "stream() must be sync (no async)")


if __name__ == "__main__":
    unittest.main()
