"""LlamaCppProvider — synchronous adapter over the V1 llama-server OpenAI-compatible API.

This is the first concrete `RuntimeProvider` (Ticket 2). It consumes the V1 llama-server
HTTP surface (`/health`, `/v1/models`, `/v1/chat/completions`, SSE `[DONE]`) through the
runtime-neutral `RuntimeProvider`/`ModelDescriptor` contract, using ONLY the stdlib's
`urllib` for transport — it never imports modal and never touches V1 deployment internals.

Design constraints (DeveloperPrompt / TaskPlan §2):
- Synchronous only (locked): `urllib` matches V1; `stream()` is a sync generator that emits
  a terminal `StreamEvent(done=True)` exactly once as the final event.
- Auth is the proxy pair ONLY: exactly `Modal-Key` (wk-…) + `Modal-Secret` (ws-…). Never
  Bearer or SDK credentials. Diagnostics redact header values (`redact_headers`).
- `base_url` is ALWAYS caller-provided (from `modal.Server.get_url()`, never hand-built).
- Timeouts: request 120 s (`request_timeout`), stream 180 s (`stream_timeout`), with a
  bounded deadline. 503 (scale-to-zero/loading) retried with exponential backoff capped at
  `max_delay`, hard deadline, never retried unbounded, never retries non-503.
- Capabilities: exactly the verified phase-one set `{"chat", "stream"}`. `tool_calls` is
  future-only; V1's validators do not verify tool-call behavior, so it is neither declared
  nor conformance-enforced. V1's accepted first-version unknown-alias behavior is preserved
  as an OBSERVED backend behavior, not a declared capability or a local server change.

Fail-closed error mapping: 4xx (auth/unknown model), timeout/deadline exhaustion,
malformed/invalid JSON, missing/empty content, and invalid SSE all raise `ProviderError`
with safe (redacted) diagnostics.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Mapping, Sequence

# Reuse the Ticket 1 contract through the same importlib.util seam the repo tests use, so
# this module has NO modal/V1 import (runtime-neutral pure Python). `contracts` is bound to
# sys.modules below so the RuntimeProvider Protocol isinstance checks work.
import importlib.util
import sys
from pathlib import Path

if "contracts" in sys.modules:
    # Reuse an already-loaded contract module (e.g. the test seam loaded it first), so
    # isinstance checks against the test's contract types succeed (shared object identity).
    contracts = sys.modules["contracts"]
else:
    _SPEC = importlib.util.spec_from_file_location(
        "contracts", Path(__file__).resolve().parents[1] / "contracts.py"
    )
    contracts = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
    sys.modules[_SPEC.name] = contracts  # type: ignore[union-attr]
    assert _SPEC is not None and _SPEC.loader is not None
    _SPEC.loader.exec_module(contracts)

ProviderError = contracts.ProviderError
HealthStatus = contracts.HealthStatus
Message = contracts.Message
ChatResponse = contracts.ChatResponse
StreamEvent = contracts.StreamEvent
ModelDescriptor = contracts.ModelDescriptor
VERIFIED_CAPABILITIES = contracts.VERIFIED_CAPABILITIES

# V1-aligned defaults (modal_llama_smoke.py constants 174-177).
DEFAULT_MODEL_ALIAS = "gemma-4-e2b-it-q4_0"
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_STREAM_TIMEOUT = 180
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 10.0

_HEALTH_PATH = "health"
_MODELS_PATH = "v1/models"
_CHAT_PATH = "v1/chat/completions"


def redact_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    """Return a copy of an HTTP header mapping with every value replaced by [REDACTED].

    Mirrors V1 `redact_headers` so diagnostics never leak Modal-Key/Modal-Secret or any
    other header value.
    """
    return {name: "[REDACTED]" for name in headers}


class LlamaCppProvider(contracts.RuntimeProvider):
    """Synchronous adapter over a llama-server OpenAI-compatible HTTP API (V1).

    Args:
        base_url:    the published server URL (caller-provided from modal.Server.get_url();
                     NEVER hand-built).
        modal_key:   the Modal proxy header value (wk-... token) for `Modal-Key`.
        modal_secret:the Modal proxy header value (ws-... token) for `Modal-Secret`.
        model_id:    the alias reported by `/v1/models` (V1 MODEL_ALIAS).
        context_length: declared llama-server ctx-size (V1 4096).
        format/runtime: descriptor metadata ("gguf" / "llama-cpp").
        request_timeout: bounded non-stream/deadline request timeout (V1 120 s).
        stream_timeout:  bounded stream timeout (V1 180 s).
        opener:      HTTP seam (defaults to urllib.request.urlopen) for tests; must be a
                     callable(request, timeout=...) -> response-like object exposing
                     `.status` and `.read()` (and iteration for stream responses).
        sleep/monotonic: seam for bounded 503 backoff timing (defaults to time.*).

    The provided `base_url` is used verbatim as the prefix for every path (no re-/dir/etc.
    rewriting). `health()`/`models()`/`chat()`/`return` read the JSON body; `stream()`
    iterates the SSE `data:` lines.
    """

    def __init__(
        self,
        base_url: str,
        modal_key: str,
        modal_secret: str,
        model_id: str = DEFAULT_MODEL_ALIAS,
        context_length: int = 4096,
        format: str = "gguf",
        runtime: str = "llama-cpp",
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        stream_timeout: float = DEFAULT_STREAM_TIMEOUT,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ProviderError("base_url must be an http(s) URL")
        self._base_url = base_url
        self._model_id = model_id
        self._context_length = context_length
        self._format = format
        self._runtime = runtime
        if request_timeout <= 0 or stream_timeout <= 0:
            raise ProviderError("request/stream timeouts must be positive")
        self._request_timeout = request_timeout
        self._stream_timeout = stream_timeout
        if base_delay <= 0 or max_delay < base_delay:
            raise ProviderError("503 retry backoff must be finite and bounded")
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._opener = opener
        self._sleep = sleep
        self._monotonic = monotonic
        self._headers = self._build_auth_headers(modal_key, modal_secret)
        self._descriptor = ModelDescriptor(
            model_id=model_id,
            base_url=base_url,
            format=format,
            runtime=runtime,
            capabilities=VERIFIED_CAPABILITIES,
            context_length=context_length,
        )
        self._descriptor.validate()

    # --- auth -------------------------------------------------------------

    @staticmethod
    def _build_auth_headers(key: str, secret: str) -> dict[str, str]:
        """Fail-closed proxy-header guard: exactly Modal-Key/Modal-Secret, no Bearer.

        Mirrors V1 `proxy_headers`/`validate_proxy_headers`: both credentials required and
        must be the `wk-`/`ws-` proxy token shapes.
        """
        if not key or not secret:
            raise ProviderError("both Modal proxy credentials are required")
        if not key.startswith("wk-") or not secret.startswith("ws-"):
            raise ProviderError(
                "Modal-Key must be a wk- proxy token and Modal-Secret a ws- proxy token"
            )
        return {"Modal-Key": key, "Modal-Secret": secret}

    def _request_headers(self, headers: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(self._headers)
        if headers:
            for name, value in headers.items():
                merged[name] = value
        return merged

    # --- transport seams -------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path}"

    def _redacted_request_snapshot(self, method: str, path: str) -> str:
        return f"{method} {self._url(path)} headers={redact_headers(self._headers)}"

    def _open_bounded(
        self, request: urllib.request.Request, timeout: float, path: str, method: str
    ) -> Any:
        """Open with bounded 503 backoff; raise ProviderError on deadline exhaustion."""
        attempt = lambda: self._attempt(request, timeout)
        end = self._monotonic() + timeout
        delay = self._base_delay
        while True:
            response = attempt()
            if getattr(response, "status", 0) != 503:
                return response
            if self._monotonic() + delay > end:
                raise ProviderError(
                    f"{self._redacted_request_snapshot(method, path)}: 503 retry deadline exceeded"
                )
            self._sleep(delay)
            delay = min(delay * 2, self._max_delay)

    def _attempt(self, request: urllib.request.Request, timeout: float) -> Any:
        try:
            return self._opener(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            # urllib raises HTTPError for non-2xx; expose the status so 503 retry logic can
            # treat it like a response, and only the status code is surfaced (no secret body).
            return _HttpErrorResponse(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("bounded HTTP request failed") from exc
        except ValueError as exc:
            raise ProviderError("invalid HTTP request") from exc

    def _read_json(self, response: Any, path: str, method: str) -> Any:
        """Read and decode the JSON body; raise ProviderError on malformed payloads."""
        try:
            body = response.read()
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: malformed JSON response"
            ) from exc

    def _raise_for_status_4xx(self, status: int, path: str, method: str) -> None:
        if 400 <= status <= 499:
            kind = "auth" if status == 401 else "client"
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: {kind} error (HTTP {status})"
            )

    # --- public RuntimeProvider surface ----------------------------------

    def health(self) -> HealthStatus:
        """GET /health: 200 -> ready; 503 -> bounded retry (or deadline exceed); 401 -> ProviderError.

        V1 `bounded_retry_503` treats a 503 as scale-to-zero/loading and retries it (bounded)
        rather than reporting a terminal "not ready", so `_open_bounded` handles 503s and a
        final 503 surfaces as ProviderError (deadline exceeded), never a silent not-ready.
        """
        path, method = _HEALTH_PATH, "GET"
        response = self._open_bounded(self._request(method, path), self._request_timeout, path, method)
        status = getattr(response, "status", 0)
        if status == 200:
            return HealthStatus(ready=True)
        if status == 401:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: auth error (HTTP 401)"
            )
        raise ProviderError(
            f"{self._redacted_request_snapshot(method, path)}: unexpected HTTP {status}"
        )

    def models(self) -> list[ModelDescriptor]:
        """GET /v1/models: build one ModelDescriptor from the fixed alias."""
        path, method = _MODELS_PATH, "GET"
        response = self._open_bounded(self._request(method, path), self._request_timeout, path, method)
        status = getattr(response, "status", 0)
        self._raise_for_status_4xx(status, path, method)
        if status != 200:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: unexpected HTTP {status}"
            )
        payload = self._read_json(response, path, method)
        return self._build_models(payload, method, path)

    def chat(self, messages: Sequence[Message]) -> ChatResponse:
        """POST /v1/chat/completions (non-stream): return non-empty chat content."""
        path, method = _CHAT_PATH, "POST"
        body = self._chat_body(messages, stream=False)
        response = self._open_bounded(self._request(method, path, body), self._request_timeout, path, method)
        status = getattr(response, "status", 0)
        self._raise_for_status_4xx(status, path, method)
        if status != 200:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: unexpected HTTP {status}"
            )
        payload = self._read_json(response, path, method)
        return self._build_chat(payload, method, path)

    def stream(self, messages: Sequence[Message]) -> Iterator[StreamEvent]:
        """POST /v1/chat/completions (stream): yield SSE chunks, end with done marker.

        Mirrors V1 `parse_sse`: requires at least one `data:` event and a terminal `[DONE]`,
        else ProviderError. The terminal `StreamEvent(done=True, content="")` is yielded
        exactly once as the final event.
        """
        path, method = _CHAT_PATH, "POST"
        body = self._chat_body(messages, stream=True)
        response = self._open_bounded(self._request(method, path, body), self._stream_timeout, path, method)
        status = getattr(response, "status", 0)
        self._raise_for_status_4xx(status, path, method)
        if status != 200:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: unexpected HTTP {status}"
            )
        seen_event = False
        for raw_line in response:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise ProviderError(
                    f"{self._redacted_request_snapshot(method, path)}: unexpected streaming line"
                )
            payload = line[5:].strip()
            if payload == "[DONE]":
                if not seen_event:
                    raise ProviderError(
                        f"{self._redacted_request_snapshot(method, path)}: stream requires data "
                        "event(s) and terminal marker"
                    )
                yield StreamEvent(content="", done=True)
                return
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProviderError("invalid JSON SSE event") from exc
            if not isinstance(event, dict):
                raise ProviderError("SSE event must be an object")
            seen_event = True
            delta = event.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "") if isinstance(delta, dict) else ""
            yield StreamEvent(content=content if isinstance(content, str) else "")
        raise ProviderError(
            f"{self._redacted_request_snapshot(method, path)}: stream requires data event(s) "
            "and terminal marker"
        )

    def capabilities(self) -> ModelDescriptor:
        return self._descriptor

    # --- request / payload builders --------------------------------------

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        return urllib.request.Request(
            self._url(path),
            data=data,
            headers=self._request_headers({"Content-Type": "application/json"} if data else {}),
            method=method,
        )

    def _chat_body(self, messages: Sequence[Message], stream: bool) -> dict[str, Any]:
        wire = [{"role": m.role, "content": m.content} for m in messages]
        return {
            "model": self._model_id,
            "messages": wire,
            "stream": stream,
        }

    def _build_models(self, payload: Any, method: str, path: str) -> list[ModelDescriptor]:
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ProviderError(f"{self._redacted_request_snapshot(method, path)}: models response must contain a data list")
        ids = {item.get("id") for item in entries if isinstance(item, dict)}
        if self._model_id not in ids:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: models response does not contain "
                f"the fixed alias {self._model_id!r}"
            )
        return [self._descriptor]

    def _build_chat(self, payload: Any, method: str, path: str) -> ChatResponse:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: chat response schema mismatch"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(
                f"{self._redacted_request_snapshot(method, path)}: chat response content must be non-empty"
            )
        return ChatResponse(content=content, model_id=self._model_id)


class _HttpErrorResponse:
    """Minimal status-only surrogate for urllib's raised HTTPError (status is enough for
    bounded 503 retry; the bodies are never inspected or logged)."""

    __slots__ = ("status",)

    def __init__(self, status: int):
        self.status = status
