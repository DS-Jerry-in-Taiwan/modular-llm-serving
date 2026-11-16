"""V2 resident request measurement and evidence helpers.

This module is intentionally separate from the V1 cold job path.  A resident
engine stays alive across requests, so each request gets a bounded resource
sampling window and a response metric snapshot.  Missing provider timings are
recorded as ``unavailable`` instead of being inferred.
"""
from __future__ import annotations

from collections.abc import Callable
import copy
from typing import Any

from .telemetry import ResourceSampler, response_metrics

SCHEMA_VERSION = 1


def engine_pid(engine: Any) -> int | None:
    """Return the owned llama-server PID when the adapter exposes one.

    Production resident engines wrap ``LocalServer``; test engines need not
    expose a process.  In the latter case RSS is explicitly unavailable.
    """
    process = getattr(engine, "process", None)
    if process is None:
        server = getattr(engine, "server", None)
        process = getattr(server, "process", None)
    pid = getattr(process, "pid", None)
    return pid if type(pid) is int and pid > 0 else None


def _unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def resident_response_metrics(result: Any) -> dict[str, dict[str, Any]]:
    """Normalize a provider response without changing V1 telemetry behavior."""
    if not isinstance(result, dict) or "elapsed_seconds" not in result or "response" not in result:
        return {
            "request_seconds": _unavailable("not returned by resident engine"),
            "generated_tokens_per_second": _unavailable("not returned by resident engine"),
        }
    try:
        metrics = response_metrics(result)
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            "request_seconds": _unavailable("not returned by resident engine"),
            "generated_tokens_per_second": _unavailable("not returned by resident engine"),
        }
    # Keep the V1 metric names and nested status/value shape so evidence can be
    # compared by downstream baseline tooling.
    return {name: copy.deepcopy(metrics[name]) for name in
            ("request_seconds", "generated_tokens_per_second")}


class ResidentRequestMeasurement:
    """Capture resource samples around one resident generation call."""

    def __init__(self, *, sampler_factory: Callable[[], ResourceSampler] = ResourceSampler):
        self.sampler = sampler_factory()
        self._started = False
        self._stopped = False

    def start(self, engine: Any) -> None:
        if self._started:
            raise RuntimeError("resident measurement already started")
        self.sampler.pid = engine_pid(engine)
        self.sampler.start()
        self._started = True

    def finish(self, result: Any = None) -> dict[str, Any]:
        if not self._started:
            return {
                "schema_version": SCHEMA_VERSION,
                "resources": _unavailable("measurement did not start"),
                "response_metrics": resident_response_metrics(result),
            }
        if not self._stopped:
            self.sampler.stop()
            self._stopped = True
        return {
            "schema_version": SCHEMA_VERSION,
            "resources": self.sampler.summary(),
            "response_metrics": resident_response_metrics(result),
        }

    def samples(self) -> list[dict[str, Any]]:
        # Return a copy so callers cannot mutate sampler-owned evidence.
        with self.sampler.lock:
            return copy.deepcopy(self.sampler.samples)
