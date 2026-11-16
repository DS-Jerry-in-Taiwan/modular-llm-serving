"""Bounded, local admission for the V2 resident inference engine.

The queue deliberately owns request admission and lifecycle bookkeeping, but not
the engine lifecycle.  :class:`ResidentSupervisor` remains the owner of the
engine; a worker starts it (when necessary) and obtains the ready engine through
``require_ready`` before invoking the injected deterministic runner.

This is a process-local seam for the V2 service.  It is intentionally not a
distributed queue and does not pretend to make an arbitrary, non-cooperative
engine cancellable.  Runners receive a ``threading.Event`` and must observe it
at useful interruption points.  A request becomes terminal only after its
runner and the optional cleanup hook have returned.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import inspect
import math
import threading
import time
from typing import Any, Callable, Deque

from inference_lab.v2_contracts import InferenceRequest

from .supervisor import EngineState, ResidentSupervisor


class AdmissionError(RuntimeError):
    """Base error for admission and request execution failures."""


class AdmissionRejected(AdmissionError):
    """The request did not enter the queue.

    ``code`` is stable enough for a transport adapter to map to an error
    envelope, while the exception itself remains useful to local callers.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class AdmissionClosed(AdmissionRejected):
    def __init__(self) -> None:
        super().__init__("admission_closed", "admission queue is closed")


class AdmissionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"

    @property
    def terminal(self) -> bool:
        return self in {
            AdmissionStatus.COMPLETED,
            AdmissionStatus.FAILED,
            AdmissionStatus.CANCELLED,
            AdmissionStatus.DEADLINE_EXCEEDED,
        }


@dataclass(frozen=True)
class AdmissionResult:
    """Immutable terminal (or observed) state returned by ``JobHandle.result``."""

    request_id: str
    run_id: str
    status: AdmissionStatus
    value: Any = None
    error: BaseException | None = None

    @property
    def terminal(self) -> bool:
        return self.status.terminal


@dataclass
class _Job:
    request: InferenceRequest
    run_id: str
    deadline: float | None
    cancel_event: threading.Event
    done: threading.Event
    status: AdmissionStatus = AdmissionStatus.QUEUED
    value: Any = None
    error: BaseException | None = None
    cancellation_requested: bool = False
    deadline_requested: bool = False
    cleanup_done: bool = False


def estimate_prompt_tokens(prompt: str) -> int:
    """Return a deterministic, conservative local token estimate.

    The real tokenizer is provider-specific and intentionally outside this
    admission seam.  Four UTF-8 bytes per token is a stable approximation for
    admission tests; deployments can inject their pinned tokenizer through
    ``token_counter``.
    """

    if not isinstance(prompt, str) or not prompt:
        return 0
    return max(1, (len(prompt.encode("utf-8")) + 3) // 4)


class JobHandle:
    """A thread-safe reference to one admitted request."""

    def __init__(self, queue: "AdmissionQueue", job: _Job):
        self._queue = queue
        self._job = job

    @property
    def request_id(self) -> str:
        return self._job.request.request_id

    @property
    def run_id(self) -> str:
        return self._job.run_id

    @property
    def idempotency_key(self) -> str | None:
        return self._job.request.idempotency_key

    @property
    def status(self) -> AdmissionStatus:
        with self._queue._condition:
            return self._job.status

    @property
    def done(self) -> bool:
        return self._job.done.is_set()

    def cancel(self) -> bool:
        """Request cancellation, returning whether this call changed state.

        Repeated cancellation is intentionally harmless and returns ``False``
        after the first request.  Active jobs remain non-terminal until the
        runner and cleanup hook return.
        """

        return self._queue.cancel(self.request_id)

    def result(self, timeout: float | None = None) -> AdmissionResult:
        if not self._job.done.wait(timeout):
            raise TimeoutError("inference request has not reached a terminal state")
        with self._queue._condition:
            return self._queue._result_locked(self._job)

    def wait(self, timeout: float | None = None) -> AdmissionResult:
        return self.result(timeout)


Runner = Callable[..., Any]
Cleanup = Callable[..., Any]


class AdmissionQueue:
    """FIFO admission with one active request and bounded queued backpressure.

    ``max_queue`` counts waiting requests, not the one active request.  A
    request's context budget is checked synchronously before it can start the
    supervisor or occupy a queue slot.  ``runner`` is called as
    ``runner(engine, request, cancel_event)``; for convenience, two- and
    one-argument runners and engine methods are accepted as well.
    """

    def __init__(
        self,
        supervisor: ResidentSupervisor,
        *,
        context_size: int,
        max_queue: int = 16,
        runner: Runner | None = None,
        token_counter: Callable[[str], int] = estimate_prompt_tokens,
        cleanup: Cleanup | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(supervisor, ResidentSupervisor):
            raise TypeError("supervisor must be a ResidentSupervisor")
        if type(context_size) is not int or context_size <= 0:
            raise ValueError("context_size must be a positive integer")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a non-negative integer")
        if not callable(token_counter) or (runner is not None and not callable(runner)):
            raise TypeError("token_counter and runner must be callable")
        if cleanup is not None and not callable(cleanup):
            raise TypeError("cleanup must be callable")

        self.supervisor = supervisor
        self.context_size = context_size
        self.max_queue = max_queue
        self.runner = runner
        self.token_counter = token_counter
        self.cleanup = cleanup
        self.clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._pending: Deque[_Job] = deque()
        self._jobs: dict[str, _Job] = {}
        self._idempotency: dict[str, tuple[tuple[Any, ...], _Job]] = {}
        self._active: _Job | None = None
        self._closed = False
        self._worker = threading.Thread(target=self._work, name="v2-admission", daemon=True)
        self._worker.start()

    @property
    def active(self) -> JobHandle | None:
        with self._condition:
            return JobHandle(self, self._active) if self._active is not None else None

    @property
    def queued(self) -> int:
        with self._condition:
            return sum(job.status is AdmissionStatus.QUEUED for job in self._pending)

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def submit(
        self,
        request: InferenceRequest,
        *,
        deadline: float | None = None,
        timeout: float | None = None,
        prompt_tokens: int | None = None,
    ) -> JobHandle:
        """Admit a request or raise :class:`AdmissionRejected`.

        ``deadline`` is an absolute value from the queue's monotonic clock.
        ``timeout`` is a relative convenience and cannot be combined with it.
        An existing idempotency key returns its original handle, including
        after terminal completion; a changed request under that key is a
        conflict and is rejected.
        """

        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        if deadline is not None and timeout is not None:
            raise ValueError("deadline and timeout are mutually exclusive")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            deadline = self.clock() + timeout
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))):
            raise ValueError("deadline must be a monotonic timestamp")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be a finite monotonic timestamp")

        fingerprint = self._fingerprint(request)
        with self._condition:
            if request.idempotency_key is not None:
                previous = self._idempotency.get(request.idempotency_key)
                if previous is not None:
                    old_fingerprint, job = previous
                    if old_fingerprint != fingerprint:
                        raise AdmissionRejected("idempotency_conflict", "idempotency key is bound to another request")
                    return JobHandle(self, job)
            if self._closed:
                raise AdmissionClosed()

            if deadline is not None and deadline <= self.clock():
                raise AdmissionRejected("deadline_exceeded", "request deadline has already expired")

            tokens = self._prompt_tokens(request.prompt, prompt_tokens)
            if tokens + request.max_new_tokens > self.context_size:
                raise AdmissionRejected(
                    "context_budget_exceeded",
                    f"request needs {tokens + request.max_new_tokens} tokens, context budget is {self.context_size}",
                )
            # ``max_queue`` counts waiting work.  A zero-depth queue still
            # accepts a first request when the worker can take it directly;
            # once anything is pending or active, it applies backpressure.
            if len(self._pending) >= self.max_queue and (self._pending or self._active is not None):
                raise AdmissionRejected("queue_full", "inference admission queue is full")

            job = _Job(
                request=request,
                run_id=request.request_id,
                deadline=deadline,
                cancel_event=threading.Event(),
                done=threading.Event(),
            )
            self._pending.append(job)
            self._jobs[request.request_id] = job
            if request.idempotency_key is not None:
                self._idempotency[request.idempotency_key] = (fingerprint, job)
            self._condition.notify()
            return JobHandle(self, job)

    def cancel(self, request_id: str) -> bool:
        with self._condition:
            job = self._jobs.get(request_id)
            if job is None or job.status.terminal:
                return False
            if job.status is AdmissionStatus.QUEUED:
                job.cancellation_requested = True
                job.cancel_event.set()
                try:
                    self._pending.remove(job)
                except ValueError:
                    # The worker may have dequeued it between the status check
                    # and removal; in that case it will observe the event.
                    pass
                self._finish_locked(job, AdmissionStatus.CANCELLED)
                self._condition.notify()
                return True
            if job.status is AdmissionStatus.RUNNING:
                if job.cancellation_requested:
                    return False
                job.cancellation_requested = True
                job.cancel_event.set()
                return True
            return False

    def get(self, request_id: str) -> JobHandle | None:
        with self._condition:
            job = self._jobs.get(request_id)
            return JobHandle(self, job) if job is not None else None

    def shutdown(self, *, stop_engine: bool = False, timeout: float | None = None) -> None:
        """Cancel waiting/active work and optionally stop the resident engine."""

        with self._condition:
            self._closed = True
            for job in self._pending:
                if job.status is AdmissionStatus.QUEUED:
                    job.cancel_event.set()
                    job.cancellation_requested = True
                    self._finish_locked(job, AdmissionStatus.CANCELLED)
            if self._active is not None and not self._active.status.terminal:
                self._active.cancel_event.set()
                self._active.cancellation_requested = True
            self._condition.notify_all()
        if timeout is not None:
            self._worker.join(timeout)
        else:
            self._worker.join()
        if stop_engine and self.supervisor.state is EngineState.READY:
            self.supervisor.stop()

    close = shutdown

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return
                job = self._pending.popleft()
                if job.status.terminal:
                    continue
                if job.deadline is not None and job.deadline <= self.clock():
                    self._finish_locked(job, AdmissionStatus.DEADLINE_EXCEEDED)
                    continue
                job.status = AdmissionStatus.RUNNING
                self._active = job
            self._execute(job)
            with self._condition:
                if self._active is job:
                    self._active = None
                self._condition.notify_all()

    def _execute(self, job: _Job) -> None:
        value: Any = None
        error: BaseException | None = None
        deadline_timer: threading.Timer | None = None

        def expire() -> None:
            with self._condition:
                if job.status is AdmissionStatus.RUNNING and not job.done.is_set():
                    job.deadline_requested = True
                    job.cancel_event.set()

        try:
            if job.deadline is not None and job.deadline <= self.clock():
                job.deadline_requested = True
                job.cancel_event.set()
            elif job.deadline is not None:
                deadline_timer = threading.Timer(max(0.0, job.deadline - self.clock()), expire)
                deadline_timer.daemon = True
                deadline_timer.start()
            if not job.cancel_event.is_set():
                self.supervisor.start()
            if not job.cancel_event.is_set():
                engine = self.supervisor.require_ready()
                value = self._invoke_runner(engine, job)
        except BaseException as exc:  # preserve runner failures in the result
            error = exc
        finally:
            if deadline_timer is not None:
                deadline_timer.cancel()

        with self._condition:
            cancelled = job.cancellation_requested
            deadline = job.deadline_requested or (
                job.deadline is not None and job.deadline <= self.clock()
            )
            if deadline:
                job.deadline_requested = True
                job.cancel_event.set()
        try:
            if self.cleanup is not None:
                self._invoke_cleanup(job, value, error)
                job.cleanup_done = True
        except BaseException as exc:
            error = exc

        with self._condition:
            if deadline:
                status = AdmissionStatus.DEADLINE_EXCEEDED
            elif cancelled or job.cancel_event.is_set():
                status = AdmissionStatus.CANCELLED
            elif error is None:
                status = AdmissionStatus.COMPLETED
            else:
                status = AdmissionStatus.FAILED
            job.value = value
            job.error = error
            self._finish_locked(job, status)

    def _invoke_runner(self, engine: Any, job: _Job) -> Any:
        if self.runner is not None:
            return self._invoke_compatible(self.runner, engine, job.request, job.cancel_event)
        for name in ("generate", "infer", "complete"):
            method = getattr(engine, name, None)
            if callable(method):
                return self._invoke_compatible(method, job.request, job.cancel_event)
        if callable(engine):
            return self._invoke_compatible(engine, job.request, job.cancel_event)
        raise AdmissionError("engine has no generate/infer/complete runner")

    @staticmethod
    def _invoke_compatible(call: Callable[..., Any], *args: Any) -> Any:
        """Call local fake runners with 3-, 2-, or 1-argument signatures."""

        try:
            signature = inspect.signature(call)
        except (TypeError, ValueError):
            return call(*args)
        positional = [p for p in signature.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        variadic = any(p.kind is p.VAR_POSITIONAL for p in signature.parameters.values())
        if variadic or len(positional) >= len(args):
            return call(*args)
        return call(*args[:len(positional)]) if positional else call()

    def _invoke_cleanup(self, job: _Job, value: Any, error: BaseException | None) -> Any:
        return self._invoke_compatible(self.cleanup, job.request, value, error, job.cancel_event)  # type: ignore[arg-type]

    def _finish_locked(self, job: _Job, status: AdmissionStatus) -> None:
        if job.status.terminal:
            return
        job.status = status
        job.done.set()

    def _result_locked(self, job: _Job) -> AdmissionResult:
        return AdmissionResult(job.request.request_id, job.run_id, job.status, job.value, job.error)

    def _prompt_tokens(self, prompt: str, supplied: int | None) -> int:
        try:
            tokens = self.token_counter(prompt) if supplied is None else supplied
        except Exception as exc:
            raise AdmissionRejected("invalid_token_count", "prompt token count could not be determined") from exc
        if type(tokens) is not int or tokens < 0:
            raise AdmissionRejected("invalid_token_count", "prompt token count must be a non-negative integer")
        return tokens

    @staticmethod
    def _fingerprint(request: InferenceRequest) -> tuple[Any, ...]:
        return (
            request.prompt,
            request.model_id,
            request.config_id,
            request.max_new_tokens,
            request.stream,
            request.contract_version,
        )


# A descriptive alias keeps transport code free to use either naming convention.
AdmissionController = AdmissionQueue


__all__ = [
    "AdmissionClosed",
    "AdmissionController",
    "AdmissionError",
    "AdmissionQueue",
    "AdmissionRejected",
    "AdmissionResult",
    "AdmissionStatus",
    "JobHandle",
    "estimate_prompt_tokens",
]
