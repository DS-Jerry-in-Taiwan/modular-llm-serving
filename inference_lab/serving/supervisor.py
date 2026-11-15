"""Owned resident engine lifecycle; transport and provider adapters plug in later."""
from __future__ import annotations

from enum import Enum
import threading
from typing import Protocol


class EngineState(str, Enum):
    STOPPED = "stopped"; STARTING = "starting"; READY = "ready"; STOPPING = "stopping"; FAILED = "failed"


class SupervisorError(RuntimeError): pass


class Engine(Protocol):
    def start(self) -> None: ...
    def ready(self) -> bool: ...
    def stop(self) -> None: ...


class ResidentSupervisor:
    """Own exactly one engine.  Terminal state changes happen after stop returns."""
    def __init__(self, factory):
        self._factory = factory; self._engine = None; self._state = EngineState.STOPPED; self._lock = threading.RLock()

    @property
    def state(self) -> EngineState:
        with self._lock: return self._state

    def start(self) -> None:
        with self._lock:
            if self._state is EngineState.READY: return
            if self._state in {EngineState.STARTING, EngineState.STOPPING}: raise SupervisorError("engine transition in progress")
            self._state = EngineState.STARTING
            engine = None
            try:
                engine = self._factory()
                if self._engine is not None: raise SupervisorError("owned engine already exists")
                self._engine = engine
                engine.start()
                if not engine.ready(): raise SupervisorError("engine is not ready")
                self._state = EngineState.READY
            except Exception:
                self._state = EngineState.FAILED
                try:
                    if engine is not None: engine.stop()
                finally:
                    self._engine = None
                raise

    def require_ready(self) -> Engine:
        with self._lock:
            if self._state is not EngineState.READY or self._engine is None: raise SupervisorError("engine is not ready")
            return self._engine

    def stop(self) -> None:
        with self._lock:
            if self._state is EngineState.STOPPED: return
            if self._state is EngineState.STOPPING: return
            engine = self._engine; self._state = EngineState.STOPPING
            try:
                if engine is not None: engine.stop()
            except Exception:
                self._state = EngineState.FAILED
                raise
            self._engine = None; self._state = EngineState.STOPPED
