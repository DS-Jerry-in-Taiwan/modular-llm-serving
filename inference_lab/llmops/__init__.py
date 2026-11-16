"""Local LLMOps registry and evaluation gates for V2."""
from .registry import Registry, RegistryError
from .evaluation_gate import EvaluationGate, GateDecision, EvaluationGateError
from .durable import (
    DurableJobError,
    DurableJobStore,
    EvidenceRef,
    InvalidTransitionError,
    JobConflictError,
    JobEvent,
    JobRecord,
    ProvenanceError,
    RECOVERY_POLICY,
)

__all__ = [
    "Registry", "RegistryError", "EvaluationGate", "GateDecision", "EvaluationGateError",
    "DurableJobError", "DurableJobStore", "EvidenceRef", "InvalidTransitionError",
    "JobConflictError", "JobEvent", "JobRecord", "ProvenanceError", "RECOVERY_POLICY",
    "CIGateError", "CIGateResult", "evaluate_fixture", "write_report",
]


def __getattr__(name):
    # Keep ``python -m inference_lab.llmops.ci_gate`` free of runpy's
    # "already in sys.modules" warning while retaining convenient package
    # imports for callers.
    if name in {"CIGateError", "CIGateResult", "evaluate_fixture", "write_report"}:
        from . import ci_gate
        return getattr(ci_gate, name)
    raise AttributeError(name)
