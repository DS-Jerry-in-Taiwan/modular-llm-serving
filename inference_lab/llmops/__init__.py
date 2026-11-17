"""Local LLMOps registry and evaluation gates for V2."""
from .registry import Registry, RegistryError
from .evaluation_gate import EvaluationGate, GateDecision, EvaluationGateError
from .evaluation_adapter import (
    ComparisonGateAdapter,
    EvaluationAdapterError,
    EvaluationGateAdapter,
    GateInput,
    GateThresholds,
    comparison_gate_input,
    map_comparison_to_gate_input,
)
from .e2e05_gate import (
    E2E05Gate,
    E2E05GateError,
    E2E05GateRunner,
    E2E05Registration,
    register_e2e05_manifests,
    verify_fixture_evidence,
)
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
    "ComparisonGateAdapter", "EvaluationAdapterError", "EvaluationGateAdapter",
    "GateInput", "GateThresholds", "comparison_gate_input", "map_comparison_to_gate_input",
    "E2E05Gate", "E2E05GateError", "E2E05GateRunner", "E2E05Registration",
    "register_e2e05_manifests", "verify_fixture_evidence",
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
