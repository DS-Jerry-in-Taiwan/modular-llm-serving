"""Offline, GPU-free evaluation gate for V2 CI fixtures.

The command consumes a small JSON fixture containing expected/actual values and
an immutable evidence reference.  It never imports a provider, starts a model,
or probes CUDA.  A report is always emitted for a well-formed fixture and has a
stable schema, source commit, and evidence digest.  Missing or changed evidence
produces a failing report and a non-zero command exit status.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from .durable import EvidenceRef, ProvenanceError


CI_GATE_SCHEMA_VERSION = 1
CI_GATE_NAME = "v2-offline-evaluation"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CIGateError(ValueError):
    """Malformed fixture or report request."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_strict_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_strict_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _commit(value: Any) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise CIGateError("source_commit must be a 40-character lowercase SHA")
    return value


def _load_fixture(path: str | Path) -> tuple[dict[str, Any], str]:
    fixture_path = Path(path)
    try:
        raw = fixture_path.read_bytes()
        fixture = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CIGateError(f"cannot read fixture: {fixture_path}") from exc
    if not isinstance(fixture, dict) or fixture.get("schema_version") != CI_GATE_SCHEMA_VERSION:
        raise CIGateError("unsupported fixture schema_version")
    if not isinstance(fixture.get("cases"), list) or not fixture["cases"]:
        raise CIGateError("fixture cases must be a non-empty list")
    seen: set[str] = set()
    for case in fixture["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise CIGateError("fixture case IDs must be unique non-empty strings")
        if "expected" not in case or "actual" not in case:
            raise CIGateError(f"fixture case lacks expected/actual: {case.get('id')}")
        seen.add(case["id"])
    return fixture, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CIGateResult:
    report: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.report["status"] == "pass"


def evaluate_fixture(path: str | Path, *, source_commit: str | None = None) -> CIGateResult:
    """Evaluate one fixture without GPU/provider dependencies."""
    fixture, fixture_sha256 = _load_fixture(path)
    commit = _commit(source_commit if source_commit is not None else fixture.get("source_commit"))
    reasons: list[str] = []
    evidence_digest: str | None = None
    evidence_value = fixture.get("evidence")
    try:
        evidence = EvidenceRef.from_value(evidence_value)
        if evidence.source_commit != commit:
            raise ProvenanceError("evidence source_commit does not match fixture source_commit")
        evidence.verify()
        evidence_digest = evidence.sha256
    except ProvenanceError as exc:
        reasons.append(f"provenance: {exc}")

    case_results: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        passed = _strict_equal(case["actual"], case["expected"])
        case_results.append({"id": case["id"], "status": "pass" if passed else "fail"})
        if not passed:
            reasons.append(f"case failed: {case['id']}")
    report = {
        "schema_version": CI_GATE_SCHEMA_VERSION,
        "gate": CI_GATE_NAME,
        "status": "pass" if not reasons else "fail",
        "source_commit": commit,
        "fixture_sha256": fixture_sha256,
        "evidence_sha256": evidence_digest,
        "case_results": case_results,
        "reasons": reasons,
    }
    return CIGateResult(report)


def write_report(result: CIGateResult, path: str | Path) -> None:
    """Write a derived report; the fixture and raw evidence are untouched."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline V2 fixture evaluation gate")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--report", type=Path, help="write the JSON gate report")
    parser.add_argument("--source-commit", help="override the fixture source commit")
    args = parser.parse_args(argv)
    try:
        result = evaluate_fixture(args.fixture, source_commit=args.source_commit)
    except CIGateError as exc:
        print(json.dumps({"schema_version": CI_GATE_SCHEMA_VERSION, "gate": CI_GATE_NAME, "status": "fail", "reasons": [str(exc)]}, sort_keys=True))
        return 2
    if args.report:
        write_report(result, args.report)
    print(json.dumps(result.report, ensure_ascii=False, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
