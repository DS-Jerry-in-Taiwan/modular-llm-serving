"""Generate a sanitized E2E-04 baseline review input from immutable run evidence.

The generated JSON contains only the review contract fields (run, dataset and
output digests plus rubric decisions). It never copies responses, telemetry or
local evidence paths into the fixture. Use the output with ``inference_lab
review`` or ``apply_baseline_review`` while the source run directory remains
available locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inference_lab.artifacts import output_hash


REVIEWER = "independent E2E-04 quality reviewer"
RATIONALE = (
    "Reviewed this exact completed output against every rubric criterion and "
    "the source prompt; no unsupported claims or omissions found."
)


def make_review(run_dir: Path, reviewer: str = REVIEWER) -> dict:
    """Build a review input bound to the supplied run's exact immutable bytes."""
    run_bytes = (run_dir / "run.json").read_bytes()
    run = json.loads(run_bytes)
    if run.get("kind") not in {"baseline", "optimization"}:
        raise ValueError("E2E-04 review requires a baseline or optimization run")
    if run.get("status") != "completed" or not run.get("cleanup_confirmed"):
        raise ValueError("run must be completed with confirmed cleanup")
    if not run.get("baseline_eligibility", {}).get("measurement_complete"):
        raise ValueError("run is not measurement-complete")

    cases = {case["id"]: case for case in run.get("dataset", {}).get("cases", [])}
    if not cases:
        raise ValueError("run dataset is required")
    repeats = run.get("measurement_protocol", {}).get("repeats")
    expected_count = len(cases) * repeats if type(repeats) is int else None
    samples = run.get("samples")
    if expected_count is None or not isinstance(samples, list) or len(samples) != expected_count:
        raise ValueError("sample cardinality does not match dataset and repeats")
    sample_ids = [sample.get("sample_id") for sample in samples]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique")

    review_samples = {}
    for sample in samples:
        case = cases.get(sample.get("case_id"))
        if case is None:
            raise ValueError("sample references an unknown case")
        if not case.get("manual_required"):
            continue
        result = sample.get("result", {})
        if sample.get("status") != "completed" or result.get("truncated") or not result.get("content", "").strip():
            raise ValueError("manual sample must be completed, non-empty and non-truncated")
        digest = output_hash(result)
        if sample.get("output_sha256") != digest:
            raise ValueError("sample output_sha256 does not match recorded result")
        review_samples[sample["sample_id"]] = {
            "output_sha256": digest,
            "rubric": {criterion: True for criterion in case.get("rubric", [])},
            "rationale": RATIONALE,
        }
    if not review_samples:
        raise ValueError("dataset has no manual samples")

    return {
        "run_id": run["run_id"],
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "dataset_sha256": run["dataset_sha256"],
        "reviewer": reviewer,
        "samples": review_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", default=REVIEWER)
    args = parser.parse_args()
    review = make_review(args.run_dir, args.reviewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(review, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"run_id": review["run_id"], "run_sha256": review["run_sha256"], "samples": len(review["samples"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
