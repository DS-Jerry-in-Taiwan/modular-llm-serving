# E2E-04 review fixtures

These sanitized inputs contain only the `apply_baseline_review` contract:
`run_id`, exact `run_sha256`, `dataset_sha256`, per-sample output hashes, and
rubric decisions. They do not include model responses, telemetry, host details,
or local evidence paths.

Regenerate a fixture from an externally preserved run directory with:

```bash
python3 scripts/e2e04_review_fixture.py \
  --run-dir RUN_DIRECTORY \
  --output tests/fixtures/e2e04/review.json
```

The source run must remain immutable and available locally when applying the
fixture. The two checked-in inputs bind the completed V1 24-layer and V1
40-layer runs requested by E2E-04. The resident V2 record uses a different
schema and is intentionally not passed to `apply_baseline_review`.
