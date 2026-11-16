# Local inference lab

Current release: **v2.0.0 — Serviceable Inference**. The repository now contains
two explicit local execution paths: the preserved V1 cold benchmark and the V2
resident inference service. See [CHANGELOG.md](CHANGELOG.md) for the release
scope and validation record.

Run reproducible local Qwen3.8-27B GGUF experiments and a loopback comparison demo on the same pinned llama.cpp runtime. The local path uses an external model/runtime cache and owns each inference process only for its run. The original Gemma/Modal deployment remains a separate preserved path.

- [Architecture and interfaces](ARCHITECTURE.md): module boundaries, execution, cancellation, evidence, and limitations.
- [Runbook](INFERENCE_LAB.md): configuration, acquisition, preflight, CLI experiments, review, and demo commands.
- [V2 architecture decisions](docs/architecture/v2-serving-adr.md): resident serving, admission, streaming, LLMOps, and Demo/API integration.
- [Fresh V1 convergence comparison](results/convergence-v1-20261114/report.md) and [reproduction guide](results/convergence-v1-20261114/REPRODUCE.md): new reviewed 6×2 GPU pair with the converged source layout.
- [Historical recorded comparison](results/qwen38-rtx3080-20261114-v2/report.md) and [reproduction guide](results/qwen38-rtx3080-20261114-v2/REPRODUCE.md): historical measurements, preserved as originally recorded.
- [Repository guidance](AGENTS.md): local checks and the separate Modal deployment contract.

From the repository root:

```bash
venv/bin/python -m inference_lab --help
venv/bin/python -m inference_lab.demo --help
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
```

Real inference needs the verified external model, pinned external runtime, usable GPU access and matching driver libraries described in the runbook. Unit tests do not prove GPU compatibility. Do not run CLI experiments and the web demo against the GPU simultaneously: job serialization is local to one demo process.

## V2 resident demo

The default demo remains the V1 cold benchmark. To exercise the reusable V2
service path, start the same loopback server with resident execution and a
durable job ledger:

```bash
venv/bin/python -m inference_lab.web.server --execution-mode resident \
  --state-path docs/local-inference/v2-jobs.sqlite3
```

Open `http://127.0.0.1:7860/`. The UI labels the selected protocol explicitly:
V1 uses `/api/*` and cold processes, while V2 uses `/api/v2/*`, one resident
engine, bounded admission, cancellation/deadlines, streaming terminal events,
and durable local job state. V2 accepts the versioned request fields documented
in [the API ADR](docs/architecture/v2-integration-api-adr.md); it does not accept
V1 benchmark fields such as `variants` or `max_tokens`.
