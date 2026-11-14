# Local inference lab

Run reproducible local Qwen3.8-27B GGUF experiments and a loopback comparison demo on the same pinned llama.cpp runtime. The local path uses an external model/runtime cache and owns each inference process only for its run. The original Gemma/Modal deployment remains a separate preserved path.

- [Architecture and interfaces](ARCHITECTURE.md): module boundaries, execution, cancellation, evidence, and limitations.
- [Runbook](INFERENCE_LAB.md): configuration, acquisition, preflight, CLI experiments, review, and demo commands.
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
