# Reproduce the V1 convergence validation

This fresh pair validates the converged `inference_lab/` implementation on the same local RTX 3080 host. It is separate from the original historical comparisons. Both fresh runs completed twelve nontruncated measured samples after one warmup, confirmed owned-process cleanup, and passed an independent AI review of each repeated summary/code output plus exact typed JSON checks. AI review is identified as such; no human quality validation is claimed.

| Run | ID | Observed layers | Whole run seconds | Measured request seconds |
| --- | --- | ---: | ---: | ---: |
| Baseline | `20261114T131732Z-1327935c51` | 24 | 387.58 | 343.08 |
| Candidate | `20261114T132436Z-34b235865f` | 40 | 244.40 | 218.10 |

The measured-request latency ratio is 1.57× on these fixtures. Whole-run time includes verification/startup/warmup/cleanup; it is a different metric. The only declared config change is `runtime.gpu_layers: 24 → 40`. Sampled device-wide GPU peaks are 7,396,524,032 → 11,134,042,112 bytes. Host CPU load differed; the sequential two-repeat pair does not establish statistical significance or a portable speed guarantee.

## Exact identity

The measurement dependency source aggregate is `799021a7634ec65dd81ece0323caea9199fc4b02f3124fce21cf8a645f898ccc`. The source snapshot records code commit `2d1ccd0f483b67efcb78cd795efbcc242f30d438` and visible working-tree status during documentation/test edits. No measurement source changed between runs. The aggregate covers the actual nine execution modules, including package initializers and evaluation/scoring dependencies; per-file hashes, model/runtime/dataset identity and exact run/review/manifest digests are preserved in [comparison.json](comparison.json).

Model checksum: `880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673`, size 17,442,399,968 bytes. External runtime commit: `fa6769818708afd9807b22183ccda112fd563427`; binary SHA-256 `b1c085cc4bad3d44145503c0f202c4052d72922e8e35630b518e323a14d87e4e`. The registered repository submodule was not used or changed.

## Run a fresh pair

Use the same GPU host with actual device access and verified external cache. Stop the demo and other inference jobs first: there is no cross-process GPU lock. These library/cache paths are host-specific and never replace system driver files. Use new output directories; existing evidence is not overwritten.

```bash
LD_LIBRARY_PATH=/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu \
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m inference_lab baseline \
  --config experiments/qwen38-27b-q4.json --dataset experiments/tasks-v1.json \
  --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server \
  --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/agent_context/inference-lab/convergence/runs --repeats 2

# Replace NEW_BASELINE_ID with the completed fresh baseline ID.
LD_LIBRARY_PATH=/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu \
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m inference_lab baseline \
  --config experiments/qwen38-27b-q4-gpu40.json --dataset experiments/tasks-v1.json \
  --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server \
  --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/agent_context/inference-lab/convergence/runs --repeats 2 \
  --baseline-run-id NEW_BASELINE_ID
```

Keep measurement code frozen across the pair. Independently inspect the actual new outputs against every rubric and append a review for each run using the [runbook's review contract](../../INFERENCE_LAB.md). Use the new baseline/candidate directories in the compare command and a unique destination. To validate the preserved local pair without modifying it:

```bash
venv/bin/python -m inference_lab compare \
  --baseline-dir docs/agent_context/inference-lab/convergence/runs/20261114T131732Z-1327935c51 \
  --candidate-dir docs/agent_context/inference-lab/convergence/runs/20261114T132436Z-34b235865f \
  --output-dir NEW_UNIQUE_COMPARISON_DIRECTORY
```

Raw evidence is ignored local data and is not bundled in Git. Preserve/export both raw run directories, including `manifest.json`, `run.json`, sample JSON, `resource-samples.json`, `server.log`, and appended `review.json`, if transporting this result to another machine. The tracked report alone cannot recreate those original responses. Historical `results/qwen38-rtx3080-20261114*` and earlier raw runs retain their original bytes and source identities.
