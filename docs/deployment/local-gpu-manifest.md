# Local GPU deployment manifest runbook

The deployment seam records the local GPU and CUDA identity before a serving
process is started. It is read-only: it does not download a model, build a
runtime, start a service, or read `.env` or credentials.

## Preflight

From the repository root, run this with a new output path:

```bash
venv/bin/python -m inference_lab.deployment preflight \
  --config experiments/qwen38-27b-q4.json \
  --output docs/agent_context/inference-lab/deployment-manifest.json
```

The command exits `0` only when NVML reports one matching GPU and CUDA reports
at least one device. Without GPU access it writes a `status: blocked` manifest
and exits `2`; this is the expected fail-closed result on ordinary CI hosts.
Use `--nvml-library` and `--cuda-library` only for explicit, isolated driver
library paths supplied by the host operator.

## Acceptance gates

**A gate (blocking):** manifest schema validates; model and llama.cpp revisions
are immutable SHAs; GPU/CUDA probes are successful; the target GPU family and
single-device policy match; the output is created exclusively and is not
overwritten. A blocked or missing manifest is a deployment stop.

**B gate (documented limitation):** readiness does not prove model loading,
GPU layer offload, throughput, or response quality. Those require the separate
runtime smoke and evaluation paths on an actual GPU. Memory is a device-wide
observation and may include unrelated workloads.

Never hand-edit a manifest or reuse it after changing model/runtime/config
identity. Create a new path and retain the old file as immutable evidence.
