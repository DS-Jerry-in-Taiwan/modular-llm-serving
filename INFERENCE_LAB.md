# Local inference experiments

This additive experiment path is separate from the fixed Gemma/Modal deployment. First target: Qwen3.8-27B on RTX 3080 12GB with system RAM offload. Successful unit tests are not GPU or model-load evidence.

The first interface uses a versioned JSON configuration and CLI so the same headless experiment can be repeated and compared without an interactive UI.

## T1: read-only preflight

```sh
venv/bin/python -m inference_lab preflight \
  --config experiments/qwen38-27b-q4.json \
  --check-network \
  --output docs/agent_context/inference-lab/preflight-unique.json
```

Use a unique output path: previous reports are never overwritten. Exit 0 means ready for **runtime validation**, not a successful inference. Exit 2 means blocked; inspect the structured report. Run on the actual GPU host with device/library access. No `.env`, secret, model download, service startup, Docker mutation or cloud operation is performed. Network checks only read pinned HF metadata; omit `--check-network` for offline inspection, which cannot claim model verification.

The versioned JSON configuration pins a public model revision, file size and LFS SHA-256. It pins llama.cpp b10896 (the quantizer release named by the model publisher), built/downloaded into an **external** runtime directory; do not change the repo's existing llama.cpp submodule or use the existing Modal provider for localhost auth. Runtime architecture compatibility still requires a real T2 smoke test.

## Proposed executable plan and resource caps

1. Establish genuine NVML/CUDA functionality. PCI/device presence alone is insufficient. A zero-byte `nvidia-smi` or driver `.so` file is not usable GPU support. Repair by isolated matching userspace libraries only after agreeing the execution approach; never replace system driver files automatically.
2. Resolve the immutable quantized model and verify filename/size/SHA-256 after acquisition. Current Q4_K_M is 17,442,399,968 bytes; it needs CPU offload on 12GB. No image projector is needed for initial text tasks.
3. Use an external pinned llama.cpp build, single request slot and 4096-token context. Start with 24 GPU layers and 8 CPU threads, validate actual allocation, and stop before OOM rather than silently switching to CPU-only or a different model.
4. Smoke cap: 600 seconds. Then three text categories, at most six initial prompts, one warmup and two measured repeats; total initial experiment cap 3600 seconds. These are operational budgets, not performance acceptance thresholds.
5. Download cap 25,000,000,000 bytes; runtime storage reserve 5 GiB. Cache downloaded model outside Git. Do not start acquisition until the host is runnable.
6. Later tickets add immutable run records, task scoring, baseline repetitions and optimization comparison. Baselines and usability thresholds come from actual tests, not guessed tokens/s.

## Host observation on 2026-11-14

Escalated read-only checks identified RTX 3080 12GB via PCI, `/dev/nvidia0`, NVIDIA kernel driver 580.178.04, i5-13600K, 62 GiB RAM (~45 GiB available), and ~545 GiB available disk. `nvidia-smi` and matching libcuda/NVML files were **zero bytes**, so the initial NVML/CUDA probes failed `file too short`. Matching driver userspace libraries were subsequently recovered into a private cache without changing system files; use the explicit library options below and preserve both observations. Successful probes still do not prove inference compatibility. Ordinary sandbox checks additionally hide GPU nodes; report both observations separately.

Private matching library paths for this host:

```sh
venv/bin/python -m inference_lab preflight \
  --config experiments/qwen38-27b-q4.json --check-network \
  --cuda-library /home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  --nvml-library /home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --output docs/agent_context/inference-lab/preflight-private-driver-unique.json
```

These paths are host-specific operational inputs, not portable configuration defaults. Run with actual device access; changing library paths does not bypass sandbox device restrictions. Check the new report for NVML and CUDA success before T2. For a future runtime process, its dynamic linker also needs the isolated driver library directory (for example through `LD_LIBRARY_PATH` scoped to that process).

The official Qwen model and chosen quantized artifact were verified via the real HF API, not only a search index. The config records their immutable revisions. These metadata checks prove artifact availability, not inference performance.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
```

## T2: acquire and smoke

Acquire the pinned artifact outside Git (existing complete files are fully re-verified; partial files are never silently resumed or overwritten):

```sh
venv/bin/python -m inference_lab acquire-model \
  --config experiments/qwen38-27b-q4.json \
  --destination /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf
```

Run one local-only, ticket-owned service, ask a short non-thinking smoke prompt, then stop it on success or failure:

```sh
LD_LIBRARY_PATH=/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu \
venv/bin/python -m inference_lab smoke \
  --config experiments/qwen38-27b-q4.json \
  --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server \
  --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/agent_context/inference-lab/runs
```

The host needs actual GPU device access. The CLI never reads root `.env` or uses Modal authentication. Each unique run directory retains `run.json` and `server.log`; failed runs remain failed, cancellation is recorded after cleanup. Exact model checksum and runtime commit/version are checked before startup. An observed positive GPU offload count and valid NVML evidence are required; CPU-only output cannot pass. HTTP wall deadlines and an overall watchdog bound GPU activity. The process group belongs only to this invocation and is stopped at completion, including TERM-ignoring children. Original deployment source, pinned submodule and unrelated workloads are not changed.

The smoke asks for at most 64 generated tokens, marks truncation explicitly, and rejects an empty or truncated answer. This verifies the inference path, not summary/code quality or performance acceptance. Later task suites evaluate real usefulness.

### Verified first GPU smoke (2026-11-14)

Run `20261114T052455Z-995842ddd5` used the pinned Q4_K_M artifact and build10896/fa676981 runtime on the real RTX 3080. The server reported **24/66 layers offloaded**, a 5906.28 MiB GPU model buffer and ~6247 MiB total owned GPU buffers. NVML sampled ~7.40 GB device-wide used memory during the request (including other GPU allocations). The exact answer was `inference ok`, with `finish_reason=stop`, 4 generated tokens, 4.42 seconds request time and 5.03 seconds readiness time. The owned server was stopped and cleanup confirmed. This tiny, cache-warm smoke is not a workload baseline or a general speed estimate.

Two earlier failed attempts remain in the evidence history: one exposed the need to retry `/health` loading HTTP 503 within the deadline; another lacked offload evidence because the pinned runtime maps loader INFO to trace verbosity4. The harness now retries only known readiness503 and explicitly sets log verbosity4. Missing evidence was treated as failure, not presented as proof that hardware was CPU-only.

Local evidence is retained under ignored `docs/agent_context/inference-lab/`: model acquisition checksum receipt and per-run JSON/log files. No credentials are included; model/runtime caches live outside the checkout.
