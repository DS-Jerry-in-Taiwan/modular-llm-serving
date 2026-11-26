# Local inference experiments

This local experiment path is separate from the fixed Gemma/Modal deployment. First target: Qwen3.8-27B on RTX 3080 12GB with system RAM offload. Successful unit tests are not GPU or model-load evidence.

The CLI uses versioned JSON configuration for repeatable headless experiments. The loopback demo shares the core runtime and telemetry. See [repository entrypoint](README.md) and [architecture/interfaces](ARCHITECTURE.md) for the canonical module layout and ownership contracts.

## T1: read-only preflight

```sh
venv/bin/python -m inference_lab preflight \
  --config experiments/qwen38-27b-q4.json \
  --check-network \
  --output docs/agent_context/inference-lab/preflight-unique.json
```

Use a unique output path: previous reports are never overwritten. Exit 0 means ready for **runtime validation**, not a successful inference. Exit 2 means blocked; inspect the structured report. Run on the actual GPU host with device/library access. No `.env`, secret, model download, service startup, Docker mutation or cloud operation is performed. Network checks only read pinned HF metadata; omit `--check-network` for offline inspection, which cannot claim model verification.

The versioned JSON configuration pins a public model revision, file size and LFS SHA-256. It pins llama.cpp b10896 (the quantizer release named by the model publisher), built/downloaded into an **external** runtime directory; do not change the repo's existing llama.cpp submodule or use the existing Modal provider for localhost auth. Runtime architecture compatibility still requires a real T2 smoke test.

## Execution protocol and resource caps

1. Establish genuine NVML/CUDA functionality. PCI/device presence alone is insufficient. A zero-byte `nvidia-smi` or driver `.so` file is not usable GPU support. Repair only after agreeing the execution approach, and never replace system driver files automatically. On 2026-11-14 the repair used isolated matching userspace libraries; on 2026-11-26 the system driver packages were installed by an agreed manual step (see the host update below).
2. Resolve the immutable quantized model and verify filename/size/SHA-256 after acquisition. Current Q4_K_M is 17,442,399,968 bytes; it needs CPU offload on 12GB. No image projector is needed for initial text tasks.
3. Use an external pinned llama.cpp build, single request slot and 4096-token context. Start with 24 GPU layers and 8 CPU threads, validate actual allocation, and stop before OOM rather than silently switching to CPU-only or a different model.
4. Smoke cap: 600 seconds. Then three text categories, at most six initial prompts, one warmup and two measured repeats; total initial experiment cap 3600 seconds. These are operational budgets, not performance acceptance thresholds.
5. Download cap 25,000,000,000 bytes; runtime storage reserve 5 GiB. Cache downloaded model outside Git. Do not start acquisition until the host is runnable.
6. Immutable run records, task scoring, repeated baselines and reviewed single-factor comparison are implemented in `inference_lab/evaluation/`. Baselines and usability thresholds come from actual tests, not guessed tokens/s.

## Host observation on 2026-11-14

Escalated read-only checks identified RTX 3080 12GB via PCI, `/dev/nvidia0`, NVIDIA kernel driver 580.178.04, i5-13600K, 62 GiB RAM (~45 GiB available), and ~545 GiB available disk. `nvidia-smi` and matching libcuda/NVML files were **zero bytes**, so the initial NVML/CUDA probes failed `file too short`. Matching driver userspace libraries were subsequently recovered into a private cache without changing system files; The 2026-11-26 host update below replaced this with the repaired system driver; the private cache is now a fallback. Preserve both observations. Successful probes still do not prove inference compatibility. Ordinary sandbox checks additionally hide GPU nodes; report both observations separately.

The 2026-11-14 T2–T5 runs below used this private cache through `LD_LIBRARY_PATH`. The `REPRODUCE.md` files under `results/` record that historical setup and are intentionally unchanged.

## Host update on 2026-11-26: system driver repaired

The container's system userspace driver now matches the host kernel module. `libnvidia-compute-580`, `nvidia-utils-580`, `nvidia-kernel-common-580` and `nvidia-firmware-580-580.178.04` were installed at exactly `580.178.04` from Ubuntu `noble-updates` and put on `apt-mark hold`. `nvidia-smi` works, a CUDA kernel test passed, and `preflight` without library options reported NVML `ok` (RTX 3080, 12 GiB) and CUDA `ok` (1 device). The private cache is no longer required. It remains on disk as a fallback.

When the host NVIDIA driver changes, read the new version from `/proc/driver/nvidia/version` and compare it with `dpkg -l 'libnvidia-compute-*'`. The packages are on hold, so `apt-mark unhold` them first. Install the packages of the matching major version (for 590, `libnvidia-compute-590`, `nvidia-utils-590`, `nvidia-kernel-common-590` and `nvidia-firmware-590-<version>`) at the exact version, `apt-mark hold` the new packages, and regenerate the Docker CDI spec (below).

Default preflight (system libraries):

```sh
venv/bin/python -m inference_lab preflight \
  --config experiments/qwen38-27b-q4.json --check-network \
  --output docs/agent_context/inference-lab/preflight-system-driver-20261126.json
```

Fallback, only if the system libraries are broken again: add `--cuda-library` and `--nvml-library` pointing at `/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu/libcuda.so.1` and `.../libnvidia-ml.so.1`, and prefix runtime commands with `LD_LIBRARY_PATH=/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu`. These paths are host-specific, not portable defaults. Changing library paths does not bypass sandbox device restrictions. Check the new report for NVML and CUDA success before T2.

### Docker GPU access (2026-11-26)

Docker runs nested inside this LXC container. `--gpus all` first failed with `bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not permitted`. The fix sets `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml` (original kept as `config.toml.bak-20261126`), which relaxes per-container GPU device isolation. A CDI spec was also generated at `/etc/cdi/nvidia.yaml`; regenerate it after any driver change:

```sh
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

`--gpus all` and `--device nvidia.com/gpu=all` expose the GPU. `--runtime nvidia` alone exposes it only when the image sets `NVIDIA_VISIBLE_DEVICES` (NVIDIA CUDA base images, such as the `server-cuda` image below, set it to `all`); otherwise add `-e NVIDIA_VISIBLE_DEVICES=all`. Some graphics libraries in the CDI spec are zero-byte placeholders; CUDA compute is unaffected.

One manual check used `ghcr.io/ggml-org/llama.cpp:server-cuda` (image build not recorded) with the pinned Q4_K_M model mounted read-only, `-ngl 24 -c 4096 -t 8 -np 1`, bound to `127.0.0.1:18181`. Port 18080 is used by an unrelated Label Studio container. The server became healthy in about 30 seconds and used about 6.6 GB device VRAM. A non-thinking Traditional Chinese prompt returned a coherent answer at about 3.15 generated tokens/s (92 tokens) and 8.3 prompt tokens/s (24 tokens). The container was then removed. This is a single unrecorded observation, not a harness run or baseline, and the Docker image is not the pinned `fa676981` runtime.

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

## T3: task usefulness suite and independent review

```sh
venv/bin/python -m inference_lab suite \
  --config experiments/qwen38-27b-q4.json --dataset experiments/tasks-v1.json \
  --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server \
  --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/agent_context/inference-lab/runs
```

The six fixed cases cover two summaries, two exact JSON extractions and two Python questions. Each response is capped at192 new tokens, thinking disabled, within a1200-second suite cap. One model hash and one model load cover the whole suite. The run saves the exact dataset snapshot/hash, original prompts/responses, generation parameters, failure/truncation status and per-case evaluation. Each case gets its own immutable JSON plus a final run receipt and derived report.

Execution completion is distinct from quality. JSON checks require exact keys, order where applicable, value types and values; extra prose, duplicate keys, incorrect types and truncation fail. Summary/code fact filters only flag candidates: they never mark manual quality pass. An independent reviewer reads actual responses and decides every source/semantic rubric criterion with a rationale. Pending review is not pass, and quality failures remain visible even if inference execution completed.

Submit a JSON review using `python -m inference_lab review --run-dir RUN_DIR --review-file REVIEW_INPUT`. The review must name the reviewer and bind `run_id`, `run_sha256` (SHA-256 of immutable run.json bytes), and `dataset_sha256`; each manual case entry has `output_sha256` (UTF-8 content hash), a `rubric` mapping every criterion to a boolean, and a nonempty `rationale`. Failed, missing, empty or truncated responses cannot receive manual pass. The accepted review is appended once as `review.json`; original run/case evidence is not rewritten. `report.md` is a derived view and can be regenerated.

Host load and competing workloads can affect timing. Until repeated baseline measurements are established, individual suite timings are descriptive observations only.

### Verified first usefulness suite (2026-11-14)

Run `20261114T053113Z-57fe28c22d` completed all six real GPU cases in183.06 seconds, with no truncated responses and cleanup confirmed. Both JSON extractions passed exact typed-value checks. An independent AI acceptor reviewed the four summary/code responses against every rubric item, bound that review to the exact run/dataset/output hashes, and independently executed the proposed Python repair. All six quality outcomes passed for these small fixtures. This is a limited usefulness sample, not a general model capability guarantee or repeated baseline. The review explicitly identifies AI review and does not claim human validation.

## T4: repeated baseline

Use the same suite command arguments with `baseline` instead of `suite`, plus `--repeats 2`. This starts a new owned server, runs one unmeasured warmup, then each of six cases twice in repeat-major dataset order. Every request fixes `cache_prompt=false`; actual engine-reported cached tokens are saved separately. A new process is not a cold filesystem cache, and startup time is labelled accordingly.

A pre-start immutable `manifest.json` captures config/dataset/source identities, exact runtime/model identities, host environment, protocol and baseline ID. Source hashes follow the baseline runner’s actual transitive local relative imports and package initializers, using paths relative to `inference_lab/` (including `evaluation/baseline.py` and `evaluation/tasks.py`), even if the working tree has changes; git commit and dirty status remain visible. Old manifests retain their original source identities. The original run, manifest, samples and append-only review are never overwritten. `progress.json` is explicitly mutable progress, and `report.md` is a derived view.

GPU device-wide used memory, host `MemTotal-MemAvailable`, and owned server process `VmRSS` are sampled every0.25 seconds, including model load when a process exists. Reported maxima are **sampled peaks**, not continuous maxima, and GPU/system values include unrelated workloads. Missing metrics are unavailable, never zero. Non-stream TTFT is unavailable because first-token arrival was not observed. Request elapsed time and engine token throughput are distinct metrics; actual cached tokens and input/output token counts are retained per sample.

A baseline is measurement-complete only when every planned sample completed without truncation and owned-service cleanup succeeded. Baseline CLI exits nonzero otherwise. It is not quality-eligible until all exact JSON outcomes and independently reviewed summary/code outputs pass for this exact run; prior suite reviews cannot be reused silently. The review sidecar binds run bytes, dataset and each repeated output. Use the same `review` CLI; baseline review input has a `samples` mapping keyed by sample ID instead of `cases`.

Per-case aggregates retain count/mean/median/min/max. Two repeats illustrate observed variation; they do not establish statistical significance or a production SLA. Usability thresholds are determined from the experiments rather than guessed in advance.

### Verified first baseline (2026-11-14)

Baseline `20261114T054137Z-0e00a3ea0c` completed12/12 nontruncated measurements after one warmup, with all observed cached-prompt counts zero and cleanup confirmed. The independent AI acceptor reviewed the exact eight repeated summary/code outputs against the rubric, with per-output and run/dataset hashes; both JSON tasks passed strict checks in both repeats. The baseline is measurement-complete and quality-reviewed eligible through its appended review.

Observed total run308.61 seconds, readiness3.69 seconds. Per-task mean request times: summaries13.99/13.58s, JSON10.27/12.87s, code45.44/49.20s. Engine generation throughput was approximately3.0–3.13 tokens/s for these fixtures. The998 sampled resource observations gave device-wide GPU peak7,403,274,240 bytes and owned-server RSS peak17,728,733,184 bytes. These are scoped measurements on this host with possible competing CPU workloads, not portable performance promises. Raw evidence and independent acceptance receipts remain in the local ignored evidence directory.

## T5: single-factor optimization comparison

The GPU40 candidate changes only `runtime.gpu_layers` from24 to40 in `experiments/qwen38-27b-q4-gpu40.json`. Run it with the baseline command plus `--baseline-run-id BASELINE_ID`; the new run is an optimization candidate, not a replacement baseline. Independently review its exact new outputs before comparing.

```sh
venv/bin/python -m inference_lab compare \
  --baseline-dir docs/agent_context/inference-lab/runs/20261114T054137Z-0e00a3ea0c \
  --candidate-dir docs/agent_context/inference-lab/runs/20261114T054955Z-b62d4eb4a9 \
  --output-dir NEW_UNIQUE_COMPARISON_DIRECTORY
```

The comparator checks model/runtime/source/hardware/dataset/protocol identity, actual prompts and generation settings, observed cached-token counts, complete nontruncated sample coverage, pre-start/final manifest consistency, artifact hashes and independent review bindings. It accepts only the declared GPU-layer change and refuses quality-pending or incompatible runs. Each comparison is written once to a new directory.

Final versioned result: `results/qwen38-rtx3080-20261114-v2/report.md`, with structured comparison and `REPRODUCE.md`. The original generated comparison is retained as history. The tracked result includes exact local evidence digests and provenance; raw runs remain ignored and do not travel in Git. The reproduction guide includes optional evidence export and host-specific runtime/library details.

The measured12-request total improved from290.67s to199.39s (1.46× observed latency ratio), while sampled device VRAM rose from7.40GB to11.14GB. Both runs passed the12 fixed short text samples. The40-layer setting is a tested starting point for these tasks; it does not establish long-context, full-agent or media-generation capability. All services were stopped, caches retained for reuse, and no cloud resources or system driver files were changed.

## Local live comparison demo

The loopback-only web interface runs the pinned Qwen3.8-27B Q4_K_M model on the local GPU. It compares the 24 GPU-layer baseline with the 40 GPU-layer configuration, sequentially, with confirmed owned-process cleanup between configurations. Start from the repository root (system driver libraries; see the 2026-11-26 host update for the private-cache fallback):

```bash
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m inference_lab.demo \
  --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server \
  --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/local-inference/demo-runs --port 7860
```

Open `http://127.0.0.1:7860`. The interface is available immediately while asynchronous startup verifies the full pinned model checksum, runtime identity, and actual GPU telemetry. Requests remain disabled until readiness passes. Configuration paths can be supplied through `--baseline-config` and `--optimized-config`; they must pin the same model and differ only in 24 versus 40 GPU layers. HTTP clients cannot choose paths or runtime arguments.

If the service runs on a remote inference machine, its port 7860 remains bound to that machine’s loopback interface. On your own computer, forward it with `ssh -N -L 7860:127.0.0.1:7860 user@host`, replacing `user@host` with your configured SSH destination, then open `http://127.0.0.1:7860` locally. An SSH alias such as `llm-server` can replace the destination only if you have configured it; no alias is assumed or verified here.

Each request accepts 1–3000 prompt characters and 16–256 output tokens (default 192). Generation uses temperature 0, seed 42, and `cache_prompt=false`. One job in this demo process can own the GPU; a second job receives HTTP 409. There is no cross-process GPU lock: stop the demo before running CLI measurements, and do not run independent inference processes concurrently. Cancellation and Ctrl-C/SIGTERM stop the owned process group and prevent the next configuration from starting. Each configuration has a 600-second deadline, the whole job at most 1200 seconds, and each generation request at most 120 seconds. Cleanup failure blocks further runs.

The interface reports observed loading/generating/cleanup phases and non-stream request time; it does not estimate progress or TTFT. GPU memory is a sampled maximum for the whole device, including other processes. Live runs have no warmup and filesystem cache is uncontrolled, so their timings are exploratory. A token-limit truncation is explicitly shown; completing execution does not establish response quality. Arbitrary user prompts receive no automatic quality verdict. `/reference` displays the existing historical controlled benchmark separately.

The latest 20 run summaries are retained in memory for the current server process. Full immutable manifests, per-configuration raw responses, server logs, resource samples and final run records remain under the ignored output directory after restart; browser history is not reconstructed from disk. Prompts and outputs are local evidence and are not sent to a cloud service. The server only binds `127.0.0.1`, accepts same-origin JSON POST requests up to 64 KB, and serves a fixed allowlist from `inference_lab/web/static/`. `inference_lab/demo.py` is the permanent public launcher; HTTP and job implementation live in `web/server.py` and `web/jobs.py`.


## Converged implementation and fresh validation

Configuration validation lives in `config.py`; read-only diagnostics remain in `preflight.py`; `runtime.py` owns bounded llama-server processes; `telemetry.py` owns probes, sampling and metrics; `artifacts.py` owns hashing, unique runs and exclusive JSON writes. Evaluation code is canonical under `evaluation/`; the old flat task/baseline/comparison modules are removed. The web path calls shared core modules directly and does not score arbitrary live prompts.

Disk-full and permission failures in resource/raw/variant/final evidence writes produce terminal failed jobs while preserving cleanup and ownership rules. Unconfirmed process cleanup keeps the demo blocked and retains the owned-server reference. If final receipt persistence fails after successful variants, the job reports failure while those completed variant artifacts remain intact. An unwritable output filesystem cannot guarantee a final disk receipt.

Fresh post-convergence baseline/candidate evidence is recorded separately under ignored `docs/agent_context/inference-lab/convergence/`; historical `results/` and earlier raw run directories remain unchanged. Fresh results must be reviewed against their own exact output/run/dataset hashes before comparison.


### Fresh V1 convergence GPU validation (2026-11-14)

The converged implementation ran a fresh baseline `20261114T131732Z-1327935c51` and GPU40 candidate `20261114T132436Z-34b235865f` with identical measurement source aggregate `799021a7634ec65dd81ece0323caea9199fc4b02f3124fce21cf8a645f898ccc`. Each completed all twelve nontruncated samples after one warmup with cleanup confirmed and observed 24/40 offloaded layers. An independent AI acceptor reviewed every repeated summary/code output with exact run/dataset/output bindings; strict JSON checks passed for both repeated extraction cases.

The fresh measured request total was **343.08s → 218.10s (1.57× observed ratio)**. Whole runs were 387.58s and 244.40s, distinct from measured request time; the combined GPU pair used 631.98s within the 3600-second budget. Sampled device-wide GPU peaks were 7.40GB → 11.13GB. These short fixed fixtures and two sequential repeats retain the host-load/cache/statistical limitations above. They are not a comparison against the original historical pair.

See the new [fresh report](results/convergence-v1-20261114/report.md), [exact comparison identities](results/convergence-v1-20261114/comparison.json), and [reproduction instructions](results/convergence-v1-20261114/REPRODUCE.md). The earlier [historical result](results/qwen38-rtx3080-20261114-v2/report.md) remains unchanged. Raw fresh runs and append-only reviews remain under ignored `docs/agent_context/inference-lab/convergence/runs/`.
