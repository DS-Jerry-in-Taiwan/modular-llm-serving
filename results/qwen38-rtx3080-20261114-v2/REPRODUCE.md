# Reproduce this local inference experiment

See the root `INFERENCE_LAB.md` for preflight, acquisition, suite, review and baseline commands. Configuration files are `experiments/qwen38-27b-q4.json` and `experiments/qwen38-27b-q4-gpu40.json`; the sole optimization is GPU layers24→40. Dataset: `experiments/tasks-v1.json`.

## Runtime and model acquisition

- Model artifact repository `bartowski/Qwen3.8-27B-GGUF`, revision `87410e7b9a5c21e6ca70c0c9cc5237de909fe621`, `Qwen3.8-27B-Q4_K_M.gguf`,17442399968 bytes; SHA256 `880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673`.
- Runtime official source `https://github.com/ggml-org/llama.cpp`, tag `b10896`, commit `fa6769818708afd9807b22183ccda112fd563427`. Acquire/build in an external directory; do not change the repository's historical pinned submodule.
- This host built with CMake Release, `GGML_CUDA=ON`, CUDA architecture86, server target, GNU13.3 for CPU and GNU12.4 CUDA host compiler with CUDA12.0. The actual `llama-server --version` and binary SHA256 are in `comparison.json`. Binary hashes will differ with toolchains; a different binary requires a fresh baseline before within-build optimization comparison.
- Host runtime manifest retained locally at `/home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/manifest.json`, with source archive hash and build provenance.
- Matching driver userspace libraries580.178.04 were extracted into a private cache because this host's system CUDA/NVML files were zero-byte placeholders. No system driver files were changed. The `LD_LIBRARY_PATH` below is specific to this host; other hosts must use their own verified matching driver libraries and must pass preflight.

Source archive SHA256 used on this host: `ffb3574c5b19cef1cbb9af22c8b153f6716ed7d2ac0a7e2b4d459db34fd61575`. Configure an external source/build checkout with the recorded choices:

```sh
cmake -S EXTERNAL_PINNED_SOURCE -B EXTERNAL_BUILD -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=OFF -DCUDA_cuda_driver_LIBRARY=/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu/libcuda.so
cmake --build EXTERNAL_BUILD --target llama-server -j12
```

Confirm the build resolves a CUDA-compatible host compiler (this host used GNU12.4 for CUDA) and `llama-server --version` reports the pinned commit before acquisition/run. On another GPU, choose its actual supported architecture and create a fresh hardware-specific baseline.

## Run fresh measurements

Replace external model/runtime paths as appropriate. These commands create new immutable run IDs; they cannot reproduce wall-clock timings exactly or the old run IDs.

```sh
export LD_LIBRARY_PATH=/home/ubuntu/.cache/inference-lab/runtime/nvidia-580.178.04/usr/lib/x86_64-linux-gnu
venv/bin/python -m inference_lab acquire-model --config experiments/qwen38-27b-q4.json --destination /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf
venv/bin/python -m inference_lab baseline --config experiments/qwen38-27b-q4.json --dataset experiments/tasks-v1.json --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf --output-root docs/agent_context/inference-lab/runs --repeats 2
venv/bin/python -m inference_lab baseline --config experiments/qwen38-27b-q4-gpu40.json --dataset experiments/tasks-v1.json --binary /home/ubuntu/.cache/inference-lab/runtime/llama-fa676981/build/bin/llama-server --model /home/ubuntu/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf --output-root docs/agent_context/inference-lab/runs --repeats 2 --baseline-run-id NEW_BASELINE_RUN_ID
```

Each run fixes one warmup, six prompts×two measured repeats, no requested prompt cache, no thinking, temperature0 and seed42. Baseline/candidate must use the exact same relevant source, runtime binary, model, dataset, generation protocol and hardware. Only the declared GPU-layer difference is accepted. New independent reviews must bind the new run/dataset/output hashes; old reviews do not carry over.

After independent rubric review of each run, use `python -m inference_lab review --run-dir RUN_DIR --review-file NEW_REVIEW.json`, then `python -m inference_lab compare --baseline-dir NEW_BASELINE_DIR --candidate-dir NEW_CANDIDATE_DIR --output-dir NEW_COMPARISON_DIR`.

## Evidence retention

The raw run manifests, per-sample outputs, logs, resource samples, immutable reviews and acceptance receipts are local under ignored `docs/agent_context/inference-lab/`. They are intentionally not claimed as included in a fresh clone. This tracked summary retains their exact run/review/manifest digests plus model/runtime/source identities for audit on the original host; new measurements must produce their own evidence. The first comparison artifact is retained as generation history; this v2 artifact adds portable provenance and reproduction notes without changing any measurements.

## Tested recommendation and limits

For these six short text fixtures, use the40-layer configuration as the tested starting point on this RTX3080: the12 measured requests totalled199.39s versus290.67s at24 layers, an observed1.46× latency ratio (31.4% less request time). Device-wide sampled VRAM peak rose from6.895GiB to10.377GiB; about1.623GiB of the12GiB physical total remained at that sampled peak. This is observed headroom for this4096-context workload, not a guarantee for longer prompts or other competing GPU applications.

Useful demonstrated applications: short Chinese summaries preserving source facts, exact JSON extraction, Python explanations and a small repair. Both configurations passed all12 fixed measurements under independent AI rubric review/strict JSON checks. Complex repository agents, long documents, image/video generation and TTS were not tested. Two repeats, synthetic short fixtures and sequential runs limit generalization. The second code task changed from149 to150 output tokens; the report exposes that difference rather than presenting identical token work.

## Optional export of raw evidence

A fresh clone includes this summary but not the ignored raw evidence. On the original host, export the two complete run directories (without model binaries) when moving an experiment:

```sh
tar -czf /tmp/qwen38-rtx3080-raw-evidence.tar.gz -C docs/agent_context/inference-lab/runs 20261114T054137Z-0e00a3ea0c 20261114T054955Z-b62d4eb4a9
sha256sum /tmp/qwen38-rtx3080-raw-evidence.tar.gz
```

Retain the archive alongside the tracked comparison; verify run/review/manifest digests from comparison.json after extraction. For fresh measurements, store their own run directories and review receipts rather than copying these quality decisions.
