# AGENTS.md — Repository Guidance

This is a **Modal + llama.cpp `llama-server` smoke/deployment-validation** project using a
single fixed GGUF model — NOT a generic production platform. Local checks are done with the
`venv/` Python 3.12.3 + Modal 1.5.3 environment.

## Local verification commands

```bash
venv/bin/python -m py_compile modal_llama_smoke.py tests/test_modal_llama_smoke.py
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -v
```

Both must exit 0. `modal_llama_smoke.py` is the fail-closed contract; `tests/` is its local suite.

## Deployment contract

- Deploy via direct `modal deploy` only — no `modal serve` (dev step omitted).
  (`modal_llama_smoke.py` module docstring)
- `modal_llama_smoke.py` is the deployment entrypoint; it is packaged into the image at build
  time as `/app/modal_llama_smoke.py` (add_local_file `copy=True`, last image instruction,
  absolute remote path).
- `src/llama.cpp` is a parent-registered Git submodule (listed in `.gitmodules`) pinned at
  commit `cd03ec7642f192c13689c7f20096dcca38dc5e33`; it is used as the Docker build context
  (`src/llama.cpp/.devops/cuda.Dockerfile`, `server` target).
- `.opencode` is intentionally excluded from the parent baseline: it is not a tracked
  submodule and is not listed in `.gitmodules`, because it contains sensitive/local OpenCode
  data.
- The server is native llama-server binding `--host 0.0.0.0 --port 8080`; the dockerfile
  `server` target uses `ENTRYPOINT ["/app/llama-server"]` with a `:8080 /health` HEALTHCHECK.
- The deployed Server URL comes from `modal.Server.get_url()` only — never hand-built.
- CUDA 12.8.1 / Ubuntu 24.04, GPU T4.

## Model (fixed GGUF)

- Volume is mounted at `/models`; the model blob must live at the **Volume ROOT**:
  `MODEL_PATH = /models/<MODEL_FILE>`.
- Fixed GGUF: `gemma-4-E2B-it-Q4_0.gguf` with exact size and SHA-256 in
  `modal_llama_smoke.py` (`MODEL_SIZE`, `MODEL_SHA256`) and verified via
  `upload_verified_model`/`verify_model`.

## Operational gotchas

- Keep the Docker build context CLEAN (no `build/`, `.git`, model files, or products in the
  `src/llama.cpp` context).
- Retries are bounded with a hard deadline: `bounded_retry_503`,
  `STARTUP_TIMEOUT_SECONDS=600`, `T4_ACTIVE_CAP_SECONDS=900`,
  `HEALTH_POLL_SECONDS`/`HEALTH_MAX_ATTEMPTS`. Never unbounded retries.
- Auth is the proxy pair only: Modal-Key (`wk-…`) + Modal-Secret (`ws-…`). Bearer and SDK API
  credentials (`ak-`/`as-`) are rejected.
- Never read the root `.env` file; do not expose secrets anywhere.
- Cloud mutation / deployment / release operations belong to the Releaser with HITL approval.

## llama.cpp submodule guidance

When working inside the submodule, read `src/llama.cpp/AGENTS.md` and
`src/llama.cpp/CLAUDE.md` and reference, do not duplicate. Note `src/llama.cpp` does not
accept fully/predominantly AI-generated PRs; AI is permitted only in an assistive capacity.

Do not change the submodule — its files, its pinned commit, or the `.gitmodules` entry —
unless explicitly asked.
