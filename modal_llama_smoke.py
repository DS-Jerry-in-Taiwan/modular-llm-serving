"""Fail-closed local contract for the governed Modal llama.cpp smoke test.

Deploy/call flow (verified Modal 1.5.3, see docs/research/modal-deploy-call-flow.md):
- @app.server() is a class with @modal.enter()/@modal.exit(); HTTP is served by the
  native llama-server subprocess, never a method/Web Function decorator.
- The image applies .entrypoint([]) because the repo server target declares
  ENTRYPOINT ["/app/llama-server"], which conflicts with the subprocess lifecycle.
- llama-server binds --host 0.0.0.0 --port 8080 and --port equals @app.server(port=...);
  readiness blocks on /health inside @modal.enter().
- The deployed Server URL comes from modal.Server.get_url() only; never hand-built.
- Volume I/O uses 1.5.3-correct Volume.batch_upload().put_file() and Volume.listdir()
  only; the nonexistent upload_file/read_dir are rejected fail-closed.
- Scale-to-zero (Modal proxy 503) and model loading (/health 503) use bounded
  retry/backoff with a hard deadline; never unbounded retries.
- Auth is the proxy pair only: Modal-Key (wk-...) + Modal-Secret (ws-...). Bearer and
  SDK API credentials (ak-/as-) are rejected.
- Deployment is direct `modal deploy` only; the `modal serve` dev step is omitted.
- Source provenance (Human-confirmed Option A 2026-08-03): /app/llama-server --version is
  recorded verbatim and accepted only as `version: 0 (unknown)` or `version: N (cd03ec764)`;
  the full 40-char commit is never required in --version output. The M1 identity bundle
  (object ID + same-ID rehydration, dockerfile/context/definition hashes, base digests) must
  equal a fresh Human-approved baseline recomputed over a CLEAN llama.cpp context under the
  revised BUILD_IGNORE; the superseded unclean-tree hashes (d26bf384.../d0c3f01b...) never pass,
  and Option B build-input changes are rejected.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

TRACKING_ID = "20260802-001"
MODAL_VERSION = "1.5.3"
LLAMA_COMMIT = "cd03ec7642f192c13689c7f20096dcca38dc5e33"
CUDA_VERSION = "12.8.1"
UBUNTU_VERSION = "24.04"
CUDA_ARCH = "75"
DOCKERFILE_PATH = "src/llama.cpp/.devops/cuda.Dockerfile"
CONTEXT_DIR = "src/llama.cpp"
# Modal-managed Python layer REQUIRED (2026-08-04): Image.from_dockerfile must add a Python
# 3.12 runtime so @app.server()/@modal.enter()/@modal.exit() can satisfy the deploy runtime
# check. The M1-approved native CUDA image had no Modal Python layer and deploy aborted with
# `ConflictError: unable to determine the version of Python installed in the Image`. This new
# from_dockerfile argument changes build_definition_sha256 ONLY (Dockerfile/content/base
# digests unchanged) and requires a fresh Human-approved M1 bundle + M1 rebuild + M5 redeploy.
ADD_PYTHON = "3.12"
# Option A deployment-file packaging (2026-08-05, root cause of M5 attempt #4): the deployed
# @app.server() entrypoint is the repo file modal_llama_smoke.py (repo root, OUTSIDE the
# src/llama.cpp dockerfile context), and @app.server() imports the module at deploy/schedule
# time (before any container runs). image im-In6cnEw5GM4PrXzFOunw1R did NOT package it, so every
# container crash-looped `ModuleNotFoundError: No module named 'modal_llama_smoke'` before GPU
# init -> proxy envoy 503 empty body for the whole bounded window (auth was correct; 503 not 401).
# Fix = Option A: chain .add_local_file(local_path, remote_path, copy=True) as the LAST image
# instruction after .entrypoint([]), so copy=True bakes the module into an image layer at build
# time (a copy=False startup mount is too late for the @app.server() import). remote_path MUST be
# absolute (/app/modal_llama_smoke.py). Copy=False startup-mount packaging, switching context_dir
# to repo root, and hand-written in-repo-outside @app.server() bundles are all forbidden.
PACKAGED_MODULE_LOCAL_PATH = "modal_llama_smoke.py"
PACKAGED_MODULE_REMOTE_PATH = "/app/modal_llama_smoke.py"
PACKAGED_MODULE_COPY = True
# The packaged-file record baked into build_definition_sha256. Deterministic and STATIC at
# contract time: size and sha256 are derived ONCE from the repo module file at import, so the
# build-definition hash is reproducible and reflects the exact deployment module that will be
# baked into the image. Dockerfile SHA-256, clean src/llama.cpp context-manifest SHA-256, and
# both base digests are UNCHANGED by this packaging (the module is not in the dockerfile
# context); only build_definition_sha256 (and the Modal Image object_id from the added layer)
# change, requiring a fresh Human-approved M1 rebuild + M5 redeploy.
_MANUAL_PACKAGED_MODULE_BYTES = Path(__file__).resolve().read_bytes()
PACKAGED_MODULE_SIZE = len(_MANUAL_PACKAGED_MODULE_BYTES)
PACKAGED_MODULE_SHA256 = hashlib.sha256(_MANUAL_PACKAGED_MODULE_BYTES).hexdigest()
PACKAGED_FILE_RECORD = {
    "local_path": PACKAGED_MODULE_LOCAL_PATH,
    "remote_path": PACKAGED_MODULE_REMOTE_PATH,
    "size": PACKAGED_MODULE_SIZE,
    "sha256": PACKAGED_MODULE_SHA256,
    "copy": PACKAGED_MODULE_COPY,
}
# Revised Modal Build Context Ignore (root cause of the M1 cloud-build failure, 2026-08-03):
# Image.from_dockerfile packs context via its explicit `ignore=` parameter and does NOT read
# llama.cpp's `.dockerignore`. The old `(".git", ".git/**")` copied the pre-existing local
# build tree (src/llama.cpp/build/ with a local-path-bound CMakeCache.txt, compile_commands.json)
# into the container, so COPY . . carried those products in and cmake failed
# ("current CMakeCache directory /app/build vs source /home/ubuntu/..."). This tuple must keep
# the packed context a CLEAN llama.cpp source tree: no build/, no compile_commands.json, no
# .o/.a/.so/.dylib/.dll, no .cache, no model files, no .git (Option A: in-container version
# stays `unknown`; source provenance is proven by the identity bundle, never the version string).
BUILD_IGNORE = (
    "build", "build/*", "build*/",       # pre-existing local build output tree (root cause)
    ".git", ".git/**",                    # Option A: excluded so in-container version stays `unknown`
    ".cache", ".cache/**",                # tool caches
    "*.o", "*.a",                         # object/archive artifacts
    "*.so", "*.dylib", "*.dll",           # built shared/static native products
    "models/**",                          # model artifacts (never in build context)
    "compile_commands.json",              # local-path-bound compile DB
    "CMakeCache.txt", "CMakeFiles/**",    # local-path-bound cmake state
    ".vs/", ".vscode/", ".github/", ".gitignore", ".DS_Store",  # non-source metadata
    "Dockerfile",                         # upstream dockerignore parity; Dockerfile via path arg
)
# M1 pre-build baseline (docs/agent_context/release_20260802-001/m1_prebuild_evidence.json;
# Human-confirmed Option A 2026-08-03). Dockerfile SHA-256 and both CUDA base manifest-list
# digests are FIXED and unchanged by this revision. The context-manifest and build-definition
# hashes are NOT fixed constants: they are recomputed over the CLEAN context under the revised
# BUILD_IGNORE and must be bound to a FRESH Human-approved M1 bundle. The superseded values
# below (d26bf384.../d0c3f01b...) were computed against an UNclean tree (included local build/)
# and must never pass; any identity still carrying them proves the clean-context recompute did
# not happen and is a hard STOP.
M1_BASELINE_DOCKERFILE_SHA256 = "988a9f2c7e7f1cc1ccbb4a5ece75e22bafb7e20c8ac0a09aba061ec5eb617bb7"
M1_BASELINE_BASE_DEVEL_REF = "nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
M1_BASELINE_BASE_RUNTIME_REF = "nvidia/cuda@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867"
# Superseded/invalid context-manifest baselines. Their presence is a hard STOP: it proves the
# corrected relative-path ignore recompute (Fix 2026-08-03) did not happen. The two sets are
# (a) the unclean-tree `d26bf384…` (computed with the old 2-entry ignore that leaked build/),
# and (b) `128f2633…` (computed with the revised 22-entry ignore but BEFORE the absolute-path
# .git/** glob-matching fix — it leaked .git internals and was not reproducible on the clean
# tree). Any identity still carrying any of these is rejected.
SUPERSEDED_CONTEXT_MANIFEST_SHA256 = frozenset({
    "d26bf384f9e150d68a270afda2c6b4d89c2f2f6024d6c6df4e23544fdb8d7cdc",
    "128f2633d0eeaafddc4a3e5a5f432ad6d3486eb853331a5cd31579599409aee0",
})
# Superseded/invalid build-definition baselines (same reasoning; these embed the superseded
# context-manifest hashes and were produced by the buggy walker). `a7d18a04…` was the approved
# CLEAN-context definition computed BEFORE the 2026-08-04 `add_python="3.12"` build-contract change
# (it bakes no add_python field). A definition still equal to it proves the add_python recompute did
# not happen and is a hard STOP (the deploy would fail again without the Modal Python layer).
SUPERSEDED_BUILD_DEFINITION_SHA256 = frozenset({
    "d0c3f01be484d520c54b2f2c457231bb7d3521153795de36af5994ece04d6087",
    "bec115c3b4089e5d9efb36c519c9e167b6e90fdc1458b5c4d3cacfc261c59606",
    "a7d18a046491809bbb2639d1e3445653ff51d9fd17ef3adf3b759b9c1311870a",
})
# Post-add_python PRE-packaging build definitions are also superseded (2026-08-05, Option A
# packaging). compute_build_definition_sha256 now ALWAYS bakes the packaged_file record, so any
# definition computed WITHOUT it (a build that would crash-loop `ModuleNotFoundError` because the
# @app.server() module is absent from the image) can never equal the recomputed expected value and
# is structurally rejected. The strongest fail-closed guarantee is structural (the recompute always
# includes packaged_file), not a numeric constant list (no fabricated historical hash values are
# asserted). A definition that still omits packaged_file is a hard STOP because M5 would redeploy on
# an image lacking /app/modal_llama_smoke.py.
# Option A version observation: /app/llama-server --version prints only the build number
# and short commit (common/arg.cpp:1048); build-info bakes only `git rev-parse --short`
# (cmake/build-info.cmake:21, scripts/build-info.sh:15). The approved BUILD_IGNORE
# excludes .git, so the in-container observation is `version: 0 (unknown)`; a best-case
# local build prints `version: N (cd03ec764)` (N = build number). The full 40-char commit
# is never required inside --version output.
LLAMA_SHORT_COMMIT = LLAMA_COMMIT[:9]
# Option A version observation (multi-line-accepted 2026-08-04): the real `/app/llama-server --version`
# raw is multi-line (e.g. `load_backend: ...\nversion: 0 (unknown)\nbuilt with GNU 14.2.0 for Linux x86_64`).
# PASS requires at least one line that exactly matches `version: 0 (unknown)` (in-container, .git excluded)
# or `version: N (cd03ec764)` (best-case local build). The full 40-char commit is never required. Non-canonical
# service/backend/build lines elsewhere in the raw do not affect the verdict; FAIL only when NO canonical
# `version:` line matches (missing line, empty `version:`, or a foreign-hash `version:`).
VERSION_OBSERVATION_RE = re.compile(
    rf"version: (?:0 \(unknown\)|[0-9]+ \({re.escape(LLAMA_SHORT_COMMIT)}\))"
)
GPU = "T4"
GPU_COUNT = 1
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
BUFFER_CONTAINERS = 0
TARGET_CONCURRENCY = 1
SCALEDOWN_WINDOW_SECONDS = 60
STARTUP_TIMEOUT_SECONDS = 600
SERVER_PORT = 8080
REQUEST_TIMEOUT_SECONDS = 120
STREAM_TIMEOUT_SECONDS = 180
CLEANUP_TIMEOUT_SECONDS = 120
T4_ACTIVE_CAP_SECONDS = 900
REQUEST_START_CUTOFF_SECONDS = 780
CLEANUP_TRIGGER_SECONDS = 840
HEALTH_POLL_SECONDS = 2
HEALTH_MAX_ATTEMPTS = STARTUP_TIMEOUT_SECONDS // HEALTH_POLL_SECONDS
MODEL_REPOSITORY = "ggml-org/gemma-4-E2B-it-GGUF"
MODEL_REVISION = "b4243c156154b6dca9324415f8c7ccc098b4aed1"
MODEL_FILE = "gemma-4-E2B-it-Q4_0.gguf"
MODEL_SIZE = 2_841_481_184
MODEL_SHA256 = "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52"
MODEL_LICENSE = "Apache-2.0"
MODEL_PATH = f"/models/{MODEL_FILE}"
MODEL_ALIAS = "gemma-4-e2b-it-q4_0"
MODEL_URL = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
APP_NAME = "modal-llamacpp-smoke-20260802-001"
VOLUME_NAME = "modal-llamacpp-smoke-model-20260802-001"
MUTATION_GATES = frozenset({f"M{number}" for number in range(1, 8)})
EVIDENCE_FIELD_NAMES = (
    "test_id", "UTC timestamp", "phase", "final image digest", "llama.cpp commit",
    "model revision/hash", "GPU type/count", "PASS/FAIL",
    "HTTP status where applicable", "configured timeout", "measured duration",
    "sanitized evidence location", "failure reason", "cleanup-triggered flag",
)
EVIDENCE_TEST_IDS = (
    "AUTH-01", "AUTH-02", "AUTH-03", "AUTH-04", "AUTH-05",
    "HEALTH-01", "MODEL-01", "CHAT-01", "CHAT-02", "STREAM-01",
    "COLD-01", "WARM-01", "COST-01", "COST-02", "CLEAN-01",
    "CLEAN-02", "CLEAN-03", "CLEAN-04", "CLEAN-05",
)
OCI_DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")
COST_TERM_NAMES = (
    "image_build_cpu_usd", "image_build_memory_usd", "image_build_storage_usd",
    "image_build_network_usd", "model_download_compute_usd",
    "model_download_network_usd", "volume_storage_usd", "non_gpu_validation_usd",
    "t4_cpu_usd", "t4_memory_usd", "endpoint_proxy_usd", "cleanup_usd",
    "all_other_applicable_usd", "contingency_usd",
)


class ContractError(RuntimeError):
    """Raised before a prohibited or unverifiable operation can start."""


@dataclass(frozen=True)
class Approval:
    gate: str
    tracking_id: str
    resource_set: str
    approved_by: str
    approved_at_utc: str


@dataclass(frozen=True)
class ImageIdentity:
    modal_image_object_id: str
    rehydrated_object_id: str
    build_definition_sha256: str
    context_manifest_sha256: str
    dockerfile_sha256: str
    base_devel_ref: str
    base_runtime_ref: str
    source_commit: str
    context_clean: bool
    modal_version: str
    build_log_location: str
    build_started_at_utc: str
    build_finished_at_utc: str
    llama_server_version: str  # raw verbatim --version observation; Option A forms only
    cuda_version: str
    ubuntu_version: str

    def validate(self, approved_baseline: "M1ApprovedBaseline | None" = None) -> None:
        if (
            not IMAGE_ID_RE.fullmatch(self.modal_image_object_id)
            or self.rehydrated_object_id != self.modal_image_object_id
        ):
            raise ContractError("a stable, same-ID Modal Image rehydration is required")
        for digest in (
            self.build_definition_sha256,
            self.context_manifest_sha256,
            self.dockerfile_sha256,
        ):
            validate_sha256(digest)
        expected_definition = compute_build_definition_sha256(
            self.dockerfile_sha256,
            self.context_manifest_sha256,
            self.base_devel_ref,
            self.base_runtime_ref,
        )
        if self.build_definition_sha256 != expected_definition:
            raise ContractError("build-definition hash does not bind the approved inputs")
        if (
            self.context_manifest_sha256 in SUPERSEDED_CONTEXT_MANIFEST_SHA256
            or self.build_definition_sha256 in SUPERSEDED_BUILD_DEFINITION_SHA256
        ):
            raise ContractError(
                "context/build-definition hashes still equal a superseded baseline "
                "(unclean-tree or pre-ignore-fix); the corrected relative-path "
                "clean-context recompute is required"
            )
        if approved_baseline is None:
            raise ContractError(
                "an M1-approved baseline (fresh Human-approved clean-context recompute) "
                "is required to bind source provenance"
            )
        approved_baseline.validate()
        if (
            self.dockerfile_sha256 != approved_baseline.dockerfile_sha256
            or self.context_manifest_sha256 != approved_baseline.context_manifest_sha256
            or self.build_definition_sha256 != approved_baseline.build_definition_sha256
        ):
            raise ContractError(
                "identity bundle must equal the M1-approved baseline "
                "(recomputed over the clean context under the revised ignore policy)"
            )
        if (
            self.base_devel_ref != M1_BASELINE_BASE_DEVEL_REF
            or self.base_runtime_ref != M1_BASELINE_BASE_RUNTIME_REF
        ):
            raise ContractError("base digests must equal the approved manifest-list digests")
        if self.source_commit != LLAMA_COMMIT or not self.context_clean:
            raise ContractError("exact clean llama.cpp source context is required")
        if self.modal_version != MODAL_VERSION:
            raise ContractError(f"Modal SDK {MODAL_VERSION} identity evidence is required")
        if not is_sanitized_location(self.build_log_location):
            raise ContractError("sanitized authoritative build-log evidence is required")
        started = parse_utc(self.build_started_at_utc)
        finished = parse_utc(self.build_finished_at_utc)
        if finished < started:
            raise ContractError("image build timestamps are inconsistent")
        validate_version_observation(self.llama_server_version)
        if self.cuda_version != CUDA_VERSION or self.ubuntu_version != UBUNTU_VERSION:
            raise ContractError("CUDA or Ubuntu runtime observation mismatch")

    def evidence_ref(self, approved_baseline: "M1ApprovedBaseline") -> str:
        self.validate(approved_baseline)
        return (
            f"modal-image:{self.modal_image_object_id}"
            f"#definition-sha256:{self.build_definition_sha256}"
            f"#context-sha256:{self.context_manifest_sha256}"
        )


@dataclass(frozen=True)
class M1ApprovedBaseline:
    """A fresh Human-approved M1 build-identity baseline.

    Because the doc-only revision changed the build context (revised BUILD_IGNORE), the
    context-manifest and build-definition hashes are NOT fixed constants; they are recomputed
    at M1 over the clean llama.cpp tree and must be bound to a fresh Human-approved bundle
    before M1. This baseline records that approved recompute (tracking ID bound; dockerfile
    hash and base digests remain the fixed approved values). Any value still equal to the
    superseded clean-tree baseline is rejected by the identity bundle creator.
    """

    tracking_id: str
    dockerfile_sha256: str
    context_manifest_sha256: str
    build_definition_sha256: str
    approved_at_utc: str
    evidence_location: str

    def validate(self) -> None:
        if self.tracking_id != TRACKING_ID:
            raise ContractError("M1 baseline is not bound to this tracking ID")
        if self.dockerfile_sha256 != M1_BASELINE_DOCKERFILE_SHA256:
            raise ContractError("M1 baseline must carry the fixed approved Dockerfile hash")
        validate_sha256(self.context_manifest_sha256)
        validate_sha256(self.build_definition_sha256)
        if (
            self.context_manifest_sha256 in SUPERSEDED_CONTEXT_MANIFEST_SHA256
            or self.build_definition_sha256 in SUPERSEDED_BUILD_DEFINITION_SHA256
        ):
            raise ContractError(
                "M1 baseline may not equal a superseded hash (unclean-tree or pre-ignore-fix); "
                "the corrected relative-path clean-context recompute is required"
            )
        # Option A packaging (2026-08-05): the definition must recompute to bind the packaged
        # deployment file (add_local_file copy=True). A pre-packaging definition (which omits the
        # packaged_file record and would ship an image lacking the @app.server() module, crashing
        # ModuleNotFoundError) cannot match the recomputed value and is a hard STOP, even if a
        # Human-approved baseline claims it. Base refs are the fixed approved constants here.
        expected = compute_build_definition_sha256(
            self.dockerfile_sha256,
            self.context_manifest_sha256,
            M1_BASELINE_BASE_DEVEL_REF,
            M1_BASELINE_BASE_RUNTIME_REF,
        )
        if self.build_definition_sha256 != expected:
            raise ContractError(
                "M1 baseline build-definition does not bind the packaged deployment file "
                "(Option A add_local_file copy=True); the pre-packaging definition is invalid"
            )
        parse_utc(self.approved_at_utc)
        if not is_sanitized_location(self.evidence_location):
            raise ContractError("M1 baseline requires sanitized approval evidence")


@dataclass(frozen=True)
class CostPreflight:
    observed_at_utc: str
    rates_observed_at_utc: str
    max_age_seconds: int
    currency: str
    attributable_baseline_usd: Decimal
    available_credits_usd: Decimal
    approved_cap_usd: Decimal
    t4_rate_usd_per_second: Decimal
    image_build_cpu_usd: Decimal
    image_build_memory_usd: Decimal
    image_build_storage_usd: Decimal
    image_build_network_usd: Decimal
    model_download_compute_usd: Decimal
    model_download_network_usd: Decimal
    volume_storage_usd: Decimal
    non_gpu_validation_usd: Decimal
    t4_cpu_usd: Decimal
    t4_memory_usd: Decimal
    endpoint_proxy_usd: Decimal
    cleanup_usd: Decimal
    all_other_applicable_usd: Decimal
    contingency_usd: Decimal
    actual_spend_to_date_usd: Decimal
    remaining_worst_case_usd: Decimal
    official_rate_evidence: str
    attributable: bool
    t4_entitled: bool
    t4_capacity_available: bool
    max_gpu_seconds: int = T4_ACTIVE_CAP_SECONDS

    def validate(self, now: datetime | None = None) -> Decimal:
        current = now or datetime.now(timezone.utc)
        if self.currency != "USD" or self.max_age_seconds <= 0:
            raise ContractError("fresh USD pricing is required")
        for timestamp in (self.observed_at_utc, self.rates_observed_at_utc):
            age = (current - parse_utc(timestamp)).total_seconds()
            if age < 0 or age > self.max_age_seconds:
                raise ContractError("cost or rate snapshot is stale or future-dated")
        decimal_values = [
            self.attributable_baseline_usd,
            self.available_credits_usd,
            self.approved_cap_usd,
            self.t4_rate_usd_per_second,
            self.actual_spend_to_date_usd,
            self.remaining_worst_case_usd,
            *(getattr(self, name) for name in COST_TERM_NAMES),
        ]
        if any(not isinstance(value, Decimal) for value in decimal_values):
            raise ContractError("every cost input must be an explicit Decimal")
        if any(value < 0 for value in decimal_values):
            raise ContractError("cost inputs must be explicit and non-negative")
        if self.approved_cap_usd > Decimal("1.00") or self.max_gpu_seconds != T4_ACTIVE_CAP_SECONDS:
            raise ContractError("USD 1 and 900-second hard caps are fixed")
        if (
            not isinstance(self.official_rate_evidence, str)
            or not is_sanitized_location(self.official_rate_evidence)
            or not all((self.attributable, self.t4_entitled, self.t4_capacity_available))
        ):
            raise ContractError("official attributable rates and exact T4 capacity are required")
        estimate = self.t4_rate_usd_per_second * self.max_gpu_seconds + sum(
            (getattr(self, name) for name in COST_TERM_NAMES), Decimal("0")
        )
        if estimate > self.approved_cap_usd:
            raise ContractError("worst-case estimate exceeds the approved cap")
        if self.actual_spend_to_date_usd + self.remaining_worst_case_usd > self.approved_cap_usd:
            raise ContractError("actual spend plus remaining worst case exceeds USD 1")
        if self.available_credits_usd < self.remaining_worst_case_usd:
            raise ContractError("credits are insufficient for the remaining worst case")
        return estimate


@dataclass(frozen=True)
class HFSecretProvenance:
    secret_name: str
    tracking_id: str
    m2_approval: Approval
    created_at_utc: str
    evidence_location: str

    def validate(self) -> None:
        parse_utc(self.created_at_utc)
        require_approval(self.m2_approval, "M2", TRACKING_ID)
        if (
            not is_sanitized_location(self.secret_name)
            or self.tracking_id != TRACKING_ID
            or not is_sanitized_location(self.evidence_location)
        ):
            raise ContractError("test-created HF Secret provenance is required")


@dataclass(frozen=True)
class ModelAccessObservation:
    public_status: str
    observed_at_utc: str
    immutable_url: str
    authoritative: bool

    def validate(self) -> None:
        parse_utc(self.observed_at_utc)
        if self.immutable_url != MODEL_URL or not self.authoritative:
            raise ContractError("authoritative fixed-object access evidence is required")
        if self.public_status not in {"public-success", "auth-denied"}:
            raise ContractError("public access result must be success or auth denial")


@dataclass
class T4TimerGuard:
    local_start: float
    authoritative_start: float | None = None
    telemetry_available: bool = True

    def elapsed(self, now: float) -> float:
        elapsed = now - min(
            self.local_start,
            self.authoritative_start if self.authoritative_start is not None else self.local_start,
        )
        if elapsed < 0 or elapsed > T4_ACTIVE_CAP_SECONDS:
            raise ContractError("T4 active-service cap exceeded or timer invalid")
        return elapsed

    def observe(self, now: float, active_t4_count: int) -> float:
        if not self.telemetry_available or active_t4_count != GPU_COUNT:
            raise ContractError("authoritative telemetry must show exactly one active T4")
        return self.elapsed(now)

    def allow_request(self, now: float, active_t4_count: int = GPU_COUNT) -> float:
        elapsed = self.observe(now, active_t4_count)
        if elapsed >= REQUEST_START_CUTOFF_SECONDS:
            raise ContractError("no new request may start at or after 780 seconds")
        return elapsed

    def cleanup_required(self, now: float, active_t4_count: int = GPU_COUNT) -> bool:
        return self.observe(now, active_t4_count) >= CLEANUP_TRIGGER_SECONDS

    def remaining(self, now: float, active_t4_count: int = GPU_COUNT) -> float:
        return T4_ACTIVE_CAP_SECONDS - self.observe(now, active_t4_count)


@dataclass(frozen=True)
class ResourceObservation:
    observed_at_utc: str
    tracking_id: str
    active_t4_count: int
    app_state: str
    token_state: str
    hf_secret_state: str
    volume_state: str
    evidence_location: str

    def validate(self) -> None:
        parse_utc(self.observed_at_utc)
        if self.tracking_id != TRACKING_ID or self.active_t4_count not in {0, 1}:
            raise ContractError("resource inventory is not bound to this tracking ID")
        if self.app_state not in {"running", "stopped", "absent"}:
            raise ContractError("invalid App inventory state")
        if self.token_state not in {"active", "revoked", "absent"}:
            raise ContractError("invalid proxy-token inventory state")
        if self.hf_secret_state not in {"present", "absent", "not-created", "foreign"}:
            raise ContractError("invalid HF Secret inventory state")
        if self.volume_state not in {"present", "absent", "approved-retained"}:
            raise ContractError("invalid Volume inventory state")
        if not is_sanitized_location(self.evidence_location):
            raise ContractError("sanitized resource inventory evidence is required")


@dataclass(frozen=True)
class SpendObservation:
    observed_at_utc: str
    tracking_id: str
    currency: str
    attributable_baseline_usd: Decimal
    attributable_spend_usd: Decimal
    evidence_location: str

    def validate(self, preflight: CostPreflight, now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        age = (current - parse_utc(self.observed_at_utc)).total_seconds()
        if age < 0 or age > preflight.max_age_seconds:
            raise ContractError("spend observation is stale or future-dated")
        if (
            self.tracking_id != TRACKING_ID
            or self.currency != "USD"
            or not isinstance(self.attributable_baseline_usd, Decimal)
            or self.attributable_baseline_usd != preflight.attributable_baseline_usd
            or not isinstance(self.attributable_spend_usd, Decimal)
            or self.attributable_spend_usd < self.attributable_baseline_usd
            or self.attributable_spend_usd - self.attributable_baseline_usd > preflight.approved_cap_usd
            or not is_sanitized_location(self.evidence_location)
        ):
            raise ContractError("valid attributable USD spend evidence is required")


@dataclass(frozen=True)
class TimerObservation:
    observed_at_utc: str
    tracking_id: str
    observed_monotonic: float
    active_t4_count: int
    evidence_location: str

    def validate(self, guard: T4TimerGuard) -> float:
        parse_utc(self.observed_at_utc)
        if self.tracking_id != TRACKING_ID or not is_sanitized_location(self.evidence_location):
            raise ContractError("timer observation is not bound to this tracking ID")
        return guard.observe(self.observed_monotonic, self.active_t4_count)


@dataclass(frozen=True)
class CleanupTerminalArtifact:
    tracking_id: str
    observed_at_utc: str
    app_state: str
    active_t4_count: int
    token_state: str
    hf_secret_state: str
    volume_state: str
    evidence_location: str

    def validate(self) -> None:
        parse_utc(self.observed_at_utc)
        if (
            self.tracking_id != TRACKING_ID
            or self.app_state not in {"stopped", "absent"}
            or self.active_t4_count != 0
            or self.token_state not in {"revoked", "absent"}
            or self.hf_secret_state not in {"absent", "not-created", "foreign"}
            or self.volume_state not in {"absent", "approved-retained"}
            or not is_sanitized_location(self.evidence_location)
        ):
            raise ContractError("cleanup terminal artifact is incomplete or unbound")


@dataclass(frozen=True)
class EvidenceAuthority:
    tracking_id: str
    preflight: CostPreflight
    timer_guard: T4TimerGuard
    timer: TimerObservation
    resources: ResourceObservation
    spend: SpendObservation
    cleanup: CleanupTerminalArtifact

    def validate(self, now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        observations = (
            (self.preflight, CostPreflight),
            (self.timer_guard, T4TimerGuard),
            (self.timer, TimerObservation),
            (self.resources, ResourceObservation),
            (self.spend, SpendObservation),
            (self.cleanup, CleanupTerminalArtifact),
        )
        if self.tracking_id != TRACKING_ID or any(
            not isinstance(value, expected) for value, expected in observations
        ):
            raise ContractError("evidence authority is malformed or unbound")
        self.preflight.validate(current)
        elapsed = self.timer.validate(self.timer_guard)
        self.resources.validate()
        self.spend.validate(self.preflight, current)
        self.cleanup.validate()
        if elapsed > T4_ACTIVE_CAP_SECONDS:
            raise ContractError("final T4 elapsed exceeds the hard cap")
        if self.resources.active_t4_count != self.timer.active_t4_count:
            raise ContractError("timer and resource observations are inconsistent")
        if (
            self.cleanup.observed_at_utc != self.spend.observed_at_utc
            or self.cleanup.evidence_location != self.spend.evidence_location
        ):
            raise ContractError("final spend and cleanup observations are not bound")


@dataclass(frozen=True)
class EvidenceRow:
    test_id: str
    utc_timestamp: str
    phase: str
    final_image_digest: str
    llama_cpp_commit: str
    model_revision_hash: str
    gpu_type_count: str
    pass_fail: str
    http_status_where_applicable: int | None
    configured_timeout: float
    measured_duration: float
    sanitized_evidence_location: str
    failure_reason: str
    cleanup_triggered_flag: bool

    def validate(self) -> None:
        if self.test_id not in EVIDENCE_TEST_IDS:
            raise ContractError("unknown evidence test_id")
        parse_utc(self.utc_timestamp)
        validate_modal_identity_ref(self.final_image_digest)
        if self.llama_cpp_commit != LLAMA_COMMIT:
            raise ContractError("unexpected llama.cpp commit")
        if self.model_revision_hash != f"{MODEL_REVISION}/{MODEL_SHA256}":
            raise ContractError("unexpected model revision/hash")
        if self.gpu_type_count != "T4/1" or self.pass_fail not in {"PASS", "FAIL"}:
            raise ContractError("invalid GPU identity or result")
        status = self.http_status_where_applicable
        if status is not None and not 100 <= status <= 599:
            raise ContractError("HTTP status is outside its valid range")
        if self.configured_timeout <= 0 or not 0 <= self.measured_duration <= self.configured_timeout:
            raise ContractError("evidence timing is outside its configured bound")
        if not self.phase or not is_sanitized_location(self.sanitized_evidence_location):
            raise ContractError("phase and sanitized evidence location are required")
        if self.pass_fail == "FAIL" and not self.failure_reason:
            raise ContractError("failed evidence requires a failure reason")

    def as_ordered_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(zip(EVIDENCE_FIELD_NAMES, asdict(self).values(), strict=True))


class CleanupBackend(Protocol):
    def state(self, timeout: float) -> Mapping[str, Any]: ...
    def stop_app(self, timeout: float) -> None: ...
    def revoke_token(self, timeout: float) -> None: ...
    def delete_hf_secret(self, timeout: float) -> None: ...
    def delete_volume(self, timeout: float) -> None: ...


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ContractError("timestamp must be ISO-8601 UTC") from exc
    if parsed.tzinfo != timezone.utc:
        raise ContractError("timestamp must be UTC")
    return parsed


def validate_sha256(value: str) -> str:
    if not SHA256_RE.fullmatch(value) or value == "0" * 64:
        raise ContractError("non-placeholder SHA-256 evidence is required")
    return value


def validate_version_observation(raw: str) -> str:
    """Accept the observable /app/llama-server --version forms (Option A, multi-line 2026-08-04).

    The raw output is recorded verbatim (returned as-is); it is never flattened, truncated,
    normalized, guessed, or fabricated. The real raw is multi-line (e.g. `load_backend: ...
    \\nversion: 0 (unknown)\\nbuilt with GNU 14.2.0 for Linux x86_64`), so acceptance is
    PER-LINE:

    - PASS when any single line equals `version: 0 (unknown)` — expected in-container because
      the approved BUILD_IGNORE excludes `.git`, so build-info falls back to "unknown";
    - PASS when any single line equals `version: N (cd03ec764)` — best-case local build; N any
      build number and `cd03ec764` the short hash of the pinned commit;
    - FAIL/STOP only when NO canonical `version:` line matches anywhere in the raw (the canonical
      line is missing, its hash is foreign, or it is empty). A stray non-canonical line elsewhere
      in the multi-line raw is NOT a failure by itself.

    The full 40-char commit is never required inside --version output; its absence is not a failure.
    """
    if not isinstance(raw, str):
        raise ContractError("llama-server version observation must be a string")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if VERSION_OBSERVATION_RE.fullmatch(line):
            return raw
    raise ContractError(
        "llama-server version observation has no canonical "
        "'version: 0 (unknown)' or 'version: N (cd03ec764)' line "
        "in its (possibly multi-line) raw output"
    )


def validate_oci_ref(image_ref: str) -> str:
    if not OCI_DIGEST_RE.fullmatch(image_ref) or image_ref.endswith("0" * 64):
        raise ContractError("base image must be a resolved OCI SHA-256 reference")
    return image_ref


def validate_modal_identity_ref(value: str) -> str:
    match = re.fullmatch(
        r"modal-image:([^#]+)#definition-sha256:([0-9a-f]{64})#context-sha256:([0-9a-f]{64})",
        value,
    )
    if not match or not IMAGE_ID_RE.fullmatch(match[1]):
        raise ContractError("validated Modal-native image identity reference is required")
    validate_sha256(match[2])
    validate_sha256(match[3])
    return value


def inspect_modal_153_api(modal_module: Any) -> Mapping[str, str]:
    if getattr(modal_module, "__version__", None) != MODAL_VERSION:
        raise ContractError(f"Modal SDK {MODAL_VERSION} is required")
    required = {
        "Image.from_dockerfile": (modal_module.Image.from_dockerfile, {"path", "context_dir", "build_args", "add_python", "ignore"}),
        "Image.entrypoint": (modal_module.Image.entrypoint, {"entrypoint_commands"}),
        # Option A packaging (2026-08-05): Image.add_local_file(local_path, remote_path, *, copy=False)
        # must exist with those three parameters so the deployment module can be baked into the image
        # at build time (copy=True) for the @app.server() import.
        "Image.add_local_file": (modal_module.Image.add_local_file, {"local_path", "remote_path", "copy"}),
        "Image.build": (modal_module.Image.build, {"app"}),
        "Image.from_id": (modal_module.Image.from_id, {"image_id"}),
        "Secret.from_name": (modal_module.Secret.from_name, {"name", "required_keys"}),
        "Volume.batch_upload": (modal_module.Volume.batch_upload, {"force"}),
        "Volume.listdir": (modal_module.Volume.listdir, {"path", "recursive"}),
    }
    signatures = {}
    for name, (callable_object, parameters) in required.items():
        signature = inspect.signature(callable_object)
        if not parameters.issubset(signature.parameters):
            raise ContractError(f"Modal 1.5.3 API mismatch for {name}")
        signatures[name] = str(signature)
    if not isinstance(getattr(modal_module.Image, "object_id", None), property):
        raise ContractError("Modal Image.object_id property is required")
    if hasattr(modal_module.Volume, "upload_file") or hasattr(modal_module.Volume, "read_dir"):
        raise ContractError("Modal 1.5.3 Volume has no upload_file/read_dir; batch_upload().put_file()/listdir() are required")
    if not hasattr(modal_module.Server, "get_url"):
        raise ContractError("modal.Server.get_url is required for the deployed Server URL")
    return signatures


def native_build_args(base_devel_ref: str, base_runtime_ref: str) -> dict[str, str]:
    return {
        "BASE_CUDA_DEV_CONTAINER": validate_oci_ref(base_devel_ref),
        "BASE_CUDA_RUN_CONTAINER": validate_oci_ref(base_runtime_ref),
        "CUDA_DOCKER_ARCH": CUDA_ARCH,
    }


def _ignore_match(relative: str, pattern: str) -> bool:
    """Return True if a context-relative path is excluded by an ignore pattern.

    Matching is performed on paths made RELATIVE to the build-context root (never absolute or
    cwd-relative), which is the only way glob patterns like `.git/**`, `build*/`, `.cache/**`,
    and `models/**` behave correctly. Semantics follow llama.cpp's `.dockerignore` flattened to
    Modal `ignore=` patterns:
    - `<name>/**`: exclude the directory and everything beneath it;
    - `<name>/`: exclude the directory and everything beneath it;
    - `*.o|*.a|*.so|*.dylib|*.dll`: exclude any matching file at any depth;
    - otherwise: exact relative-path or exact basename match at any depth.
    """
    rel = relative.rstrip("/")
    if pattern.endswith("**"):
        base = pattern[:-2].rstrip("/")
        return rel == base or rel.startswith(base + "/")
    if pattern.endswith("/"):
        base = pattern.rstrip("/")
        return rel == base or rel.startswith(base + "/")
    if pattern in ("*.o", "*.a", "*.so", "*.dylib", "*.dll"):
        return rel.endswith(pattern[1:])
    return rel == pattern or rel.endswith("/" + pattern)


def compute_context_manifest_sha256(
    context_dir: str | Path,
    ignore: tuple[str, ...] = BUILD_IGNORE,
) -> str:
    """Deterministic SHA-256 manifest of context files under a clean-source ignore policy.

    Fixed (2026-08-03): paths are made RELATIVE to the context root before ignore matching so
    patterns like `.git/**`, `build*/`, `.cache/**`, `models/**` apply correctly regardless of
    recursion depth (the prior `rglob("*")`+`Path.match()` approach leaked `.git` and other
    ignored products because absolute/cwd-relative paths do not match these globs). Ignored
    directories are pruned during descent (never descended into) for correctness and speed.
    The result is deterministic and reproducible on a clean llama.cpp source tree.
    """
    root = Path(context_dir)
    if not root.is_dir():
        raise ContractError("build context directory is unavailable")
    records = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirpath_rel = os.path.relpath(dirpath, root)
            kept_dirs = []
            for name in dirnames:
                rel = os.path.join(dirpath_rel, name) if dirpath_rel != "." else name
                if any(_ignore_match(rel, pattern) for pattern in ignore):
                    continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs
            for filename in filenames:
                rel = os.path.join(dirpath_rel, filename) if dirpath_rel != "." else filename
                if any(_ignore_match(rel, pattern) for pattern in ignore):
                    continue
                data = (Path(dirpath) / filename).read_bytes()
                records.append((rel.replace(os.sep, "/"), len(data), hashlib.sha256(data).hexdigest()))
    except OSError as exc:
        raise ContractError("build context cannot be hashed authoritatively") from exc
    records.sort(key=lambda record: record[0])
    encoded = json.dumps(records, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_build_definition_sha256(
    dockerfile_sha256: str,
    context_manifest_sha256: str,
    base_devel_ref: str,
    base_runtime_ref: str,
) -> str:
    # The definition bakes the add_python argument (2026-08-04): Image.from_dockerfile(..., add_python="3.12", ...)
    # adds the Modal-managed Python runtime required by @app.server()/@modal.enter()/@modal.exit(). Omitting it
    # changes build_definition_sha256 and is a hard STOP. Dockerfile path/hash, context manifest, and base
    # digests are unchanged by this argument.
    # It also bakes the Option A packaged deployment-file record (2026-08-05): the added layer for
    # .add_local_file("modal_llama_smoke.py", "/app/modal_llama_smoke.py", copy=True) is a NEW image layer,
    # so build_definition_sha256 must reflect the packaged file (path, size, sha256, remote path, copy=True).
    # A definition that omits this record does not bind the deployment module and is a hard STOP.
    definition = {
        "add_python": ADD_PYTHON,
        "build_args": native_build_args(base_devel_ref, base_runtime_ref),
        "context_dir": CONTEXT_DIR,
        "context_manifest_sha256": validate_sha256(context_manifest_sha256),
        "dockerfile_path": DOCKERFILE_PATH,
        "dockerfile_sha256": validate_sha256(dockerfile_sha256),
        "ignore": BUILD_IGNORE,
        "modal_version": MODAL_VERSION,
        "packaged_file": PACKAGED_FILE_RECORD,
    }
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def define_native_image(
    modal_module: Any,
    approval: Approval,
    base_devel_ref: str,
    base_runtime_ref: str,
) -> Any:
    """Define the lazy M1 image; eager build remains a separately approved caller action.

    `.entrypoint([])` is required: the repo Dockerfile `server` target declares
    ENTRYPOINT ["/app/llama-server"], which would preempt the subprocess lifecycle
    under @app.server().

    Option A packaging (2026-08-05, M5 attempt #4 root cause): `.add_local_file(..., copy=True)`
    MUST be the LAST image instruction, chained after `.entrypoint([])`. copy=True bakes the repo
    deployment module modal_llama_smoke.py into an image layer at BUILD time so @app.server() can
    import it at deploy/schedule time (a copy=False startup mount is too late). remote_path is
    absolute. From_dockerfile remains the runtime source; Image.from_registry is never used for the
    deployment module or the final image.
    """
    require_approval(approval, "M1", TRACKING_ID)
    inspect_modal_153_api(modal_module)
    return (
        modal_module.Image.from_dockerfile(
            DOCKERFILE_PATH,
            context_dir=CONTEXT_DIR,
            build_args=native_build_args(base_devel_ref, base_runtime_ref),
            add_python=ADD_PYTHON,
            ignore=BUILD_IGNORE,
        )
        .entrypoint([])
        # Last image instruction: bake the deployment module at build time.
        .add_local_file(
            PACKAGED_MODULE_LOCAL_PATH,
            PACKAGED_MODULE_REMOTE_PATH,
            copy=PACKAGED_MODULE_COPY,
        )
    )


def build_native_image(
    image: Any,
    app: Any,
    approval: Approval,
    preflight: CostPreflight,
    now: datetime | None = None,
) -> Any:
    """Run the injected eager build only behind the exact M1 approval."""
    require_approval(approval, "M1", TRACKING_ID)
    preflight.validate(now)
    return image.build(app)


def rehydrate_image(
    modal_module: Any,
    identity: ImageIdentity,
    approved_baseline: M1ApprovedBaseline,
) -> Any:
    identity.validate(approved_baseline)
    inspect_modal_153_api(modal_module)
    image = modal_module.Image.from_id(identity.modal_image_object_id)
    if getattr(image, "object_id", None) != identity.modal_image_object_id:
        raise ContractError("Modal Image same-ID rehydration failed")
    return image


def verify_remote_deployment_module(
    image: Any,
    op: Callable[[str, float], bool],
    approval: Approval,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Bounded read-only pre-redeploy gate: confirm the packaged deployment module is in the image.

    M5 MUST redeploy the repo formal entrypoint modal_llama_smoke.py, which is baked into the image at
    /app/modal_llama_smoke.py (copy=True) at build time. Because @app.server() imports the module at
    deploy/schedule time, an image WITHOUT /app/modal_llama_smoke.py crash-loops `ModuleNotFoundError`
    and the proxy returns a persistent envoy 503 — the M5 attempt #4 root cause. This gate performs a
    bounded read-only check that the file is present and HARD-FAILS if absent. It never mutates cloud
    state. `op` is an injectable bounded read-only existence probe (e.g. listing/exec scoped to the file
    path); it must return True only when the packaged module is present.
    """
    require_approval(approval, "M1", TRACKING_ID)
    if not isinstance(image, object) or not callable(op):
        raise ContractError("pre-redeploy gate needs the built image and a bounded file-exists probe")
    if timeout <= 0:
        raise ContractError("pre-redeploy file-exists check must be bounded")
    try:
        present = op(PACKAGED_MODULE_REMOTE_PATH, timeout)
    except ContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed on any probe failure
        raise ContractError(
            f"pre-redeploy file-exists check failed for {PACKAGED_MODULE_REMOTE_PATH}"
        ) from exc
    if present is not True:
        raise ContractError(
            f"packaged deployment module {PACKAGED_MODULE_REMOTE_PATH} is ABSENT from the image; "
            "M5 would crash-loop ModuleNotFoundError (proxy envoy 503). Rebuild with "
            "add_local_file(... copy=True) and retry."
        )
    return PACKAGED_MODULE_REMOTE_PATH


def select_hf_secret(
    observation: ModelAccessObservation,
    approval: Approval | None,
    secret_name: str | None,
    modal_module: Any,
) -> Any | None:
    observation.validate()
    if observation.public_status == "public-success":
        if approval is not None or secret_name is not None:
            raise ContractError("HF Secret is prohibited when immutable public access succeeds")
        return None
    require_approval(approval, "M2", TRACKING_ID)
    if not secret_name or not is_sanitized_location(secret_name):
        raise ContractError("purpose-scoped HF Secret name is required")
    inspect_modal_153_api(modal_module)
    return modal_module.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])


def build_evidence_matrix(
    rows: Iterable[EvidenceRow],
    identity: ImageIdentity | None = None,
    authority: EvidenceAuthority | None = None,
    approved_baseline: M1ApprovedBaseline | None = None,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    materialized = list(rows)
    if len(materialized) != len(EVIDENCE_TEST_IDS):
        raise ContractError("evidence matrix requires exactly 19 input rows")
    test_ids = [row.test_id for row in materialized]
    if len(set(test_ids)) != len(test_ids):
        raise ContractError("evidence matrix rejects duplicate test IDs")
    if set(test_ids) != set(EVIDENCE_TEST_IDS):
        raise ContractError("evidence matrix rejects missing or extra test IDs")
    indexed = {row.test_id: row for row in materialized}
    if any(row.pass_fail == "PASS" for row in indexed.values()):
        if identity is None:
            raise ContractError("PASS requires authoritative Modal image identity")
        identity.validate(approved_baseline)
        if approved_baseline is None:
            raise ContractError("PASS evidence requires an M1-approved baseline")
        if any(row.final_image_digest != identity.evidence_ref(approved_baseline) for row in indexed.values()):
            raise ContractError("evidence rows do not match the authoritative image identity")
        if authority is None:
            raise ContractError("PASS requires structured authoritative observations")
        authority.validate(now_utc)
        required_locations = {
            "COST-01": authority.preflight.official_rate_evidence,
            "COST-02": authority.spend.evidence_location,
            "COLD-01": authority.resources.evidence_location,
            "WARM-01": authority.timer.evidence_location,
            **{test_id: authority.cleanup.evidence_location for test_id in EVIDENCE_TEST_IDS[-5:]},
        }
        for test_id, expected in required_locations.items():
            if (
                indexed[test_id].pass_fail == "PASS"
                and indexed[test_id].sanitized_evidence_location != expected
            ):
                raise ContractError(f"{test_id} is not bound to its authority artifact")
    return [indexed[test_id].as_ordered_dict() for test_id in EVIDENCE_TEST_IDS]


def verify_model(path: str | Path) -> str:
    model = Path(path)
    digest = hashlib.sha256()
    size = 0
    try:
        with model.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ContractError("model is unavailable for integrity verification") from exc
    observed_hash = digest.hexdigest()
    if size != MODEL_SIZE or observed_hash != MODEL_SHA256:
        raise ContractError("model size or SHA-256 mismatch")
    return observed_hash


def upload_verified_model(
    path: str | Path,
    upload: Callable[[str | Path], Any],
    approval: Approval,
    preflight: CostPreflight,
    now: datetime | None = None,
) -> Any:
    require_approval(approval, "M3", TRACKING_ID)
    preflight.validate(now)
    verify_model(path)
    return upload(path)


def upload_verified_model_to_volume(
    volume: Any,
    local_path: str | Path,
    approval: Approval,
    preflight: CostPreflight,
    remote_path: str = MODEL_PATH,
    now: datetime | None = None,
) -> str:
    """M3 upload using the verified 1.5.3 Volume.batch_upload().put_file() API only."""
    require_approval(approval, "M3", TRACKING_ID)
    preflight.validate(now)
    verify_model(local_path)
    with volume.batch_upload() as upload:
        upload.put_file(local_path, remote_path)
    return remote_path


def list_volume_entries(volume: Any, path: str = "/") -> list[Any]:
    """Verified 1.5.3 Volume.listdir(); the nonexistent read_dir is never used."""
    return list(volume.listdir(path))


def create_proxy_token(
    create: Callable[[], Any],
    approval: Approval,
    preflight: CostPreflight,
    now: datetime | None = None,
) -> Any:
    require_approval(approval, "M4", TRACKING_ID)
    preflight.validate(now)
    return create()


def start_verified_server(
    path: str | Path = MODEL_PATH,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> subprocess.Popen[Any]:
    verify_model(path)
    return popen(llama_server_command())


def require_approval(approval: Approval | None, gate: str, resource_set: str) -> None:
    if gate not in MUTATION_GATES:
        raise ContractError("unknown mutation gate")
    if approval is None or (
        approval.gate != gate
        or approval.tracking_id != TRACKING_ID
        or approval.resource_set != resource_set
        or not approval.approved_by
    ):
        raise ContractError(f"{gate} approval is required for this resource set")
    parse_utc(approval.approved_at_utc)


def proxy_headers(key: str | None, secret: str | None) -> dict[str, str]:
    if not key or not secret:
        raise ContractError("both Modal proxy credentials are required")
    if not key.startswith("wk-") or not secret.startswith("ws-"):
        raise ContractError("Modal-Key must be a wk- proxy token and Modal-Secret a ws- proxy token")
    return {"Modal-Key": key, "Modal-Secret": secret}


def validate_proxy_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Fail-closed client guard: exactly Modal-Key/Modal-Secret, no Bearer, no API credentials."""
    lowered = {name.lower(): value for name, value in headers.items()}
    if "authorization" in lowered:
        raise ContractError("Authorization/Bearer is not a custom Server call header")
    if set(lowered) != {"modal-key", "modal-secret"}:
        raise ContractError("exactly Modal-Key and Modal-Secret headers are required")
    return proxy_headers(lowered["modal-key"], lowered["modal-secret"])


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {name: "[REDACTED]" for name in headers}


def llama_server_command() -> tuple[str, ...]:
    return (
        "/app/llama-server", "--model", MODEL_PATH, "--host", "0.0.0.0",
        "--port", str(SERVER_PORT), "--alias", MODEL_ALIAS, "--n-gpu-layers", "all",
        "--ctx-size", "4096", "--parallel", "1", "--timeout", "120",
        "--offline", "--no-webui",
    )


def wait_until_ready(
    process: subprocess.Popen[Any],
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = HEALTH_MAX_ATTEMPTS,
    interval: float = HEALTH_POLL_SECONDS,
) -> None:
    if attempts <= 0 or interval <= 0:
        raise ContractError("readiness polling must be finite and positive")
    for _ in range(attempts):
        return_code = process.poll()
        if return_code is not None:
            raise ContractError(f"llama-server exited before readiness with code {return_code}")
        try:
            with opener("http://127.0.0.1:8080/health", timeout=interval) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            pass
        sleep(interval)
    raise ContractError("llama-server readiness deadline exceeded")


def terminate_process(process: subprocess.Popen[Any], timeout: float = 10) -> None:
    if timeout <= 0:
        raise ContractError("termination timeout must be positive")
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def retrieve_server_url(get_url: Callable[[], str | None]) -> str:
    """The deployed Server URL must come from modal.Server.get_url(); never hand-built."""
    url = get_url()
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ContractError("deployed Server URL must come from modal.Server.get_url()")
    return url


def bounded_retry_503(
    attempt: Callable[[], tuple[int, Any]],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> tuple[int, Any]:
    """Bounded retry for Modal proxy 503 (scale-to-zero) and /health 503 (loading).

    Uses exponential backoff capped at max_delay and a hard deadline; never retries
    unboundedly and never retries non-503 responses.
    """
    if base_delay <= 0 or max_delay < base_delay or (deadline is not None and deadline <= 0):
        raise ContractError("503 retry backoff must be finite and bounded")
    end = deadline if deadline is not None else monotonic() + REQUEST_TIMEOUT_SECONDS
    delay = base_delay
    while True:
        status, payload = attempt()
        if status != 503:
            return status, payload
        if monotonic() + delay > end:
            raise ContractError("scale-to-zero/loading 503 retry deadline exceeded")
        sleep(delay)
        delay = min(delay * 2, max_delay)


def validate_models_response(payload: Any) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ContractError("models response must contain a data list")
    if MODEL_ALIAS not in {item.get("id") for item in payload["data"] if isinstance(item, dict)}:
        raise ContractError("models response does not contain the fixed alias")


def validate_chat_response(payload: Any) -> None:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ContractError("chat response does not match the expected schema") from exc
    if not isinstance(content, str) or not content.strip():
        raise ContractError("chat response content must be non-empty")


def auth_cases(key: str, secret: str, invalid_key: str, invalid_secret: str) -> tuple[tuple[str, dict[str, str]], ...]:
    if not invalid_key or not invalid_secret or (invalid_key, invalid_secret) == (key, secret):
        raise ContractError("an explicit invalid proxy pair distinct from the valid pair is required")
    return (
        ("AUTH-01", {}),
        ("AUTH-02", {"Modal-Key": key}),
        ("AUTH-03", {"Modal-Secret": secret}),
        ("AUTH-04", {"Modal-Key": invalid_key, "Modal-Secret": invalid_secret}),
        ("AUTH-05", proxy_headers(key, secret)),
    )


def proxy_boundary_request(
    headers: Mapping[str, str],
    authenticate: Callable[[Mapping[str, str]], bool],
    backend: Callable[[], Any],
) -> tuple[int, Any | None]:
    if not authenticate(headers):
        return 401, None
    return 200, backend()


def run_guarded_smoke(
    operation: Callable[[float], Any],
    cleanup_action: Callable[[], Any],
    m6_approval: Approval,
    m7_approval: Approval,
    preflight: CostPreflight,
    timer: T4TimerGuard,
    observe_runtime: Callable[[], tuple[SpendObservation, bool, int]],
    now_monotonic: Callable[[], float] = time.monotonic,
    now_utc: Callable[[], datetime] | None = None,
    poll_interval: float = 0.05,
) -> Any:
    """Run M6 while independently polling budget, telemetry, and deadlines."""
    require_approval(m7_approval, "M7", TRACKING_ID)
    utc_clock = now_utc or (lambda: datetime.now(timezone.utc))
    try:
        require_approval(m6_approval, "M6", TRACKING_ID)
        if poll_interval <= 0:
            raise ContractError("runtime polling interval must be positive")
        preflight.validate(utc_clock())

        def check_runtime() -> float:
            spend, telemetry_available, active_t4_count = observe_runtime()
            if not telemetry_available:
                raise ContractError("authoritative T4 telemetry was lost")
            spend.validate(preflight, utc_clock())
            attributable = spend.attributable_spend_usd - spend.attributable_baseline_usd
            if attributable + preflight.remaining_worst_case_usd > preflight.approved_cap_usd:
                raise ContractError("runtime spend threshold exceeded")
            now = now_monotonic()
            elapsed = timer.observe(now, active_t4_count)
            remaining = min(timer.remaining(now, active_t4_count), CLEANUP_TRIGGER_SECONDS - elapsed)
            if remaining <= 0:
                raise ContractError("cleanup trigger reached")
            return remaining

        timeout = check_runtime()
        result: list[Any] = []
        failure: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(operation(timeout))
            except BaseException as exc:
                failure.append(exc)

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(min(poll_interval, check_runtime()))
        check_runtime()
        if failure:
            raise failure[0]
        return result[0] if result else None
    finally:
        cleanup_action()


def endpoint_request(
    test_id: str,
    request: urllib.request.Request,
    approval: Approval,
    preflight: CostPreflight,
    timer: T4TimerGuard,
    now_monotonic: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    active_t4_count: int = GPU_COUNT,
    now_utc: datetime | None = None,
) -> tuple[int, Any]:
    require_approval(approval, "M6", TRACKING_ID)
    preflight.validate(now_utc)
    elapsed = timer.allow_request(now_monotonic, active_t4_count)
    timeout = min(
        timeout,
        timer.remaining(now_monotonic, active_t4_count),
        CLEANUP_TRIGGER_SECONDS - elapsed,
    )
    if test_id not in {"HEALTH-01", "MODEL-01", "CHAT-01", "CHAT-02", "STREAM-01"}:
        raise ContractError("unknown endpoint test")
    if timeout <= 0:
        raise ContractError("endpoint timeout must fit the remaining T4 cap")
    try:
        with opener(request, timeout=timeout) as response:
            status = response.status
            if test_id == "STREAM-01":
                if status != 200:
                    raise ContractError("stream endpoint did not return HTTP 200")
                return status, parse_sse(response)
            payload = json.loads(response.read())
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise ContractError("bounded endpoint request failed") from exc
    if test_id == "HEALTH-01":
        if status != 200:
            raise ContractError("health endpoint did not become ready")
    elif test_id == "MODEL-01":
        if status != 200:
            raise ContractError("models endpoint did not return HTTP 200")
        validate_models_response(payload)
    elif test_id == "CHAT-01":
        if status != 200:
            raise ContractError("chat endpoint did not return HTTP 200")
        validate_chat_response(payload)
    elif not 400 <= status <= 499:
        raise ContractError("invalid-model chat must return controlled 4xx")
    return status, payload


def parse_sse(lines: Iterable[bytes | str]) -> tuple[list[dict[str, Any]], bool]:
    events: list[dict[str, Any]] = []
    done = False
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ContractError("unexpected streaming response line")
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError("invalid JSON SSE event") from exc
        if not isinstance(event, dict):
            raise ContractError("SSE event must be an object")
        events.append(event)
    if not done or not events:
        raise ContractError("stream requires data event(s) and terminal marker")
    return events, done


def run_bounded_result(operation: Callable[[float], Any], timeout: float, name: str) -> Any:
    if timeout <= 0:
        raise ContractError(f"{name} operation deadline exceeded")
    result: list[Any] = []
    failure: list[BaseException] = []

    def invoke() -> None:
        try:
            result.append(operation(timeout))
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise ContractError(f"{name} operation deadline exceeded")
    if failure:
        raise failure[0]
    return result[0] if result else None


def run_bounded_operation(operation: Callable[[float], None], timeout: float, name: str) -> None:
    if timeout <= 0:
        raise ContractError(f"{name} operation deadline exceeded")
    failure: list[BaseException] = []

    def invoke() -> None:
        try:
            operation(timeout)
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise ContractError(f"{name} operation deadline exceeded")
    if failure:
        raise failure[0]


def cleanup(
    backend: CleanupBackend,
    m7_approval: Approval,
    preflight: CostPreflight,
    timer: T4TimerGuard,
    final_spend: SpendObservation,
    hf_secret_provenance: HFSecretProvenance | None = None,
    retention: Mapping[str, str] | None = None,
    attempts: int = 6,
    sleep: Callable[[float], None] = time.sleep,
    operation_timeout: float = CLEANUP_TIMEOUT_SECONDS / 5,
    monotonic: Callable[[], float] = time.monotonic,
    now_utc: datetime | None = None,
) -> Mapping[str, Any]:
    require_approval(m7_approval, "M7", TRACKING_ID)
    current_utc = now_utc or datetime.now(timezone.utc)
    preflight.validate(current_utc)
    final_spend.validate(preflight, current_utc)
    if attempts <= 0 or operation_timeout <= 0:
        raise ContractError("cleanup polling and operation timeout must be finite")
    deadline = monotonic() + CLEANUP_TIMEOUT_SECONDS

    def bounded_state() -> Mapping[str, Any]:
        return run_bounded_result(
            backend.state,
            min(operation_timeout, deadline - monotonic()),
            "state",
        )

    initial = bounded_state()
    for state_name, terminal_states, operation, operation_name in (
        ("app", {"stopped", "absent"}, backend.stop_app, "stop_app"),
        ("token", {"revoked", "absent"}, backend.revoke_token, "revoke_token"),
    ):
        if initial.get(state_name) not in terminal_states:
            run_bounded_operation(operation, min(operation_timeout, deadline - monotonic()), operation_name)
    if initial.get("hf_secret") == "present":
        if hf_secret_provenance is None:
            raise ContractError("refusing to delete an HF Secret without test ownership provenance")
        hf_secret_provenance.validate()
        if initial.get("hf_secret_name") != hf_secret_provenance.secret_name:
            raise ContractError("refusing to delete a foreign HF Secret")
        if (
            initial.get("hf_secret_tracking_id") != hf_secret_provenance.tracking_id
            or initial.get("hf_secret_created_at_utc") != hf_secret_provenance.created_at_utc
            or initial.get("hf_secret_evidence_location") != hf_secret_provenance.evidence_location
        ):
            raise ContractError("refusing to delete an HF Secret without matching creation provenance")
        run_bounded_operation(
            backend.delete_hf_secret,
            min(operation_timeout, deadline - monotonic()),
            "delete_hf_secret",
        )
    volume_retained = False
    if initial.get("volume") != "absent":
        if retention:
            required = {"owner", "reason", "retained_until_utc", "approval_id"}
            if not required.issubset(retention) or not all(retention[name] for name in required):
                raise ContractError("retention requires owner, reason, expiry, and approval ID")
            parse_utc(retention["retained_until_utc"])
            volume_retained = True
        else:
            run_bounded_operation(
                backend.delete_volume,
                min(operation_timeout, deadline - monotonic()),
                "delete_volume",
            )
    for attempt in range(attempts):
        if monotonic() >= deadline:
            raise ContractError("cleanup overall deadline exceeded")
        state = bounded_state()
        terminal = (
            state.get("active_gpu_count") == 0
            and state.get("app") in {"stopped", "absent"}
            and state.get("token") in {"revoked", "absent"}
            and state.get("hf_secret") in {"absent", "not-created", "foreign"}
            and (state.get("volume") == "absent" or volume_retained)
        )
        if terminal:
            elapsed = timer.elapsed(monotonic())
            final_spend.validate(preflight, current_utc)
            if final_spend.observed_at_utc != state.get("cost_observed_at_utc"):
                raise ContractError("final cleanup state is not bound to attributable cost")
            authoritative_elapsed = state.get("t4_elapsed_seconds")
            if (
                not isinstance(authoritative_elapsed, (int, float))
                or not 0 <= authoritative_elapsed <= T4_ACTIVE_CAP_SECONDS
                or elapsed > T4_ACTIVE_CAP_SECONDS
            ):
                raise ContractError("final cleanup state lacks bounded T4 elapsed time")
            return state
        sleep(min(2 ** attempt, 10))
    raise ContractError("cleanup terminal state deadline exceeded")


def is_sanitized_location(location: str) -> bool:
    lowered = location.lower()
    return bool(location) and not location.startswith(("/", "~")) and not any(
        marker in lowered for marker in ("modal-key", "modal-secret", "hf_token", "token=", "..")
    )


def create_modal_app(
    identity: ImageIdentity,
    image: Any,
    approval: Approval,
    preflight: CostPreflight,
    timer: T4TimerGuard,
    modal_module: Any,
    volume_name: str = VOLUME_NAME,
    now_utc: datetime | None = None,
    now_monotonic: float | None = None,
    approved_baseline: M1ApprovedBaseline | None = None,
    deployment_file_present: Callable[[], bool] | None = None,
) -> Any:
    """Construct a lazy M5 graph from an already built and same-ID image handle.

    Deployment is direct `modal deploy` only; the `modal serve` dev step is omitted.
    The Server is a class with @modal.enter()/@modal.exit(); HTTP is served by the
    native llama-server subprocess, and --port equals @app.server(port=...).

    Pre-redeploy file-exists gate (Option A, M5 attempt #4 root cause): the repo deployment
    module must be present in the rebuilt image at /app/modal_llama_smoke.py BEFORE M5 is allowed
    to deploy. M5 PASS requires authenticated /health to return 200 (not 401, not a persistent
    envoy 503) after confirming the packaged module is present. `deployment_file_present` is an
    injectable bounded read-only probe; when omitted the gate fails closed (no implicit probe is
    invented), so M5 cannot proceed without explicit pre-redeploy packaging evidence.
    """
    identity.validate(approved_baseline)
    require_approval(approval, "M5", TRACKING_ID)
    preflight.validate(now_utc)
    # Pre-redeploy file-exists gate: hard STOP if the packaged module is not confirmed present.
    if deployment_file_present is None or deployment_file_present() is not True:
        raise ContractError(
            "M5 requires confirming the packaged deployment module is present in the rebuilt "
            "image (pre-redeploy file-exists gate); an image lacking it crashes ModuleNotFoundError."
        )
    observed_now = now_monotonic if now_monotonic is not None else timer.local_start
    if timer.local_start > observed_now:
        raise ContractError("T4 timer must start no later than M5 invocation")
    if timer.cleanup_required(observed_now, GPU_COUNT):
        raise ContractError("M5 cannot start at or after the cleanup trigger")
    inspect_modal_153_api(modal_module)
    if getattr(image, "object_id", None) != identity.modal_image_object_id:
        raise ContractError("deployment image handle does not match approved identity")
    volume = modal_module.Volume.from_name(volume_name, create_if_missing=False)
    app = modal_module.App(APP_NAME)

    @app.server(
        image=image,
        gpu=GPU,
        volumes={"/models": volume},
        port=SERVER_PORT,
        min_containers=MIN_CONTAINERS,
        max_containers=MAX_CONTAINERS,
        buffer_containers=BUFFER_CONTAINERS,
        target_concurrency=TARGET_CONCURRENCY,
        scaledown_window=SCALEDOWN_WINDOW_SECONDS,
        startup_timeout=STARTUP_TIMEOUT_SECONDS,
        exit_grace_period=15,
        unauthenticated=False,
    )
    class LlamaServer:
        @modal_module.enter()
        def start(self) -> None:
            self.process = start_verified_server()
            wait_until_ready(self.process)

        @modal_module.exit()
        def stop(self) -> None:
            terminate_process(self.process)

    return app
