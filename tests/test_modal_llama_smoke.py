import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("modal_llama_smoke", ROOT / "modal_llama_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
assert SPEC.loader is not None
SPEC.loader.exec_module(smoke)

DIGEST_B = "nvidia/cuda@sha256:" + "b" * 64
DIGEST_C = "nvidia/cuda@sha256:" + "c" * 64
NOW = "2026-08-03T00:00:00Z"


def approval(gate):
    return smoke.Approval(gate, smoke.TRACKING_ID, smoke.TRACKING_ID, "operator", NOW)


def baseline(**changes):
    """A fresh Human-approved M1 baseline over a clean context.

    The context/build-definition hashes are a freshly recomputed pair over the revised
    BUILD_IGNORE — chosen here as deterministic test values that do NOT equal the superseded
    unclean-tree baseline d26bf384.../d0c3f01b....
    """
    values = {
        "tracking_id": smoke.TRACKING_ID,
        "dockerfile_sha256": smoke.M1_BASELINE_DOCKERFILE_SHA256,
        "context_manifest_sha256": "a" * 64,
        "build_definition_sha256": smoke.compute_build_definition_sha256(
            smoke.M1_BASELINE_DOCKERFILE_SHA256, "a" * 64,
            smoke.M1_BASELINE_BASE_DEVEL_REF, smoke.M1_BASELINE_BASE_RUNTIME_REF,
        ),
        "approved_at_utc": NOW,
        "evidence_location": "evidence/m1-approved-baseline.json",
    }
    return smoke.M1ApprovedBaseline(**(values | changes))


def identity(**changes):
    definition_hash = smoke.compute_build_definition_sha256(
        smoke.M1_BASELINE_DOCKERFILE_SHA256,
        "a" * 64,
        smoke.M1_BASELINE_BASE_DEVEL_REF,
        smoke.M1_BASELINE_BASE_RUNTIME_REF,
    )
    values = {
        "modal_image_object_id": "im-verified123",
        "rehydrated_object_id": "im-verified123",
        "build_definition_sha256": definition_hash,
        "context_manifest_sha256": "a" * 64,
        "dockerfile_sha256": smoke.M1_BASELINE_DOCKERFILE_SHA256,
        "base_devel_ref": smoke.M1_BASELINE_BASE_DEVEL_REF,
        "base_runtime_ref": smoke.M1_BASELINE_BASE_RUNTIME_REF,
        "source_commit": smoke.LLAMA_COMMIT,
        "context_clean": True,
        "modal_version": smoke.MODAL_VERSION,
        "build_log_location": "evidence/build-01.json",
        "build_started_at_utc": NOW,
        "build_finished_at_utc": "2026-08-03T00:01:00Z",
        "llama_server_version": f"version: 8868 ({smoke.LLAMA_COMMIT[:9]})",
        "cuda_version": smoke.CUDA_VERSION,
        "ubuntu_version": smoke.UBUNTU_VERSION,
    }
    return smoke.ImageIdentity(**(values | changes))


def costs(total=Decimal("1.00"), **changes):
    zero_terms = {name: Decimal("0") for name in smoke.COST_TERM_NAMES}
    values = {
        "observed_at_utc": NOW,
        "rates_observed_at_utc": NOW,
        "max_age_seconds": 300,
        "currency": "USD",
        "attributable_baseline_usd": Decimal("0"),
        "available_credits_usd": Decimal("1"),
        "approved_cap_usd": Decimal("1"),
        "t4_rate_usd_per_second": Decimal("0"),
        **(zero_terms | {"contingency_usd": total}),
        "actual_spend_to_date_usd": Decimal("0"),
        "remaining_worst_case_usd": total,
        "official_rate_evidence": "evidence/cost-01.json",
        "attributable": True,
        "t4_entitled": True,
        "t4_capacity_available": True,
    }
    return smoke.CostPreflight(**(values | changes))


def timer(start=0):
    return smoke.T4TimerGuard(start)


def spend(amount=Decimal("0"), baseline=Decimal("0"), location="evidence/final.json", observed_at=NOW):
    return smoke.SpendObservation(observed_at, smoke.TRACKING_ID, "USD", baseline, amount, location)


def authority():
    guard = timer()
    return smoke.EvidenceAuthority(
        smoke.TRACKING_ID,
        costs(),
        guard,
        smoke.TimerObservation(NOW, smoke.TRACKING_ID, 1, 1, "evidence/runtime.json"),
        smoke.ResourceObservation(NOW, smoke.TRACKING_ID, 1, "running", "active", "not-created", "present", "evidence/runtime.json"),
        spend(),
        smoke.CleanupTerminalArtifact(smoke.TRACKING_ID, NOW, "stopped", 0, "revoked", "not-created", "absent", "evidence/final.json"),
    )


class Clock:
    def __init__(self, values=None):
        self.values = iter(values) if values is not None else None
        self.current = 0

    def __call__(self):
        if self.values is not None:
            return next(self.values)
        self.current += 0.01
        return self.current


class ImageDefinition:
    def __init__(self, path, context_dir, build_args, add_python, ignore):
        self.path = path
        self.context_dir = context_dir
        self.build_args = build_args
        self.add_python = add_python
        self.ignore = ignore
        self.entrypoint_args = None
        self.add_local_file_path = None
        self.add_local_file_remote = None
        self.add_local_file_copy = None

    def entrypoint(self, entrypoint_commands):
        self.entrypoint_args = entrypoint_commands
        return self

    def add_local_file(self, local_path, remote_path, *, copy=False):
        # Option A packaging: record the bake; must be chained LAST (after .entrypoint([])).
        self.add_local_file_path = local_path
        self.add_local_file_remote = remote_path
        self.add_local_file_copy = copy
        return self


class FakeImage:
    def __init__(self, object_id="im-verified123"):
        self._object_id = object_id

    @property
    def object_id(self):
        return self._object_id


class _FakeBatchUpload:
    def __init__(self, volume):
        self.volume = volume

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def put_file(self, local, remote):
        self.volume.uploads.append((local, remote))


class FakeVolume:
    def __init__(self, name):
        self.name = name
        self.uploads = []
        self.listed = []
        self.entries = [f"/models/{smoke.MODEL_FILE}"]

    def batch_upload(self, force=False):
        return _FakeBatchUpload(self)

    def listdir(self, path, *, recursive=False):
        self.listed.append((path, recursive))
        return list(self.entries)


class FakeModal:
    __version__ = smoke.MODAL_VERSION

    class Image:
        @property
        def object_id(self): return ""

        @staticmethod
        def from_dockerfile(path, *, context_dir=None, build_args={}, add_python=None, ignore=()):
            return ImageDefinition(path, context_dir, build_args, add_python, ignore)

        def build(self, app): return self

        @classmethod
        def from_id(cls, image_id, client=None): return FakeImage(image_id)

        def entrypoint(self, entrypoint_commands):
            raise NotImplementedError("ImageDefinition records the entrypoint")

        @staticmethod
        def add_local_file(local_path, remote_path, *, copy=False):
            raise NotImplementedError("ImageDefinition records the packaged file")

    class Secret:
        @staticmethod
        def from_name(name, *, required_keys=[]): return (name, required_keys)

    class Volume:
        @staticmethod
        def from_name(name, create_if_missing):
            assert create_if_missing is False
            return FakeVolume(name)

        batch_upload = FakeVolume.batch_upload
        listdir = FakeVolume.listdir

    class Server:
        @staticmethod
        def get_url(self=None):
            return "https://fake-smoke-20260802-001.modal.run"

    class App:
        def __init__(self, name):
            self.name = name
            self.server_kwargs = None
            self.server_class = None

        def server(self, **kwargs):
            self.server_kwargs = kwargs
            def decorator(cls):
                self.server_class = cls
                return cls
            return decorator

    @staticmethod
    def enter():
        def decorate(fn):
            fn._modal_enter = True
            return fn
        return decorate

    @staticmethod
    def exit():
        def decorate(fn):
            fn._modal_exit = True
            return fn
        return decorate


class FakeBackend:
    def __init__(self, states):
        self.states = iter(states)
        self.current = None
        self.calls = []

    def state(self, timeout):
        self.calls.append(("state", timeout))
        try: self.current = next(self.states)
        except StopIteration: pass
        return self.current

    def stop_app(self, timeout): self.calls.append(("stop", timeout))
    def revoke_token(self, timeout): self.calls.append(("revoke", timeout))
    def delete_hf_secret(self, timeout): self.calls.append(("secret", timeout))
    def delete_volume(self, timeout): self.calls.append(("volume", timeout))


class FakeResponse:
    def __init__(self, status, payload=None, lines=()):
        self.status = status
        self.payload = payload
        self.lines = iter(lines)

    def __enter__(self): return self
    def __exit__(self, *_): return None
    def read(self): return json.dumps(self.payload).encode()
    def __iter__(self): return self.lines


class ModalLlamaSmokeTests(unittest.TestCase):
    def test_modal_153_api_and_native_definition_only(self):
        signatures = smoke.inspect_modal_153_api(FakeModal)
        self.assertEqual(set(signatures), {
            "Image.from_dockerfile", "Image.entrypoint", "Image.add_local_file",
            "Image.build", "Image.from_id", "Secret.from_name", "Volume.batch_upload",
            "Volume.listdir",
        })
        image = smoke.define_native_image(FakeModal, approval("M1"), DIGEST_B, DIGEST_C)
        self.assertEqual(image.path, smoke.DOCKERFILE_PATH)
        self.assertEqual(image.context_dir, smoke.CONTEXT_DIR)
        self.assertEqual(image.build_args["CUDA_DOCKER_ARCH"], "75")
        self.assertEqual(image.ignore, smoke.BUILD_IGNORE)
        # add_python="3.12" MUST be present in the from_dockerfile call shape (M5 FAIL_BLOCKER
        # fix: the M1-approved image had no Modal Python layer and deploy aborted with
        # ConflictError "unable to determine the version of Python installed in the Image").
        self.assertEqual(image.add_python, "3.12")
        self.assertEqual(image.add_python, smoke.ADD_PYTHON)
        self.assertEqual(image.entrypoint_args, [])
        # Option A packaging (2026-08-05, M5 attempt #4 root cause): the deployment module is
        # baked into the image with copy=True at an ABSOLUTE remote path, and the add_local_file
        # step is the LAST image instruction (after .entrypoint([])).
        self.assertEqual(image.add_local_file_path, smoke.PACKAGED_MODULE_LOCAL_PATH)
        self.assertEqual(image.add_local_file_path, "modal_llama_smoke.py")
        self.assertEqual(image.add_local_file_remote, smoke.PACKAGED_MODULE_REMOTE_PATH)
        self.assertEqual(image.add_local_file_remote, "/app/modal_llama_smoke.py")
        self.assertTrue(image.add_local_file_remote.startswith("/"))
        self.assertTrue(image.add_local_file_copy)
        self.assertEqual(smoke.PACKAGED_MODULE_COPY, True)
        self.assertEqual(smoke.PACKAGED_MODULE_LOCAL_PATH, "modal_llama_smoke.py")
        self.assertEqual(smoke.PACKAGED_MODULE_REMOTE_PATH, "/app/modal_llama_smoke.py")
        self.assertFalse(hasattr(FakeModal.Image, "from_registry"))
        self.assertFalse(hasattr(FakeModal.Volume, "upload_file"))
        self.assertFalse(hasattr(FakeModal.Volume, "read_dir"))
        self.assertTrue(hasattr(FakeModal.Server, "get_url"))
        build_image = Mock()
        smoke.build_native_image(build_image, "app", approval("M1"), costs(), datetime(2026, 8, 3, tzinfo=timezone.utc))
        build_image.build.assert_called_once_with("app")
        with self.assertRaises(smoke.ContractError):
            smoke.build_native_image(build_image, "app", approval("M2"), costs(), datetime(2026, 8, 3, tzinfo=timezone.utc))
        with self.assertRaises(smoke.ContractError):
            smoke.define_native_image(FakeModal, approval("M2"), DIGEST_B, DIGEST_C)
        bad_image = type("BadImage", (FakeModal.Image,), {"entrypoint": lambda self, x, y: self})
        bad_volume = type("BadVolume", (FakeModal.Volume,), {"upload_file": lambda self, *args: None})
        # add_local_file with a wrong signature (missing remote_path/copy) is a Modal API mismatch.
        bad_add = type("BadImage2", (FakeModal.Image,), {"add_local_file": staticmethod(lambda local_path, extra: None)})
        for attr, replacement in (("Image", bad_image), ("Volume", bad_volume), ("Image", bad_add)):
            bad_modal = type("BadModal", (), dict(FakeModal.__dict__))
            setattr(bad_modal, attr, replacement)
            with self.assertRaises(smoke.ContractError):
                smoke.inspect_modal_153_api(bad_modal)

    def test_build_definition_bakes_packaged_file_and_definition_changes(self):
        # The packaged-file record (path, size, sha256, remote path, copy=True) must be baked into
        # build_definition_sha256, so the definition reflects the new add_local_file layer. This is
        # deterministic and reproduces across calls; any definition omitting the packaged file (a
        # pre-packaging build that would crash-loop ModuleNotFoundError) is structurally rejected
        # because the recomputed expected hash now always includes the packaged_file field.
        record = smoke.PACKAGED_FILE_RECORD
        self.assertEqual(record["local_path"], smoke.PACKAGED_MODULE_LOCAL_PATH)
        self.assertEqual(record["remote_path"], smoke.PACKAGED_MODULE_REMOTE_PATH)
        self.assertEqual(record["copy"], smoke.PACKAGED_MODULE_COPY)
        self.assertEqual(record["size"], smoke.PACKAGED_MODULE_SIZE)
        self.assertEqual(record["sha256"], smoke.PACKAGED_MODULE_SHA256)
        # size matches the real repo file; sha256 is a valid non-placeholder hash of that file.
        self.assertEqual(smoke.PACKAGED_MODULE_SIZE, smoke.Path(__file__).resolve().parent.parent.joinpath(smoke.PACKAGED_MODULE_LOCAL_PATH).stat().st_size)
        smoke.validate_sha256(record["sha256"])
        # Definition is deterministic and distinct from a definition that omits packaging.
        base = dict(
            dockerfile_sha256=smoke.M1_BASELINE_DOCKERFILE_SHA256,
            context_manifest_sha256="a" * 64,
            base_devel_ref=smoke.M1_BASELINE_BASE_DEVEL_REF,
            base_runtime_ref=smoke.M1_BASELINE_BASE_RUNTIME_REF,
        )
        packaged = smoke.compute_build_definition_sha256(**base)
        self.assertEqual(packaged, smoke.compute_build_definition_sha256(**base))
        self.assertNotIn(packaged, smoke.SUPERSEDED_BUILD_DEFINITION_SHA256)
        # A definition WITHOUT the packaged_file record (simulating the pre-packaging build) must
        # NOT equal the packaged definition and is structurally rejected by identity validation.
        old_definition = {
            "add_python": smoke.ADD_PYTHON,
            "build_args": smoke.native_build_args(**{k: base[k] for k in ("base_devel_ref", "base_runtime_ref")}),
            "context_dir": smoke.CONTEXT_DIR,
            "context_manifest_sha256": "a" * 64,
            "dockerfile_path": smoke.DOCKERFILE_PATH,
            "dockerfile_sha256": smoke.M1_BASELINE_DOCKERFILE_SHA256,
            "ignore": smoke.BUILD_IGNORE,
            "modal_version": smoke.MODAL_VERSION,
        }
        old_hash = hashlib.sha256(json.dumps(old_definition, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertNotEqual(old_hash, packaged)
        baseline_obj = smoke.M1ApprovedBaseline(
            smoke.TRACKING_ID, smoke.M1_BASELINE_DOCKERFILE_SHA256, "a" * 64, packaged, NOW,
            "evidence/m1-approved-baseline.json",
        )
        baseline_obj.validate()
        # Identity validation recomputes the expected definition (always baking packaged_file), so
        # an identity/baseline carrying a packaging-omitted definition is structurally rejected — it
        # proves M1 was not recomputed for the add_local_file bundle and the image lacks the module.
        with self.assertRaises(smoke.ContractError):
            smoke.M1ApprovedBaseline(
                smoke.TRACKING_ID, smoke.M1_BASELINE_DOCKERFILE_SHA256, "a" * 64, old_hash, NOW,
                "evidence/m1-approved-baseline.json",
            ).validate()
        with self.assertRaises(smoke.ContractError):
            identity(build_definition_sha256=old_hash).validate(baseline_obj)
        self.assertNotIn(old_hash, smoke.SUPERSEDED_BUILD_DEFINITION_SHA256)

    def test_forbidden_packaging_and_deployment_patterns_rejected(self):
        # Option A packaging prohows (task_plan Section 14 / developer_prompt): the deployment
        # entrypoint is the repo formal modal_llama_smoke.py ONLY. The following are forbidden and
        # STATICALLY absent from the implementation so no cloud build/deploy can hit them:
        #  - a hand-written in-repo-outside @app.server() bundle (e.g. /tmp/.../m5_deploy_20260802.py)
        #  - include_source for the deployment module
        #  - switching the dockerfile context_dir to repo root (must stay src/llama.cpp)
        #  - Image.from_registry for the deployment module or the final image
        src = smoke.Path(ROOT / "modal_llama_smoke.py").read_text(encoding="utf-8")
        # Actual call/use sites (runtime deployment paths) must be absent. The words
        # `Image.from_registry`, `include_source`, `m5_deploy_20260802`, `from_dotenv` may appear
        # ONLY as explanatory comments/docstrings; a parenthesized call or an assignment of them
        # must never exist. Scanned here with call-shaped patterns to be precise.
        for forbidden_call in (
            "from_registry(", "include_source=", "include_source(",
            "m5_deploy_20260802.py", "from_dotenv", "load_dotenv(",
        ):
            self.assertNotIn(forbidden_call, src, f"forbidden deployment pattern {forbidden_call!r} present")
        # The dockerfile context is pinned to the clean llama.cpp tree (not repo root).
        self.assertEqual(smoke.CONTEXT_DIR, "src/llama.cpp")
        # add_local_file is the ONLY image instruction chained after .entrypoint([]): it must
        # appear AFTER the entrypoint call and be the LAST chained step (Modal 1.5.3 _image.py
        # rule: add_local_* after any runtime run_*; it must be last for buildability).
        entry_idx = src.index(".entrypoint([])")
        add_idx = src.index(".add_local_file(")
        self.assertGreater(add_idx, entry_idx, "add_local_file must be chained after .entrypoint([])")
        # There is no run_*/run_commands step after entrypoint (nothing between entrypoint and the
        # packaging step other than whitespace/comment; the packaging step is the builder's tail).
        tail_after_entry = src[entry_idx + len(".entrypoint([])"):]
        tail_after_entry = tail_after_entry.split(".add_local_file(")[1]
        # The image builder returns immediately after .add_local_file(...) — no further .run_* call.
        self.assertIn(")", tail_after_entry)
        self.assertNotIn(".run_", tail_after_entry[:tail_after_entry.index(")")+1])
        # The packaging step bakes copy=True to an ABSOLUTE path.
        self.assertIn("copy=PACKAGED_MODULE_COPY", src)
        self.assertIn(PACKAGED_MODULE_REMOTE := "/app/modal_llama_smoke.py", src)
        self.assertTrue(smoke.PACKAGED_MODULE_REMOTE_PATH.startswith("/"))
        # A non-last add_local_file (followed by a run_* step) is not expressible in the builder,
        # and the build-definition contract has no such ordering path. Verify the module source has
        # NO run_* image step at all (from_dockerfile -> entrypoint -> add_local_file only).
        self.assertNotIn(".run_", src)

    def test_add_local_file_is_last_and_has_absolute_remote_path(self):
        # Structural test against the builder: the returned image chain records only
        # from_dockerfile -> entrypoint -> add_local_file, and add_local_file has an absolute path.
        image = smoke.define_native_image(FakeModal, approval("M1"), DIGEST_B, DIGEST_C)
        self.assertEqual(image.add_local_file_remote, "/app/modal_llama_smoke.py")
        self.assertTrue(image.add_local_file_remote.startswith("/"))
        self.assertEqual(image.add_local_file_remote, smoke.PACKAGED_MODULE_REMOTE_PATH)
        # copy=True required (bakes at build time for the @app.server() import).
        self.assertTrue(smoke.PACKAGED_MODULE_COPY)

    def test_image_identity_and_same_id_rehydration_fail_closed(self):
        with tempfile.TemporaryDirectory() as context:
            root = Path(context)
            (root / "source.txt").write_text("approved", encoding="ascii")
            first = smoke.compute_context_manifest_sha256(root)
            self.assertEqual(first, smoke.compute_context_manifest_sha256(root))
            (root / "source.txt").write_text("changed", encoding="ascii")
            self.assertNotEqual(first, smoke.compute_context_manifest_sha256(root))
        valid = identity()
        bl = baseline()
        self.assertEqual(smoke.validate_modal_identity_ref(valid.evidence_ref(bl)), valid.evidence_ref(bl))
        self.assertEqual(smoke.rehydrate_image(FakeModal, valid, bl).object_id, valid.modal_image_object_id)
        for changes in (
            {"modal_image_object_id": ""}, {"rehydrated_object_id": "im-other"},
            {"build_definition_sha256": "bad"}, {"context_manifest_sha256": "0" * 64},
            {"context_clean": False}, {"build_log_location": ""},
            {"source_commit": "bad"}, {"llama_server_version": "unknown"},
        ):
            with self.assertRaises(smoke.ContractError): identity(**changes).validate(baseline())
        bad_modal = type("BadModal", (), dict(FakeModal.__dict__))
        bad_modal.Image = type("Image", (FakeModal.Image,), {"from_id": classmethod(lambda cls, image_id, client=None: FakeImage("im-other"))})
        with self.assertRaises(smoke.ContractError): smoke.rehydrate_image(bad_modal, valid, bl)
        # The superseded baselines must never pass: an identity still carrying any of the
        # unclean-tree (d26bf384.../d0c3f01b...) or pre-ignore-fix (128f2633.../bec115c3...)
        # hashes is a hard STOP.
        self.assertTrue(any(h.startswith("d26bf384") for h in smoke.SUPERSEDED_CONTEXT_MANIFEST_SHA256))
        self.assertTrue(any(h.startswith("128f2633") for h in smoke.SUPERSEDED_CONTEXT_MANIFEST_SHA256))
        self.assertTrue(any(h.startswith("bec115c3") for h in smoke.SUPERSEDED_BUILD_DEFINITION_SHA256))
        # Pre-add_python clean-context definition a7d18a04... (the M5-approved bundle) is now
        # superseded: it bakes no add_python field, so the deploy failed with the Python-layer
        # ConflictError. An identity/baseline still carrying it proves the add_python recompute
        # did not happen and is a hard STOP.
        self.assertTrue(any(h.startswith("a7d18a04") for h in smoke.SUPERSEDED_BUILD_DEFINITION_SHA256))
        for stale_ctx in smoke.SUPERSEDED_CONTEXT_MANIFEST_SHA256:
            with self.assertRaises(smoke.ContractError):
                identity(context_manifest_sha256=stale_ctx).validate(baseline())
        for stale_def in smoke.SUPERSEDED_BUILD_DEFINITION_SHA256:
            with self.assertRaises(smoke.ContractError):
                identity(build_definition_sha256=stale_def).validate(baseline())
        # A baseline equal to any superseded context OR definition hash is rejected.
        for stale_ctx in smoke.SUPERSEDED_CONTEXT_MANIFEST_SHA256:
            with self.assertRaises(smoke.ContractError):
                baseline(context_manifest_sha256=stale_ctx, build_definition_sha256="e" * 64).validate()
        for stale_def in smoke.SUPERSEDED_BUILD_DEFINITION_SHA256:
            with self.assertRaises(smoke.ContractError):
                baseline(context_manifest_sha256="e" * 64, build_definition_sha256=stale_def).validate()
        # Missing approved baseline is a hard STOP (fresh Human approval is required).
        with self.assertRaises(smoke.ContractError): identity().validate()

    def test_version_observation_accepts_option_a_forms_and_rejects_others(self):
        # Real /app/llama-server --version raw is MULTI-LINE (M5 observation 2026-08-04):
        #   load_backend: loaded CPU backend from /app/libggml-cpu-icelake.so
        #   version: 0 (unknown)
        #   built with GNU 14.2.0 for Linux x86_64
        # The validator records the full raw verbatim and PASSes when ANY single line is canonical.
        real_raw = (
            "load_backend: loaded CPU backend from /app/libggml-cpu-icelake.so\n"
            "version: 0 (unknown)\n"
            "built with GNU 14.2.0 for Linux x86_64"
        )
        accepted = (
            "version: 0 (unknown)",
            "version: 8868 (cd03ec764)",
            "version: 1 (cd03ec764)",
            "version: 0 (cd03ec764)",
            real_raw,
            "load_backend: ...\nversion: 0 (unknown)\nbuilt with GNU 14.2.0",
            "built with GNU 14.2.0 for Linux x86_64\nversion: 8868 (cd03ec764)",
            "\n\nversion: 0 (unknown)\n",
            "version: 1 (cd03ec764)\nversion: 2 (cd03ec764)",
        )
        for raw in accepted:
            self.assertEqual(smoke.validate_version_observation(raw), raw)
            self.assertEqual(smoke.validate_version_observation(raw).strip(), raw.strip())
            identity(llama_server_version=raw).validate(baseline())
            # The full raw is recorded verbatim, never flattened/truncated.
            self.assertIn("version:", smoke.validate_version_observation(raw))
        # Option A: the full 40-char commit is never required inside --version output.
        for raw in accepted:
            self.assertNotIn(smoke.LLAMA_COMMIT, raw)
        rejected = (
            "", "unknown", "version: (unknown)", "version: 5 (unknown)",
            "version: abc (cd03ec764)", "version: 0 (deadbeef)", "version: 0 (UNKNOWN)",
            "version: 0 (unknown) extra", "version: 8868 (cd03ec764",
            "version: 8868 cd03ec764",
            f"version: 8868 ({smoke.LLAMA_COMMIT})", f"llama.cpp {smoke.LLAMA_COMMIT}",
            # Multi-line with NO canonical line: a foreign-hash version line among other lines,
            # an empty version line, or the canonical text missing entirely must all FAIL.
            "load_backend: ...\nversion: 0 (deadbeef)\nbuilt with GNU 14.2.0",
            "load_backend: ...\nversion: \nbuilt with GNU 14.2.0",
            "load_backend: ...\nllama version note only\nbuilt with GNU 14.2.0",
        )
        # A multi-line raw with a canonical line PLUS a stray non-canonical line is ACCEPTED per
        # the per-line rule (a stray non-canonical line is not a failure by itself).
        self.assertEqual(
            smoke.validate_version_observation("version: 0 (unknown)\nversion: 0 (deadbeef)"),
            "version: 0 (unknown)\nversion: 0 (deadbeef)",
        )
        for raw in rejected:
            with self.assertRaises(smoke.ContractError):
                smoke.validate_version_observation(raw)
            with self.assertRaises(smoke.ContractError):
                identity(llama_server_version=raw).validate(baseline())

    def test_m1_baseline_recompute_and_mismatch(self):
        # A fresh approved baseline (clean-context recompute) over the revised BUILD_IGNORE
        # carries Dockerfile = fixed 988a9f2c... and both base digests fixed; only context and
        # definition hashes are recomputed. It must NOT equal the superseded unclean-tree hashes.
        bl = baseline()
        self.assertEqual(bl.dockerfile_sha256, smoke.M1_BASELINE_DOCKERFILE_SHA256)
        self.assertNotIn(bl.context_manifest_sha256, smoke.SUPERSEDED_CONTEXT_MANIFEST_SHA256)
        self.assertNotIn(bl.build_definition_sha256, smoke.SUPERSEDED_BUILD_DEFINITION_SHA256)
        bl.validate()
        # identity must equal the approved baseline to be accepted.
        identity().validate(bl)
        identity(context_manifest_sha256=bl.context_manifest_sha256,
                 build_definition_sha256=bl.build_definition_sha256).validate(bl)
        # The add_python="3.12" build-contract change (2026-08-04) makes the fresh definition
        # hash differ from the pre-add_python M1-approved bundle a7d18a04... and from the earlier
        # unclean/pre-fix baselines. The recompute path never asserts equality with any superseded
        # value; an identity/baseline still carrying one is a hard STOP.
        recomputed = smoke.compute_build_definition_sha256(
            smoke.M1_BASELINE_DOCKERFILE_SHA256, bl.context_manifest_sha256,
            smoke.M1_BASELINE_BASE_DEVEL_REF, smoke.M1_BASELINE_BASE_RUNTIME_REF,
        )
        self.assertEqual(recomputed, bl.build_definition_sha256)
        self.assertEqual(recomputed, smoke.compute_build_definition_sha256(
            smoke.M1_BASELINE_DOCKERFILE_SHA256, bl.context_manifest_sha256,
            smoke.M1_BASELINE_BASE_DEVEL_REF, smoke.M1_BASELINE_BASE_RUNTIME_REF))
        self.assertNotIn(recomputed, smoke.SUPERSEDED_BUILD_DEFINITION_SHA256)
        for stale_def in smoke.SUPERSEDED_BUILD_DEFINITION_SHA256:
            with self.assertRaises(smoke.ContractError):
                baseline(context_manifest_sha256=bl.context_manifest_sha256,
                         build_definition_sha256=stale_def).validate()
            with self.assertRaises(smoke.ContractError):
                identity(context_manifest_sha256=bl.context_manifest_sha256,
                         build_definition_sha256=stale_def).validate(bl)
        for changes in (
            {"dockerfile_sha256": "f" * 64},
            {"context_manifest_sha256": "f" * 64},
            {"base_devel_ref": DIGEST_B},
            {"base_runtime_ref": DIGEST_C},
            {"build_definition_sha256": "e" * 64},
        ):
            with self.assertRaises(smoke.ContractError):
                identity(**changes).validate(bl)
        # A mismatched baseline (dockerfile altered) is rejected.
        bad_bl = baseline(dockerfile_sha256="f" * 64)
        with self.assertRaises(smoke.ContractError): bad_bl.validate()
        with self.assertRaises(smoke.ContractError): identity().validate(bad_bl)
        # Wrong tracking ID / unbound / missing fields are rejected.
        for changes in (
            {"tracking_id": "other"},
            {"context_manifest_sha256": "0" * 64},
            {"build_definition_sha256": "bad"},
            {"evidence_location": ""},
            {"approved_at_utc": "not-a-date"},
        ):
            with self.assertRaises(smoke.ContractError): baseline(**changes).validate()

    def test_revised_ignore_excludes_build_products_from_context_manifest(self):
        # Root cause regression: a local build tree (src/llama.cpp/build/ with CMakeCache.txt,
        # compile_commands.json, CMakeFiles/, .o/.a/.so, .cache) must be EXCLUDED from the
        # packed-context manifest under the revised BUILD_IGNORE, so the Modal context is a
        # clean llama.cpp source tree and cmake cannot fail on a host-path-bound CMakeCache.

        with tempfile.TemporaryDirectory() as context:
            root = Path(context)
            (root / "source.cpp").write_text("int main(){}", encoding="ascii")
            (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.0)", encoding="ascii")
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_HOME_DIRECTORY=/home/ubuntu/projects/llm-inference/src/llama.cpp", encoding="ascii")
            (build_dir / "compile_commands.json").write_text("[]", encoding="ascii")
            (build_dir / "CMakeFiles").mkdir()
            (build_dir / "CMakeFiles" / "x.dir").mkdir()
            (build_dir / "CMakeFiles" / "x.dir" / "main.o").write_bytes(b"\x00")
            (build_dir / "libllama.a").write_bytes(b"\x01")
            (build_dir / "libggml.so").write_bytes(b"\x02")
            (root / ".cache").mkdir()
            (root / ".cache" / "nv").write_text("cache", encoding="ascii")
            (root / "models").mkdir()
            (root / "models" / "gemma.gguf").write_text("model", encoding="ascii")
            clean = smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE)
            dirty = smoke.compute_context_manifest_sha256(root, ())
            self.assertNotEqual(clean, dirty)
            first = smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE)
            second = smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE)
            self.assertEqual(first, second)  # deterministic across runs
            self.assertEqual(clean, first)

    def test_ignore_match_relative_paths_and_directory_pruning(self):
        # Regression for the rglob absolute-path .git/**/build*/ glob-matching bug (Fix 2026-08-03):
        # ignore patterns must match on paths RELATIVE to the context root so .git internals,
        # build*/ dirs, .cache, .github, models, and nested artifacts are all excluded regardless
        # of depth; chdir-relative/absolute Path.match was leaking these into the manifest.
        with tempfile.TemporaryDirectory() as context:
            root = Path(context)
            (root / "Keep.cpp").write_text("k", encoding="ascii")
            # .git (option A: must never appear; leaks .git/** internals in the old buggy walker)
            git = root / ".git"
            git.mkdir()
            (git / "HEAD").write_text("ref: refs/heads/main", encoding="ascii")
            (git / "index").write_text("idx", encoding="ascii")
            (git / "packed-refs").write_text("refs", encoding="ascii")
            (git / "hooks").mkdir()
            (git / "hooks" / "commit-msg.sample").write_text("hi", encoding="ascii")
            # build*/ matched anywhere (nested build dir)
            nested = root / "tests" / "build"
            nested.mkdir(parents=True)
            (nested / "CMakeCache.txt").write_text("cache", encoding="ascii")
            (nested / "foo.o").write_bytes(b"\x01")
            # .cache/ and .cache/** and shallow CMakeFiles/**
            (root / ".cache").mkdir()
            (root / ".cache" / "nv").mkdir()
            (root / ".cache" / "nv" / "cache.bin").write_bytes(b"\x02")
            # compile_commands.json at depth, *.a / *.so / *.dylib
            (root / "src").mkdir()
            (root / "src" / "compile_commands.json").write_text("[]", encoding="ascii")
            (root / "src" / "libz.a").write_bytes(b"\x03")
            (root / "src" / "x.so").write_bytes(b"\x04")
            (root / "src" / "y.dylib").write_bytes(b"\x05")
            # .github/ excluded
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / ".github" / "workflows" / "ci.yml").write_text("on", encoding="ascii")
            # model files anywhere under models/
            (root / "models" / "a" / "b").mkdir(parents=True)
            (root / "models" / "a" / "b" / "g.gguf").write_text("m", encoding="ascii")
            records = smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE)
            # Remove .git/, build/, .cache/, .github/, models/, compile_commands.json, *.a/.o/.so/.dylib
            for remove in (".git", "tests/build", ".cache", "models", ".github"):
                for p in (root / remove).rglob("*"):
                    if p.is_file():
                        p.unlink()
                for d in sorted([root / remove] + list((root / remove).rglob("*")), reverse=True):
                    if d.is_dir():
                        try: d.rmdir()
                        except OSError: pass
            (root / "src" / "compile_commands.json").unlink()
            (root / "src" / "libz.a").unlink()
            (root / "src" / "x.so").unlink()
            (root / "src" / "y.dylib").unlink()
            (root / "tests" / "build").mkdir()  # re-create empty build to ensure no leak
            cleaned = smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE)
            # Adding all the ignored junk must NOT change the manifest: it is fully excluded.
            self.assertEqual(cleaned, records)
            # Removing the junk (so the tree is just Keep.cpp + empty dirs) must be identical too.
            self.assertEqual(cleaned, smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE))
            # Deterministic across runs.
            self.assertEqual(records, smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE))

    def test_option_b_build_changes_rejected(self):
        # Option B (build-info full-hash embedding) modifies the Dockerfile; any such
        # change alters the Dockerfile hash so it can no longer equal the fixed baseline.
        with tempfile.TemporaryDirectory() as context:
            root = Path(context)
            dockerfile = root / "cuda.Dockerfile"
            dockerfile.write_text("approved dockerfile", encoding="ascii")
            approved_hash = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
            dockerfile.write_text("ENV LLAMA_BUILD_COMMIT=cd03ec7642f192c13689c7f20096dcca38dc5e33", encoding="ascii")
            baked_hash = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
            self.assertNotEqual(approved_hash, baked_hash)
        with self.assertRaises(smoke.ContractError):
            identity(dockerfile_sha256=baked_hash).validate(baseline())
        # Option B (.git in context) changes the context manifest vs. the approved ignore.
        with tempfile.TemporaryDirectory() as context:
            root = Path(context)
            (root / "source.txt").write_text("approved", encoding="ascii")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="ascii")
            manifest_ignored = smoke.compute_context_manifest_sha256(root, smoke.BUILD_IGNORE)
            manifest_including_git = smoke.compute_context_manifest_sha256(root, ())
            self.assertNotEqual(manifest_ignored, manifest_including_git)
        with self.assertRaises(smoke.ContractError):
            identity(context_manifest_sha256=manifest_including_git).validate(baseline())
        # Option B (hash-baking context file) changes the context manifest.
        with tempfile.TemporaryDirectory() as context:
            root = Path(context)
            (root / "source.txt").write_text("approved", encoding="ascii")
            clean = smoke.compute_context_manifest_sha256(root)
            # Use a name that is NOT matched by any BUILD_IGNORE pattern (a `build-*` name
            # would be excluded by `build*/`), so adding it genuinely changes the manifest.
            (root / "commit-hash.txt").write_text(smoke.LLAMA_COMMIT, encoding="ascii")
            baked = smoke.compute_context_manifest_sha256(root)
            self.assertNotEqual(clean, baked)
        with self.assertRaises(smoke.ContractError):
            identity(context_manifest_sha256=baked).validate(baseline())

    def test_create_graph_consumes_matching_image_under_m5(self):
        image = FakeImage()
        app = smoke.create_modal_app(identity(), image, approval("M5"), costs(), timer(), FakeModal, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc), now_monotonic=0, approved_baseline=baseline(), deployment_file_present=lambda: True)
        self.assertEqual(app.server_kwargs["image"], image)
        self.assertEqual((app.server_kwargs["gpu"], app.server_kwargs["min_containers"], app.server_kwargs["max_containers"]), ("T4", 0, 1))
        self.assertEqual((app.server_kwargs["buffer_containers"], app.server_kwargs["target_concurrency"]), (0, 1))
        self.assertFalse(app.server_kwargs["unauthenticated"])
        for bad_image, gate in ((FakeImage("im-other"), "M5"), (image, "M4")):
            with self.assertRaises(smoke.ContractError):
                smoke.create_modal_app(identity(), bad_image, approval(gate), costs(), timer(), FakeModal, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc), now_monotonic=0, approved_baseline=baseline(), deployment_file_present=lambda: True)
        for preflight, guard, observed_now in (
            (costs(Decimal("1.01")), timer(), 0),
            (costs(), smoke.T4TimerGuard(1), 0),
            (costs(), timer(), 840),
        ):
            with self.assertRaises(smoke.ContractError):
                smoke.create_modal_app(identity(), image, approval("M5"), preflight, guard, FakeModal, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc), now_monotonic=observed_now, approved_baseline=baseline(), deployment_file_present=lambda: True)

    def test_m5_pre_redeploy_gate_fails_when_module_absent(self):
        image = FakeImage()
        # Hard fail when the packaged module is absent ("present" probe returns False).
        with self.assertRaisesRegex(smoke.ContractError, "ModuleNotFoundError"):
            smoke.create_modal_app(identity(), image, approval("M5"), costs(), timer(), FakeModal, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc), now_monotonic=0, approved_baseline=baseline(), deployment_file_present=lambda: False)
        # The gate is mandatory: omitting it fails closed (no implicit probe is invented).
        with self.assertRaisesRegex(smoke.ContractError, "file-exists gate"):
            smoke.create_modal_app(identity(), image, approval("M5"), costs(), timer(), FakeModal, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc), now_monotonic=0, approved_baseline=baseline())
        # verify_remote_deployment_module hard-fails when the bounded probe reports absent.
        # Negative/probe failure (filename-relative, no real cloud call).
        with self.assertRaises(smoke.ContractError):
            smoke.verify_remote_deployment_module(image, lambda _p, _t: False, approval("M1"), timeout=5)
        # Wrong approval gate and non-callable probe also fail closed.
        with self.assertRaises(smoke.ContractError):
            smoke.verify_remote_deployment_module(image, lambda _p, _t: True, approval("M2"), timeout=5)
        with self.assertRaises(smoke.ContractError):
            smoke.verify_remote_deployment_module(image, None, approval("M1"), timeout=5)
        # A probe that raises fails closed too (no swallowed side effect).
        def boom(_p, _t):
            raise RuntimeError("probe unavailable")
        with self.assertRaises(smoke.ContractError):
            smoke.verify_remote_deployment_module(image, boom, approval("M1"), timeout=5)

    def test_server_is_class_with_enter_exit_and_matching_port(self):
        image = FakeImage()
        app = smoke.create_modal_app(identity(), image, approval("M5"), costs(), timer(), FakeModal, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc), now_monotonic=0, approved_baseline=baseline(), deployment_file_present=lambda: True)
        self.assertTrue(inspect.isclass(app.server_class))
        self.assertTrue(callable(app.server_class.start))
        self.assertTrue(callable(app.server_class.stop))
        self.assertTrue(getattr(app.server_class.start, "_modal_enter", False))
        self.assertTrue(getattr(app.server_class.stop, "_modal_exit", False))
        self.assertEqual(app.server_kwargs["port"], smoke.SERVER_PORT)
        command = smoke.llama_server_command()
        self.assertIn("--host", command)
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
        self.assertIn("--port", command)
        self.assertEqual(command[command.index("--port") + 1], str(app.server_kwargs["port"]))
        self.assertEqual(command[command.index("--port") + 1], str(smoke.SERVER_PORT))
        self.assertEqual(command[command.index("--parallel") + 1], "1")

    def test_server_url_comes_from_modal_get_url(self):
        get_url = Mock(return_value="https://smoke-20260802-001.modal.run")
        self.assertEqual(smoke.retrieve_server_url(get_url), "https://smoke-20260802-001.modal.run")
        get_url.assert_called_once_with()
        for hand_built in ("http://127.0.0.1:8080", "http://localhost:8080", "", None, "ftp://modal.run"):
            with self.assertRaises(smoke.ContractError):
                smoke.retrieve_server_url(Mock(return_value=hand_built))

    def test_volume_uses_verified_153_batch_upload_and_listdir(self):
        volume = FakeModal.Volume.from_name("vol", False)
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        with patch.object(smoke, "MODEL_SIZE", 2), patch.object(smoke, "MODEL_SHA256", hashlib.sha256(b"ok").hexdigest()):
            with tempfile.NamedTemporaryFile() as model:
                model.write(b"ok"); model.flush()
                local_path = model.name
                self.assertEqual(
                    smoke.upload_verified_model_to_volume(volume, model.name, approval("M3"), costs(), now=now),
                    smoke.MODEL_PATH,
                )
        self.assertEqual(volume.uploads, [(local_path, smoke.MODEL_PATH)])
        self.assertEqual(smoke.list_volume_entries(volume, "/"), volume.entries)
        self.assertEqual(volume.listed, [("/", False)])
        for gate in (None, approval("M2")):
            volume.uploads = []
            with patch.object(smoke, "verify_model"):
                with self.assertRaises(smoke.ContractError):
                    smoke.upload_verified_model_to_volume(volume, "model.gguf", gate, costs(), now=now)
            self.assertEqual(volume.uploads, [])
        blocked_volume = Mock()
        with self.assertRaises(smoke.ContractError):
            smoke.upload_verified_model_to_volume(blocked_volume, "missing.gguf", approval("M3"), costs(), now=now)
        blocked_volume.batch_upload.assert_not_called()

    def test_bounded_503_retry_distinguishes_and_stops(self):
        attempt = Mock(side_effect=[(503, None), (200, {"status": "ok"})])
        sleeper = Mock()
        status, payload = smoke.bounded_retry_503(attempt, sleep=sleeper, monotonic=lambda: 0.0, deadline=10.0)
        self.assertEqual((status, payload), (200, {"status": "ok"}))
        sleeper.assert_called_once_with(1.0)
        attempt = Mock(return_value=(401, None))
        sleeper = Mock()
        self.assertEqual(smoke.bounded_retry_503(attempt, sleep=sleeper, monotonic=lambda: 0.0, deadline=10.0), (401, None))
        sleeper.assert_not_called()
        attempt = Mock(return_value=(503, None))
        sleeper = Mock()
        with self.assertRaisesRegex(smoke.ContractError, "deadline exceeded"):
            smoke.bounded_retry_503(attempt, sleep=sleeper, monotonic=lambda: 0.0, deadline=1.5)
        self.assertEqual(sleeper.call_args_list, [((1.0,),)])
        with self.assertRaises(smoke.ContractError):
            smoke.bounded_retry_503(Mock(return_value=(503, None)), monotonic=lambda: 0.0, base_delay=0, deadline=1)
        attempt = Mock(return_value=(503, None))
        with self.assertRaisesRegex(smoke.ContractError, "deadline exceeded"):
            smoke.bounded_retry_503(attempt, sleep=lambda _: None, monotonic=lambda: 0.0, deadline=0.5)
        self.assertEqual(attempt.call_count, 1)

    def test_optional_hf_secret_requires_denial_and_m2(self):
        public = smoke.ModelAccessObservation("public-success", NOW, smoke.MODEL_URL, True)
        denied = smoke.ModelAccessObservation("auth-denied", NOW, smoke.MODEL_URL, True)
        self.assertIsNone(smoke.select_hf_secret(public, None, None, FakeModal))
        self.assertEqual(smoke.select_hf_secret(denied, approval("M2"), "hf-smoke-secret", FakeModal), ("hf-smoke-secret", ["HF_TOKEN"]))
        for observation, gate, name in (
            (denied, None, "hf-smoke-secret"), (denied, approval("M1"), "hf-smoke-secret"),
            (public, approval("M2"), "hf-smoke-secret"),
            (smoke.ModelAccessObservation("auth-denied", NOW, smoke.MODEL_URL, False), approval("M2"), "hf-smoke-secret"),
        ):
            with self.assertRaises(smoke.ContractError): smoke.select_hf_secret(observation, gate, name, FakeModal)

    def test_m1_through_m7_are_independent(self):
        self.assertEqual(smoke.MUTATION_GATES, frozenset({"M1", "M2", "M3", "M4", "M5", "M6", "M7"}))
        for gate in smoke.MUTATION_GATES:
            smoke.require_approval(approval(gate), gate, smoke.TRACKING_ID)
            wrong = "M1" if gate != "M1" else "M2"
            with self.assertRaises(smoke.ContractError): smoke.require_approval(approval(wrong), gate, smoke.TRACKING_ID)

    def test_cost_boundaries_and_complete_formula(self):
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        self.assertEqual(costs(Decimal("0")).validate(now), Decimal("0"))
        self.assertEqual(costs().validate(now), Decimal("1.00"))
        for candidate in (
            costs(Decimal("1.01")), costs(currency="EUR"), costs(attributable=False),
            costs(official_rate_evidence=False), costs(t4_capacity_available=False),
            costs(available_credits_usd=Decimal("0.99")),
            costs(actual_spend_to_date_usd=Decimal("0.1"), remaining_worst_case_usd=Decimal("0.91")),
            costs(observed_at_utc=(now - timedelta(hours=1)).isoformat()),
            costs(image_build_cpu_usd=Decimal("-0.01")),
            costs(image_build_cpu_usd=None),
        ):
            with self.assertRaises(smoke.ContractError): candidate.validate(now)

    def test_t4_timer_exact_boundaries(self):
        guard = smoke.T4TimerGuard(0)
        self.assertEqual(guard.allow_request(779), 779)
        with self.assertRaises(smoke.ContractError): guard.allow_request(780)
        self.assertFalse(guard.cleanup_required(839))
        self.assertTrue(guard.cleanup_required(840))
        self.assertEqual(guard.remaining(900), 0)
        with self.assertRaises(smoke.ContractError): guard.observe(901, 1)
        with self.assertRaises(smoke.ContractError): guard.observe(1, 2)
        guard.telemetry_available = False
        with self.assertRaises(smoke.ContractError): guard.observe(1, 1)
        self.assertTrue(smoke.T4TimerGuard(10, 5).cleanup_required(845))

    def test_model_provenance_and_dual_checksum_boundaries(self):
        self.assertIn(smoke.MODEL_REVISION, smoke.MODEL_URL)
        upload, popen = Mock(), Mock()
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        with self.assertRaises(smoke.ContractError): smoke.upload_verified_model("missing.gguf", upload, approval("M3"), costs(), now)
        with self.assertRaises(smoke.ContractError): smoke.start_verified_server("missing.gguf", popen)
        upload.assert_not_called(); popen.assert_not_called()
        with patch.object(smoke, "MODEL_SIZE", 2), patch.object(smoke, "MODEL_SHA256", hashlib.sha256(b"ok").hexdigest()):
            with tempfile.NamedTemporaryFile() as model:
                model.write(b"ok"); model.flush()
                smoke.upload_verified_model(model.name, upload, approval("M3"), costs(), now)
                smoke.start_verified_server(model.name, popen)
        upload.assert_called_once(); popen.assert_called_once()
        for gate in (None, approval("M2")):
            upload.reset_mock()
            with patch.object(smoke, "verify_model"):
                with self.assertRaises(smoke.ContractError):
                    smoke.upload_verified_model("model.gguf", upload, gate, costs(), now)
            upload.assert_not_called()
        creator = Mock(return_value="proxy-token")
        self.assertEqual(smoke.create_proxy_token(creator, approval("M4"), costs(), now), "proxy-token")
        for gate in (None, approval("M3")):
            creator.reset_mock()
            with self.assertRaises(smoke.ContractError):
                smoke.create_proxy_token(creator, gate, costs(), now)
            creator.assert_not_called()

    def test_auth_proxy_backend_and_redaction(self):
        headers = smoke.proxy_headers("wk-key", "ws-secret")
        cases = dict(smoke.auth_cases("wk-key", "ws-secret", "wk-bad", "ws-bad"))
        backend = Mock(return_value={"ok": True})
        statuses = {test_id: smoke.proxy_boundary_request(case, lambda supplied: supplied == headers, backend)[0] for test_id, case in cases.items()}
        self.assertEqual(statuses, {"AUTH-01": 401, "AUTH-02": 401, "AUTH-03": 401, "AUTH-04": 401, "AUTH-05": 200})
        self.assertEqual(backend.call_count, 1)
        self.assertNotIn("wk-key", str(smoke.redact_headers(headers)))
        self.assertNotIn("ws-secret", str(smoke.redact_headers(headers)))
        for key, secret in (("ak-key", "ws-secret"), ("as-key", "ws-secret"), ("key", "ws-secret"), ("wk-key", "secret"), (None, "ws-secret")):
            with self.assertRaises(smoke.ContractError):
                smoke.proxy_headers(key, secret)
        self.assertEqual(smoke.validate_proxy_headers(headers), headers)
        for bad in (
            {"Modal-Key": "wk-key", "Modal-Secret": "ws-secret", "Authorization": "Bearer x"},
            {"Authorization": "Bearer x"},
            {"Modal-Key": "wk-key"},
            {"Modal-Key": "wk-key", "Modal-Secret": "ws-secret", "X-Extra": "1"},
        ):
            with self.assertRaises(smoke.ContractError):
                smoke.validate_proxy_headers(bad)

    def test_guarded_smoke_cleanup_on_success_failure_and_boundaries(self):
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        cleanup_action = Mock()
        operation = Mock(return_value="ok")
        observation = Mock(return_value=(spend(), True, 1))
        self.assertEqual(
            smoke.run_guarded_smoke(
                operation, cleanup_action, approval("M6"), approval("M7"),
                costs(), timer(), observation, Clock([1, 2]), lambda: now,
            ),
            "ok",
        )
        self.assertEqual(operation.call_args.args, (839,))
        self.assertGreaterEqual(observation.call_count, 2)
        cleanup_action.assert_called_once()

        for m6, m7, preflight, guard, observation, clock in (
            (approval("M5"), approval("M7"), costs(), timer(), Mock(return_value=(spend(), True, 1)), Clock([1])),
            (approval("M6"), approval("M6"), costs(), timer(), Mock(return_value=(spend(), True, 1)), Clock([1])),
            (approval("M6"), approval("M7"), costs(Decimal("1.01")), timer(), Mock(return_value=(spend(), True, 1)), Clock([1])),
            (approval("M6"), approval("M7"), costs(), timer(), Mock(return_value=(spend(), True, 1)), Clock([840])),
            (approval("M6"), approval("M7"), costs(), timer(), Mock(return_value=(spend(), False, 1)), Clock([1])),
            (approval("M6"), approval("M7"), costs(), timer(), Mock(return_value=(spend(), True, 2)), Clock([1])),
        ):
            blocked, cleanup_action = Mock(), Mock()
            with self.assertRaises(smoke.ContractError):
                smoke.run_guarded_smoke(blocked, cleanup_action, m6, m7, preflight, guard, observation, clock, lambda: now)
            blocked.assert_not_called()
            if m7.gate == "M7": cleanup_action.assert_called_once()
            else: cleanup_action.assert_not_called()

    def test_guarded_smoke_mid_operation_breach_triggers_cleanup(self):
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        for breach in (
            (spend(Decimal("0.75")), True, 1),
            (spend(), False, 1),
            (spend(), True, 2),
        ):
            started = __import__("threading").Event()
            release = __import__("threading").Event()
            cleanup_action = Mock(side_effect=release.set)

            def operation(_):
                started.set()
                release.wait(0.1)

            observations = iter(((spend(), True, 1), breach))

            def observe():
                if started.is_set():
                    return next(observations)
                return spend(), True, 1

            with self.assertRaises(smoke.ContractError):
                smoke.run_guarded_smoke(
                    operation, cleanup_action, approval("M6"), approval("M7"),
                    costs(Decimal("0.5")), timer(), observe, Clock(), lambda: now,
                    poll_interval=0.001,
                )
            cleanup_action.assert_called_once()

    def test_api_sse_and_invalid_alias_then_health(self):
        cases = (
            ("MODEL-01", FakeResponse(200, {"data": [{"id": smoke.MODEL_ALIAS}]}), 200),
            ("CHAT-01", FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]}), 200),
            ("STREAM-01", FakeResponse(200, lines=[b'data: {"delta":"ok"}', b"data: [DONE]"]), 200),
        )
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        for test_id, response, expected in cases:
            status, _ = smoke.endpoint_request(test_id, smoke.urllib.request.Request("https://proxy.invalid/test"), approval("M6"), costs(), timer(), 1, Mock(return_value=response), timeout=3, now_utc=now)
            self.assertEqual(status, expected)
        responses = iter((FakeResponse(400, {"error": "bad alias"}), FakeResponse(200, {"status": "ok"})))
        calls = []
        def opener(request, **kwargs):
            calls.append(request.full_url)
            return next(responses)
        self.assertEqual(smoke.endpoint_request("CHAT-02", smoke.urllib.request.Request("https://proxy.invalid/chat"), approval("M6"), costs(), timer(), 1, opener, 3, now_utc=now)[0], 400)
        self.assertEqual(smoke.endpoint_request("HEALTH-01", smoke.urllib.request.Request("https://proxy.invalid/health"), approval("M6"), costs(), timer(), 1, opener, 3, now_utc=now)[0], 200)
        self.assertEqual(calls, ["https://proxy.invalid/chat", "https://proxy.invalid/health"])
        blocked = Mock(return_value=FakeResponse(200, {"status": "ok"}))
        for gate, elapsed in ((approval("M5"), 1), (approval("M6"), 780)):
            with self.assertRaises(smoke.ContractError):
                smoke.endpoint_request("HEALTH-01", smoke.urllib.request.Request("https://proxy.invalid/health"), gate, costs(), timer(), elapsed, blocked, 3, now_utc=now)
        blocked.assert_not_called()
        bounded = Mock(return_value=FakeResponse(200, {"status": "ok"}))
        smoke.endpoint_request("HEALTH-01", smoke.urllib.request.Request("https://proxy.invalid/health"), approval("M6"), costs(), timer(), 779, bounded, 200, now_utc=now)
        self.assertEqual(bounded.call_args.kwargs["timeout"], 61)
        for invalid in (["data: [DONE]"], ["data: {}"], ["bad"]):
            with self.assertRaises(smoke.ContractError): smoke.parse_sse(invalid)

    def test_evidence_schema_cardinality_order_and_authority(self):
        valid_identity = identity()
        auth = authority()
        locations = {
            "COST-01": auth.preflight.official_rate_evidence,
            "COST-02": auth.spend.evidence_location,
            "COLD-01": auth.resources.evidence_location,
            "WARM-01": auth.timer.evidence_location,
            **{test_id: auth.cleanup.evidence_location for test_id in smoke.EVIDENCE_TEST_IDS[-5:]},
        }
        rows = [smoke.EvidenceRow(
            test_id, NOW, "cloud", valid_identity.evidence_ref(baseline()), smoke.LLAMA_COMMIT,
            f"{smoke.MODEL_REVISION}/{smoke.MODEL_SHA256}", "T4/1", "PASS",
            200, 10, 0, locations.get(test_id, f"evidence/{test_id}.json"), "", False,
        ) for test_id in smoke.EVIDENCE_TEST_IDS]
        self.assertEqual(list(rows[0].as_ordered_dict()), list(smoke.EVIDENCE_FIELD_NAMES))
        now = datetime(2026, 8, 3, tzinfo=timezone.utc)
        matrix = smoke.build_evidence_matrix(rows, valid_identity, auth, baseline(), now)
        self.assertEqual([row["test_id"] for row in matrix], list(smoke.EVIDENCE_TEST_IDS))
        for bad_rows in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(smoke.ContractError): smoke.build_evidence_matrix(bad_rows, valid_identity, auth, baseline(), now)
        for kwargs in ({}, {"identity": valid_identity}):
            with self.assertRaises(smoke.ContractError): smoke.build_evidence_matrix(rows, now_utc=now, approved_baseline=baseline(), **kwargs)
        malformed = smoke.EvidenceAuthority(
            smoke.TRACKING_ID, costs(), timer(),
            smoke.TimerObservation(NOW, "foreign", 1, 1, "evidence/runtime.json"),
            auth.resources, auth.spend, auth.cleanup,
        )
        with self.assertRaises(smoke.ContractError): smoke.build_evidence_matrix(rows, valid_identity, malformed, baseline(), now)
        malformed_type = smoke.EvidenceAuthority(
            smoke.TRACKING_ID, True, timer(), auth.timer, auth.resources, auth.spend, auth.cleanup,
        )
        with self.assertRaises(smoke.ContractError): smoke.build_evidence_matrix(rows, valid_identity, malformed_type, baseline(), now)
        unbound_final = smoke.EvidenceAuthority(
            smoke.TRACKING_ID, costs(), timer(), auth.timer, auth.resources, auth.spend,
            smoke.CleanupTerminalArtifact(smoke.TRACKING_ID, NOW, "stopped", 0, "revoked", "not-created", "absent", "evidence/other.json"),
        )
        with self.assertRaises(smoke.ContractError): smoke.build_evidence_matrix(rows, valid_identity, unbound_final, baseline(), now)
        unbound = list(rows)
        unbound[smoke.EVIDENCE_TEST_IDS.index("COST-02")] = smoke.EvidenceRow(
            **(unbound[smoke.EVIDENCE_TEST_IDS.index("COST-02")].__dict__ | {"sanitized_evidence_location": "evidence/other.json"})
        )
        with self.assertRaises(smoke.ContractError): smoke.build_evidence_matrix(unbound, valid_identity, auth, baseline(), now)
        for observed_at, validation_time in (
            (NOW, now + timedelta(seconds=301)),
            ((now + timedelta(seconds=1)).isoformat(), now),
        ):
            stale_or_future = smoke.EvidenceAuthority(
                smoke.TRACKING_ID,
                costs(observed_at_utc=observed_at, rates_observed_at_utc=observed_at),
                timer(), auth.timer, auth.resources,
                spend(observed_at=observed_at),
                smoke.CleanupTerminalArtifact(smoke.TRACKING_ID, observed_at, "stopped", 0, "revoked", "not-created", "absent", "evidence/final.json"),
            )
            with self.assertRaises(smoke.ContractError):
                smoke.build_evidence_matrix(rows, valid_identity, stale_or_future, baseline(), validation_time)

    def test_cleanup_m7_secret_terminal_state_and_races(self):
        dirty = {"active_gpu_count": 1, "app": "running", "token": "active", "hf_secret": "present", "hf_secret_name": "hf-smoke-secret", "hf_secret_tracking_id": smoke.TRACKING_ID, "hf_secret_created_at_utc": NOW, "hf_secret_evidence_location": "evidence/hf-secret.json", "volume": "present"}
        clean = {"active_gpu_count": 0, "app": "stopped", "token": "revoked", "hf_secret": "absent", "volume": "absent", "cost_observed_at_utc": NOW, "t4_elapsed_seconds": 0.07}
        provenance = smoke.HFSecretProvenance("hf-smoke-secret", smoke.TRACKING_ID, approval("M2"), NOW, "evidence/hf-secret.json")
        backend = FakeBackend([dirty, dirty, clean])
        self.assertEqual(smoke.cleanup(backend, approval("M7"), costs(), timer(), spend(), provenance, attempts=2, sleep=lambda _: None, monotonic=Clock(), now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc)), clean)
        self.assertEqual([name for name, _ in backend.calls if name != "state"], ["stop", "revoke", "secret", "volume"])
        absent = {"active_gpu_count": 0, "app": "absent", "token": "absent", "hf_secret": "not-created", "volume": "absent", "cost_observed_at_utc": NOW, "t4_elapsed_seconds": 0.03}
        backend = FakeBackend([absent, absent])
        self.assertEqual(smoke.cleanup(backend, approval("M7"), costs(), timer(), spend(), attempts=1, monotonic=Clock(), now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc)), absent)
        self.assertEqual([name for name, _ in backend.calls if name != "state"], [])
        foreign = dirty | {"hf_secret_name": "human-secret"}
        backend = FakeBackend([foreign])
        with self.assertRaisesRegex(smoke.ContractError, "foreign"):
            smoke.cleanup(backend, approval("M7"), costs(), timer(), spend(), provenance, attempts=1, monotonic=Clock(), now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc))
        self.assertNotIn("secret", [name for name, _ in backend.calls])
        with self.assertRaises(smoke.ContractError): smoke.cleanup(FakeBackend([dirty]), approval("M6"), costs(), timer(), spend(), provenance, attempts=1, monotonic=Clock(), now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc))
        stale = (datetime(2026, 8, 3, tzinfo=timezone.utc) - timedelta(seconds=301)).isoformat()
        for final_spend, preflight, final_elapsed in (
            (spend(Decimal("1.01")), costs(), 0),
            (spend(), costs(), None),
            (spend(), costs(), 901),
            (spend(observed_at=stale), costs(), 0),
            (spend(Decimal("0.2"), baseline=Decimal("0.1")), costs(attributable_baseline_usd=Decimal("0.2")), 0),
        ):
            final = absent | {"cost_observed_at_utc": final_spend.observed_at_utc, "t4_elapsed_seconds": final_elapsed}
            with self.assertRaises(smoke.ContractError):
                smoke.cleanup(FakeBackend([final, final]), approval("M7"), preflight, timer(), final_spend, attempts=1, monotonic=Clock(), now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc))

    def test_cleanup_operations_and_overall_deadlines(self):
        provenance = smoke.HFSecretProvenance("hf-smoke-secret", smoke.TRACKING_ID, approval("M2"), NOW, "evidence/hf-secret.json")
        dirty = {"active_gpu_count": 1, "app": "running", "token": "active", "hf_secret": "present", "hf_secret_name": "hf-smoke-secret", "hf_secret_tracking_id": smoke.TRACKING_ID, "hf_secret_created_at_utc": NOW, "hf_secret_evidence_location": "evidence/hf-secret.json", "volume": "present"}
        state_backend = FakeBackend([dirty])
        state_backend.state = Mock(side_effect=lambda _: __import__("time").sleep(0.05))
        with self.assertRaisesRegex(smoke.ContractError, "state operation deadline"):
            smoke.cleanup(state_backend, approval("M7"), costs(), timer(), spend(), provenance, attempts=1, operation_timeout=0.001, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc))
        state_backend.state.assert_called_once_with(0.001)
        for operation in ("stop_app", "revoke_token", "delete_hf_secret", "delete_volume"):
            backend = FakeBackend([dirty])
            blocker = Mock(side_effect=lambda _: __import__("time").sleep(0.05))
            setattr(backend, operation, blocker)
            with self.assertRaisesRegex(smoke.ContractError, "deadline exceeded"):
                smoke.cleanup(backend, approval("M7"), costs(), timer(), spend(), provenance, attempts=1, operation_timeout=0.001, now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc))
            blocker.assert_called_once_with(0.001)
        no_secret = dirty | {"hf_secret": "not-created"}
        backend = FakeBackend([no_secret, no_secret])
        clock = iter((0, 0, 0, 0, 0, smoke.CLEANUP_TIMEOUT_SECONDS))
        with self.assertRaisesRegex(smoke.ContractError, "overall deadline"):
            smoke.cleanup(backend, approval("M7"), costs(), timer(), spend(), attempts=1, monotonic=lambda: next(clock), now_utc=datetime(2026, 8, 3, tzinfo=timezone.utc))

    def test_process_readiness_and_termination_are_bounded(self):
        process = Mock(); process.poll.return_value = 7
        with self.assertRaisesRegex(smoke.ContractError, "code 7"): smoke.wait_until_ready(process, attempts=1, interval=.01)
        process.poll.return_value = None
        with self.assertRaisesRegex(smoke.ContractError, "deadline"):
            smoke.wait_until_ready(process, opener=Mock(side_effect=OSError), sleep=Mock(), attempts=1, interval=.01)
        process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 1), 0]
        smoke.terminate_process(process, timeout=1)
        process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
