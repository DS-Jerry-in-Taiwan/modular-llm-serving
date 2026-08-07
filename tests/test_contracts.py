"""Ticket 1 contract tests for the runtime-neutral RuntimeProvider contract.

Red → green entry for Ticket 1. These tests operate ONLY against the Protocol and the
public contract types — no provider implementation details (that is Ticket 2). A single
minimal conforming double stands in for a real provider, satisfying the
"provider-neutral conformance seams" requirement.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contracts", ROOT / "contracts.py")
contracts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contracts
assert SPEC.loader is not None
SPEC.loader.exec_module(contracts)


class _ConformingProvider(contracts.RuntimeProvider):
    """Minimal provider that conforms to the Protocol with the verified capability set."""

    def __init__(self, descriptor):
        self._descriptor = descriptor

    def health(self) -> contracts.HealthStatus:
        return contracts.HealthStatus(ready=True, detail="mock ready")

    def models(self) -> list[contracts.ModelDescriptor]:
        return [self._descriptor]

    def chat(self, messages) -> contracts.ChatResponse:
        return contracts.ChatResponse(content="hi", model_id=self._descriptor.model_id)

    def stream(self, messages):
        yield contracts.StreamEvent(content="hi")
        yield contracts.StreamEvent(content="", done=True)

    def capabilities(self) -> contracts.ModelDescriptor:
        return self._descriptor


class ModelDescriptorValidationTests(unittest.TestCase):
    """📏/🔴 — ModelDescriptor.validate() over controlled ranges and invalid shapes."""

    def _valid(self, **overrides):
        base = {
            "model_id": "gemma-4-e2b-it-q4_0",
            "base_url": "https://example-proxy.modal.run",
            "format": "gguf",
            "runtime": "llama-cpp",
            "capabilities": frozenset({"chat", "stream"}),
            "context_length": 4096,
        }
        base.update(overrides)
        return contracts.ModelDescriptor(**base)

    def test_finite_positive_context(self):
        """📏 context_length must be a finite positive int (gguf ctx-size=4096 is in range)."""
        d = self._valid()
        d.validate()  # must not raise

        for bad in (0, -1, 10.5, float("inf"), float("nan"), "4096"):
            with self.subTest(bad=bad):
                with self.assertRaises(contracts.ProviderError):
                    self._valid(context_length=bad).validate()

    def test_nonempty_model_id(self):
        """🔴 model_id must be non-empty."""
        for bad in ("", "   "):
            with self.assertRaises(contracts.ProviderError):
                self._valid(model_id=bad).validate()

    def test_capabilities_subset_of_verified_set(self):
        """🎯 capabilities must be a subset of the verified phase-one set {"chat","stream"}."""
        self._valid(capabilities=frozenset()).validate()  # empty subset is legal
        self._valid(capabilities=contracts.VERIFIED_CAPABILITIES).validate()

        # 🔴 future-only / unknown capabilities must be rejected by validate().
        with self.assertRaises(contracts.ProviderError):
            self._valid(capabilities=frozenset({"chat", "stream", "tool_calls"})).validate()
        with self.assertRaises(contracts.ProviderError):
            self._valid(capabilities=frozenset({"embeddings"})).validate()

    def test_verified_capabilities_do_not_include_tool_calls(self):
        """🎯 The declared phase-one capability set is exactly {"chat","stream"}."""
        self.assertEqual(
            contracts.VERIFIED_CAPABILITIES, frozenset({"chat", "stream"})
        )
        self.assertNotIn("tool_calls", contracts.VERIFIED_CAPABILITIES)

    def test_format_runtime_base_url_nonempty(self):
        """🔴 format/runtime/base_url are declared schema fields and must be non-empty."""
        for field in ("format", "runtime", "base_url"):
            with self.subTest(field=field):
                with self.assertRaises(contracts.ProviderError):
                    self._valid(**{field: ""}).validate()


class ModelDescriptorConstructionTests(unittest.TestCase):
    """🔲/🟢 — Descriptor construction and immutability semantics."""

    def test_frozen(self):
        """Descriptors are immutable value objects (hashable, cannot be mutated)."""
        d = contracts.ModelDescriptor(
            model_id="m",
            base_url="https://x",
            format="gguf",
            runtime="llama-cpp",
            capabilities=frozenset({"chat", "stream"}),
            context_length=4096,
        )
        with self.assertRaises(AttributeError):
            d.model_id = "other"

    def test_capabilities_are_a_frozenset(self):
        """capabilities is stored as a frozenset (set-like, hashable)."""
        d = contracts.ModelDescriptor(
            model_id="m",
            base_url="https://x",
            format="gguf",
            runtime="llama-cpp",
            capabilities={"chat", "stream"},
            context_length=4096,
        )
        self.assertIsInstance(d.capabilities, frozenset)
        self.assertEqual(d.capabilities, frozenset({"chat", "stream"}))


class ProviderErrorTests(unittest.TestCase):
    """🟢/🔴 — ProviderError is the single controlled error type and is a RuntimeError."""

    def test_is_runtime_error_subclass(self):
        self.assertTrue(issubclass(contracts.ProviderError, RuntimeError))
        with self.assertRaises(contracts.ProviderError):
            raise contracts.ProviderError("controlled")
        # A ProviderError is still catchable as RuntimeError (base error hierarchy).
        with self.assertRaises(RuntimeError):
            raise contracts.ProviderError("controlled")


class RuntimeProviderProtocolTests(unittest.TestCase):
    """🟢/🎯 — Conformance suite: does a conforming provider satisfy the Protocol?

    These assertions validate ONLY the contract surface (sync methods, return types),
    never the implementation internals — the runtime-neutral conformance seam.
    """

    def setUp(self):
        self.provider = _ConformingProvider(
            contracts.ModelDescriptor(
                model_id="gemma-4-e2b-it-q4_0",
                base_url="https://example-proxy.modal.run",
                format="gguf",
                runtime="llama-cpp",
                capabilities=contracts.VERIFIED_CAPABILITIES,
                context_length=4096,
            )
        )
        self.messages = [contracts.Message(role="user", content="hi")]

    def test_conforms_to_protocol(self):
        self.assertIsInstance(self.provider, contracts.RuntimeProvider)

    def test_health_is_sync_and_returns_healthstatus(self):
        result = self.provider.health()
        self.assertIsInstance(result, contracts.HealthStatus)
        self.assertTrue(result.ready)

    def test_models_returns_list_of_descriptors(self):
        models = self.provider.models()
        self.assertIsInstance(models, list)
        self.assertTrue(all(isinstance(m, contracts.ModelDescriptor) for m in models))
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].model_id, "gemma-4-e2b-it-q4_0")

    def test_chat_returns_nonempty_content(self):
        """🟢 non-stream chat content must be non-empty."""
        response = self.provider.chat(self.messages)
        self.assertIsInstance(response, contracts.ChatResponse)
        self.assertTrue(response.content.strip())

    def test_stream_is_a_sync_iterator_with_terminal_marker(self):
        """🟢/🔲 stream() returns a sync Iterator and terminates with done=True marker."""
        self.assertFalse(hasattr(self.provider.stream, "__await__"), "stream() must be sync (no async)")
        events = list(self.provider.stream(self.messages))
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(events[-1].done, "sync generator must yield a terminal done event")
        self.assertTrue(any(isinstance(e, contracts.StreamEvent) for e in events))

    def test_capabilities_returns_descriptor(self):
        descriptor = self.provider.capabilities()
        self.assertIsInstance(descriptor, contracts.ModelDescriptor)
        self.assertEqual(
            descriptor.capabilities, contracts.VERIFIED_CAPABILITIES
        )


if __name__ == "__main__":
    unittest.main()
