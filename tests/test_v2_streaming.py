import unittest

from inference_lab.serving import (
    CancellationToken,
    StreamProtocolError,
    stream_events,
    validate_stream,
)
from inference_lab.v2_contracts import StreamEvent


class StreamingContractTests(unittest.TestCase):
    def test_chunks_are_ordered_and_have_one_terminal(self):
        events = list(stream_events(["a", "bc"], "req-1", "run-1"))
        self.assertEqual([(e.sequence, e.kind, e.text) for e in events], [
            (0, "chunk", "a"), (1, "chunk", "bc"), (2, "terminal", "")
        ])
        self.assertEqual(validate_stream(events), tuple(events))

    def test_missing_and_duplicate_terminal_are_rejected(self):
        events = list(stream_events(["a"], "req-1", "run-1"))
        with self.assertRaises(StreamProtocolError):
            validate_stream(events[:-1])
        with self.assertRaises(StreamProtocolError):
            validate_stream([*events, events[-1]])

    def test_malformed_or_out_of_order_events_are_rejected(self):
        events = list(stream_events(["a"], "req-1", "run-1"))
        with self.assertRaises(ValueError):
            StreamEvent.from_dict({**events[0].to_dict(), "private": "credential"})
        with self.assertRaises(StreamProtocolError):
            validate_stream([StreamEvent("req-1", "run-1", 4, "chunk", "late"), events[-1]])
        with self.assertRaises(StreamProtocolError):
            validate_stream([events[0], StreamEvent("other", "run-1", 1, "terminal")])

    def test_source_failure_is_controlled_without_exception_leak(self):
        def source():
            yield "before"
            raise RuntimeError("prompt=private credential=secret")

        events = list(stream_events(source(), "req-1", "run-1"))
        self.assertEqual([e.kind for e in events], ["chunk", "error", "terminal"])
        self.assertEqual(events[1].error.code, "STREAM_FAILED")
        wire = events[1].to_dict()
        self.assertNotIn("private", str(wire))
        self.assertNotIn("secret", str(wire))

    def test_disconnect_cancels_and_calls_hook_once(self):
        token = CancellationToken()
        cleanup = []
        stream = stream_events(
            ["a", "b"], "req-1", "run-1", cancellation=token,
            on_cancel=lambda: cleanup.append(True),
        )
        self.assertEqual(next(stream).text, "a")
        token.cancel()
        events = list(stream)
        self.assertEqual([e.kind for e in events], ["error", "terminal"])
        self.assertEqual(events[0].error.code, "CLIENT_DISCONNECTED")
        self.assertEqual(len(cleanup), 1)

    def test_close_before_first_next_calls_hook_once(self):
        cleanup = []
        stream = stream_events(
            ["never consumed"], "req-1", "run-1",
            on_cancel=lambda: cleanup.append(True),
        )

        stream.close()
        stream.close()

        self.assertEqual(cleanup, [True])


if __name__ == "__main__":
    unittest.main()
