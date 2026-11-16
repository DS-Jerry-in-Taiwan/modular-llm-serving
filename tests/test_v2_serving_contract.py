import threading, unittest
from inference_lab.v2_contracts import *
from inference_lab.serving import EngineState, ResidentSupervisor, SupervisorError

class FakeEngine:
    def __init__(self): self.started = 0; self.stopped = 0; self.ready_flag = True
    def start(self): self.started += 1
    def ready(self): return self.ready_flag
    def stop(self): self.stopped += 1

class ContractTests(unittest.TestCase):
    def test_request_and_stream_are_versioned(self):
        req = InferenceRequest('req-1','hello','qwen','resident',stream=True)
        self.assertEqual(req.to_dict()['contract_version'], CONTRACT_VERSION)
        event = StreamEvent('req-1','run-1',0,'chunk','hi')
        self.assertEqual(event.to_dict()['kind'],'chunk')
        with self.assertRaises(ValueError): InferenceRequest('req-1','x','qwen','c',max_new_tokens=0)
    def test_identity_and_evidence_require_hashes(self):
        identity = ModelIdentity('qwen','rev','a'*64,10,'Q4_K_M')
        self.assertEqual(identity.size_bytes,10)
        with self.assertRaises(ValueError): ModelIdentity('qwen','rev','bad',10,'Q4')

class SupervisorTests(unittest.TestCase):
    def test_factory_failure_is_failed_and_does_not_retain_ownership(self):
        def factory(): raise RuntimeError('factory failed')
        s=ResidentSupervisor(factory)
        with self.assertRaises(RuntimeError): s.start()
        self.assertEqual(s.state,EngineState.FAILED)
        with self.assertRaises(SupervisorError): s.require_ready()

    def test_exactly_one_ready_engine_and_idempotent_stop(self):
        engines=[]
        def factory():
            e=FakeEngine(); engines.append(e); return e
        s=ResidentSupervisor(factory); self.assertEqual(s.state,EngineState.STOPPED)
        s.start(); s.start(); self.assertEqual(s.state,EngineState.READY); self.assertEqual(len(engines),1)
        self.assertIs(s.require_ready(),engines[0]); s.stop(); s.stop(); self.assertEqual(engines[0].stopped,1); self.assertEqual(s.state,EngineState.STOPPED)
    def test_failed_readiness_cleans_engine_and_blocks_ready(self):
        e=FakeEngine(); e.ready_flag=False; s=ResidentSupervisor(lambda:e)
        with self.assertRaises(SupervisorError): s.start()
        self.assertEqual(e.stopped,1); self.assertEqual(s.state,EngineState.FAILED)
        with self.assertRaises(SupervisorError): s.require_ready()
    def test_stop_failure_is_not_published_as_stopped(self):
        class Bad(FakeEngine):
            def stop(self): super().stop(); raise RuntimeError('stop failed')
        e=Bad(); s=ResidentSupervisor(lambda:e); s.start()
        with self.assertRaises(RuntimeError): s.stop()
        self.assertEqual(s.state,EngineState.FAILED)

    def test_concurrent_stop_waits_for_start_before_stopping(self):
        started = threading.Event()
        release_start = threading.Event()
        stop_returned = threading.Event()

        class SlowStart(FakeEngine):
            def start(self):
                started.set()
                self.assert_start_released = release_start.wait(2)
                if not self.assert_start_released: raise RuntimeError('start timed out')

        e=SlowStart(); s=ResidentSupervisor(lambda:e)
        start_thread = threading.Thread(target=s.start)
        start_thread.start()
        self.assertTrue(started.wait(2))
        stop_thread = threading.Thread(target=lambda: (s.stop(), stop_returned.set()))
        stop_thread.start()
        self.assertFalse(stop_returned.wait(.05))
        release_start.set()
        start_thread.join(2); stop_thread.join(2)
        self.assertFalse(start_thread.is_alive())
        self.assertFalse(stop_thread.is_alive())
        self.assertTrue(stop_returned.is_set())
        self.assertEqual(e.stopped,1)
        self.assertEqual(s.state,EngineState.STOPPED)

if __name__=='__main__': unittest.main()
