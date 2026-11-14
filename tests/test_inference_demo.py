"""No GPU needed: lifecycle serialization and hostile HTTP request boundaries."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from inference_lab import demo


class FakeSampler:
    start_time = None
    samples = []
    def start(self): self.start_time = time.monotonic()
    def stop(self): pass
    def summary(self): return {"gpu_device_used_bytes": {"status": "ok", "sampled_peak_bytes": 123}}


class FakeServer:
    instances = []
    gate = None
    live = 0
    max_live = 0
    def __init__(self, binary, model, config, directory, deadline):
        self.config, self.directory = config, directory
        self.argv, self.process, self.last_response = [], None, None
        self.load_seconds, self.cleanup_confirmed, self.killed = .1, False, False
        self.instances.append(self)
    def start(self):
        type(self).live += 1
        type(self).max_live = max(type(self).max_live, type(self).live)
        (self.directory / 'server.log').write_text('fake owned server')
    def offload_evidence(self): return {'gpu_layers_observed': self.config['gpu_layers']}
    def chat(self, prompt, **kwargs):
        if self.gate:
            while not self.gate.wait(.01):
                if self.killed: raise RuntimeError('killed')
        self.last_response = {'choices': [{'message': {'content': prompt}, 'finish_reason': 'length'}], 'usage': {'completion_tokens': 16}, 'timings': {'predicted_per_second': 2}}
        return {'content': prompt, 'response': self.last_response, 'elapsed_seconds': .2, 'truncated': True}
    def _expire(self): self.killed = True
    def stop(self):
        if not self.cleanup_confirmed: type(self).live -= 1
        self.cleanup_confirmed = True


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = json.loads((demo.ROOT / 'experiments/qwen38-27b-q4.json').read_text())
        opt = copy.deepcopy(self.config)
        opt['runtime']['gpu_layers'] = 40
        self.manager = demo.DemoManager('/fake/server', '/fake/model', self.temp.name, self.config, opt, verify_async=False)
        self.manager.status, self.manager.identity = 'ready', {'cached': True}
        FakeServer.instances, FakeServer.gate, FakeServer.live, FakeServer.max_live = [], None, 0, 0
        for target, value in [('LocalServer', FakeServer), ('ResourceSampler', FakeSampler), ('gpu_probe', lambda: {'status': 'ok'})]:
            p = patch.object(demo, target, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.manager.shutdown)

    def wait_run(self, run_id):
        self.manager.worker.join(3)
        self.assertFalse(self.manager.worker.is_alive())
        return self.manager.get(run_id)

    def test_sequential_variants_and_immutable_evidence(self):
        run_id = self.manager.submit({'prompt': 'Hello', 'max_tokens': 16})
        result = self.wait_run(run_id)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(FakeServer.max_live, 1)
        self.assertEqual([s.config['gpu_layers'] for s in FakeServer.instances], [24, 40])
        self.assertTrue(all(s.cleanup_confirmed for s in FakeServer.instances))
        self.assertTrue(all(v['truncated'] for v in result['variants']))
        self.assertEqual(result['variants'][0]['output_tokens'], 16)
        directory = Path(self.temp.name) / run_id
        original = (directory / 'run.json').read_bytes()
        self.manager.cancel(run_id)
        self.assertEqual((directory / 'run.json').read_bytes(), original)
        self.assertTrue((directory / 'baseline/raw-response.json').is_file())
        result['variants'][0]['text'] = 'mutation'
        self.assertEqual(self.manager.get(run_id)['variants'][0]['text'], 'Hello')

    def test_busy_and_cancel_stops_next_variant(self):
        FakeServer.gate = threading.Event()
        run_id = self.manager.submit({'prompt': 'Wait'})
        with self.assertRaises(BlockingIOError): self.manager.submit({'prompt': 'Other'})
        deadline = time.monotonic() + 2
        while self.manager.get(run_id)['phase'] != 'generating' and time.monotonic() < deadline: time.sleep(.01)
        self.manager.cancel(run_id)
        result = self.wait_run(run_id)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(len(FakeServer.instances), 1)
        self.assertEqual(FakeServer.live, 0)
        self.assertEqual(result['variants'][1]['status'], 'cancelled')

    def test_invalid_inputs_and_config(self):
        for body in [None, {}, {'prompt': ' '}, {'prompt': 'x'*3001}, {'prompt': 'ok', 'max_tokens': True}, {'prompt': 'ok', 'max_tokens': 257}, {'prompt': 'ok', 'variants': ['../file']}, {'prompt': 'ok', 'variants': ['baseline', 'baseline']}, {'prompt': 'ok', 'binary': '/other'}]:
            with self.subTest(body=body):
                with self.assertRaises(ValueError): self.manager.submit(body)
        opt = copy.deepcopy(self.config)
        opt['runtime'].update(gpu_layers=40, threads=16)
        with self.assertRaises(ValueError): demo.DemoManager('x', 'y', self.temp.name, self.config, opt)

    def test_cleanup_failure_blocks_future_jobs(self):
        def bad_stop(server):
            FakeServer.live = 0
            raise RuntimeError('secret/path')
        with patch.object(FakeServer, 'stop', bad_stop):
            run_id = self.manager.submit({'prompt': 'Hi'})
            result = self.wait_run(run_id)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.manager.info()['status'], 'blocked')
        self.assertEqual(len(FakeServer.instances), 1)
        self.assertNotIn('secret', json.dumps(result))
        with self.assertRaises(RuntimeError): self.manager.submit({'prompt': 'again'})

    def test_identity_is_cached_once_and_failure_blocks_submission(self):
        self.manager.status = 'checking'
        with patch.object(demo, 'verify_model', return_value={'sha256': 'pinned'}) as model_verify, patch.object(demo, 'runtime_identity', return_value={'pin_verified': True}) as runtime_verify:
            self.manager.verify()
            self.assertTrue(self.manager.info()['ready'])
            for _ in range(3): self.manager.info()
            self.assertEqual(model_verify.call_count, 1)
            self.assertEqual(runtime_verify.call_count, 1)
        with patch.object(demo, 'verify_model', side_effect=ValueError('/private/model-secret')):
            self.manager.verify()
        self.assertEqual(self.manager.info()['status'], 'blocked')
        self.assertNotIn('private', self.manager.info()['error'])
        with self.assertRaises(RuntimeError): self.manager.submit({'prompt': 'No'})

    def test_history_bound_retains_immutable_disk_evidence(self):
        ids = []
        for _ in range(21):
            run_id = self.manager.submit({'prompt': 'history', 'variants': ['baseline']})
            self.wait_run(run_id)
            ids.append(run_id)
        self.assertEqual(len(self.manager.history()['runs']), 20)
        self.assertIsNone(self.manager.get(ids[0]))
        self.assertTrue((Path(self.temp.name) / ids[0] / 'run.json').is_file())
        latest = self.manager.history()['runs'][0]
        self.assertEqual(latest['prompt'], 'history')
        self.assertEqual(latest['max_tokens'], 192)

    def test_forwarded_loopback_authority_preserves_same_origin(self):
        (Path(self.temp.name) / 'index.html').write_text('<!doctype html><title>Demo</title>')
        server = demo.DemoHTTPServer(('127.0.0.1', 0), demo.handler_for(self.manager, self.temp.name))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def request(path, hosts, origin=None, body=None, extra=None):
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
            try:
                connection.putrequest('POST' if body is not None else 'GET', path, skip_host=True)
                for host in hosts: connection.putheader('Host', host)
                if origin is not None: connection.putheader('Origin', origin)
                if extra:
                    for name, value in extra: connection.putheader(name, value)
                if body is not None:
                    connection.putheader('Content-Type', 'application/json')
                    connection.putheader('Content-Length', str(len(body)))
                connection.endheaders(body)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        for authority in ['localhost:63565', '127.0.0.1:63565', '[::1]:63565', 'LOCALHOST:63565']:
            with self.subTest(authority=authority):
                self.assertEqual(request('/', [authority])[0], 200)
                self.assertEqual(request('/api/info', [authority])[0], 200)
        body = b'{"prompt":"Forwarded preview","variants":["baseline"]}'
        status, raw = request('/api/runs', ['localhost:63565'], 'http://localhost:63565', body)
        self.assertEqual(status, 202)
        self.assertEqual(self.wait_run(json.loads(raw)['id'])['status'], 'completed')
        self.assertEqual(request('/api/runs', ['LOCALHOST:63565'], 'http://localhost:63565', body)[0], 202)
        self.manager.worker.join(2)
        for host in ['evil.example:63565', 'localhost.evil:63565', 'user@localhost:63565', 'localhost:0', 'localhost:65536', 'localhost:-1', 'localhost:abc', 'localhost:63565/path', 'localhost:63565?', 'localhost:63565#x', 'localhost:63565,evil.example', '::1:63565', '127.0.0.2:63565']:
            with self.subTest(host=host):
                self.assertEqual(request('/api/info', [host])[0], 403)
        self.assertEqual(request('/api/info', [])[0], 403)
        self.assertEqual(request('/api/info', ['localhost:63565', 'localhost:63565'])[0], 403)
        for origin in ['http://evil.example:63565', 'http://localhost:63564', 'http://127.0.0.1:63565', 'https://localhost:63565', 'http://localhost:63565/', 'http://user@localhost:63565', 'null']:
            with self.subTest(origin=origin):
                self.assertEqual(request('/api/runs', ['localhost:63565'], origin, body)[0], 403)
        self.assertEqual(request('/api/runs', ['localhost:63565'], 'http://localhost:63565', body, [('Origin', 'http://localhost:63565')])[0], 403)
        self.assertEqual(request('/api/info', ['evil.example'], extra=[('X-Forwarded-Host', 'localhost:63565')])[0], 403)

    def test_http_security_and_contract(self):
        server = demo.DemoHTTPServer(('127.0.0.1', 0), demo.handler_for(self.manager, self.temp.name))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        def request(path, body=None, headers=None):
            req = urllib.request.Request(url + path, data=body, headers=headers or {})
            try:
                with urllib.request.urlopen(req, timeout=2) as response: return response.status, json.load(response)
            except urllib.error.HTTPError as exc: return exc.code, json.load(exc)
        self.assertEqual(request('/api/info')[1]['status'], 'ready')
        self.assertEqual(request('/../../etc/passwd')[0], 404)
        self.assertEqual(request('/api/info', headers={'Host': 'evil.example'})[0], 403)
        headers = {'Content-Type': 'application/json'}
        self.assertEqual(request('/api/runs', b'{}', {**headers, 'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(request('/api/runs', b'{}', {'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(request('/api/runs', b'{' , headers)[0], 400)
        self.assertEqual(request('/api/runs', b'x'*65537, headers)[0], 413)
        self.assertEqual(request('/api/runs', b'{"prompt":"Hi","variants":["unknown"]}', headers)[0], 400)
        status, result = request('/api/runs', b'{"prompt":"Hi","variants":["baseline"]}', headers)
        self.assertEqual(status, 202)
        self.wait_run(result['id'])
        self.assertEqual(request('/api/runs/' + result['id'])[1]['status'], 'completed')
        self.assertEqual(len(request('/api/history')[1]['runs']), 1)
        self.assertEqual(request('/api/runs/missing/cancel', b'{}', headers)[0], 404)


if __name__ == '__main__': unittest.main()
