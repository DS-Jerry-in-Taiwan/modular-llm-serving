"""No GPU needed: lifecycle serialization and hostile HTTP request boundaries."""
import copy
import errno
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

from inference_lab.web import jobs, server as web_server


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
        self.config = json.loads((web_server.ROOT / 'experiments/qwen38-27b-q4.json').read_text())
        opt = copy.deepcopy(self.config)
        opt['runtime']['gpu_layers'] = 40
        self.manager = jobs.DemoManager('/fake/server', '/fake/model', self.temp.name, self.config, opt, verify_async=False)
        self.manager.status, self.manager.identity = 'ready', {'cached': True}
        FakeServer.instances, FakeServer.gate, FakeServer.live, FakeServer.max_live = [], None, 0, 0
        for target, value in [('LocalServer', FakeServer), ('ResourceSampler', FakeSampler), ('gpu_probe', lambda: {'status': 'ok'})]:
            p = patch.object(jobs, target, value)
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
        with self.assertRaises(ValueError): jobs.DemoManager('x', 'y', self.temp.name, self.config, opt)

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

    def test_evidence_write_failures_converge_and_allow_recovery(self):
        original_write = jobs.write_json_new
        for target, error_code in [(target, code) for target in ['resource-samples.json', 'raw-response.json', 'variant-run', 'top-run', 'all'] for code in [errno.ENOSPC, errno.EACCES]]:
            with self.subTest(target=target, error_code=error_code):
                def fail_selected(path, value):
                    selected = (target == 'all' and path.name != 'manifest.json') or path.name == target or (target == 'variant-run' and path.name == 'run.json' and path.parent.name == 'baseline') or (target == 'top-run' and path.name == 'run.json' and path.parent.parent == Path(self.temp.name))
                    if selected:
                        raise OSError(error_code, '/private/evidence-secret')
                    original_write(path, value)
                with patch.object(jobs, 'write_json_new', side_effect=fail_selected):
                    run_id = self.manager.submit({'prompt': 'persist', 'variants': ['baseline']})
                    result = self.wait_run(run_id)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['phase'], 'failed')
                self.assertTrue(result['cleanup_confirmed'])
                self.assertEqual(result['variants'][0]['status'], 'failed' if target != 'top-run' else 'completed')
                self.assertIsNone(self.manager.server)
                self.assertIsNone(self.manager.active)
                self.assertFalse(self.manager.info()['busy'])
                self.assertEqual(FakeServer.live, 0)
                self.assertNotIn('evidence-secret', json.dumps(result))
                recovery = self.manager.submit({'prompt': 'recovery', 'variants': ['baseline']})
                self.assertEqual(self.wait_run(recovery)['status'], 'completed')

    def test_active_reservation_is_held_until_cleanup_confirmed(self):
        entered, release = threading.Event(), threading.Event()
        original_stop = FakeServer.stop
        def gated_stop(server):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test cleanup gate timed out')
            original_stop(server)
        with patch.object(FakeServer, 'stop', gated_stop):
            run_id = self.manager.submit({'prompt': 'wait for cleanup', 'variants': ['baseline']})
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.manager.get(run_id)['phase'], 'cleaning_up')
                self.assertEqual(self.manager.get(run_id)['status'], 'running')
                self.assertIsNotNone(self.manager.server)
                self.assertEqual(self.manager.active, run_id)
                with self.assertRaises(BlockingIOError):
                    self.manager.submit({'prompt': 'too early'})
            finally:
                release.set()
            self.assertEqual(self.wait_run(run_id)['status'], 'completed')
        self.assertIsNone(self.manager.active)
        self.assertIsNone(self.manager.server)

    def test_manifest_write_failures_never_reserve_gpu(self):
        for error_code in [errno.ENOSPC, errno.EACCES]:
            with self.subTest(error_code=error_code), patch.object(jobs, 'write_json_new', side_effect=OSError(error_code, '/private/manifest-secret')):
                with self.assertRaises(OSError):
                    self.manager.submit({'prompt': 'manifest failure'})
            self.assertIsNone(self.manager.active)
            self.assertIsNone(self.manager.server)
            self.assertFalse(self.manager.jobs)
            self.assertFalse(FakeServer.instances)

    def test_cancel_reaps_real_owned_process_group(self):
        from inference_lab.runtime import LocalServer, RuntimeFailure
        binary = Path(self.temp.name) / 'owned-server'
        binary.write_text('#!/usr/bin/env python3\nimport subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])\nprint("ready",flush=True)\ntime.sleep(30)\n')
        binary.chmod(0o700)
        self.manager.binary = binary
        started = threading.Event()
        instances = []
        class RealServer(LocalServer):
            def __init__(server, *args):
                super().__init__(*args)
                instances.append(server)
            def offload_evidence(server):
                return {'gpu_layers_observed': server.config['gpu_layers']}
            def chat(server, prompt, **kwargs):
                deadline = time.monotonic() + 2
                while 'ready' not in (server.run_dir / 'server.log').read_text():
                    if time.monotonic() >= deadline:
                        raise RuntimeFailure('owned child did not start')
                    time.sleep(.01)
                started.set()
                server.process.wait(timeout=5)
                raise RuntimeFailure('owned process stopped')
        with patch.object(jobs, 'LocalServer', RealServer), patch('inference_lab.runtime.free_loopback_port', return_value=18441), patch('inference_lab.runtime.http_json', return_value={'status': 'ok'}):
            run_id = self.manager.submit({'prompt': 'cancel owned child'})
            self.assertTrue(started.wait(2))
            self.manager.cancel(run_id)
            result = self.wait_run(run_id)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(result['cleanup_confirmed'])
        self.assertEqual(result['variants'][0]['status'], 'cancelled')
        self.assertEqual(result['variants'][1]['status'], 'cancelled')
        self.assertTrue(instances[0].cleanup_confirmed)
        self.assertIsNotNone(instances[0].process.poll())
        self.assertFalse(LocalServer.group_has_live_members(instances[0].process.pid))
        self.assertIsNone(self.manager.active)
        self.assertIsNone(self.manager.server)

    def test_cleanup_failure_retains_ownership_even_when_evidence_fails(self):
        def failed_stop(server):
            raise RuntimeError('/private/cleanup-secret')
        original_write = jobs.write_json_new
        def failed_receipt(path, value):
            if path.name in {'raw-response.json', 'run.json'}:
                raise OSError('/private/write-secret')
            original_write(path, value)
        with patch.object(FakeServer, 'stop', failed_stop), patch.object(jobs, 'write_json_new', side_effect=failed_receipt):
            run_id = self.manager.submit({'prompt': 'retain ownership'})
            result = self.wait_run(run_id)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['cleanup_confirmed'])
        self.assertEqual(result['variants'][0]['status'], 'failed')
        self.assertIs(self.manager.server, FakeServer.instances[0])
        self.assertEqual(self.manager.active, run_id)
        self.assertTrue(self.manager.info()['busy'])
        self.assertFalse(self.manager.info()['ready'])
        with self.assertRaises(RuntimeError):
            self.manager.submit({'prompt': 'must stay blocked'})
        self.assertNotIn('secret', json.dumps(result))
        # Test teardown owns the fake process after the failure has been inspected.
        self.manager.server.stop()

    def test_identity_is_cached_once_and_failure_blocks_submission(self):
        self.manager.status = 'checking'
        with patch.object(jobs, 'verify_model', return_value={'sha256': 'pinned'}) as model_verify, patch.object(jobs, 'runtime_identity', return_value={'pin_verified': True}) as runtime_verify:
            self.manager.verify()
            self.assertTrue(self.manager.info()['ready'])
            for _ in range(3): self.manager.info()
            self.assertEqual(model_verify.call_count, 1)
            self.assertEqual(runtime_verify.call_count, 1)
        with patch.object(jobs, 'verify_model', side_effect=ValueError('/private/model-secret')):
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
        server = web_server.DemoHTTPServer(('127.0.0.1', 0), web_server.handler_for(self.manager, self.temp.name))
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

    def test_default_static_and_historical_reference_routes(self):
        server = web_server.DemoHTTPServer(('127.0.0.1', 0), web_server.handler_for(self.manager))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        static_root = Path(web_server.__file__).parent / 'static'
        url = f'http://127.0.0.1:{server.server_port}'
        for route, filename in [('/', 'index.html'), ('/app.js', 'app.js'), ('/style.css', 'style.css')]:
            with self.subTest(route=route), urllib.request.urlopen(url + route, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), (static_root / filename).read_bytes())
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
        with urllib.request.urlopen(url + '/reference', timeout=2) as response:
            self.assertEqual(response.status, 200)
            body = response.read().decode('utf-8')
            self.assertIn('收斂後 V1 固定測試', body)
            self.assertIn('20261114T131732Z-1327935c51', body)
            self.assertIn('343.08s', body)

    def test_reference_api_projects_latest_comparison_without_local_paths(self):
        source = json.loads((web_server.ROOT / web_server.REFERENCE_DIRECTORY / 'comparison.json').read_text())
        server = web_server.DemoHTTPServer(('127.0.0.1', 0), web_server.handler_for(self.manager))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}/api/reference'
        with urllib.request.urlopen(url, timeout=2) as response:
            self.assertEqual(response.status, 200)
            data = json.load(response)
        self.assertEqual(data['status'], 'available')
        self.assertEqual(data['label'], '收斂後 V1 固定測試')
        self.assertEqual(data['baseline_run_id'], source['baseline_run_id'])
        self.assertEqual(data['candidate_run_id'], source['candidate_run_id'])
        self.assertEqual(data['measured_request_totals_seconds'], source['measured_request_totals_seconds'])
        self.assertEqual(data['quality']['samples_per_configuration'], 12)
        self.assertEqual(data['reference_url'], '/reference')
        for case in data['cases']:
            self.assertEqual(case['baseline_seconds'], source['per_case'][case['id']]['baseline_request_seconds'])
            self.assertEqual(case['candidate_seconds'], source['per_case'][case['id']]['candidate_request_seconds'])
            self.assertEqual(case['latency_ratio'], source['per_case'][case['id']]['latency_ratio'])
        for key in ('baseline', 'candidate'):
            self.assertEqual(data['gpu_peak_bytes'][key], source['resources'][key]['gpu_device_used_bytes']['sampled_peak_bytes'])
        public = json.dumps(data)
        for private in ('/home/', 'evidence_locations', 'binary', 'source', 'git_status', 'docs/agent_context'):
            self.assertNotIn(private, public)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url, headers={'Host': 'evil.example'}), timeout=2)
        self.assertEqual(error.exception.code, 403)

    def test_reference_unavailable_for_invalid_or_missing_evidence(self):
        source = json.loads((web_server.ROOT / web_server.REFERENCE_DIRECTORY / 'comparison.json').read_text())
        root = Path(self.temp.name) / 'reference-root'
        directory = root / web_server.REFERENCE_DIRECTORY
        directory.mkdir(parents=True)
        report, comparison = directory / 'report.md', directory / 'comparison.json'
        variants = [None, [], {}, {**source, 'quality': {'baseline': 'pending', 'candidate': 'pass'}},
                    {**source, 'measured_request_totals_seconds': {'baseline': float('nan'), 'candidate': 1, 'latency_ratio': 1}},
                    {**source, 'baseline_run_id': '/private/evidence-secret'}]
        with patch.object(web_server, 'ROOT', root):
            self.assertEqual(web_server.reference_info()['status'], 'unavailable')
            report.write_text('Fixed report')
            for value in variants:
                with self.subTest(value=type(value).__name__):
                    comparison.write_text(json.dumps(value))
                    result = web_server.reference_info()
                    self.assertEqual(set(result), {'status', 'label', 'scope', 'reference_url'})
                    self.assertEqual(result['status'], 'unavailable')
            for invalid in ['{malformed', '[' * 2000, 'x' * (web_server.REFERENCE_MAX_BYTES + 1)]:
                comparison.write_text(invalid)
                self.assertEqual(web_server.reference_info()['status'], 'unavailable')
            comparison.write_text(json.dumps(source))
            self.assertEqual(web_server.reference_info()['status'], 'available')
            for content in [b'', b'\xff', b'x' * (web_server.REFERENCE_MAX_BYTES + 1)]:
                report.write_bytes(content)
                self.assertEqual(web_server.reference_info()['status'], 'unavailable')
            report.write_text('<script>alert("escaped")</script>')
            server = web_server.DemoHTTPServer(('127.0.0.1', 0), web_server.handler_for(self.manager))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                url = f'http://127.0.0.1:{server.server_port}'
                with urllib.request.urlopen(url + '/reference', timeout=2) as response:
                    text = response.read().decode()
                    self.assertNotIn('<script>', text)
                    self.assertIn('&lt;script&gt;', text)
                comparison.write_text('{malformed')
                with urllib.request.urlopen(url + '/api/reference', timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.load(response)['status'], 'unavailable')
            finally:
                server.shutdown()
                server.server_close()

    def test_http_security_and_contract(self):
        server = web_server.DemoHTTPServer(('127.0.0.1', 0), web_server.handler_for(self.manager, self.temp.name))
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
