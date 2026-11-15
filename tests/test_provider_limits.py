from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tess_memory.provider_limits import ProviderGate, LimitedBackend, ModelBudget
from tess_memory.distill import CallBudgetExceeded


class ProviderLimitTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / 'gate.db'

    def test_processes_share_global_and_provider_limits(self):
        ProviderGate(self.path, model_limit=2, provider_limits={'a': 1, 'b': 2})
        code = '''import os,sys,time
from pathlib import Path
from tess_memory.provider_limits import ProviderGate
gate=ProviderGate(Path(sys.argv[1]),model_limit=2,provider_limits={'a':1,'b':2},poll_seconds=0.005)
for i in range(3):
    with gate.acquire(sys.argv[2],'model',owner=str(os.getpid()),instance=sys.argv[3]):
        time.sleep(0.04)
'''
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        processes = [subprocess.Popen([sys.executable, '-c', code, str(self.path), 'a' if i % 2 else 'b', str(i)],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(4)]
        for process in processes:
            output, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, error.decode())
        with sqlite3.connect(self.path) as db:
            rows = db.execute('SELECT provider,started,ended FROM call_receipts').fetchall()
        self.assertEqual(len(rows), 12)
        events = sorted([(start, 1, provider) for provider, start, end in rows]
                        + [(end, -1, provider) for provider, start, end in rows])
        active = peak = 0
        providers = {'a': 0, 'b': 0}
        for _, delta, provider in events:
            active += delta
            providers[provider] += delta
            peak = max(peak, active)
            self.assertLessEqual(active, 2)
            self.assertLessEqual(providers['a'], 1)
        self.assertEqual((active, peak), (0, 2))

    def test_crashed_owner_leases_are_recovered(self):
        gate = ProviderGate(self.path, model_limit=1)
        code = '''import os,sys
from pathlib import Path
from tess_memory.provider_limits import ProviderGate
with ProviderGate(Path(sys.argv[1]),model_limit=1).acquire('p','model',owner='crashed',instance='old'):
    os._exit(0)
'''
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        subprocess.run([sys.executable, '-c', code, str(self.path)], env=env, check=True, timeout=5)
        with gate.acquire('p', 'model', owner='new', instance='new'):
            self.assertEqual(gate.status()['active'][0]['instance'], 'new')
        self.assertTrue(any(r['outcome'] == 'worker_lost' for r in gate.status()['completed']))

    def test_stale_heartbeat_does_not_overbook_a_live_call(self):
        gate = ProviderGate(self.path, model_limit=1, poll_seconds=0.005)
        entered = threading.Event()
        def second():
            with gate.acquire('p', 'model', owner='second', instance='two'):
                entered.set()
        with ThreadPoolExecutor(1) as pool:
            with gate.acquire('p', 'model', owner='first', instance='one'):
                with gate.connect() as db:
                    db.execute('UPDATE call_leases SET heartbeat=0')
                future = pool.submit(second)
                self.assertFalse(entered.wait(0.1))
            future.result(timeout=3)
        self.assertTrue(entered.is_set())

    def test_cooling_provider_does_not_block_another_provider(self):
        gate = ProviderGate(self.path, model_limit=1, poll_seconds=0.005)
        gate.cooldown('a', seconds=0.2)
        entered = threading.Event()
        def cooling():
            with gate.acquire('a', 'model', owner='a', instance='one'):
                entered.set()
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(cooling)
            time.sleep(0.02)
            with gate.acquire('b', 'model', owner='b', instance='two'):
                self.assertFalse(entered.is_set())
            future.result(timeout=3)

    def test_model_budget_is_shared_by_primary_and_fallback(self):
        class Backend:
            name = 'stub'
            def complete(self, **kwargs):
                return 'OK'
        gate = ProviderGate(self.path)
        budget = ModelBudget(2)
        a = LimitedBackend(Backend(), gate, budget, provider='a', owner='test', instance='one')
        b = LimitedBackend(Backend(), gate, budget, provider='b', owner='test', instance='one')
        self.assertEqual(a.complete(system='s', prompt='p'), 'OK')
        self.assertEqual(b.complete(system='s', prompt='p'), 'OK')
        with self.assertRaises(CallBudgetExceeded):
            b.complete(system='s', prompt='p')
        self.assertEqual(budget.calls, 2)
        self.assertFalse(gate.status()['active'])

    def test_deadline_drains_active_call_and_removes_waiter_without_new_call(self):
        started, release = threading.Event(), threading.Event()
        class Backend:
            name = 'stub'
            def complete(self, **kwargs):
                started.set()
                self_test.assertTrue(release.wait(3))
                return 'paid output'
        self_test = self
        gate = ProviderGate(self.path, model_limit=1, poll_seconds=0.005)
        budget = ModelBudget(100)
        backend = LimitedBackend(Backend(), gate, budget, provider='p', owner='o', instance='i')
        with ThreadPoolExecutor(2) as pool:
            active = pool.submit(backend.complete, system='s', prompt='first')
            self.assertTrue(started.wait(2))
            waiting = pool.submit(backend.complete, system='s', prompt='second')
            budget.draining.set()
            with self.assertRaises(CallBudgetExceeded):
                waiting.result(timeout=2)
            self.assertFalse(active.done())
            release.set()
            self.assertEqual(active.result(timeout=2), 'paid output')
        self.assertEqual(budget.calls, 1)
        self.assertFalse(gate.status()['waiting'])
        self.assertEqual(gate.status()['completed'][0]['calls'], 1)

    def test_monotonic_deadline_stops_admission(self):
        with mock.patch('tess_memory.provider_limits.time.monotonic', return_value=11):
            budget = ModelBudget(128, deadline=10)
            with self.assertRaises(CallBudgetExceeded):
                budget.take()
        self.assertEqual(budget.calls, 0)
        self.assertTrue(budget.draining.is_set())

    def test_source_becoming_active_while_waiting_never_reaches_model(self):
        from functools import partial
        from tess_memory.distill import DistillationPool
        from tess_memory.source_activity import require_quiet, SourceNotQuiet
        from test_incremental_ingest import Backend
        path = self.root/'source.jsonl'
        path.write_text('{}\n')
        os.utime(path, (time.time()-3600, time.time()-3600))
        gate = ProviderGate(self.path, model_limit=1, poll_seconds=0.005)
        budget = ModelBudget(10)
        primary, fallback = Backend(), Backend()
        backend = LimitedBackend(primary, gate, budget, provider='p', owner='target', instance='one')
        with gate.acquire('p', 'model', owner='occupied', instance='other'), \
             DistillationPool(backend, fallback, workers=1) as pool:
            calls = pool.session('target', admission_guard=partial(require_quiet, path, 'transcript', 900))
            future = calls.submit('system', 'prompt', 'segment stable')
            until = time.monotonic()+2
            while not gate.status()['waiting'] and time.monotonic()<until:
                time.sleep(0.005)
            self.assertTrue(gate.status()['waiting'])
            path.write_text('{"new":"activity"}\n')
            with self.assertRaises(SourceNotQuiet):
                future.result(timeout=2)
            self.assertEqual(primary.calls, [])
            self.assertEqual(fallback.calls, [])
            self.assertEqual(budget.calls, 0)
            self.assertEqual(gate.status()['waiting'], [])

    def test_embedding_deferral_is_not_retried_or_wrapped_as_a_network_error(self):
        from tess_memory.embed import OpenAICompatEmbedder
        from tess_memory.model_usage import guard_source_requests
        from tess_memory.source_activity import require_quiet, SourceNotQuiet
        path = self.root/'hot.jsonl'
        path.write_text('{}\n')
        embedder = OpenAICompatEmbedder.__new__(OpenAICompatEmbedder)
        embedder._url, embedder._key, embedder._model, embedder._dims = 'https://unused.test', 'fake', 'fake', 4
        with guard_source_requests(lambda: require_quiet(path, 'transcript', 900)), \
             mock.patch('httpx.post') as post, mock.patch('time.sleep') as sleep:
            with self.assertRaises(SourceNotQuiet):
                embedder._post(['text'])
        post.assert_not_called()
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
