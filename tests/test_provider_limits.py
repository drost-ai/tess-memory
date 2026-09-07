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

from agentic_memory.provider_limits import ProviderGate, LimitedBackend, ModelBudget
from agentic_memory.distill import CallBudgetExceeded


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
from agentic_memory.provider_limits import ProviderGate
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
from agentic_memory.provider_limits import ProviderGate
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


if __name__ == '__main__':
    unittest.main()
