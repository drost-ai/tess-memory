from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from agentic_memory.config import Config
from agentic_memory.distill import DistillationPool, SESSION_SYSTEM
from agentic_memory.map_batch import BATCH_INSTRUCTIONS
from agentic_memory.readers import ParsedSource
from agentic_memory.stable_distill import stable_digest, stable_map_items
from agentic_memory.transcript import SessionTranscript, Turn


class Backend:
    name = 'stub'
    def __init__(self):
        self.calls = []
        self.units = Counter()
        self.lock = threading.Lock()
        self.invalid = False
    def complete(self, *, system, prompt):
        with self.lock:
            self.calls.append((system, prompt))
        if prompt.startswith(BATCH_INSTRUCTIONS):
            payload = json.loads(prompt[len(BATCH_INSTRUCTIONS):].split('\n===== END UNIT DATA =====')[0])
            with self.lock:
                self.units.update(row['id'] for row in payload['units'])
            ids = [row['id'] for row in payload['units']]
            if self.invalid:
                ids = ids[:1]
            return json.dumps({'results': [{'id': key, 'summary': 'Unit ' + key, 'memories': []} for key in ids]})
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt.startswith('A completed persisted-message'):
            with self.lock:
                self.units[key] += 1
        return json.dumps({'summary': 'Unit ' + key, 'memories': []})


class MapBatchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / 'cache.db'
        self.cfg = Config(segment_chars=3000, segment_overlap=256, llm_workers=4, segment_workers=4,
                          incremental_batch_chars=32000, incremental_batch_items=8)
        self.st = SessionTranscript('one', 'p', '/unused', source_type='codex')
        self.st.turns = [Turn('user', f'Request {i}: ' + 'valuable source detail ' * 15) for i in range(8)]
        self.backend = Backend()

    def snapshot(self):
        return ParsedSource(self.st, list(range(1, len(self.st.turns)+1)), ['turn_complete'] * len(self.st.turns), [], False, '', 0)

    def pool(self, backend=None):
        return DistillationPool(backend or self.backend, workers=4, cache_path=self.cache, share_requests=True)

    def test_batch_preserves_each_unit_reduces_input_and_reuses_results_after_append(self):
        snapshot = self.snapshot()
        items = list(stable_map_items(self.cfg, snapshot))
        with self.pool() as pool:
            stable_digest(self.cfg, snapshot, pool.session('one'))
            self.assertEqual((pool.calls, pool.batch_requests, pool.batched_units), (2, 1, 8))
        batch = next(prompt for _, prompt in self.backend.calls if prompt.startswith(BATCH_INSTRUCTIONS))
        self.assertLess(len(batch)+len(SESSION_SYSTEM), sum(len(i.prompt)+len(SESSION_SYSTEM) for i in items))
        for turn in self.st.turns:
            self.assertIn(turn.text, batch)
        self.assertEqual(set(self.backend.units), {i.identifier for i in items})
        self.st.turns.append(Turn('user', 'One new request with distinct useful content'))
        self.cfg.incremental_batch_chars = 0  # derived units also work with batching disabled
        with self.pool() as pool:
            stable_digest(self.cfg, self.snapshot(), pool.session('one'))
            self.assertEqual(pool.calls, 3)  # new map, right branch, changed root
        self.assertTrue(all(self.backend.units[i.identifier] == 1 for i in items))
        with self.pool() as pool:
            stable_digest(self.cfg, self.snapshot(), pool.session('one'))
            self.assertEqual(pool.calls, 0)

    def test_existing_paid_single_request_wins_and_is_not_rewritten(self):
        item = next(stable_map_items(self.cfg, self.snapshot()))
        paid = {'summary': 'Existing paid answer', 'memories': []}
        with self.pool() as pool:
            key = pool.cache.key('one', self.backend.name, SESSION_SYSTEM, item.prompt)
            pool.cache.put(key, paid)
            stable_digest(self.cfg, self.snapshot(), pool.session('one'))
            self.assertEqual(pool.batched_units, 7)
            self.assertEqual(pool.cache.get(key), paid)
            self.assertEqual(self.backend.units[item.identifier], 0)
            self.assertIn('Existing paid answer', self.backend.calls[-1][1])

    def test_overlapping_batches_share_individual_work_without_deadlock(self):
        items = list(stable_map_items(self.cfg, self.snapshot()))
        with self.pool() as pool, ThreadPoolExecutor(2) as sessions:
            futures = [sessions.submit(lambda name=name, group=group: pool.session(name).submit_batch(group).result())
                       for name, group in [('one', items[:3]), ('two', items[1:4])]]
            for future in futures:
                self.assertEqual(len(future.result(timeout=4)), 3)
        self.assertEqual(self.backend.units, Counter({i.identifier: 1 for i in items[:4]}))

    def test_invalid_partial_batch_is_never_published_as_individual_successes(self):
        items = list(stable_map_items(self.cfg, self.snapshot()))
        self.backend.invalid = True
        with self.pool() as pool:
            with self.assertRaises(RuntimeError):
                pool.session('one').submit_batch(items[:3]).result()
            self.assertEqual(pool.checkpointed_requests, 0)
            for item in items[:3]:
                self.assertIsNone(pool.session('one').cached_result(SESSION_SYSTEM, item.prompt, 'segment stable'))
        self.backend.invalid = False
        with self.pool() as pool:
            self.assertEqual(len(pool.session('one').submit_batch(items[:3]).result()), 3)
            self.assertEqual(pool.calls, 1)

    def test_saved_batch_recovers_if_individual_publication_was_interrupted(self):
        from unittest import mock
        items = list(stable_map_items(self.cfg, self.snapshot()))[:3]
        with self.pool() as pool:
            original = pool.cache.put_if_absent
            def interrupted(key, result):
                if 'request_key' in result:
                    raise RuntimeError('interrupted derived-unit publication')
                return original(key, result)
            with mock.patch.object(pool.cache, 'put_if_absent', side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    pool.session('one').submit_batch(items).result()
            self.assertEqual(pool.calls, 1)
        with self.pool() as pool:
            self.assertEqual(len(pool.session('one').submit_batch(items).result()), 3)
            self.assertEqual(pool.calls, 0)
