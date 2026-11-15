"""Exact request sharing without merging sessions or replacing paid answers."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

from tess_memory.checkpoints import DistillationCheckpoints
from tess_memory.distill import CallBudgetExceeded, DistillationPool
from test_parallel_distill import Backend, response


class SharedCheckpointTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)/'checkpoints.db'

    def pool(self, primary=None, fallback=None, **kwargs):
        return DistillationPool(primary or Backend(), fallback, workers=4,
                                cache_path=self.path, share_requests=True, **kwargs)

    @staticmethod
    def call(pool, namespace, prompt='same input', system='same instructions'):
        return pool.session(namespace).submit(system, prompt, 'segment stable').result(timeout=5)

    def test_concurrent_identical_sessions_use_one_provider_call_and_survive_restart(self):
        started, release = threading.Event(), threading.Event()
        def model(prompt):
            started.set()
            self.assertTrue(release.wait(3))
            return response()
        backend = Backend(fn=model)
        with self.pool(backend) as pool:
            sessions = [pool.session(str(i)) for i in range(4)]
            futures = [s.submit('system', 'identical', 'segment stable') for s in sessions]
            self.assertTrue(started.wait(3))
            release.set()
            self.assertEqual(len({f.result(timeout=5)['summary'] for f in futures}), 1)
            self.assertEqual((pool.calls, pool.shared_cache_hits), (1, 3))
            self.assertEqual(pool.checkpointed_requests, 1)
            self.assertEqual(pool._request_locks, {})
            for session in sessions:
                self.assertIn(DistillationCheckpoints.key(session.namespace, backend.name, 'system', 'identical'), session.keys)
        with self.pool(backend) as pool:
            pool.session('new fork').submit('system', 'identical', 'segment stable').result()
            self.assertEqual(pool.calls, 0)
            self.assertEqual(pool.checkpointed_requests, 0)
        self.assertEqual(len(backend.calls), 1)

    def test_distinct_prompts_still_run_concurrently(self):
        barrier = threading.Barrier(4)
        backend = Backend(fn=lambda _: (barrier.wait(timeout=3), response())[1])
        with self.pool(backend) as pool:
            futures = [pool.session(str(i)).submit('system', str(i), 'segment stable') for i in range(4)]
            for f in futures:
                self.assertTrue(f.result(timeout=5)['summary'])
            self.assertEqual(pool.peak_calls, 4)

    def test_legacy_fallback_is_preserved_even_if_shared_primary_already_exists(self):
        primary, fallback = Backend(), Backend('fallback')
        cache = DistillationCheckpoints(self.path)
        legacy = json.loads(response('paid fallback answer'))
        key = cache.key('old session', fallback.name, 'same instructions', 'same input')
        cache.put(key, legacy)
        with self.pool(primary, fallback) as pool:
            self.call(pool, 'other session')
            old = pool.session('old session')
            self.assertEqual(old.cached_result('same instructions', 'same input', 'segment stable'), legacy)
            self.assertTrue(old.fallback_used)
            self.assertEqual(pool.calls, 1)
            self.assertEqual(cache.get(key), legacy)
        self.assertEqual(len(fallback.calls), 0)

    def test_cache_only_promotion_reuses_paid_results_without_invoking_model(self):
        backend = Backend(fn=lambda _: self.fail('model invoked by cache-only migration'))
        cache = DistillationCheckpoints(self.path)
        expected = json.loads(response('existing paid answer'))
        cache.put(cache.key('old', backend.name, 's', 'p'), expected)
        with self.pool(backend) as pool:
            self.assertEqual(pool.session('old').cached_result('s', 'p', 'segment stable'), expected)
            self.assertIsNone(pool.session('old').cached_result('s', 'missing', 'segment stable'))
            self.assertEqual(pool.session('new').cached_result('s', 'p', 'segment stable'), expected)
            self.assertEqual((pool.calls, pool.cache_promotions, pool.shared_cache_hits), (0, 1, 1))
            cache.pin('retained session', pool.session('unused').keys | {cache.shared_key(backend.name, 's', 'p')})
            cache.prune(float('inf'))
            self.assertEqual(pool.session('after prune').cached_result('s', 'p', 'segment stable'), expected)
        # New work also leaves the original session-key format readable.
        self.assertEqual(cache.get(cache.key('after prune', backend.name, 's', 'p')), expected)

    def test_existing_distinct_paid_answers_are_not_overwritten_by_sharing(self):
        cache = DistillationCheckpoints(self.path)
        a, b = json.loads(response('answer A')), json.loads(response('answer B'))
        cache.put(cache.key('a', 'primary', 's', 'p'), a)
        cache.put(cache.key('b', 'primary', 's', 'p'), b)
        with self.pool() as pool:
            self.assertEqual(pool.session('a').cached_result('s', 'p', 'segment stable'), a)
            self.assertEqual(pool.session('b').cached_result('s', 'p', 'segment stable'), b)
            self.assertEqual(pool.session('new').cached_result('s', 'p', 'segment stable'), a)
            self.assertEqual(pool.calls, 0)
        self.assertEqual(cache.get(cache.key('b', 'primary', 's', 'p')), b)

    def test_prompt_model_system_and_store_boundaries_remain_distinct(self):
        backend = Backend()
        with self.pool(backend) as pool:
            self.call(pool, 'a')
            self.call(pool, 'b')
            self.call(pool, 'c', prompt='changed')
            self.call(pool, 'd', system='changed')
            backend.cache_identity = 'different model and reasoning effort'
            self.call(pool, 'e')
            self.assertEqual(pool.calls, 4)
        with DistillationPool(backend, workers=1, cache_path=self.path.parent/'other-store.db', share_requests=True) as pool:
            self.call(pool, 'a')
            self.assertEqual(pool.calls, 1)

    def test_concurrent_fallback_is_shared_and_budget_is_not_spent_on_waiters(self):
        primary = Backend(fn=lambda _: (_ for _ in ()).throw(RuntimeError('unavailable')))
        fallback = Backend('fallback')
        with self.pool(primary, fallback, max_calls=2) as pool:
            sessions = [pool.session(str(i)) for i in range(4)]
            futures = [s.submit('system', 'p', 'segment stable') for s in sessions]
            for f in futures:
                self.assertTrue(f.result(timeout=5)['summary'])
            self.assertTrue(all(s.fallback_used for s in sessions))
            self.assertEqual(pool.calls, 2)
            with self.assertRaises(CallBudgetExceeded):
                self.call(pool, 'different', prompt='uncached')
            self.assertEqual(pool._request_locks, {})
        self.assertEqual((len(primary.calls), len(fallback.calls)), (1, 1))

    def test_failed_requests_release_waiters_and_are_not_shared_as_success(self):
        backend = Backend(fn=lambda _: response(''))
        with self.pool(backend) as pool:
            futures = [pool.session(str(i)).submit('s', 'p', 'segment stable') for i in range(2)]
            for f in futures:
                with self.assertRaises(RuntimeError):
                    f.result(timeout=5)
            self.assertEqual(pool._request_locks, {})
            backend.fn = None
            self.assertTrue(pool.session('retry').submit('s', 'p', 'segment stable').result()['summary'])
            self.assertEqual(pool.calls, 3)


if __name__ == '__main__':
    unittest.main()
