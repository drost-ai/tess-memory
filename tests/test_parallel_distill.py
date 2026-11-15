"""Concurrency, ordering and restart guarantees without real model calls."""
from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest import mock

from tess_memory.config import CodexSettings, Config, DataStore, load_config
from tess_memory.distill import DistillationPool, distill_session, distill_session_segmented
from tess_memory.ingest import ingest
from tess_memory.store import MemoryStore
from test_codex import RecordingEmbedder, append, message, wrapped, write_rollout


def response(summary="Durable summary"):
    return json.dumps({"title": "Useful work", "summary": summary,
                       "outcome": "Complete", "memories": []})


class Backend:
    def __init__(self, name="primary", fn=None):
        self.name = name
        self.fn = fn
        self.calls = []
        self.lock = threading.Lock()

    def complete(self, *, system, prompt):
        with self.lock:
            self.calls.append(prompt)
        return self.fn(prompt) if self.fn else response()


def segmented(pool, *, namespace="codex:one", workers=3, text="abcdef" * 15):
    calls = pool.session(namespace)
    digest = distill_session_segmented(pool.backends[0], "project", "2026-09-06", text,
                                      30, 0, 20, calls=calls, workers=workers)
    return digest, calls


class ParallelDistillTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.cache = self.root / "distillation-cache.db"

    def test_segments_overlap_in_time_and_reduce_in_source_order(self):
        barrier = threading.Barrier(3)
        finished = []

        def model(prompt):
            match = re.match(r"Segment (\d+) of", prompt)
            if match:
                idx = int(match[1])
                barrier.wait(timeout=3)  # serial execution cannot pass
                time.sleep((3 - idx) * 0.02)
                finished.append(idx)
                return response(f"summary-{idx}")
            self.assertEqual(finished, [3, 2, 1])
            self.assertLess(prompt.index("summary-1"), prompt.index("summary-2"))
            self.assertLess(prompt.index("summary-2"), prompt.index("summary-3"))
            return response("Merged in transcript order")

        with DistillationPool(Backend(fn=model), workers=3) as pool:
            digest, _ = segmented(pool)
        self.assertEqual(digest.summary, "Merged in transcript order")

    def test_shared_cap_covers_maps_single_sessions_reducers_and_fallback(self):
        lock = threading.Lock()
        active = peak = 0
        rendezvous = threading.Event()

        def model(prompt, fail=False):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 3:
                    rendezvous.set()
            try:
                self.assertTrue(rendezvous.wait(3))
                time.sleep(0.005)
                if fail:
                    raise RuntimeError("primary unavailable")
                return response()
            finally:
                with lock:
                    active -= 1

        primary = Backend(fn=lambda p: model(p, True))
        fallback = Backend("fallback", model)
        with DistillationPool(primary, fallback, workers=3) as pool, ThreadPoolExecutor(5) as sessions:
            futures = [sessions.submit(segmented, pool, namespace=f"codex:{i}", workers=4)
                       for i in range(4)]
            single = sessions.submit(distill_session, primary, "project", "date", "small",
                                     calls=pool.session("small"))
            for future in futures:
                self.assertTrue(future.result(timeout=5)[1].fallback_used)
            self.assertTrue(single.result(timeout=5).summary)
        self.assertEqual(peak, 3)
        self.assertEqual(active, 0)

    def test_only_failed_segment_uses_fallback_and_restart_reuses_all_successes(self):
        def model(prompt):
            if prompt.startswith("Segment 2 of"):
                raise RuntimeError("temporary transport failure")
            return response()

        primary, fallback = Backend(fn=model), Backend("fallback")
        with DistillationPool(primary, fallback, workers=3, cache_path=self.cache) as pool:
            _, calls = segmented(pool)
            self.assertTrue(calls.fallback_used)
        self.assertEqual(len(primary.calls), 4)  # three maps and the reducer
        self.assertEqual(len(fallback.calls), 1)
        self.assertTrue(fallback.calls[0].startswith("Segment 2 of"))
        with DistillationPool(primary, fallback, workers=3, cache_path=self.cache) as pool:
            _, calls = segmented(pool)
            self.assertTrue(calls.fallback_used)
        self.assertEqual((len(primary.calls), len(fallback.calls)), (4, 1))

    def test_failed_reduce_resumes_from_saved_maps_after_reopening(self):
        def fail_reduce(prompt):
            if not prompt.startswith("Segment "):
                raise RuntimeError("reduce interrupted")
            return response()

        backend = Backend(fn=fail_reduce)
        with DistillationPool(backend, workers=3, cache_path=self.cache) as pool:
            with self.assertRaisesRegex(RuntimeError, "reduce failed"):
                segmented(pool)
        backend.fn = None
        with DistillationPool(backend, workers=3, cache_path=self.cache) as pool:
            segmented(pool)
        self.assertEqual(sum(p.startswith("Segment ") for p in backend.calls), 3)
        self.assertEqual(len(backend.calls), 5)

    def test_failed_map_does_not_reduce_and_completed_siblings_survive(self):
        barrier = threading.Barrier(3)

        def fail_one(prompt):
            self.assertTrue(prompt.startswith("Segment "))
            barrier.wait(timeout=3)
            if prompt.startswith("Segment 2 of"):
                raise RuntimeError("interrupted")
            return response()

        backend = Backend(fn=fail_one)
        with DistillationPool(backend, workers=3, cache_path=self.cache) as pool:
            with self.assertRaisesRegex(RuntimeError, "segment 2/3 failed"):
                segmented(pool)
        backend.fn = None
        with DistillationPool(backend, workers=3, cache_path=self.cache) as pool:
            segmented(pool)
        maps = Counter(re.match(r"Segment (\d+)", p)[1] for p in backend.calls if p.startswith("Segment "))
        self.assertEqual(maps, {"1": 1, "2": 2, "3": 1})

    def test_checkpoint_invalidates_for_prompt_system_provider_and_namespace(self):
        backend = Backend()
        with DistillationPool(backend, workers=2, cache_path=self.cache) as pool:
            def call(namespace="one", system="system", prompt="prompt"):
                return pool.session(namespace).submit(system, prompt, "session").result()
            call()
            call()
            self.assertEqual(len(backend.calls), 1)
            call(prompt="changed content")
            call(system="new instructions")
            call(namespace="different session")
            backend.cache_identity = "changed-model-and-effort"
            call()
        self.assertEqual(len(backend.calls), 5)
        with DistillationPool(backend, workers=2, cache_path=self.root / "other" / "cache.db") as pool:
            pool.session("one").submit("system", "prompt", "session").result()
        self.assertEqual(len(backend.calls), 6)

    def test_empty_or_malformed_success_is_not_checkpointed(self):
        backend = Backend(fn=lambda _: response(""))
        for bad in (response(""), '{"summary":"a", "memories":[null]}'):
            backend.fn = lambda _, result=bad: result
            with DistillationPool(backend, workers=1, cache_path=self.cache) as pool:
                with self.assertRaises(RuntimeError):
                    pool.session("one").submit("system", bad, "segment 1/1").result()
            backend.fn = None
            with DistillationPool(backend, workers=1, cache_path=self.cache) as pool:
                self.assertTrue(pool.session("one").submit("system", bad, "segment 1/1").result()["summary"])
        self.assertEqual(len(backend.calls), 4)

    def test_primary_rate_limit_cooldown_is_shared_between_sessions(self):
        primary = Backend(fn=lambda _: (_ for _ in ()).throw(RuntimeError("Codex request was rate limited")))
        fallback = Backend("fallback")
        with DistillationPool(primary, fallback, workers=2) as pool:
            pool.session("one").submit("system", "first", "session").result()
            pool.session("two").submit("system", "second", "session").result()
            self.assertEqual((len(primary.calls), len(fallback.calls)), (1, 2))
            with mock.patch("tess_memory.distill.time.monotonic", return_value=time.monotonic() + 61):
                pool.session("three").submit("system", "third", "session").result()
        self.assertEqual((len(primary.calls), len(fallback.calls)), (2, 3))

    def test_per_session_window_is_bounded_even_when_global_pool_is_larger(self):
        lock = threading.Lock()
        active = peak = 0

        def model(prompt):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return response()

        with DistillationPool(Backend(fn=model), workers=8) as pool:
            segmented(pool, workers=2, text="abcdef" * 50)
        self.assertEqual(peak, 2)

    def test_codex_replaces_finished_session_without_waiting_for_slow_neighbor(self):
        home = self.root / "codex"
        paths = [write_rollout(home, str(i)) for i in range(3)]
        cfg = Config(data_stores=[DataStore("codex", home, type="codex")],
                     data_dir=self.root / "memory", codex=CodexSettings(batch_size=2), llm_workers=2)
        third_started = threading.Event()
        calls = 0
        lock = threading.Lock()

        def model(prompt):
            nonlocal calls
            with lock:
                index = calls
                calls += 1
            if index == 0:
                self.assertTrue(third_started.wait(3), "batch barrier prevented the third session from starting")
            if index == 2:
                third_started.set()
            return response()

        with closing(MemoryStore(cfg.db_path)) as store:
            stats = ingest(cfg, store, RecordingEmbedder(), Backend(fn=model), refresh_aggregates=False)
            self.assertEqual(stats.distilled, len(paths))
            self.assertEqual(stats.errors, 0)

    def test_parallel_ingest_stores_fallback_status_and_default_instances_create_no_cache(self):
        home = self.root / "codex"
        write_rollout(home)
        cfg = Config(data_stores=[DataStore("codex", home, type="codex")], data_dir=self.root / "memory",
                     llm_workers=3, segment_workers=3, distillation_cache=True,
                     max_transcript_chars=300, segment_chars=300, segment_overlap=0)
        primary = Backend(fn=lambda _: (_ for _ in ()).throw(RuntimeError("offline primary")))
        with closing(MemoryStore(cfg.db_path)) as store:
            stats = ingest(cfg, store, RecordingEmbedder(), primary, Backend("fallback"), refresh_aggregates=False)
            self.assertEqual((stats.distilled, stats.fallback, stats.errors), (1, 1, 0))
            self.assertEqual(store.file_status(str(home / "sessions/rollout-one.jsonl")), "fallback")
        self.assertTrue((cfg.data_dir / "distillation-cache.db").is_file())
        cfg.data_dir = self.root / "legacy-memory"
        cfg.segment_workers = 1
        cfg.distillation_cache = False
        with closing(MemoryStore(cfg.db_path)) as store:
            self.assertEqual(ingest(cfg, store, RecordingEmbedder(), Backend(), refresh_aggregates=False).distilled, 1)
        self.assertFalse((cfg.data_dir / "distillation-cache.db").exists())

    def test_config_defaults_and_validation(self):
        path = self.root / "settings.toml"
        path.write_text("")
        self.assertEqual((load_config(path).segment_workers, load_config(path).distillation_cache), (1, False))
        path.write_text("[ingest]\nsegment_workers=16\ndistillation_cache=true\n")
        self.assertEqual((load_config(path).segment_workers, load_config(path).distillation_cache), (16, True))
        for entry in ("segment_workers=0", "segment_workers=-2", "segment_workers=true",
                      "segment_workers=1.5", 'distillation_cache="yes"', "distillation_cache=1"):
            path.write_text("[ingest]\n" + entry)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_embedding_failure_retries_using_request_checkpoints_without_certifying_source(self):
        home = self.root / "codex"
        path = write_rollout(home)
        cfg = Config(data_stores=[DataStore("codex", home, type="codex")], data_dir=self.root / "memory",
                     llm_workers=3, segment_workers=3, distillation_cache=True,
                     max_transcript_chars=300, segment_chars=300, segment_overlap=0)
        backend = Backend()
        embedder = RecordingEmbedder()
        with closing(MemoryStore(cfg.db_path)) as store:
            with mock.patch.object(embedder, "embed_passages", side_effect=RuntimeError("offline embeddings")):
                with self.assertRaisesRegex(RuntimeError, "offline embeddings"):
                    ingest(cfg, store, embedder, backend, refresh_aggregates=False)
            self.assertIsNone(store.get_session("codex:one"))
            self.assertIsNone(store.file_state(str(path)))
            calls_before_retry = len(backend.calls)
            self.assertEqual(ingest(cfg, store, embedder, backend, refresh_aggregates=False).distilled, 1)
            self.assertEqual(len(backend.calls), calls_before_retry)

    def test_parallel_replacement_failure_preserves_prior_paid_memory_and_remains_pending(self):
        home = self.root / "codex"
        path = write_rollout(home)
        cfg = Config(data_stores=[DataStore("codex", home, type="codex")], data_dir=self.root / "memory",
                     llm_workers=3, segment_workers=3, distillation_cache=True,
                     max_transcript_chars=300, segment_chars=300, segment_overlap=0)
        backend = Backend()
        embedder = RecordingEmbedder()
        with closing(MemoryStore(cfg.db_path)) as store:
            ingest(cfg, store, embedder, backend, refresh_aggregates=False)
            saved_session = dict(store.get_session("codex:one"))
            saved_state = store.file_state(str(path))
            append(path, wrapped("response_item", message("assistant", "New durable conclusion. " * 30), 50))
            backend.fn = lambda _: (_ for _ in ()).throw(RuntimeError("both providers offline"))
            failed = ingest(cfg, store, embedder, backend, Backend("fallback", backend.fn), refresh_aggregates=False)
            self.assertEqual((failed.preserved, failed.errors), (1, 1))
            self.assertEqual(dict(store.get_session("codex:one")), saved_session)
            self.assertEqual(store.file_state(str(path)), saved_state)
            backend.fn = None
            self.assertEqual(ingest(cfg, store, embedder, backend, refresh_aggregates=False).distilled, 1)


if __name__ == "__main__":
    unittest.main()
