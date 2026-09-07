from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from agentic_memory import plan
from agentic_memory.store import MemoryRow, MemoryStore


class Embedder:
    descriptor = "test:shared-store:2"

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector

    def embed_passages(self, texts):
        return np.tile(np.asarray(self.vector, dtype=np.float32), (len(texts), 1))


class BlockingEmbedder(Embedder):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed_passages(self, texts):
        self.started.set()
        if not self.release.wait(5):
            raise RuntimeError("test did not release the embedding request")
        return super().embed_passages(texts)


class SharedStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "memory.db"
        self.a, self.b = MemoryStore(path), MemoryStore(path)
        self.addCleanup(self.a.close)
        self.addCleanup(self.b.close)

    def test_cache_tracks_other_connections_even_with_unchanged_row_count(self):
        self.a.add_memories([MemoryRow("memory", "fact", "first", vector=np.asarray([1, 0]))])
        self.a.commit()
        ids, matrix = self.b.embedding_matrix()
        np.testing.assert_array_equal(matrix, [[1, 0]])
        self.a.delete_scoped("memory")
        self.a.add_memories([MemoryRow("memory", "fact", "replacement", vector=np.asarray([0, 1]))])
        self.a.commit()
        new_ids, matrix = self.b.embedding_matrix()
        self.assertEqual(len(ids), len(new_ids))
        np.testing.assert_array_equal(matrix, [[0, 1]])
        self.a.delete_scoped("memory")
        self.a.commit()
        self.assertEqual(len(self.b.embedding_matrix()[0]), 0)

    def test_empty_cache_sees_ingestion_and_local_writes_still_invalidate(self):
        self.assertEqual(len(self.b.embedding_matrix()[0]), 0)
        self.a.add_memories([MemoryRow("memory", "fact", "new", vector=np.asarray([1, 0]))])
        self.a.commit()
        self.assertEqual(len(self.b.embedding_matrix()[0]), 1)
        self.b.add_memories([MemoryRow("memory", "fact", "local", vector=np.asarray([0, 1]))])
        self.b.commit()
        self.assertEqual(len(self.b.embedding_matrix()[0]), 2)

    def test_slow_embedding_does_not_hold_the_database_write_lock(self):
        slow = BlockingEmbedder()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(plan.add_item, self.a, slow, project="repo", kind="task", title="First task")
            try:
                self.assertTrue(slow.started.wait(2))
                self.assertTrue(self.b.fts_search("First"))
                second = pool.submit(plan.add_item, self.b, Embedder(), project="repo", kind="task", title="Second task")
                self.assertEqual(second.result(timeout=2)["title"], "Second task")
            finally:
                slow.release.set()
            future.result(timeout=2)

    def test_late_embedding_cannot_overwrite_a_concurrent_edit(self):
        slow = BlockingEmbedder()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(plan.add_item, self.a, slow, project="repo", kind="task", title="Old title")
            try:
                self.assertTrue(slow.started.wait(2))
                item = self.b.plan_items_for("repo")[0]
                plan.update_item(self.b, Embedder((0, 1)), item["id"], title="New title")
            finally:
                slow.release.set()
            future.result(timeout=2)
        row = self.b.db.execute("SELECT m.content,e.vector FROM memories m JOIN embeddings e ON e.memory_id=m.id WHERE m.plan_id=?", (item["id"],)).fetchone()
        self.assertIn("New title", row["content"])
        np.testing.assert_array_equal(np.frombuffer(row["vector"], dtype=np.float32), [0, 1])

    def test_write_failure_rolls_back_plan_and_history_without_leaving_a_lock(self):
        with mock.patch.object(self.a, "plan_log", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                plan.add_item(self.a, Embedder(), project="repo", kind="task", title="Not committed")
        self.assertFalse(self.a.db.in_transaction)
        self.assertEqual(self.b.plan_items_for("repo"), [])
        self.assertEqual(self.b.db.execute("SELECT COUNT(*) FROM plan_events").fetchone()[0], 0)
        plan.add_item(self.b, Embedder(), project="repo", kind="task", title="Still writable")

    def test_embedding_failure_preserves_keyword_memory_and_plan_state(self):
        failing = Embedder()
        failing.embed_passages = mock.Mock(side_effect=RuntimeError("offline"))
        item = plan.add_item(self.a, failing, project="repo", kind="task", title="Durable keyword item")
        self.assertEqual(self.b.plan_get(item["id"])["title"], "Durable keyword item")
        self.assertTrue(self.b.fts_search("Durable"))
        self.assertEqual(len(self.b.embedding_matrix()[0]), 0)


if __name__ == "__main__":
    unittest.main()
