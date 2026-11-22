from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from tess_memory import plan
from tess_memory.store import MemoryRow, MemoryStore


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

    def test_two_writers_appending_to_one_item_both_keep_their_text(self):
        item = plan.add_item(self.a, Embedder(), project="repo", kind="task", title="Ship it", body="Goal: ship.")
        # Each connection read the item before the other wrote, as two agents would.
        self.a.plan_get(item["id"]), self.b.plan_get(item["id"])
        plan.update_item(self.a, Embedder(), item["id"], append="Claude: tests pass.")
        updated, changes = plan.update_item(self.b, Embedder(), item["id"], append="  Codex: deployed.\n", status="complete")
        self.assertEqual(updated["body"], "Goal: ship.\n\nClaude: tests pass.\n\nCodex: deployed.")
        self.assertEqual(changes, ["status planned -> complete", "body appended"])
        self.assertIn("Codex: deployed.", self.a.db.execute("SELECT content FROM memories WHERE plan_id=?", (item["id"],)).fetchone()[0])
        empty = plan.add_item(self.a, Embedder(), project="repo", kind="note", title="Scratch")
        self.assertEqual(plan.update_item(self.a, Embedder(), empty["id"], append="First line.")[0]["body"], "First line.")
        self.assertEqual(plan.update_item(self.a, Embedder(), empty["id"], append="   ")[1], [])
        with self.assertRaisesRegex(ValueError, "not both"):
            plan.update_item(self.a, Embedder(), item["id"], body="Replaced.", append="Added.")

    def test_a_note_follows_the_work_it_is_filed_under(self):
        def add(kind, title, parent=None):
            return plan.add_item(self.a, Embedder(), project="repo", kind=kind, title=title, parent_id=parent)["id"]
        def done(item):
            plan.update_item(self.a, Embedder(), item, status="complete")
        epic = add("epic", "Release")
        shipped, open_slice = add("slice", "Wizard", epic), add("slice", "Website", epic)
        record, reminder, standalone = add("note", "Ran it on a second Mac", shipped), add("note", "Check auth first", open_slice), add("note", "Ideas")
        done(shipped)
        board = plan.render_board(self.a, "repo")
        # The record went with the finished slice; the note under open work and the one at the top stay.
        self.assertEqual([line.strip().split("]")[1].strip() for line in board.splitlines()[1:-1]],
                         ["Release", "Website", "Check auth first", "Ideas"])
        self.assertEqual(board.splitlines()[0], "repo — 4 planned, 1 complete")
        self.assertEqual(board.splitlines()[-1], "(1 complete/dropped, 1 note under finished work hidden — plan_board include_done=true)")
        done(open_slice), done(epic)
        board = plan.render_board(self.a, "repo")
        self.assertEqual(board.splitlines(), ["repo — 1 planned, 3 complete", f"#{standalone} [note | planned | p3] Ideas",
                                              "(3 complete/dropped, 2 notes under finished work hidden — plan_board include_done=true)"])
        # Nothing stored changed: the notes are as they were left, and shown when everything is.
        everything = plan.render_board(self.a, "repo", include_done=True)
        self.assertIn(f"#{record} [note | planned | p3] Ran it on a second Mac", everything)
        self.assertEqual(everything.splitlines()[0], "repo — 3 planned, 3 complete")
        # A task filed under a note is still work, and keeps that note on the board.
        task = add("task", "Try the idea", record)
        self.assertIn("Ran it on a second Mac", plan.render_board(self.a, "repo"))
        done(task)
        self.assertNotIn("Ran it on a second Mac", plan.render_board(self.a, "repo"))

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
