"""Image records: recorded by a host, found by the one search, and never
touched when a session is ingested again. Temp stores, a synthetic embedder."""
from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tess_memory import images, retrieval
from tess_memory.store import MemoryRow, MemoryStore

ROOFTOP = "A night photograph on a rooftop terrace. A woman in a red dress laughs beside a man in a navy sweater; city towers glow behind them."
YACHT = "A daytime photograph on a white yacht. A man in a navy sweater sits on the deck with open sea behind him."


class Embedder:
    """Two dimensions: how much a text is about a rooftop, and about a yacht."""
    descriptor = "test:images:2"

    def _vector(self, text):
        low = text.lower()
        return np.array([1.0 if ("rooftop" in low or "terrace" in low) else 0.05,
                         1.0 if ("yacht" in low or "boat" in low) else 0.05], dtype=np.float32)

    def embed_passages(self, texts):
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, query):
        return self._vector(query)


class ImageRecordTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "memory.db"
        self.store = MemoryStore(self.path)
        self.addCleanup(lambda: self.store.close())
        self.embedder = Embedder()
        from tess_memory.config import Config
        patched = mock.patch.object(retrieval, "CONFIG", Config(data_dir=Path(tmp.name)))  # search reads its weights from the settings
        patched.start()
        self.addCleanup(patched.stop)

    def session(self, session_id="s-trip", project="assistant"):
        self.store.upsert_session(id=session_id, project=project, title="Summer trip", summary="Photos from the trip.", outcome="Shared.",
                                  started_at="2026-07-04T20:00:00", ended_at="2026-07-04T23:00:00")
        self.store.db.commit()

    def record(self, ref="img-rooftop", description=ROOFTOP, **fields):
        return images.record_image(self.store, self.embedder, ref=ref, description=description, **fields)

    def count(self, sql):
        return self.store.db.execute(sql).fetchone()[0]

    def test_a_record_is_kept_whole_and_found_with_its_id_path_and_session(self):
        self.session()
        saved = self.record(session_id="s-trip", path="/home/you/.agent/images/img-rooftop.webp", name="IMG_0042.jpg",
                            sent_at="2026-11-20T23:10:00+00:00", taken_at="2026-07-04T21:15:09-04:00",
                            facts={"camera": "T-1 Pro", "latitude": 43.65})
        self.record(ref="img-yacht", description=YACHT, project="assistant", path="/home/you/.agent/images/img-yacht.webp")
        self.assertEqual((saved["project"], saved["embedded"]), ("assistant", True))  # filed with its session
        self.assertEqual(images.get_image(self.store, "img-rooftop")["facts"], {"camera": "T-1 Pro", "latitude": 43.65})

        hits = retrieval.search(self.store, self.embedder, "woman in a red dress on a rooftop", levels=("image",))
        self.assertEqual(hits[0]["level"], "image")
        content = hits[0]["content"]
        for expected in ("image img-rooftop", "IMG_0042.jpg", "File: /home/you/.agent/images/img-rooftop.webp",
                         "Sent in session s-trip", "sent 2026-11-20", ROOFTOP):
            self.assertIn(expected, content)
        # By meaning as well as by words: nothing in the yacht's description says "boat".
        self.assertIn("img-yacht", retrieval.search(self.store, self.embedder, "people on a boat", levels=("image",))[0]["content"])
        # And among everything else, when no level is asked for.
        self.assertTrue(any(hit["level"] == "image" for hit in retrieval.search(self.store, self.embedder, "red dress rooftop")))
        self.assertEqual([row["ref"] for row in self.store.images_for_session("s-trip")], ["img-rooftop"])

    def test_recording_the_same_image_again_replaces_it_and_removing_it_forgets_it(self):
        self.record(project="assistant")
        again = self.record(description="A corrected description: the same terrace at dusk, not night.", project="assistant")
        self.assertTrue(again["description"].startswith("A corrected description"))
        rows = self.store.db.execute(
            "SELECT m.content, e.vector FROM memories m LEFT JOIN embeddings e ON e.memory_id=m.id WHERE m.image_ref='img-rooftop'").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertTrue("at dusk" in rows[0]["content"] and "red dress" not in rows[0]["content"] and rows[0]["vector"] is not None)
        self.assertEqual(self.count("SELECT COUNT(*) FROM image_records"), 1)

        self.assertTrue(images.remove_image(self.store, "img-rooftop"))
        self.assertFalse(images.remove_image(self.store, "img-rooftop"))
        self.assertIsNone(images.get_image(self.store, "img-rooftop"))
        self.assertEqual(self.count("SELECT COUNT(*) FROM memories WHERE level='image'"), 0)
        self.assertEqual(self.count("SELECT COUNT(*) FROM embeddings"), 0)

    def test_ingesting_its_session_again_never_touches_an_image_record(self):
        self.session()
        self.record(session_id="s-trip")
        with self.store.write_transaction():
            self.store.replace_session_rows("s-trip", [
                MemoryRow(level="memory", kind="fact", content="They went up to the roof.", session_id="s-trip", project="assistant")])
        with self.store.write_transaction():
            self.store.delete_session_memories("s-trip")
        self.assertIsNotNone(images.get_image(self.store, "img-rooftop"))
        self.assertEqual(len(retrieval.search(self.store, self.embedder, "rooftop terrace", levels=("image",))), 1)

    def test_what_cannot_be_recorded_is_refused_and_a_failed_embedding_keeps_the_record(self):
        for fields in ({"ref": "", "description": ROOFTOP}, {"ref": "two words", "description": ROOFTOP}, {"ref": "ok", "description": "   "},
                       {"ref": "ok", "description": ROOFTOP, "facts": {"blob": "x" * 70_000}}):
            with self.assertRaises(ValueError):
                images.record_image(self.store, self.embedder, **fields)
        self.assertEqual(self.count("SELECT COUNT(*) FROM image_records"), 0)

        failing = mock.Mock(descriptor=self.embedder.descriptor)
        failing.embed_passages.side_effect = RuntimeError("synthetic: the embedding endpoint is down")
        with contextlib.redirect_stderr(io.StringIO()):
            saved = images.record_image(self.store, failing, ref="img-yacht", description=YACHT)
        self.assertFalse(saved["embedded"])
        # Still found by its words.
        self.assertIn("img-yacht", self.store.db.execute("SELECT content FROM memories WHERE image_ref='img-yacht'").fetchone()["content"])

    def test_a_host_applies_operations_in_order_and_one_refusal_does_not_stop_the_rest(self):
        results = images.apply(self.store, self.embedder, [
            {"op": "record", "ref": "img-rooftop", "description": ROOFTOP, "project": "assistant"},
            {"op": "record", "ref": "bad ref", "description": ROOFTOP},
            {"op": "paint", "ref": "img-x"},
            {"op": "record", "ref": "img-yacht", "description": YACHT, "facts": ["not", "an", "object"]},
            {"op": "remove", "ref": "img-rooftop"},
            {"op": "remove", "ref": "never-recorded"},
        ])
        self.assertEqual([result["ok"] for result in results], [True, False, False, False, True, True])
        self.assertEqual((results[0]["project"], results[0]["embedded"], results[4]["removed"], results[5]["removed"]),
                         ("assistant", True, True, False))
        self.assertEqual(self.count("SELECT COUNT(*) FROM image_records"), 0)

    def test_a_store_from_before_image_records_gains_them_when_opened(self):
        self.store.close()
        db = sqlite3.connect(self.path)
        db.executescript("DROP INDEX idx_memories_image; ALTER TABLE memories DROP COLUMN image_ref; DROP TABLE image_records;")
        db.close()
        self.store = MemoryStore(self.path)
        self.assertTrue(self.record(project="assistant")["embedded"])

    def test_the_command_line_applies_a_hosts_lines_and_prints_a_result_each(self):
        from tess_memory import cli

        self.store.close()
        lines = (json.dumps({"op": "record", "ref": "img-rooftop", "description": ROOFTOP, "project": "assistant"}) + "\n\n"
                 + json.dumps({"op": "remove", "ref": "nothing"}) + "\n")
        out = io.StringIO()
        config = mock.Mock(db_path=self.path)
        with mock.patch.object(cli, "CONFIG", config), mock.patch("tess_memory.embed.make_embedder", return_value=self.embedder), \
                mock.patch("sys.stdin", io.StringIO(lines)), contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["images", "apply"]), 0)
        printed = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(printed, [{"ref": "img-rooftop", "ok": True, "project": "assistant", "embedded": True},
                                   {"ref": "nothing", "ok": True, "removed": False}])
        self.store = MemoryStore(self.path)
        self.assertIsNotNone(images.get_image(self.store, "img-rooftop"))


if __name__ == "__main__":
    unittest.main()
