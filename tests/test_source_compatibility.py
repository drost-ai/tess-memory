"""Existing sources retain their parsing, project ownership, and output policy."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tess_memory.attribution import source_adapter
from tess_memory.config import Config, DataStore
from tess_memory.ingest import ingest
from tess_memory.plan import add_item, resolve_project_policy
from tess_memory.store import MemoryStore
from tess_memory.transcript import render_for_llm


class LocalEmbedder:
    descriptor = "test:legacy-compatibility:4"

    def embed_passages(self, texts):
        return np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (len(texts), 1))


class ExistingSourceCompatibilityTests(unittest.TestCase):
    def test_claude_tess_and_assistant_keep_their_existing_behavior(self):
        for flavor in ["claude-code", "inline", "full-transcript", "python-sidecar", "transcript"]:
            with self.subTest(flavor=flavor), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                user = "Keep the existing project memory and planning state. " * 10
                answer = "The original integration continues to work. " * 10
                content = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
                source = root / "source"
                source.mkdir()
                if flavor == "claude-code":
                    project = source / "-work-claude"
                    project.mkdir()
                    (project / "session.jsonl").write_text("".join(json.dumps({
                        "type": m["role"], "message": {"content": m["content"]}, "cwd": "/work/claude",
                        "timestamp": "2026-08-01T00:00:00Z"}) + "\n" for m in content))
                    ds = DataStore("claude", source, type="claude-code")
                    expected = "-work-claude"
                elif flavor == "transcript":
                    (source / "session.jsonl").write_text("".join(json.dumps({"type": "message", **m,
                        "persisted_at": "2026-08-01T00:00:00Z"}) + "\n" for m in content))
                    ds = DataStore("assistant-export", source, type="transcript", project="assistant")
                    expected = "assistant"
                else:
                    meta = {"id": "session", "cwd": "/work/tess", "createdAt": "2026-08-01T00:00:00Z", "updatedAt": "2026-08-01T01:00:00Z"}
                    if flavor == "inline": meta["messages"] = content
                    (source / "session.json").write_text(json.dumps(meta))
                    if flavor == "full-transcript":
                        directory = source / "session"
                        directory.mkdir()
                        (directory / "full-transcript-with-tools.jsonl").write_text("".join(json.dumps({"type": "message", **m}) + "\n" for m in content))
                    if flavor == "python-sidecar":
                        (source / "session.messages.jsonl").write_text("".join(json.dumps({"type": "message", "message_index": i, "message": m}) + "\n" for i, m in enumerate(content)))
                    ds = DataStore("tess", source, type="tess-core", project="assistant" if flavor == "python-sidecar" else None)
                    expected = "assistant" if flavor == "python-sidecar" else "-work-tess"
                cfg = Config(data_stores=[ds], data_dir=root / "memory")
                store = MemoryStore(cfg.db_path)
                embedder = LocalEmbedder()
                try:
                    authored = add_item(store, embedder, project=expected, kind="task", title="Existing plan remains")
                    original_plan = dict(store.plan_get(authored["id"]))
                    found, parse = source_adapter(ds)
                    st = parse(found[0][0], ds.resolve_project(found[0][1]), min_chars=0)
                    self.assertEqual(st.source_type, "")
                    self.assertIsNone(st.content_fingerprint)
                    self.assertIn(user.strip(), render_for_llm(st, 100000, cfg.tool_result_snippet))
                    result = ingest(cfg, store, embedder, None)
                    self.assertEqual(result.extractive, 1)
                    self.assertEqual(store.get_session("session")["project"], expected)
                    self.assertEqual(dict(store.plan_get(authored["id"])), original_plan)
                    self.assertEqual(ingest(cfg, store, embedder, None).unchanged, 1)
                    self.assertIsNone(store.get_meta("codex:normalized:session"))
                    self.assertFalse((cfg.data_dir / "codex-ingest.lock").exists())
                    if expected == "assistant":
                        with self.assertRaises(ValueError):
                            resolve_project_policy("other", default_project="assistant", lock_writes_to_default=True, write=True)
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()


class RenamedSourceTypeTests(unittest.TestCase):
    """Stores written while engine sessions were called 'agent-core' sources."""

    def test_opening_a_store_relabels_recorded_engine_sources(self):
        import sqlite3, tempfile
        from tess_memory.store import MemoryStore
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.db"
            store = MemoryStore(path)
            store.record_source(path="/s/one.json", session_id="one", project="p", source_type="tess-core",
                                signature="sig", status="indexed", state={"indexed_units": ["u"]}, now=1.0, indexed=True)
            store.commit(); store.close()
            with sqlite3.connect(path) as db:
                db.execute("UPDATE ingest_sources SET source_type='agent' || '-core'")
            store = MemoryStore(path)
            self.assertEqual([r[0] for r in store.db.execute("SELECT source_type FROM ingest_sources")], ["tess-core"])
            self.assertEqual(json.loads(store.source_progress("/s/one.json")["state"]), {"indexed_units": ["u"]})
            store.close()

    def test_a_settings_file_with_the_old_type_says_what_to_change(self):
        from tess_memory.attribution import source_adapter
        from tess_memory.config import DataStore
        with self.assertRaisesRegex(ValueError, "is now 'tess-core'"):
            source_adapter(DataStore("tess", Path("/nowhere"), type="agent" + "-core"))
