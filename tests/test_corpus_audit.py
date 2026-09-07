from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from agentic_memory.audit import activity_state, inventory
from agentic_memory.config import Config, DataStore
from agentic_memory.evaluation import evaluate
from agentic_memory.store import MemoryRow, MemoryStore
from test_codex import append, message, wrapped, write_rollout
from agentic_memory.codex import CodexCatalog, CodexParseError


class CorpusAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_inventory_counts_body_files_and_does_not_create_missing_database(self):
        home = self.root / "source"
        home.mkdir()
        meta = home / "one.json"
        meta.write_text(json.dumps({"id": "one", "cwd": "/work/one"}))
        body = home / "one.messages.jsonl"
        body.write_text('{"type":"message","message":{"role":"user","content":"private content"}}\n')
        cfg = Config(data_stores=[DataStore("test", home, type="agent-core")], data_dir=self.root / "memory")
        result = inventory(cfg)
        self.assertFalse(cfg.data_dir.exists())
        self.assertEqual(result["sources"][0]["source_bytes"], meta.stat().st_size + body.stat().st_size)
        self.assertNotIn("private content", json.dumps(result))

    def test_unlabelled_assistant_is_not_falsely_reported_as_complete_or_running(self):
        path = self.root / "session.jsonl"
        path.write_text('{"type":"message","role":"user","content":"hello"}\n'
                        '{"type":"message","role":"assistant","content":"reply"}\n')
        self.assertEqual(activity_state(path, "transcript"), "unknown")
        path.write_text('{"type":"assistant","message":{"role":"assistant","stop_reason":"end_turn"}}\n')
        self.assertEqual(activity_state(path, "claude-code"), "complete")

    def test_previously_unknown_codex_records_preserve_evidence_and_remove_opaque_state(self):
        home = self.root / "codex"
        path = write_rollout(home)
        append(path, wrapped("compacted", {"replacement_history": [
            {"type": "compaction_summary", "encrypted_content": "opaque-secret"}]}))
        append(path, wrapped("event_msg", {"type": "error", "message": "transport failed",
                                           "codex_error_info": "http_connection"}))
        append(path, wrapped("response_item", {"type": "function_call", "name": "view_image",
                                               "call_id": "image1", "arguments": '{"path":"/tmp/plot.png"}'}))
        append(path, wrapped("event_msg", {"type": "view_image_tool_call", "call_id": "image1",
                                           "path": "/tmp/plot.png"}))
        st, report = CodexCatalog([DataStore("c", home, type="codex")]).inspect(path)
        text = "\n".join(t.text for t in st.turns)
        self.assertNotIn("opaque-secret", text)
        self.assertIn("transport failed", text)
        self.assertIn("http_connection", text)
        self.assertEqual(text.count("/tmp/plot.png"), 1)
        append(path, wrapped("event_msg", {"type": "future_unknown"}))
        with self.assertRaises(CodexParseError):
            CodexCatalog([DataStore("c", home, type="codex")]).inspect(path)

    def test_readonly_store_rejects_writes_without_migration(self):
        db = self.root / "memory.db"
        with closing(MemoryStore(db)) as store:
            store.upsert_session(id="one", project="project")
        before = db.read_bytes()
        with closing(MemoryStore(db, readonly=True)) as store:
            self.assertEqual(store.get_session("one")["project"], "project")
            with self.assertRaises(sqlite3.OperationalError):
                store.upsert_session(id="two", project="project")
        self.assertEqual(db.read_bytes(), before)

    def test_evaluation_uses_source_labels_and_checks_cached_vector_identity(self):
        cfg_file = self.root / "settings.toml"
        cfg_file.write_text('[storage]\ndata_dir="' + str(self.root) + '"\n')
        with closing(MemoryStore(self.root / "memory.db")) as store:
            store.set_meta("embedder", "synthetic:2d")
            store.upsert_session(id="one", project="p")
            mid = store.add_memories([MemoryRow(level="memory", kind="decision", project="p", session_id="one",
                                               content="Atlas selected SQLite", vector=np.array([1, 0], dtype=np.float32))])[0]
        query = "Which database did Atlas select?"
        cases = [{"id": "positive", "settings": str(cfg_file), "query": query, "project": "p",
                  "expected_memory_ids": [mid]},
                 {"id": "negative", "settings": str(cfg_file), "query": "missing", "project": "absent"}]
        path = self.root / "cases.json"
        path.write_text(json.dumps(cases))
        vectors = self.root / "queries.npz"
        meta = {"positive": {"query_sha256": hashlib.sha256(query.encode()).hexdigest(), "descriptor": "synthetic:2d"}}
        np.savez(vectors, positive=np.array([1, 0], dtype=np.float32), __metadata__=json.dumps(meta))
        report = evaluate(path, vectors_path=vectors)
        self.assertEqual((report["cases"], report["passed"]), (2, 2))
        self.assertNotIn(query, json.dumps(report))
        cases[0]["query"] = "different question"
        path.write_text(json.dumps(cases))
        with self.assertRaisesRegex(ValueError, "embedding space"):
            evaluate(path, vectors_path=vectors)

    def test_evaluation_applies_temporal_bounds_in_keyword_and_hybrid_modes(self):
        cfg_file = self.root / 'settings.toml'
        cfg_file.write_text('[storage]\ndata_dir="' + str(self.root) + '"\n')
        with closing(MemoryStore(self.root / 'memory.db')) as store:
            store.set_meta('embedder', 'synthetic:2d')
            ids = store.add_memories([MemoryRow(level='memory', kind='decision', project='p',
                                               content='Atlas cache choice', ts=stamp,
                                               vector=np.array([1, 0], dtype=np.float32))
                                      for stamp in ('2026-01-01', '2026-09-01')])
        query = 'Atlas cache choice'
        cases = [{'id': 'old', 'before': '2026-02-01', 'expected_memory_ids': [ids[0]]},
                 {'id': 'new', 'after': '2026-08-01', 'expected_memory_ids': [ids[1]]}]
        metadata = {}
        for case in cases:
            case.update(settings=str(cfg_file), query=query, project='p')
            metadata[case['id']] = {'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
                                    'descriptor': 'synthetic:2d'}
        path = self.root/'cases.json'
        path.write_text(json.dumps(cases))
        vectors = self.root/'queries.npz'
        np.savez(vectors, old=np.array([1, 0]), new=np.array([1, 0]), __metadata__=json.dumps(metadata))
        for keyword_only in (True, False):
            report = evaluate(path, vectors_path=vectors, keyword_only=keyword_only, limit=1)
            self.assertEqual(report['passed'], 2)
            self.assertEqual([r['returned_memory_ids'] for r in report['results']], [[ids[0]], [ids[1]]])


if __name__ == "__main__":
    unittest.main()
