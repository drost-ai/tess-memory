from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sqlite3
import zlib

from tess_memory.config import Config, DataStore
from tess_memory.readers import CompletedSourceReader
from tess_memory.source_journal import SourceReadError
from test_codex import append, message, wrapped, write_rollout


class IncrementalReaderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def reader(self, kind="codex"):
        self.home = self.root / kind
        self.home.mkdir(exist_ok=True)
        self.ds = DataStore("test", self.home, type=kind)
        self.cfg = Config(data_stores=[self.ds], data_dir=self.root / (kind + "-memory"))
        return CompletedSourceReader(self.cfg)

    def test_codex_keeps_published_prefix_while_next_explicit_turn_is_running(self):
        self.reader()
        path = write_rollout(self.home)
        reader = CompletedSourceReader(self.cfg)
        first = reader.read(self.ds, path, "p")
        self.assertEqual(len(first.unit_keys()), 1)
        self.assertEqual(CompletedSourceReader(self.cfg).read(self.ds, path, "p").parsed_records, 0)
        append(path, wrapped("event_msg", {"type": "task_started", "turn_id": "next"}))
        append(path, wrapped("response_item", message("user", "New pending request")))
        second = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        self.assertEqual(second.unit_keys(), first.unit_keys())
        self.assertTrue(second.pending_tail)
        append(path, wrapped("response_item", message("assistant", "New complete answer")))
        append(path, wrapped("event_msg", {"type": "task_complete", "turn_id": "next"}))
        third = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        self.assertEqual(third.unit_keys()[:1], first.unit_keys())
        self.assertEqual(len(third.unit_keys()), 2)
        self.assertEqual(third.parsed_records, 2)
        self.assertFalse(third.pending_tail)

    def test_codex_identity_updates_after_restart_change_the_correct_old_unit(self):
        self.reader()
        path = write_rollout(self.home, items=[message("user", "User request", id="u"),
                                              message("assistant", "First answer", id="a")])
        first = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        append(path, wrapped("event_msg", {"type": "agent_message", "message": "First answer"}))
        CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        append(path, wrapped("response_item", message("assistant", "Corrected answer", id="a")))
        last = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        self.assertNotEqual(last.unit_keys(), first.unit_keys())
        self.assertIn("Corrected answer", '\n'.join(t.text for t in last.transcript.turns))

    def test_codex_partial_record_keeps_complete_prefix(self):
        self.reader()
        path = write_rollout(self.home)
        with path.open('ab') as f:
            f.write(b'{"type":')
        result = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        self.assertTrue(result.pending_tail)
        self.assertEqual(len(result.unit_keys()), 1)

    def test_realtime_session_close_publishes_spoken_transcript(self):
        self.reader()
        path = write_rollout(self.home, items=[])
        append(path, wrapped('realtime_item', {'type':'realtime_session_started','id':'start'}))
        append(path, wrapped('realtime_item', {'type':'transcript_segment','role':'user','id':'user','text':'A spoken request'}))
        append(path, wrapped('realtime_item', {'type':'transcript_segment','role':'assistant','id':'answer','text':'A spoken answer'}))
        append(path, wrapped('realtime_item', {'type':'realtime_session_closed','id':'closed','outcome':'completed'}))
        result = CompletedSourceReader(self.cfg).read(self.ds, path, 'p')
        self.assertEqual(result.unit_kinds, ['realtime_closed'])
        self.assertEqual([t.text for t in result.transcript.turns], ['A spoken request','A spoken answer'])
        self.assertFalse(result.pending_tail)

    def test_claude_completed_reason_excludes_pending_tool_exchange(self):
        reader = self.reader("claude-code")
        path = self.home / "session.jsonl"
        records = [
            {"type": "user", "message": {"content": "First task"}},
            {"type": "assistant", "message": {"content": "Done", "stop_reason": "end_turn"}},
            {"type": "user", "message": {"content": "Next task"}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "pwd"}}], "stop_reason": "tool_use"}},
        ]
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        result = reader.read(self.ds, path, "p")
        self.assertEqual(result.unit_kinds, ["turn_complete"])
        self.assertTrue(result.pending_tail)
        self.assertNotIn("Next task", '\n'.join(t.text for t in result.transcript.turns))

    def test_claude_schema_attachment_does_not_break_cached_prefix_or_later_turns(self):
        reader = self.reader("claude-code")
        path = self.home / "session.jsonl"
        append(path, {"type": "user", "message": {"content": "Original request"}})
        append(path, {"type": "assistant", "message": {"content": "Original answer", "stop_reason": "end_turn"}})
        first = reader.read(self.ds, path, "p")
        append(path, {"type": "attachment", "attachment": {"tools": [{"schema": {
            "input_schema": {"properties": {
                "type": {"type": "string"}, "optional": {"type": ["string", "null"]},
            }},
        }}]}})
        append(path, {"type": "user", "message": {"content": "Next request"}})
        append(path, {"type": "assistant", "message": {"content": "Next answer", "stop_reason": "end_turn"}})
        second = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        self.assertEqual(second.unit_keys()[:1], first.unit_keys())
        self.assertEqual(second.unit_kinds, ["turn_complete", "turn_complete"])
        self.assertEqual([t.text for t in second.transcript.turns],
                         ["Original request", "Original answer", "Next request", "Next answer"])
        self.assertEqual(second.parsed_records, 3)
        cached = CompletedSourceReader(self.cfg).read(self.ds, path, "p")
        self.assertEqual(cached.parsed_records, 0)
        self.assertEqual(cached.unit_keys(), second.unit_keys())

    def test_tess_core_body_changes_without_metadata_change_are_seen_and_repair_applies(self):
        reader = self.reader("tess-core")
        path = self.home / "one.json"
        path.write_text(json.dumps({"id": "one", "cwd": "/work/p"}))
        body = self.home / "one/full-transcript-with-tools.jsonl"
        body.parent.mkdir()
        body.write_text(''.join(json.dumps({"type": "message", "message_index": i, "role": role, "content": text}) + '\n'
                                for i, (role, text) in enumerate([("user", "request"), ("assistant", "old answer")])))
        before = path.stat()
        first = reader.read(self.ds, path, "p")
        append(body, {"type": "repair", "truncated_to_message_index": 1})
        append(body, {"type": "message", "message_index": 1, "role": "assistant", "content": "new answer"})
        last = reader.read(self.ds, path, "p")
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertNotEqual(first.unit_keys(), last.unit_keys())
        self.assertIn("new answer", '\n'.join(t.text for t in last.transcript.turns))
        self.assertNotIn("old answer", '\n'.join(t.text for t in last.transcript.turns))

    def test_sidecar_uncommitted_tail_and_truncation_do_not_leak(self):
        reader = self.reader("tess-core")
        path = self.home / "one.json"
        path.write_text(json.dumps({"id": "one", "cwd": "/work/p"}))
        body = self.home / "one.messages.jsonl"
        body.write_text('')
        for i, (role, text) in enumerate([("user", "request"), ("assistant", "old")]):
            append(body, {"type": "message", "message_index": i, "message": {"role": role, "content": text}})
        append(body, {"type": "commit", "message_count": 2})
        first = reader.read(self.ds, path, "p")
        append(body, {"type": "truncate", "message_count": 1})
        append(body, {"type": "message", "message_index": 1, "message": {"role": "assistant", "content": "new"}})
        second = reader.read(self.ds, path, "p")
        self.assertEqual(first.unit_keys(), second.unit_keys())
        self.assertTrue(second.pending_tail)
        append(body, {"type": "commit", "message_count": 2})
        third = reader.read(self.ds, path, "p")
        self.assertNotEqual(first.unit_keys(), third.unit_keys())
        self.assertFalse(third.pending_tail)

    def test_inline_snapshot_strips_images_and_uses_fixed_project(self):
        reader = self.reader("tess-core")
        self.ds.project = "assistant"
        path = self.home / "one.json"
        path.write_text(json.dumps({"id": "one", "cwd": "/random/cwd", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "request"}, {"type": "image", "data": "private image"}]},
            {"role": "assistant", "content": "answer"}]}))
        result = reader.read(self.ds, path, "assistant")
        self.assertEqual(result.transcript.project, "assistant")
        self.assertEqual(len(result.unit_keys()), 1)
        with sqlite3.connect(reader.journal.path) as db:
            cached = '\n'.join(zlib.decompress(row[0]).decode() for row in db.execute("SELECT state FROM readers"))
        self.assertNotIn("private image", cached)


if __name__ == "__main__":
    unittest.main()
