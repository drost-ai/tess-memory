from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agentic_memory.source_journal import SourceChanged, SourceJournal, SourceReadError


class SourceJournalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / "source.jsonl"
        self.journal = SourceJournal(self.root / "cache.db")
        self.parsed = []

    def write(self, values):
        self.path.write_text(''.join(json.dumps({"n": v}) + '\n' for v in values))

    def append(self, value):
        with self.path.open('a') as f:
            f.write(json.dumps({"n": value}) + '\n')

    def read(self, **kwargs):
        def consume(state, record, position, start, end):
            state["values"].append(record["n"])
            self.parsed.append(record["n"])
        return self.journal.read_jsonl(self.path, "policy-v1", initial=lambda: {"values": []},
                                       consume=kwargs.pop("consume", consume), **kwargs)

    def test_append_and_restart_parse_only_new_records(self):
        self.write([1, 2])
        self.assertEqual(self.read()[0]["values"], [1, 2])
        self.journal = SourceJournal(self.root / "cache.db")
        self.assertEqual(self.read()[1].parsed_records, 0)
        self.append(3)
        state, revision = self.read()
        self.assertEqual(state["values"], [1, 2, 3])
        self.assertEqual(self.parsed, [1, 2, 3])
        self.assertEqual(revision.parsed_records, 1)
        self.assertTrue(revision.verify())

    def test_rewrite_and_truncation_reset_the_cached_reader(self):
        self.write([1, 2])
        self.read()
        self.write([8, 2, 3])
        self.assertEqual(self.read()[0]["values"], [8, 2, 3])
        self.write([4])
        self.assertEqual(self.read()[0]["values"], [4])

    def test_partial_tail_is_pending_and_never_advances_cursor(self):
        self.write([1])
        boundary = self.path.stat().st_size
        with self.path.open('ab') as f:
            f.write(b'{"n":')
        state, revision = self.read()
        self.assertTrue(revision.partial_record)
        self.assertEqual(revision.offset, boundary)
        self.assertEqual(state["values"], [1])
        with self.path.open('ab') as f:
            f.write(b'2}\n')
        self.assertEqual(self.read()[0]["values"], [1, 2])
        self.assertEqual(self.parsed, [1, 2])

    def test_complete_invalid_record_fails_without_overwriting_good_cache(self):
        self.write([1])
        self.read()
        self.path.write_text(self.path.read_text() + '{invalid}\n')
        with self.assertRaises(SourceReadError):
            self.read()
        self.write([1, 2])
        self.assertEqual(self.read()[0]["values"], [1, 2])
        self.assertEqual(self.parsed, [1, 2])

    def test_append_during_parse_can_publish_verified_prefix(self):
        self.write([1])
        def consume(state, record, *_):
            state["values"].append(record["n"])
            self.append(2)
        state, revision = self.read(consume=consume)
        self.assertEqual(state["values"], [1])
        self.assertTrue(revision.verify())
        self.assertEqual(self.read()[0]["values"], [1, 2])

    def test_rewrite_during_parse_or_after_model_work_invalidates_revision(self):
        self.write([1])
        def consume(state, record, *_):
            state["values"].append(record["n"])
            self.write([2])
        with self.assertRaises(SourceChanged):
            self.read(consume=consume)
        _, revision = self.read()
        self.write([3])
        self.assertFalse(revision.verify())

    def test_boundaries_do_not_leak_parent_future_and_missing_ordinals_fail(self):
        self.path.write_text(''.join(json.dumps({"n": i, "ordinal": i}) + '\n' for i in range(4)))
        self.assertEqual(self.read(end_ordinal=2)[0]["values"], [0, 1])
        self.path.write_text(self.path.read_text() + '{"n":4,"ordinal":4}\n')
        self.assertEqual(self.read(end_ordinal=2)[0]["values"], [0, 1])
        self.write([1])
        with self.assertRaises(SourceReadError):
            self.read(end_ordinal=2)

    def test_snapshot_requires_whole_valid_object_and_detects_replacement(self):
        path = self.root / "snapshot.json"
        path.write_text('{"n":1}')
        state, first = self.journal.read_snapshot(path, "v1", lambda value: value)
        self.assertEqual(state, {"n": 1})
        self.assertEqual(self.journal.read_snapshot(path, "v1", lambda _: self.fail())[1].parsed_records, 0)
        path.write_text('{"n":')
        with self.assertRaises(SourceReadError):
            self.journal.read_snapshot(path, "v1", lambda value: value)
        self.assertFalse(first.verify())


if __name__ == "__main__":
    unittest.main()
