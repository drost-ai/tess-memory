from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tess_memory import server
from tess_memory.config import Config
from tess_memory.transcript_reader import (
    MIN_PAGE_BYTES,
    get_session_source_path,
    read_transcript_page,
    resolve_transcript_path,
)


class TranscriptReaderTests(unittest.TestCase):
    def test_pages_reconstruct_utf8_source_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            source = "a" * 255 + "🔮" + "\n" + "β" * 300
            path.write_text(source, encoding="utf-8")

            parts: list[str] = []
            cursor = 0
            while True:
                page = read_transcript_page(
                    path, cursor=cursor, max_bytes=MIN_PAGE_BYTES
                )
                parts.append(page.content)
                if page.complete:
                    self.assertIsNone(page.next_cursor)
                    break
                self.assertIsNotNone(page.next_cursor)
                self.assertGreater(page.next_cursor, cursor)
                cursor = page.next_cursor

            self.assertEqual("".join(parts), source)

    def test_resolver_rejects_sources_outside_configured_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            allowed = base / "allowed"
            allowed.mkdir()
            outside = base / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(PermissionError):
                resolve_transcript_path(outside, [allowed])

    def test_resolver_prefers_agent_core_full_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "abc.json"
            metadata.write_text("{}", encoding="utf-8")
            full = root / "abc" / "full-transcript-with-tools.jsonl"
            full.parent.mkdir()
            full.write_text('{"type":"message"}\n', encoding="utf-8")

            self.assertEqual(resolve_transcript_path(metadata, [root]), full.resolve())

    def test_source_lookup_uses_existing_session_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "memory.db"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE sessions(id TEXT, source_path TEXT)")
            connection.execute(
                "INSERT INTO sessions VALUES(?, ?)",
                ("session-1", "/transcripts/session-1.jsonl"),
            )
            connection.commit()
            connection.close()

            self.assertEqual(
                get_session_source_path(db, "session-1"),
                "/transcripts/session-1.jsonl",
            )
            self.assertIsNone(get_session_source_path(db, "missing"))

    def test_mcp_endpoint_reads_existing_session_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "session-1.jsonl"
            transcript.write_text('{"type":"message","role":"user"}\n', encoding="utf-8")
            db = root / "memory.db"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE sessions(id TEXT, source_path TEXT)")
            connection.execute(
                "INSERT INTO sessions VALUES(?, ?)", ("session-1", str(transcript))
            )
            connection.commit()
            connection.close()

            class FakeDataStore:
                path = root

            with mock.patch.object(server, "CONFIG", Config(data_dir=root, data_stores=[FakeDataStore()])):
                result = json.loads(
                    server.get_session_transcript(
                        "session-1", cursor=0, max_bytes=MIN_PAGE_BYTES
                    )
                )

            self.assertEqual(result["session_id"], "session-1")
            self.assertEqual(result["source_format"], "jsonl")
            self.assertTrue(result["complete"])
            self.assertIsNone(result["next_cursor"])
            self.assertEqual(result["content"], transcript.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
