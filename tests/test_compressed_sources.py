from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from backports import zstd

from tess_memory.codex import CodexCatalog
from tess_memory.config import Config, DataStore
from tess_memory.ingest import ingest
from tess_memory.readers import CompletedSourceReader
from tess_memory.source_journal import SourceChanged, SourceReadError
from tess_memory.store import MemoryStore
from tess_memory.transcript_reader import resolve_transcript_path, read_transcript_page
from test_codex import write_rollout, message, append, wrapped, RecordingEmbedder
from test_incremental_ingest import Backend


class CompressedSourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.home = self.root / 'codex'
        self.ds = DataStore('codex', self.home, type='codex')
        self.cfg = Config(data_stores=[self.ds], data_dir=self.root/'memory', incremental_ingestion=True)

    def compress(self, path, *, remove=True):
        archive = path.with_name(path.name + '.zst')
        archive.write_bytes(zstd.compress(path.read_bytes()))
        if remove:
            path.unlink()
        return archive

    def test_discovery_normalization_and_cached_reads_match_plain_rollouts(self):
        path = write_rollout(self.home)
        first = CompletedSourceReader(self.cfg).read(self.ds, path, 'p')
        archive = self.compress(path)
        catalog = CodexCatalog([self.ds])
        self.assertEqual(catalog.errors, {})
        self.assertEqual(list(catalog.sources), [archive])
        reader = CompletedSourceReader(self.cfg)
        result = reader.read(self.ds, archive, 'p')
        self.assertEqual(result.unit_keys(), first.unit_keys())
        self.assertEqual([t.text for t in result.transcript.turns], [t.text for t in first.transcript.turns])
        self.assertTrue(result.revisions[0].verify())
        self.assertEqual(result.revisions[0].path, str(archive))
        self.assertEqual(CompletedSourceReader(self.cfg).read(self.ds, archive, 'p').parsed_records, 0)

    def test_live_plain_copy_wins_over_its_archive(self):
        path = write_rollout(self.home)
        self.compress(path, remove=False)
        self.assertEqual(list(CodexCatalog([self.ds]).sources), [path])

    def test_compressed_parent_uses_decoded_byte_boundary_and_segment_id(self):
        segment = '11111111-1111-1111-1111-111111111111'
        parent = write_rollout(self.home, sid='same', items=[message('user', 'Earlier request '*100),
                                                          message('assistant', 'Earlier answer '*100)])
        parent = parent.rename(parent.with_name('rollout-same_' + segment + '.jsonl'))
        boundary = parent.stat().st_size
        append(parent, wrapped('response_item', message('user', 'EXCLUDED LATER REQUEST')))
        append(parent, wrapped('response_item', message('assistant', 'EXCLUDED LATER ANSWER')))
        archive = self.compress(parent)
        self.assertGreater(boundary, archive.stat().st_size)
        child = write_rollout(self.home, sid='child', meta={'id':'same', 'history_base':{
            'rollout_id':segment, 'end_byte_offset':boundary}}, items=[message('user','Current request'), message('assistant','Current answer')])
        result = CompletedSourceReader(self.cfg).read(self.ds, child, 'p')
        text = '\n'.join(t.text for t in result.transcript.turns)
        self.assertIn('Earlier request', text)
        self.assertIn('Current answer', text)
        self.assertNotIn('EXCLUDED LATER', text)
        self.assertTrue(all(r.verify() for r in result.revisions))

    def test_truncated_archive_fails_without_saving_reader_state(self):
        archive = self.compress(write_rollout(self.home))
        reader = CompletedSourceReader(self.cfg)
        archive.write_bytes(archive.read_bytes()[:-2])
        with self.assertRaises(OSError):
            reader.read(self.ds, archive, 'p')
        with reader.journal.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM readers').fetchone()[0], 0)

    def test_decoded_record_limit_is_enforced(self):
        archive = self.compress(write_rollout(self.home, items=[message('user','x'*10000)]))
        cfg = replace(self.cfg, codex=replace(self.cfg.codex, max_record_bytes=1000))
        with self.assertRaises(SourceReadError):
            CompletedSourceReader(cfg).read(self.ds, archive, 'p')

    def test_impossible_decoded_history_boundary_fails(self):
        archive = self.compress(write_rollout(self.home))
        reader = CompletedSourceReader(self.cfg)
        with self.assertRaises(SourceReadError):
            reader.catalog._read(reader.catalog.sources[archive], end_byte=10**6)

    def test_rewritten_archive_invalidates_cached_normalization(self):
        path = write_rollout(self.home)
        archive = self.compress(path)
        before = CompletedSourceReader(self.cfg).read(self.ds, archive, 'p')
        path = write_rollout(self.home, items=[message('user','Changed request'),message('assistant','Changed answer')])
        self.compress(path)
        after = CompletedSourceReader(self.cfg).read(self.ds, archive, 'p')
        self.assertNotEqual(before.unit_keys(), after.unit_keys())
        self.assertFalse(before.revisions[0].verify())

    def test_archive_relocation_preserves_paid_rows_and_makes_no_model_calls(self):
        path = write_rollout(self.home)
        store = MemoryStore(self.cfg.db_path)
        self.addCleanup(store.close)
        embedder, backend = RecordingEmbedder(), Backend()
        ingest(self.cfg, store, embedder, backend, refresh_aggregates=False)
        rows = [tuple(r) for r in store.db.execute('SELECT m.*,e.vector FROM memories m JOIN embeddings e ON e.memory_id=m.id ORDER BY m.id')]
        calls = len(backend.calls), len(embedder.calls)
        archive = self.compress(path)
        stats = ingest(self.cfg, store, embedder, backend, refresh_aggregates=False)
        self.assertEqual((stats.model_calls, stats.embedded_passages), (0,0))
        self.assertEqual((len(backend.calls),len(embedder.calls)), calls)
        self.assertEqual(rows, [tuple(r) for r in store.db.execute('SELECT m.*,e.vector FROM memories m JOIN embeddings e ON e.memory_id=m.id ORDER BY m.id')])
        self.assertEqual(store.get_session('codex:one')['source_path'], str(archive))
        old = store.source_progress(str(path))
        self.assertEqual(old['status'], 'superseded')
        self.assertEqual(json.loads(old['state'])['superseded_by'], str(archive))
        self.assertIsNotNone(old['indexed_at'])

    def test_transcript_pages_resolve_archived_sources_and_reconstruct_utf8(self):
        path = write_rollout(self.home, items=[message('user','Words and café '*100),message('assistant','Answer '*100)])
        raw = path.read_bytes().replace(b'\\u00e9', 'é'.encode())
        path.write_bytes(raw)
        archive = self.compress(path)
        self.assertEqual(resolve_transcript_path(path, [self.home]), archive)
        cursor, parts = 0, []
        while True:
            page = read_transcript_page(archive, cursor=cursor, max_bytes=256)
            parts.append(page.content)
            self.assertEqual(page.source_size_bytes, len(raw))
            if page.complete:
                break
            cursor = page.next_cursor
        self.assertEqual(''.join(parts).encode(), raw)


if __name__ == '__main__':
    unittest.main()
