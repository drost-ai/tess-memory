from contextlib import closing
from dataclasses import replace
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from agentic_memory import retrieval
from agentic_memory.config import Config
from agentic_memory.store import MemoryRow, MemoryStore


class QueryEmbedder:
    descriptor = 'synthetic:8d'
    def __init__(self, vector):
        self.vector = vector
    def embed_query(self, query):
        return self.vector


class ScopedVectorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.path = self.root / 'memory.db'
        self.store = MemoryStore(self.path)
        self.addCleanup(self.store.close)
        self.other = MemoryStore(self.path)
        self.addCleanup(self.other.close)
        self.rng = np.random.default_rng(73)
        self.query = self.rng.normal(size=8).astype(np.float32)
        self.query /= np.linalg.norm(self.query)
        self.store.set_meta('embedder', QueryEmbedder.descriptor)
        self.store.commit()

    def add(self, project='p', vector=None, kind='fact', level='memory', ts='2026-08-01'):
        if vector is None:
            vector = self.rng.normal(size=8).astype(np.float32)
            vector /= np.linalg.norm(vector)
        mid = self.other.add_memories([MemoryRow(project=project, level=level, kind=kind, content=f'row {self.rng.integers(100000)}',
                                                vector=vector, ts=ts)])[0]
        self.other.commit()
        return mid

    def candidates(self, **kw):
        with self.store.read_snapshot() as cacheable:
            return self.store.semantic_candidates(self.query, limit=10, cacheable=cacheable, batch_rows=7, **kw)

    def test_streaming_matches_exact_full_matrix_and_respects_memory_bound(self):
        for _ in range(101):
            self.add()
        ids, matrix = self.store.embedding_matrix()
        scores = matrix @ self.query
        expected = [int(ids[i]) for i in np.lexsort((ids, -scores))[:10]]
        actual = self.candidates(cache_bytes=100)
        self.assertEqual(list(actual), expected)
        index = self.store._scoped_vectors.status()
        self.assertLessEqual(index['cache_bytes'], 100)
        self.assertLessEqual(index['max_batch_bytes'], 7 * (8 * 4 + 8))
        self.assertEqual(index['vectors_read'], 101)

    def test_project_query_does_not_read_unrelated_vectors(self):
        for _ in range(6):
            self.add('small')
        for _ in range(70):
            self.add('other')
        self.candidates(project='small')
        self.assertEqual(self.store._scoped_vectors.rows_read, 6)
        reads = self.store._scoped_vectors.rows_read
        self.add('other')
        self.other.set_meta('unrelated', 'write')
        self.other.commit()
        self.candidates(project='small')
        self.assertEqual(self.store._scoped_vectors.rows_read, reads)
        self.assertEqual(self.store._scoped_vectors.hits, 1)

    def test_other_connection_updates_and_deletes_invalidate_relevant_scope(self):
        mid = self.add(vector=-self.query)
        self.assertAlmostEqual(self.candidates(project='p')[mid], -1, places=5)
        with self.other.write_transaction():
            self.other.db.execute('UPDATE embeddings SET vector=? WHERE memory_id=?', (self.query.tobytes(), mid))
        self.assertAlmostEqual(self.candidates(project='p')[mid], 1, places=5)
        with self.other.write_transaction():
            self.other._delete_memory_ids([mid])
        self.assertEqual(self.candidates(project='p'), {})

    def test_reassignment_invalidates_both_scopes_without_changing_vector(self):
        mid = self.add('old', self.query)
        self.candidates(project='old')
        self.candidates(project='new')
        with self.other.write_transaction():
            self.other.db.execute("UPDATE memories SET project='new' WHERE id=?", (mid,))
        self.assertEqual(self.candidates(project='old'), {})
        self.assertIn(mid, self.candidates(project='new'))

    def test_kind_level_and_date_filters_apply_before_top_n_and_track_updates(self):
        desired = self.add('p', -self.query, kind='decision', ts='2026-08-20')
        for _ in range(30):
            self.add('p', self.query, kind='fact', ts='2026-08-01')
        actual = self.candidates(project='p', kinds=('decision',), levels=('memory',), after='2026-08-10')
        self.assertEqual(list(actual), [desired])
        with self.other.write_transaction():
            self.other.db.execute("UPDATE memories SET kind='fact' WHERE id=?", (desired,))
        self.assertEqual(self.candidates(project='p', kinds=('decision',), after='2026-08-10'), {})

    def test_rolled_back_write_cannot_poison_cache_when_revision_number_is_reused(self):
        original = self.add('p', -self.query)
        self.candidates(project='p')
        self.store.db.execute('BEGIN IMMEDIATE')
        transient = self.store.add_memories([MemoryRow(project='p', level='memory', kind='fact', content='transient', vector=self.query)])[0]
        self.assertIn(transient, self.candidates(project='p'))
        self.store.db.rollback()
        replacement = self.add('p', -self.query)
        values = self.candidates(project='p')
        self.assertEqual(set(values), {original, replacement})
        self.assertAlmostEqual(values[replacement], -1, places=5)

    def test_retrieval_does_not_use_full_matrix_and_embeds_outside_read_transaction(self):
        self.add('p', self.query)
        test = self
        class Embedder(QueryEmbedder):
            def embed_query(self, query):
                test.assertFalse(test.store.db.in_transaction)
                return self.vector
        with mock.patch.object(retrieval, 'CONFIG', Config(data_dir=self.root)), \
             mock.patch.object(self.store, 'embedding_matrix', side_effect=AssertionError('no full matrix')):
            hits = retrieval.search(self.store, Embedder(self.query), 'row', project='p')
        self.assertTrue(hits)
        self.assertFalse(self.store.db.in_transaction)

    def test_old_readonly_store_without_revision_table_streams_without_cache(self):
        self.add('p')
        with self.other.write_transaction():
            for name in ('vector_scope_ai', 'vector_scope_au', 'vector_scope_ad', 'vector_scope_md', 'vector_scope_mu'):
                self.other.db.execute('DROP TRIGGER ' + name)
            self.other.db.execute('DROP TABLE vector_scope_versions')
        self.candidates(project='p')
        self.candidates(project='p')
        self.assertEqual(self.store._scoped_vectors.hits, 0)

    def test_external_process_vector_commit_is_visible_to_warm_cache(self):
        mid = self.add('p', -self.query)
        self.candidates(project='p')
        code = 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute("UPDATE embeddings SET vector=? WHERE memory_id=?",(bytes.fromhex(sys.argv[2]),int(sys.argv[3]))); c.commit()'
        subprocess.run([sys.executable, '-c', code, str(self.path), self.query.tobytes().hex(), str(mid)], check=True, timeout=5)
        self.assertAlmostEqual(self.candidates(project='p')[mid], 1, places=5)


if __name__ == '__main__':
    unittest.main()
