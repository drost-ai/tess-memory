"""Exact scoped cosine search with a bounded LRU and streaming overflow."""
from __future__ import annotations

from collections import OrderedDict
import heapq

import numpy as np


class ScopedVectorSearch:
    def __init__(self):
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.hits = self.misses = self.rows_read = self.max_batch_bytes = 0

    def _trim(self, budget):
        while self.cache and self.cache_bytes > budget:
            _, (ids, matrix) = self.cache.popitem(last=False)
            self.cache_bytes -= ids.nbytes + matrix.nbytes

    @staticmethod
    def revision(store, project):
        if store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vector_scope_versions'").fetchone() is None:
            # A read-only client may open a pre-migration database. Stream it
            # exactly rather than caching with an unreliable revision token.
            return None
        scope = 'P:' + project if project else 'G'
        row = store.db.execute('SELECT revision FROM vector_scope_versions WHERE scope=?', (scope,)).fetchone()
        return row[0] if row else 0

    def search(self, store, vector, *, limit, project=None, levels=None, kinds=None,
               after=None, before=None, cacheable=True, cache_bytes=64 * 1024 * 1024, batch_rows=512):
        if limit <= 0:
            return {}
        q = np.asarray(vector, dtype=np.float32)
        if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
            raise ValueError('query vector is invalid')
        self._trim(cache_bytes)
        revision = self.revision(store, project)
        # A restricted reader's allowed rows move with memberships and project
        # mappings, which no vector revision tracks: stream, never cache.
        restricted = getattr(store, "restricted", False)
        cacheable = cacheable and revision is not None and cache_bytes > 0 and not restricted
        key = (project, tuple(sorted(levels or ())), tuple(sorted(kinds or ())), after, before, revision)
        if cacheable and key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            ids, matrix = self.cache[key]
            if matrix.shape[1] != len(q):
                raise ValueError('stored vectors differ from the query dimension')
            scores = matrix @ q
            order = np.lexsort((ids, -scores))[:limit]
            return {int(ids[i]): float(scores[i]) for i in order if np.isfinite(scores[i])}
        self.misses += 1
        where, args = store._memory_filters(project, levels, kinds, after, before)
        if where:
            sql = ('SELECT m.id,e.vector FROM memories m JOIN embeddings e ON e.memory_id=m.id WHERE 1=1'
                   + where + ' ORDER BY m.id')
        else:
            sql = 'SELECT memory_id AS id,vector FROM embeddings ORDER BY memory_id'
        cursor = store.db.execute(sql, args)
        heap = []
        collected_ids, collected_vectors = [], []
        size = 0
        keep = cacheable
        try:
            while rows := cursor.fetchmany(max(1, batch_rows)):
                ids = np.array([r['id'] for r in rows], dtype=np.int64)
                values = [np.frombuffer(r['vector'], dtype=np.float32) for r in rows]
                if any(len(v) != len(q) for v in values):
                    raise ValueError('stored vectors differ from the query dimension')
                matrix = np.vstack(values)
                if not np.isfinite(matrix).all():
                    raise ValueError('stored vectors contain invalid values')
                self.rows_read += len(rows)
                self.max_batch_bytes = max(self.max_batch_bytes, matrix.nbytes + ids.nbytes)
                scores = matrix @ q
                for i in np.lexsort((ids, -scores))[:limit]:
                    score = float(scores[i])
                    if not np.isfinite(score):
                        continue
                    item = (score, -int(ids[i]))
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                if keep:
                    size += matrix.nbytes + ids.nbytes
                    if size <= cache_bytes:
                        collected_ids.append(ids)
                        collected_vectors.append(matrix)
                    else:
                        keep = False
                        collected_ids.clear()
                        collected_vectors.clear()
        finally:
            cursor.close()
        if keep and collected_ids:
            ids = np.concatenate(collected_ids)
            matrix = np.vstack(collected_vectors)
            self.cache[key] = (ids, matrix)
            self.cache_bytes += ids.nbytes + matrix.nbytes
            self._trim(cache_bytes)
        return {-mid: score for score, mid in sorted(heap, key=lambda item: (-item[0], -item[1]))}

    def status(self):
        return {'cache_entries': len(self.cache), 'cache_bytes': self.cache_bytes,
                'cache_hits': self.hits, 'cache_misses': self.misses,
                'vectors_read': self.rows_read, 'max_batch_bytes': self.max_batch_bytes,
                'algorithm': 'exact scoped cosine with bounded streaming'}
