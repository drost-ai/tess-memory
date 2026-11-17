"""Move a store to another embedding model without losing or redoing anything.

A store's vectors all come from one model; vectors from two models cannot be
compared. Changing the embedding model therefore used to mean rebuilding the
store. This does it in place instead: every stored row (memories, summaries,
chunks, plan items) is embedded again with the new model, and nothing else is
touched. No session is read or distilled again, and the plan is not involved.

The new vectors are staged beside the old ones, a batch at a time, so the work
can be stopped and resumed and the store stays fully usable meanwhile: search
and ingestion keep running on the old vectors. Only when every row has a new
vector does one transaction replace them all and record the new model. If it
is interrupted before that, the store is exactly as it was.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .readers import content_key
from .store import MemoryStore, same_vector_space

TARGET = "reembed:target"
PAGE = 2000


class _Incomplete(Exception):
    """Rows were added or changed while embedding; go round again."""


def _prepare(store: MemoryStore, target: str) -> None:
    with store.write_transaction():
        store.db.execute("CREATE TABLE IF NOT EXISTS reembed_vectors(key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        if store.get_meta(TARGET) != target:  # staged for another model: not usable for this one
            store.db.execute("DELETE FROM reembed_vectors")
            store.set_meta(TARGET, target)


def _pages(store: MemoryStore):
    last = 0
    while True:
        rows = store.db.execute("SELECT id, content FROM memories WHERE id>? ORDER BY id LIMIT ?", (last, PAGE)).fetchall()
        if not rows:
            return
        yield rows
        last = rows[-1]["id"]


def status(store: MemoryStore, target: str) -> dict[str, Any]:
    """Where the store stands against ``target``, without changing anything."""
    current = store.get_meta("embedder")
    rows = store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    staged = 0
    if store.get_meta(TARGET) == target:
        staged = store.db.execute("SELECT COUNT(*) FROM reembed_vectors").fetchone()[0]
    return {"current": current, "target": target, "rows": rows, "staged": staged,
            "needed": current is not None and not same_vector_space(current, target)}


def reembed(store: MemoryStore, embedder: Any, *, progress: Callable[[int, int], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Re-embed the whole store with ``embedder`` and switch to it.

    Returns ``{"state": "current" | "switched" | "stopped", ...}``. ``stopped``
    keeps what was staged, and the next call continues from there."""
    target = embedder.descriptor
    current = store.get_meta("embedder")
    if current is None or same_vector_space(current, target):
        return {"state": "current", "current": current, "target": target, "embedded": 0}
    _prepare(store, target)
    batch_size = max(1, int(getattr(embedder, "batch_size", 64)))
    embedded = 0
    while True:
        staged = {row[0] for row in store.db.execute("SELECT key FROM reembed_vectors")}
        total = store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        missing: dict[str, str] = {}
        covered = 0
        for rows in _pages(store):
            for row in rows:
                key = content_key(row["content"])
                if key in staged:
                    covered += 1
                else:
                    missing.setdefault(key, row["content"])
        pending = list(missing.items())
        for offset in range(0, len(pending), batch_size):
            if should_stop and should_stop():
                return {"state": "stopped", "current": current, "target": target, "embedded": embedded}
            batch = pending[offset:offset + batch_size]
            vectors = embedder.embed_passages([text for _, text in batch])
            if len(vectors) != len(batch):
                raise ValueError("embedder returned a different number of vectors than requested passages")
            values = []
            for (key, _), vector in zip(batch, vectors, strict=True):
                vector = np.asarray(vector, dtype=np.float32)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    raise ValueError("embedder returned an invalid vector")
                values.append((key, vector.tobytes()))
            with store.write_transaction():
                store.db.executemany("INSERT OR REPLACE INTO reembed_vectors(key, vector) VALUES(?,?)", values)
            embedded += len(batch)
            if progress:
                progress(min(total, covered + offset + len(batch)), total)
        try:
            switched = _switch(store, target)
        except _Incomplete:
            continue  # ingestion published more while we worked: embed those too
        if progress:
            progress(switched, switched)
        return {"state": "switched", "current": current, "target": target, "embedded": embedded, "rows": switched}


def _switch(store: MemoryStore, target: str) -> int:
    """Replace every vector and record the new model, all or nothing."""
    with store.write_transaction():
        db = store.db
        count = 0
        for rows in _pages(store):
            values = []
            for row in rows:
                staged = db.execute("SELECT vector FROM reembed_vectors WHERE key=?", (content_key(row["content"]),)).fetchone()
                if staged is None:
                    raise _Incomplete()  # rolls the whole switch back
                values.append((row["id"], staged["vector"]))
            db.executemany("INSERT OR REPLACE INTO embeddings(memory_id, vector) VALUES(?,?)", values)
            count += len(values)
        db.execute("DELETE FROM embeddings WHERE memory_id NOT IN (SELECT id FROM memories)")
        store.set_meta("embedder", target)
        db.execute("DELETE FROM meta WHERE key=?", (TARGET,))
        db.execute("DROP TABLE reembed_vectors")
        store._emb_cache = None
    return count
