"""Hybrid retrieval: semantic + keyword + importance + recency.

Scoring: candidates come from the union of the top semantic matches (cosine
over normalized embeddings) and the top BM25 matches (FTS5). Each signal is
min-max normalized within the candidate set, then combined:

    score = 0.55*semantic + 0.20*bm25 + 0.15*importance + 0.10*recency

Recency decays exponentially with a 30-day half-life from each memory's
timestamp anchor; undated memories (project profiles) get a neutral 0.5.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from .embed import Embedder
from .store import MemoryStore

HALF_LIFE_DAYS = 30.0

W_SEM, W_BM25, W_IMP, W_REC = 0.55, 0.20, 0.15, 0.10


def _recency(ts: str | None, now: datetime) -> float:
    if not ts:
        return 0.5
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    age_days = max(0.0, (now - t).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)


def _norm(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def search(
    store: MemoryStore,
    embedder: Embedder,
    query: str,
    project: str | None = None,
    levels: tuple[str, ...] | None = None,
    kinds: tuple[str, ...] | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 8,
    candidates: int = 100,
    expand_chunks: bool = True,
) -> list[dict]:
    store.assert_embedder(embedder.descriptor)
    allowed = store.filter_memory_ids(
        project=project, levels=levels, kinds=kinds, after=after, before=before
    )

    # Semantic signal
    ids, mat = store.embedding_matrix()
    sem: dict[int, float] = {}
    if len(ids):
        qv = embedder.embed_query(query)
        sims = mat @ qv
        if allowed is not None:
            mask = np.array([i in allowed for i in ids])
            sims = np.where(mask, sims, -np.inf)
        top = np.argsort(-sims)[:candidates]
        for idx in top:
            if np.isfinite(sims[idx]):
                sem[int(ids[idx])] = float(sims[idx])

    # Keyword signal
    bm25 = store.fts_search(query, limit=candidates)
    if allowed is not None:
        bm25 = {k: v for k, v in bm25.items() if k in allowed}

    pool = set(sem) | set(bm25)
    if not pool:
        return []

    sem_n, bm25_n = _norm(sem), _norm(bm25)
    rows = store.get_memories(list(pool))
    now = datetime.now(timezone.utc)

    scored = []
    for r in rows:
        mid = r["id"]
        score = (
            W_SEM * sem_n.get(mid, 0.0)
            + W_BM25 * bm25_n.get(mid, 0.0)
            + W_IMP * float(r["importance"] or 0.3)
            + W_REC * _recency(r["ts"], now)
        )
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])

    # Light diversity: at most 3 results from the same session.
    out, per_session = [], {}
    for score, r in scored:
        sid = r["session_id"] or f"_{r['level']}_{r['project']}"
        if per_session.get(sid, 0) >= 3:
            continue
        per_session[sid] = per_session.get(sid, 0) + 1
        content = r["content"]
        if expand_chunks and r["level"] == "chunk":
            # Chunks are embedded small for retrieval precision but read big:
            # merge the hit with its neighboring conversation windows.
            expanded = store.expand_chunk(r["id"], r["session_id"])
            if expanded:
                content = expanded
        out.append(
            {
                "id": r["id"],
                "score": round(score, 4),
                "level": r["level"],
                "kind": r["kind"],
                "content": content,
                "importance": r["importance"],
                "ts": r["ts"],
                "project": r["project"],
                "session_id": r["session_id"],
                "session_title": r["session_title"] if "session_title" in r.keys() else None,
            }
        )
        if len(out) >= limit:
            break
    return out
