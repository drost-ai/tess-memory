"""Hybrid retrieval: semantic + keyword + importance + recency.

Scoring: candidates come from the union of the top semantic matches (cosine
over normalized embeddings) and the top BM25 matches (FTS5). Each signal is
min-max normalized within the candidate set, then combined with the weights
from settings.toml `[retrieval]` (defaults: 0.55 semantic, 0.20 bm25,
0.15 importance, 0.10 recency).

Recency decays exponentially (default 30-day half-life) from each memory's
timestamp anchor; undated memories (project profiles) get a neutral 0.5.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from .config import CONFIG
from .embed import Embedder
from .store import MemoryStore


def _recency(ts: str | None, now: datetime, half_life_days: float) -> float:
    if not ts:
        return 0.5
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    age_days = max(0.0, (now - t).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / half_life_days)


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
    limit: int | None = None,
    candidates: int | None = None,
    expand_chunks: bool = True,
) -> list[dict]:
    limit = limit if limit is not None else CONFIG.default_limit
    candidates = candidates if candidates is not None else CONFIG.candidate_pool
    if limit <= 0 or not store.has_memories(
        project=project, levels=levels, kinds=kinds, after=after, before=before
    ):
        return []
    store.assert_embedder(embedder.descriptor)
    allowed = store.filter_memory_ids(
        project=project, levels=levels, kinds=kinds, after=after, before=before
    )

    # Semantic signal
    ids, mat = store.embedding_matrix()
    if allowed is not None and len(ids):
        mask = np.array([i in allowed for i in ids])
        ids, mat = ids[mask], mat[mask]
    sem: dict[int, float] = {}
    if len(ids):
        qv = embedder.embed_query(query)
        sims = mat @ qv
        top = np.argsort(-sims)[:candidates]
        for idx in top:
            if np.isfinite(sims[idx]):
                sem[int(ids[idx])] = float(sims[idx])

    # Keyword signal
    bm25 = store.fts_search(query, limit=candidates, project=project,
                            levels=levels, kinds=kinds, after=after, before=before)

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
            CONFIG.w_semantic * sem_n.get(mid, 0.0)
            + CONFIG.w_bm25 * bm25_n.get(mid, 0.0)
            + CONFIG.w_importance * float(r["importance"] or 0.3)
            + CONFIG.w_recency * _recency(r["ts"], now, CONFIG.half_life_days)
        )
        scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))

    # Light diversity: cap results from the same session.
    out, per_session = [], {}
    for score, r in scored:
        sid = r["session_id"] or f"_{r['level']}_{r['project']}"
        if per_session.get(sid, 0) >= CONFIG.per_session_cap:
            continue
        per_session[sid] = per_session.get(sid, 0) + 1
        content = r["content"]
        if expand_chunks and r["level"] == "chunk":
            # Chunks are embedded small for retrieval precision but read big:
            # merge the hit with its neighboring conversation windows.
            expanded = store.expand_chunk(
                r["id"], r["session_id"], CONFIG.chunk_neighbor_chars
            )
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
