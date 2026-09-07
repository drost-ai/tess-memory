"""Run private, source-labelled retrieval cases against a read-only memory store.

Cases contain query text and expected memory/session references, so keep them
outside a public repository. Reports contain only metrics and case identifiers.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from . import retrieval
from .config import load_config
from .embed import make_embedder
from .store import MemoryStore


def evaluate(cases_path: Path, *, vectors_path: Path | None = None, limit: int = 12,
             keyword_only: bool = False) -> dict:
    cases = json.loads(cases_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation requires a nonempty case list")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    cached = np.load(vectors_path) if vectors_path else None
    metadata = json.loads(str(cached["__metadata__"])) if cached is not None else {}
    groups = defaultdict(list)
    for case in cases:
        groups[case["settings"]].append(case)
    rows = []
    original = retrieval.CONFIG
    try:
        for settings, group in groups.items():
            cfg = load_config(Path(settings).expanduser())
            retrieval.CONFIG = cfg
            store = MemoryStore(cfg.db_path, readonly=True)
            try:
                if keyword_only:
                    embedder = None
                elif cached is None:
                    embedder = make_embedder(cfg)
                else:
                    descriptor = store.get_meta("embedder")
                    class CachedQueries:
                        def __init__(self):
                            self.descriptor = descriptor
                        def embed_query(self, query):
                            expected = {"query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                                        "descriptor": self.descriptor}
                            if metadata.get(active_case["id"]) != expected:
                                raise ValueError("cached query vector does not match query and embedding space")
                            return cached[active_case["id"]]
                    embedder = CachedQueries()
                for active_case in group:
                    before = time.monotonic()
                    if keyword_only:
                        matches = store.fts_search(active_case["query"], limit=limit,
                                                   project=active_case.get("project"),
                                                   kinds=(active_case["kind"],) if active_case.get("kind") else None,
                                                   levels=(active_case["level"],) if active_case.get("level") else None)
                        hits = [dict(r) for r in store.get_memories(list(matches))]
                    else:
                        hits = retrieval.search(store, embedder, active_case["query"],
                                            project=active_case.get("project"),
                                            kinds=(active_case["kind"],) if active_case.get("kind") else None,
                                            levels=(active_case["level"],) if active_case.get("level") else None,
                                            limit=limit, expand_chunks=False)
                    mids = {h["id"] for h in hits}
                    sids = {h["session_id"] for h in hits}
                    expected = set(active_case.get("expected_memory_ids", []))
                    expected_sessions = set(active_case.get("expected_session_ids", []))
                    answerable = bool(expected or expected_sessions)
                    found = bool(mids & expected or sids & expected_sessions)
                    rows.append({"id": active_case["id"], "category": active_case.get("category", "recall"),
                                 "answerable": answerable, "supported_hit": found,
                                 "expected_memory_hit": bool(mids & expected),
                                 "returned_candidates": len(hits),
                                 "passed": found if answerable else not hits,
                                 "latency_seconds": round(time.monotonic() - before, 4),
                                 "returned_memory_ids": [h["id"] for h in hits]})
            finally:
                store.close()
                del store
                gc.collect()
    finally:
        retrieval.CONFIG = original
        if cached is not None:
            cached.close()
    times = np.array([r["latency_seconds"] for r in rows])
    positives = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"cases": len(rows), "passed": sum(r["passed"] for r in rows), "top_k": limit,
            "positive_source_recall": sum(r["supported_hit"] for r in positives) / len(positives) if positives else None,
            "negative_candidate_rate": sum(r["returned_candidates"] > 0 for r in negatives) / len(negatives) if negatives else None,
            "latency_p50_seconds": round(float(np.median(times)), 4),
            "latency_p95_seconds": round(float(np.percentile(times, 95)), 4),
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "mode": ("keyword only" if keyword_only else "static hybrid") + "; no answer-generation evaluation", "results": rows}
