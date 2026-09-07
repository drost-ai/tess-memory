"""Publish verified completed prefixes while preserving paid rows and checkpoints."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from .attribution import source_adapter
from .checkpoints import DistillationCheckpoints
from .distill import CallBudgetExceeded, DistillationPool, SESSION_SYSTEM, REDUCE_PROMPT
from .ingest import IngestStats, _extractive_digest, _log, _refresh_aggregates, _session_rows
from .readers import CompletedSourceReader, content_key, source_signature
from .source_journal import SourceChanged
from .stable_distill import STABLE_MAP_PROMPT, stable_digest
from .time_utils import week_start
from .transcript import chunk_session


def pipeline_policy(cfg, embedder, llm, fallback):
    return content_key({"version": 1, "segment_chars": cfg.segment_chars, "overlap": cfg.segment_overlap,
                        "fanout": cfg.incremental_reduce_fanout, "importance": cfg.min_importance,
                        "chunk_chars": cfg.chunk_chars, "tool_snippet": cfg.tool_result_snippet,
                        "min_session_chars": cfg.min_session_chars,
                        "embedding": embedder.descriptor,
                        "prompts": content_key([SESSION_SYSTEM, STABLE_MAP_PROMPT, REDUCE_PROMPT]),
                        "primary": getattr(llm, "cache_identity", getattr(llm, "name", None)),
                        "fallback": getattr(fallback, "cache_identity", getattr(fallback, "name", None))})


def _old_digest(old, rows):
    memories = []
    for row in rows:
        if row['level'] != 'memory':
            continue
        memory = {"content": row['content'], "kind": row['kind'],
                  "importance": round((row['importance'] or 0.3) * 10), "tags": json.loads(row['tags'] or '[]')}
        if row['ts'] and row['ts'] != old['ended_at']:
            memory['date'] = row['ts'][:10]
        memories.append(memory)
    return {"title": old['title'], "summary": old['summary'], "outcome": old['outcome'], "memories": memories}


def _bootstrap(snapshot, old, rows, cfg):
    """Seed only a prefix that can be checked against existing paid chunks."""
    if not old or not old['summary'] or old['outcome'] == '(not distilled)':
        return None
    if not isinstance(old['n_user'], int) or not isinstance(old['n_assistant'], int):
        return None
    chunks = [r['content'] for r in rows if r['level'] == 'chunk']
    if not chunks:
        return None
    st = snapshot.transcript
    keys = snapshot.unit_keys()
    users = assistants = start = 0
    for index, end in enumerate(snapshot.unit_ends, 1):
        for turn in st.turns[start:end]:
            users += turn.role == 'user'
            assistants += turn.role == 'assistant'
        start = end
        if users == old['n_user'] and assistants == old['n_assistant']:
            prefix = replace(st, turns=st.turns[:end])
            if [text for text, _ in chunk_session(prefix, cfg.chunk_chars)] == chunks:
                return {"units": keys[:index], "digest": _old_digest(old, rows)}
        if users > old['n_user'] or assistants > old['n_assistant']:
            break
    return None


def _embed_changed(rows, previous, cache, embedder):
    known = {r['content']: r['vector'] for r in previous if r['vector'] is not None}
    missing = {}
    for row in rows:
        key = content_key([embedder.descriptor, row.content])
        blob = known.get(row.content)
        if blob is None:
            blob = cache.passage(key)
        if blob is not None:
            row.vector = np.frombuffer(blob, dtype=np.float32)
        else:
            missing.setdefault(row.content, key)
    generated = {}
    if missing:
        vectors = embedder.embed_passages(list(missing))
        if len(vectors) != len(missing):
            raise ValueError("embedder returned a different number of vectors than requested passages")
        for (content, key), vector in zip(missing.items(), vectors, strict=True):
            vector = np.asarray(vector, dtype=np.float32)
            if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError("embedder returned an invalid vector")
            generated[content] = vector
        cache.save_passages([(missing[text], vec.tobytes()) for text, vec in generated.items()])
    for row in rows:
        if row.vector is None:
            row.vector = generated[row.content]
    return len(missing)


def ingest_incremental(cfg, store, embedder, llm, fallback_llm=None, *, limit=None,
                       retry_extractive=False, session_ids=None, refresh_aggregates=None):
    stats = IngestStats()
    started = time.monotonic()
    store.assert_embedder(embedder.descriptor)
    reader = CompletedSourceReader(cfg)
    cache = DistillationCheckpoints(cfg.data_dir / 'distillation-cache.db')
    policy = pipeline_policy(cfg, embedder, llm, fallback_llm)
    hold_signature = content_key(cfg.ingestion_hold_texts)
    found = []
    from .audit import source_members
    for ds in cfg.data_stores:
        sources, _ = source_adapter(ds, codex_catalog=reader.catalog)
        for path, source_project, parent in sources:
            if session_ids is not None:
                sid = (reader.catalog.sources[path.resolve()].session_id if ds.type == 'codex' else path.stem)
                if sid not in {s.removeprefix('codex:') for s in session_ids}:
                    continue
            try:
                modified = max(p.stat().st_mtime_ns for p in source_members(path, ds.type))
                found.append((modified, ds, path, ds.resolve_project(source_project), parent))
            except OSError:
                stats.deferred_active += 1
    found.sort(key=lambda row: row[0], reverse=True)
    stats.scanned = len(found)
    if reader.catalog:
        stats.errors += len(reader.catalog.errors)
    _log(f"[incremental] {stats.scanned} sources; completed-prefix processing")

    def record(ds, path, project, signature, status, state, sid=None, *, indexed=False, error=None):
        with store.write_transaction():
            store.record_source(path=str(path), session_id=sid, project=project, source_type=ds.type,
                                signature=signature, status=status, state=state, now=time.time(),
                                indexed=indexed, error=error)

    def work(snapshot, seed, pool):
        if llm is None:
            return _extractive_digest(snapshot.transcript), 'extractive', None, None
        calls = pool.session(f"incremental/{snapshot.transcript.source_type}/{snapshot.transcript.session_id}")
        keys = snapshot.unit_keys()
        if seed and keys[:len(seed['units'])] != seed['units']:
            seed = None
        digest = stable_digest(cfg, snapshot, calls, seed=seed['digest'] if seed else None,
                               seed_units=len(seed['units']) if seed else 0)
        return digest, 'fallback' if calls.fallback_used or (seed and seed.get('status') == 'fallback') else 'distilled', calls, seed

    def adopt(ds, path, project, signature, state, st, old, pending, status):
        with store.write_transaction():
            if old['project'] != project:
                store.reattribute_session_project(st.session_id, project, commit=False)
            store.upsert_session(**{**dict(old), 'project': project, 'cwd': st.cwd,
                                     'git_branch': st.git_branch, 'parent_session': st.parent_session,
                                     'source_path': str(path)})
            info = path.stat()
            store.mark_file(str(path), 0 if pending else info.st_mtime, 0 if pending else info.st_size,
                            st.session_id, status or 'distilled')
            store.record_source(path=str(path), session_id=st.session_id, project=project, source_type=ds.type,
                                signature=signature, status='pending_tail' if pending else 'indexed',
                                state=state, now=time.time(), indexed=True)

    pool = DistillationPool(llm, fallback_llm, workers=cfg.llm_workers,
                            cache_path=cfg.data_dir / 'distillation-cache.db', log=_log,
                            max_calls=cfg.max_model_calls_per_run, share_requests=True) if llm else None
    futures = {}
    attempted = 0
    visited = 0
    iterator = iter(found)
    exhausted = False
    seen_sessions = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, cfg.llm_workers)) as coordinators:
            while futures or not exhausted:
                while not exhausted and len(futures) < max(1, cfg.codex.batch_size):
                    if limit is not None and attempted >= limit:
                        exhausted = True
                        break
                    try:
                        _, ds, path, project, parent = next(iterator)
                        visited += 1
                    except StopIteration:
                        exhausted = True
                        break
                    progress = store.source_progress(str(path))
                    state = json.loads(progress['state']) if progress else {}
                    tracked = store.session_for_file(str(path))
                    known_id = progress['session_id'] if progress and progress['session_id'] else tracked['id'] if tracked else None
                    signature = None
                    try:
                        signature = content_key([source_signature(path, ds, cfg, catalog=reader.catalog if ds.type == 'codex' else None), policy])
                        if known_id:
                            if known_id in seen_sessions and seen_sessions[known_id] != str(path):
                                state['shadowed_by'] = seen_sessions[known_id]
                                record(ds, path, project, signature, 'shadowed', state, known_id)
                                stats.unchanged += 1
                                continue
                            seen_sessions[known_id] = str(path)
                        old_status = store.file_status(str(path))
                        if (cfg.source_retry_backoff and progress and progress['status'] == 'error' and progress['observed_signature'] == signature
                                and progress['retry_at'] > time.time() and not retry_extractive):
                            stats.retries_deferred += 1
                            continue
                        if retry_extractive and old_status != 'extractive':
                            stats.unchanged += 1
                            continue
                        if (progress and progress['status'] == 'held' and progress['observed_signature'] == signature
                                and state.get('hold_signature') == hold_signature):
                            stats.unchanged += 1
                            continue
                        if (progress and progress['observed_signature'] == signature
                                and progress['project'] == project
                                and progress['status'] in {'indexed', 'pending_tail', 'trivial'} and not retry_extractive):
                            if progress['session_id']:
                                seen_sessions.setdefault(progress['session_id'], str(path))
                            stats.unchanged += 1
                            continue
                        attempted += 1
                        snapshot = reader.read(ds, path, project, parent)
                        stats.parsed_records += snapshot.parsed_records
                        signature = content_key([snapshot.observed_signature, policy])
                        st = snapshot.transcript
                        if st and any(pattern in turn.text for pattern in cfg.ingestion_hold_texts for turn in st.turns):
                            state['hold_signature'] = hold_signature
                            record(ds, path, project, signature, 'held', state, st.session_id,
                                   error='source content matches an operator export hold')
                            continue
                        if st is None:
                            prior = store.session_for_file(str(path))
                            if not snapshot.pending_tail and prior is not None:
                                raise SourceChanged('previously indexed source now has no eligible activity')
                            status = 'pending_tail' if snapshot.pending_tail else 'trivial'
                            record(ds, path, project, signature, status, state, prior['id'] if prior else None)
                            stats.preserved += prior is not None
                            if status == 'pending_tail':
                                stats.deferred_active += 1
                            else:
                                stats.skipped_trivial += 1
                            continue
                        keys = snapshot.unit_keys()
                        if known_id and known_id != st.session_id:
                            raise SourceChanged('source identity differs from its published session')
                        if st.session_id in seen_sessions and seen_sessions[st.session_id] != str(path):
                            state['shadowed_by'] = seen_sessions[st.session_id]
                            record(ds, path, project, signature, 'shadowed', state, st.session_id)
                            stats.unchanged += 1
                            continue
                        seen_sessions[st.session_id] = str(path)
                        old = store.get_session(st.session_id)
                        if (old and snapshot.pending_tail and
                                (st.n_user < (old['n_user'] or 0) or st.n_assistant < (old['n_assistant'] or 0))):
                            stats.preserved += 1
                            record(ds, path, project, signature, 'pending_tail', state, st.session_id)
                            continue
                        if st.content_chars < cfg.min_session_chars:
                            if old is not None:
                                raise SourceChanged('previously indexed source became trivial')
                            record(ds, path, project, signature, 'trivial', state, st.session_id)
                            stats.skipped_trivial += 1
                            continue
                        known_status = old_status or (store.file_status(old['source_path']) if old else None)
                        previous = store.session_rows_with_vectors(st.session_id)
                        if (not retry_extractive and state.get('policy') == policy and state.get('indexed_units') == keys and old is not None):
                            state.update(pending_tail=snapshot.pending_tail,
                                         revisions=[asdict(r) for r in snapshot.revisions])
                            adopt(ds, path, project, signature, state, st, old, snapshot.pending_tail, known_status)
                            stats.unchanged += 1
                            continue
                        seed = state.get('seed') if state.get('policy') == policy else None
                        if seed is None and not state.get('indexed_units') and not retry_extractive:
                            seed = _bootstrap(snapshot, old, previous, cfg)
                            if seed:
                                seed['status'] = known_status
                        if llm is None and old is not None and old['outcome'] != '(not distilled)':
                            stats.preserved += 1
                            record(ds, path, project, signature, 'awaiting_model', state, st.session_id)
                            continue
                        if seed and seed['units'] == keys and old is not None and not retry_extractive:
                            state = {"policy": policy, "indexed_units": keys, "seed": seed,
                                     "pending_tail": snapshot.pending_tail, "revisions": [asdict(r) for r in snapshot.revisions],
                                     "boundary_kinds": snapshot.unit_kinds}
                            adopt(ds, path, project, signature, state, st, old, snapshot.pending_tail, known_status)
                            stats.unchanged += 1
                            continue
                        record(ds, path, project, signature, 'processing', state, st.session_id)
                        future = coordinators.submit(work, snapshot, seed, pool)
                        futures[future] = (ds, path, project, parent, snapshot, old, previous, state, seed, signature)
                    except Exception as exc:
                        stats.errors += 1
                        record(ds, path, project, signature, 'error', state, error=str(exc)[:500])
                        _log(f"[incremental] source failed: {path.name}: {exc}")
                if not futures:
                    continue
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    ds, path, project, parent, snapshot, old, previous, state, seed, signature = futures.pop(future)
                    st = snapshot.transcript
                    try:
                        digest, status, calls, seed = future.result()
                        rows = _session_rows(st, digest, cfg.chunk_chars)
                        stats.embedded_passages += _embed_changed(rows, previous, cache, embedder)
                        # Re-read through the same journal after slow provider
                        # work. Appends can advance the tail; earlier semantic
                        # units (including repair/fork changes) must still match.
                        current = reader.read(ds, path, project, parent)
                        stats.parsed_records += current.parsed_records
                        keys = snapshot.unit_keys()
                        if (current.transcript is None or current.unit_keys()[:len(keys)] != keys
                                or current.transcript.cwd != st.cwd or current.transcript.parent_session != st.parent_session):
                            raise SourceChanged("eligible source prefix changed during provider work")
                        pending_units = len(current.unit_keys()) - len(keys)
                        pending = current.pending_tail or pending_units > 0
                        signature = content_key([current.observed_signature, policy])
                        publication = {"policy": policy, "indexed_units": keys, "seed": seed,
                                       "pending_tail": current.pending_tail, "pending_units": pending_units,
                                       "boundary_kinds": snapshot.unit_kinds,
                                       "indexed_through_at": st.ended_at, "revisions": [asdict(r) for r in snapshot.revisions],
                                       "request_keys": sorted(calls.keys) if calls else []}
                        with store.write_transaction():
                            store.upsert_session(id=st.session_id, project=st.project, cwd=st.cwd, git_branch=st.git_branch,
                                                 title=digest.title, summary=digest.summary, outcome=digest.outcome,
                                                 started_at=st.started_at, ended_at=st.ended_at, n_user=st.n_user,
                                                 n_assistant=st.n_assistant, source_path=str(path), parent_session=st.parent_session)
                            store.replace_session_rows(st.session_id, rows)
                            info = path.stat()
                            store.mark_file(str(path), 0 if pending else info.st_mtime,
                                            0 if pending else info.st_size, st.session_id, status)
                            for scope in {st.project, old['project'] if old else None}:
                                if scope:
                                    store.mark_aggregate_dirty('project', scope)
                            for stamp in {st.ended_at, old['ended_at'] if old else None}:
                                week = week_start(stamp)
                                if week:
                                    store.mark_aggregate_dirty('digest', week)
                            store.record_source(path=str(path), session_id=st.session_id, project=project, source_type=ds.type,
                                                signature=signature, status='pending_work' if pending_units else 'pending_tail' if pending else 'indexed',
                                                state=publication, now=time.time(), indexed=True)
                        if calls:
                            try:
                                cache.pin(calls.namespace, calls.keys)
                            except Exception as exc:
                                # Publication is already durable; a cache
                                # maintenance failure cannot undo its status.
                                _log(f"[incremental] checkpoint pinning deferred: {exc}")
                        if status == 'extractive':
                            stats.extractive += 1
                        else:
                            stats.distilled += 1
                            stats.fallback += status == 'fallback'
                        _log(f"[incremental] {st.session_id}: published {len(keys)} units ({status}); pending_tail={pending}")
                    except CallBudgetExceeded as exc:
                        stats.budget_deferred += 1
                        record(ds, path, project, signature, 'budget_pending', state, st.session_id, error=str(exc))
                    except SourceChanged as exc:
                        stats.deferred_active += 1
                        record(ds, path, project, None, 'source_changed', state, st.session_id, error=str(exc))
                    except Exception as exc:
                        stats.errors += 1
                        stats.preserved += old is not None
                        record(ds, path, project, signature, 'error', state, st.session_id, error=str(exc)[:500])
                        _log(f"[incremental] {st.session_id}: retained last good memory after failure: {exc}")
                done.clear()
    finally:
        for future in futures:
            future.cancel()
        if pool:
            pool.__exit__()
            stats.model_calls = pool.calls
            stats.request_cache_hits = pool.cache_hits
            stats.shared_cache_hits = pool.shared_cache_hits
            stats.cache_promotions = pool.cache_promotions
    if llm is not None and refresh_aggregates is not False:
        interval = 0 if refresh_aggregates is True else cfg.aggregate_refresh_seconds
        class Counted:
            def __init__(self, backend):
                self.backend, self.name = backend, backend.name
            def complete(self, **kwargs):
                return pool.complete(self.backend, **kwargs)
        _refresh_aggregates(cfg, store, embedder, Counted(llm), Counted(fallback_llm) if fallback_llm else None, stats, interval)
        stats.model_calls = pool.calls
    now = time.time()
    if now - float(store.get_meta('incremental:last_cache_prune') or 0) >= 86400:
        cutoff = now - cfg.cache_retention_days * 86400
        reader.journal.prune(cutoff)
        cache.prune(cutoff)
        with store.write_transaction():
            store.set_meta('incremental:last_cache_prune', str(now))
    stats.aggregates_pending = len(store.pending_aggregates(float('inf')))
    stats.source_reads = attempted
    stats.sources_remaining = max(0, stats.scanned - visited)
    stats.duration_seconds = round(time.monotonic() - started, 3)
    store.commit()
    return stats
