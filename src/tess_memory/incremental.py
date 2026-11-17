"""Publish verified completed prefixes while preserving paid rows and checkpoints."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from .attribution import source_adapter
from .checkpoints import DistillationCheckpoints
from .distill import CallBudgetExceeded, DistillationPool, SESSION_SYSTEM, REDUCE_PROMPT
from .ingest import IngestStats, _extractive_digest, _log, _refresh_aggregates, _session_rows
from .readers import CompletedSourceReader, content_key, source_signature
from .source_journal import SourceChanged, SourceReadError
from .codex import CodexParseError
from .source_activity import require_quiet, SourceNotQuiet
from .model_usage import source_namespace, guard_source_requests, check_source_admission
from .source_schedule import SCAN_CURSOR, source_scan_order
from .stable_distill import STABLE_MAP_PROMPT, stable_digest
from .time_utils import week_start
from .transcript import chunk_session


def _policy_shape(cfg, embedder):
    return {"version": 1, "segment_chars": cfg.segment_chars, "overlap": cfg.segment_overlap,
            "fanout": cfg.incremental_reduce_fanout, "importance": cfg.min_importance,
            "chunk_chars": cfg.chunk_chars, "tool_snippet": cfg.tool_result_snippet,
            "min_session_chars": cfg.min_session_chars,
            "embedding": embedder.descriptor,
            "prompts": content_key([SESSION_SYSTEM, STABLE_MAP_PROMPT, REDUCE_PROMPT])}


def pipeline_policy(cfg, embedder):
    """How sessions are distilled now: how sources are cut, the prompts and
    the embedding space. It is recorded with each publication and decides how
    NEW work is done. It never makes published work stale: a session someone
    already paid to distil is not redone because a setting, a prompt in a
    newer release, or a model changed. Only `tess-memory redistil` redoes it."""
    return content_key(_policy_shape(cfg, embedder))


# What a source carried forward from before the policy stopped naming models
# records as the models that wrote it: unknown, so never equal to a real set.
CARRIED_MODELS = "carried-forward"


def legacy_pipeline_policy(cfg, embedder, llm, fallback):
    """The models behind a publication, in the form the policy used to take
    while it still named them. Recorded with each publication so a session
    appended after a model change is seeded from its paid rows."""
    return content_key({**_policy_shape(cfg, embedder),
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
        size = max(1, getattr(embedder, 'batch_size', len(missing)))
        pending = list(missing.items())
        for offset in range(0, len(pending), size):
            batch = pending[offset:offset+size]
            check_source_admission()
            vectors = embedder.embed_passages([text for text, _ in batch])
            if len(vectors) != len(batch):
                raise ValueError("embedder returned a different number of vectors than requested passages")
            validated = []
            for (content, key), vector in zip(batch, vectors, strict=True):
                vector = np.asarray(vector, dtype=np.float32)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    raise ValueError("embedder returned an invalid vector")
                generated[content] = vector
                validated.append((key, vector.tobytes()))
            # A source can become active between HTTP batches. Keep every
            # completed batch even if the next admission must be deferred.
            cache.save_passages(validated)
    for row in rows:
        if row.vector is None:
            row.vector = generated[row.content]
    return len(missing)


def ingest_incremental(cfg, store, embedder, llm, fallback_llm=None, *, limit=None,
                       retry_extractive=False, session_ids=None, refresh_aggregates=None, redistil=False):
    stats = IngestStats()
    started = time.monotonic()
    store.assert_embedder(embedder.descriptor)
    reader = CompletedSourceReader(cfg)
    cache = DistillationCheckpoints(cfg.data_dir / 'distillation-cache.db')
    policy = pipeline_policy(cfg, embedder)
    legacy_policy = legacy_pipeline_policy(cfg, embedder, llm, fallback_llm)
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
    fair_scan = limit is not None and session_ids is None and not retry_extractive
    if session_ids is None:
        # Absent files never enter discovery. Keep their paid memory and
        # progress, but report unavailable evidence rather than a parser error.
        selected_ids = ({'codex:' + s.session_id for s in reader.catalog.sources.values()}
                        if reader.catalog else set())
        with store.write_transaction():
            for row in store.db.execute("SELECT * FROM ingest_sources WHERE status NOT IN ('superseded','shadowed','held','source_unavailable')").fetchall():
                path = Path(row['path'])
                if path.is_file() or cfg.data_store_for(path) is None:
                    continue
                if (row['session_id'] in selected_ids or
                        (path.suffix == '.jsonl' and path.with_name(path.name + '.zst').is_file())):
                    continue
                state = json.loads(row['state'])
                state['unavailable_reason'] = 'source file is absent; previous memory and progress retained'
                store.record_source(path=row['path'], session_id=row['session_id'], project=row['project'],
                    source_type=row['source_type'], signature=row['observed_signature'], status='source_unavailable',
                    state=state, now=time.time(), error=state['unavailable_reason'])
    recovery = cfg.recovery_ingestion
    scan_cursor = 'recovery:scan_cursor' if recovery else SCAN_CURSOR
    scan = source_scan_order(found, limit=limit if fair_scan else None,
                             cursor=store.get_meta(scan_cursor),
                             known_paths={r[0] for r in store.db.execute('SELECT path FROM ingest_sources')},
                             now=time.time())
    stats.scanned = len(found)
    if reader.catalog:
        stats.errors += len(reader.catalog.errors)
    _log(f"[incremental] {stats.scanned} sources; completed-prefix processing")

    def record(ds, path, project, signature, status, state, sid=None, *, indexed=False, error=None):
        with store.write_transaction():
            store.record_source(path=str(path), session_id=sid, project=project, source_type=ds.type,
                                signature=signature, status=status, state=state, now=time.time(),
                                indexed=indexed, error=error)

    def defer_quiet(ds, path, project, state, sid, exc):
        state.update(quiet_until=exc.quiet_until, observed_modified_at=exc.activity.modified_at)
        record(ds, path, project, None, 'quiet_pending', state, sid, error=str(exc))
        stats.deferred_active += 1
        stats.quiet_deferred += 1

    def source_guard(ds, path):
        quiet = max(cfg.quiet_seconds, cfg.recovery_min_age_seconds) if recovery else cfg.quiet_seconds
        return partial(require_quiet, path, ds.type, quiet) if (fair_scan or recovery) and quiet else None

    def recovered_state(snapshot, signature):
        return {'recovered_boundaries': snapshot.boundary_records(),
                'recovery_notes': snapshot.recovery_notes,
                'recovery_signature': signature, 'recovery_version': 1,
                'recovery_only': False}

    def read_source(ds, path, project, parent, state):
        return reader.read(ds, path, project, parent).retain_boundaries(state.get('recovered_boundaries'))

    def work(snapshot, seed, pool, guard):
        if llm is None:
            return _extractive_digest(snapshot.transcript), 'extractive', None, None
        calls = pool.session(source_namespace(snapshot.transcript.source_type, snapshot.transcript.session_id), admission_guard=guard)
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
            store.supersede_source_paths(st.session_id, str(path))

    pool = DistillationPool(llm, fallback_llm, workers=cfg.llm_workers,
                            cache_path=cfg.data_dir / 'distillation-cache.db', log=_log,
                            max_calls=cfg.max_model_calls_per_run, share_requests=True) if llm else None
    futures = {}
    attempted = 0
    iterator = iter(scan)
    exhausted = False
    seen_sessions = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, cfg.llm_workers)) as coordinators:
            while futures or not exhausted:
                while not exhausted and len(futures) < max(1, cfg.codex.batch_size):
                    try:
                        (_, ds, path, project, parent), advance_cursor = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    progress = store.source_progress(str(path))
                    state = json.loads(progress['state']) if progress else {}
                    tracked = store.session_for_file(str(path))
                    known_id = progress['session_id'] if progress and progress['session_id'] else tracked['id'] if tracked else None
                    if known_id is None and ds.type == 'codex':
                        known_id = 'codex:' + reader.catalog.sources[path.resolve()].session_id
                    signature = None
                    try:
                        signature = content_key([source_signature(path, ds, cfg, catalog=reader.catalog if ds.type == 'codex' else None), policy])
                        if recovery:
                            if progress and state.get('recovery_signature') == signature and state.get('recovery_version') == 1:
                                stats.unchanged += 1
                                continue
                            if progress and progress['status'] in {'indexed', 'shadowed', 'superseded'} and progress['observed_signature'] == signature:
                                stats.unchanged += 1
                                continue
                        elif cfg.source_retry_backoff and not retry_extractive and progress and progress['observed_signature'] == signature and state.get('recovery_only'):
                            stats.retries_deferred += 1
                            continue
                        if known_id:
                            if known_id in seen_sessions and seen_sessions[known_id] != str(path):
                                state['shadowed_by'] = seen_sessions[known_id]
                                record(ds, path, project, signature, 'shadowed', state, known_id)
                                stats.unchanged += 1
                                continue
                            seen_sessions[known_id] = str(path)
                        old_status = store.file_status(str(path))
                        if (not recovery and cfg.source_retry_backoff and progress and progress['status'] == 'error' and progress['observed_signature'] == signature
                                and progress['retry_at'] > time.time() and not retry_extractive):
                            stats.retries_deferred += 1
                            continue
                        if retry_extractive and old_status != 'extractive':
                            stats.unchanged += 1
                            continue
                        if (progress and progress['status'] == 'held' and progress['observed_signature'] == signature
                                and state.get('hold_signature') == hold_signature):
                            stats.unchanged += 1
                            stats.recovery_held += recovery
                            continue
                        if (not recovery and progress and progress['observed_signature'] == signature
                                and progress['project'] == project
                                and progress['status'] in {'indexed', 'pending_tail', 'trivial'} and not retry_extractive and not redistil):
                            if progress['session_id']:
                                seen_sessions.setdefault(progress['session_id'], str(path))
                            stats.unchanged += 1
                            continue
                        guard = source_guard(ds, path)
                        if guard:
                            guard()  # current files, never the discovery-time timestamp
                        state.pop('quiet_until', None)
                        state.pop('observed_modified_at', None)
                        if limit is not None and attempted >= limit:
                            # Finish the cheap metadata checks even after the
                            # read budget is spent. A rotated scan's suffix can
                            # contain already-indexed files, not just backlog.
                            stats.sources_remaining += 1
                            continue
                        attempted += 1
                        if advance_cursor:
                            # This records an attempt, never successful coverage.
                            # Persist before slow work so errors and restarts
                            # cannot keep the scan pinned to the same large file.
                            with store.write_transaction():
                                store.set_meta(scan_cursor, str(path))
                        snapshot = read_source(ds, path, project, parent, state)
                        if guard:
                            guard()  # a source can advance during a slow parse
                        stats.parsed_records += snapshot.parsed_records
                        signature = content_key([snapshot.observed_signature, policy])
                        st = snapshot.transcript
                        if st and any(pattern in turn.text for pattern in cfg.ingestion_hold_texts for turn in st.turns):
                            state['hold_signature'] = hold_signature
                            record(ds, path, project, signature, 'held', state, st.session_id,
                                   error='source content matches an operator export hold')
                            stats.recovery_held += recovery
                            continue
                        if st is None:
                            prior = store.session_for_file(str(path))
                            if not snapshot.pending_tail and prior is not None:
                                if recovery:
                                    state.update(recovered_state(snapshot, signature))
                                    state['recovery_only'] = True
                                    state['recovery_notes'] = ['The current source lacks previously indexed activity; the last good memory and original source are preserved.']
                                    record(ds, path, project, signature, 'retained_snapshot', state, prior['id'])
                                    stats.preserved += 1
                                    continue
                                raise SourceChanged('previously indexed source now has no eligible activity')
                            status = 'pending_tail' if snapshot.pending_tail else 'trivial'
                            if recovery:
                                state.update(recovered_state(snapshot, signature))
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
                            if recovery:
                                state.update(recovered_state(snapshot, signature))
                                state['recovery_notes'].append('Previously indexed activity is absent from this partial source; the last good memory is retained.')
                            record(ds, path, project, signature, 'pending_tail', state, st.session_id)
                            continue
                        if st.content_chars < cfg.min_session_chars and not recovery and old is None:
                            record(ds, path, project, signature, 'trivial', state, st.session_id)
                            stats.skipped_trivial += 1
                            continue
                        known_status = old_status or (store.file_status(old['source_path']) if old else None)
                        previous = store.session_rows_with_vectors(st.session_id)
                        if state.get('indexed_units') and state.get('policy') not in (None, policy):
                            # Published under other settings, other prompts or
                            # an earlier release. That work was paid for and is
                            # the person's memory: the source is re-read once
                            # and carried forward here while its content still
                            # matches, never redistilled because how sessions
                            # are processed has changed since. What is appended
                            # to it later is distilled the current way.
                            state['policy'] = policy
                            state.setdefault('models', CARRIED_MODELS)
                        if (not retry_extractive and not redistil and state.get('policy') == policy and state.get('indexed_units') == keys and old is not None):
                            state.update(pending_tail=snapshot.pending_tail,
                                         revisions=[asdict(r) for r in snapshot.revisions])
                            if recovery:
                                state.update(recovered_state(snapshot, signature))
                            adopt(ds, path, project, signature, state, st, old, snapshot.pending_tail, known_status)
                            stats.unchanged += 1
                            continue
                        same_policy = state.get('policy') == policy
                        seed = state.get('seed') if same_policy else None
                        indexed = state.get('indexed_units') or []
                        tracking_matches = keys[:len(indexed)] == indexed
                        seed_matches = not seed or keys[:len(seed['units'])] == seed['units']
                        # A previous reader may have used different completed
                        # boundaries for identical content. Re-verify the paid
                        # prefix before discarding that seed and redistilling.
                        # Policy changes invalidate tracked work by default.
                        # A specifically authorized old policy may adopt only
                        # an exactly verified paid prefix; request caches still
                        # retain their original model identities.
                        # Request checkpoints are keyed by model, so after a model
                        # change an appended session would pay for its earlier
                        # units again. Seed it from its verified paid rows instead.
                        other_models = same_policy and not seed and state.get('models') != legacy_policy
                        if (not retry_extractive and not redistil and
                                (not indexed or (same_policy and (not tracking_matches or not seed_matches))
                                 or other_models or state.get('policy') in cfg.adopt_published_policies)):
                            seed = _bootstrap(snapshot, old, previous, cfg)
                            if seed:
                                seed['status'] = known_status
                        if llm is None and old is not None and old['outcome'] != '(not distilled)':
                            stats.preserved += 1
                            record(ds, path, project, signature, 'awaiting_model', state, st.session_id)
                            continue
                        if seed and seed['units'] == keys and old is not None and not retry_extractive and not redistil:
                            state = {"policy": policy, "indexed_units": keys, "seed": seed,
                                     "pending_tail": snapshot.pending_tail, "revisions": [asdict(r) for r in snapshot.revisions],
                                     "boundary_kinds": snapshot.unit_kinds}
                            if recovery:
                                state.update(recovered_state(snapshot, signature))
                            adopt(ds, path, project, signature, state, st, old, snapshot.pending_tail, known_status)
                            stats.unchanged += 1
                            continue
                        if old is not None and st.content_chars < cfg.min_session_chars:
                            # Try verified-prefix adoption above even for short
                            # historical sessions. An unverifiable tiny rewrite
                            # must not replace a previously paid snapshot.
                            state.update(recovered_state(snapshot, signature))
                            state['recovery_only'] = True
                            state['recovery_notes'] = ['Current source is below the indexing threshold; previous memory retained until sufficient source evidence is available.']
                            record(ds, path, project, signature, 'retained_snapshot', state, st.session_id)
                            stats.preserved += 1
                            continue
                        if recovery:
                            state['recovery_only'] = True
                        record(ds, path, project, signature, 'processing', state, st.session_id)
                        future = coordinators.submit(work, snapshot, seed, pool, guard)
                        futures[future] = (ds, path, project, parent, snapshot, old, previous, state, seed, signature)
                    except SourceNotQuiet as exc:
                        defer_quiet(ds, path, project, state, known_id, exc)
                    except Exception as exc:
                        stats.errors += 1
                        if isinstance(exc, (CodexParseError, SourceReadError, SourceChanged)):
                            state['recovery_only'] = True
                            state['error_class'] = 'source_format'
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
                        if snapshot.recovery_notes:
                            digest.outcome = '[Partial source recovery] ' + ' '.join(snapshot.recovery_notes) + '\n' + digest.outcome
                        rows = _session_rows(st, digest, cfg.chunk_chars)
                        with guard_source_requests(source_guard(ds, path)):
                            stats.embedded_passages += _embed_changed(rows, previous, cache, embedder)
                        # Re-read through the same journal after slow provider
                        # work. Appends can advance the tail; earlier semantic
                        # units (including repair/fork changes) must still match.
                        current = read_source(ds, path, project, parent,
                                              {'recovered_boundaries': snapshot.boundary_records()} if recovery or state.get('recovered_boundaries') else state)
                        stats.parsed_records += current.parsed_records
                        keys = snapshot.unit_keys()
                        if (current.transcript is None or current.unit_keys()[:len(keys)] != keys
                                or current.transcript.cwd != st.cwd or current.transcript.parent_session != st.parent_session):
                            raise SourceChanged("eligible source prefix changed during provider work")
                        pending_units = len(current.unit_keys()) - len(keys)
                        pending = current.pending_tail or pending_units > 0
                        signature = content_key([current.observed_signature, policy])
                        publication = {"policy": policy, "models": legacy_policy, "indexed_units": keys, "seed": seed,
                                       "pending_tail": current.pending_tail, "pending_units": pending_units,
                                       "boundary_kinds": snapshot.unit_kinds,
                                       "indexed_through_at": st.ended_at, "revisions": [asdict(r) for r in snapshot.revisions],
                                       "request_keys": sorted(calls.keys) if calls else []}
                        if recovery or state.get('recovered_boundaries'):
                            publication.update(recovered_state(snapshot, signature))
                            # Only a full nightly read certifies recovery of the
                            # latest raw tail; a daytime completed-prefix read cannot.
                            if not recovery:
                                publication['recovery_signature'] = state.get('recovery_signature')
                        owner = store.owner_for(ds.user)
                        for row in rows:
                            row.owner_user = owner
                        with store.write_transaction():
                            store.upsert_session(id=st.session_id, project=st.project, cwd=st.cwd, git_branch=st.git_branch,
                                                 title=digest.title, summary=digest.summary, outcome=digest.outcome,
                                                 started_at=st.started_at, ended_at=st.ended_at, n_user=st.n_user,
                                                 n_assistant=st.n_assistant, source_path=str(path), parent_session=st.parent_session,
                                                 owner_user=owner)
                            store.ensure_project(st.project, owner=owner, team=ds.team)
                            store.replace_session_rows(st.session_id, rows)
                            store.supersede_source_paths(st.session_id, str(path))
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
                        stats.recovered_partial += bool(recovery and snapshot.recovery_notes)
                        _log(f"[incremental] {st.session_id}: published {len(keys)} units ({status}); pending_tail={pending}")
                    except SourceNotQuiet as exc:
                        defer_quiet(ds, path, project, state, st.session_id, exc)
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
            stats.checkpointed_requests = getattr(pool, 'checkpointed_requests', 0)
            stats.batch_requests = pool.batch_requests
            stats.batched_units = pool.batched_units
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
    selected = {str(row[2]) for row in found}
    for row in store.db.execute('SELECT path,status,state FROM ingest_sources'):
        state = json.loads(row['state'])
        if row['path'] not in selected:
            stats.recovery_unavailable += recovery and not Path(row['path']).exists()
            continue
        if row['status'] in {'pending_work', 'budget_pending', 'source_changed', 'processing'}:
            stats.sources_pending += not state.get('recovery_only')
        if row['status'] in {'error', 'held', 'budget_pending'} or (row['status'] == 'pending_tail' and not state.get('recovery_signature')):
            stats.recovery_pending += 1
    stats.duration_seconds = round(time.monotonic() - started, 3)
    store.commit()
    return stats
