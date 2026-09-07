"""Ingestion pipeline: scan transcripts → distill → embed → store.

Incremental: files are keyed by (mtime, size); unchanged files are skipped,
changed sessions are fully re-distilled (their old memories are replaced).
Streaming: each session is embedded and committed as soon as its
distillation completes, so memory is queryable while a long ingest runs.
Dirty project/week aggregates survive runs and refresh on a configurable cadence.
An unsuccessful replacement never downgrades an existing model distillation.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .attribution import source_adapter
from .config import Config
from .distill import (
    CallBudgetExceeded,
    DistillationPool,
    SessionDigest,
    distill_profile,
    distill_session,
    distill_session_segmented,
    distill_week,
)
from .embed import Embedder
from .llm import AgentCoreBackend, RefusalError
from .store import MemoryRow, MemoryStore
from .time_utils import week_start as _week_start
from .transcript import SessionTranscript, chunk_session, render_for_llm


@dataclass
class IngestStats:
    scanned: int = 0
    unchanged: int = 0
    skipped_trivial: int = 0
    distilled: int = 0
    fallback: int = 0     # distilled, but by the fallback profile after a refusal/failure
    extractive: int = 0
    errors: int = 0
    profiles: int = 0
    digests: int = 0
    codex_unchanged: int = 0  # file changed, normalized activity did not
    deferred_active: int = 0
    preserved: int = 0       # retained a previous model distillation
    aggregates_pending: int = 0
    duration_seconds: float = 0.0
    # The current text-only backend/embedder interfaces do not expose billed
    # usage. Unknown is explicit, never estimated from characters or prices.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    parsed_records: int = 0
    model_calls: int = 0
    request_cache_hits: int = 0
    embedded_passages: int = 0
    budget_deferred: int = 0


_LOG_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _LOG_LOCK:
        print(msg, file=sys.stderr, flush=True)


# A bracketed harness preamble at the start of a user message: either a paired
# block ("[Turn context] … [/Turn context]") or a lone tag ("[Cron Cron (@cron)
# id:0 | 2026-02-06 08:00 PST]").
_METADATA_BLOCK = re.compile(r"^\[([A-Za-z][^\]]*)\]\s*(?:.*?\[/\1\]\s*)?", re.S)


def _strip_leading_metadata(text: str) -> str:
    """Peel harness preambles off a user message so extractive titles and
    summaries start at the actual request instead of '[Turn context]…'."""
    prev = None
    while text != prev and text.lstrip().startswith("["):
        prev = text
        text = _METADATA_BLOCK.sub("", text.lstrip(), count=1)
    return text.strip()


def _extractive_digest(st: SessionTranscript) -> SessionDigest:
    """LLM-free fallback: shallow but never fails."""
    first_user = _strip_leading_metadata(
        next((t.text for t in st.turns if t.role == "user"), "")
    )
    last_assistant = next(
        (t.text for t in reversed(st.turns) if t.role == "assistant"), ""
    )
    summary = f"User asked: {first_user[:400]}"
    if last_assistant:
        summary += f"\nSession ended with: {last_assistant[:400]}"
    return SessionDigest(
        title=st.title or first_user[:80] or "Untitled session",
        summary=summary,
        outcome="(not distilled)",
        memories=[],
    )


def _session_rows(st: SessionTranscript, digest: SessionDigest, chunk_chars: int) -> list[MemoryRow]:
    rows: list[MemoryRow] = []
    base = dict(session_id=st.session_id, project=st.project, ts=st.ended_at)
    if digest.summary:
        rows.append(
            MemoryRow(
                level="session",
                kind="summary",
                content=f"{digest.title}: {digest.summary} Outcome: {digest.outcome}",
                importance=0.6,
                **base,
            )
        )
    for m in digest.memories:
        # A memory tied to a specific day (distiller-extracted) is stamped
        # with that day, not the session end — long episodic sessions would
        # otherwise skew every memory's recency toward their last message.
        row_kwargs = {**base, "ts": m.date or base["ts"]}
        rows.append(
            MemoryRow(
                level="memory", kind=m.kind, content=m.content,
                importance=m.importance, tags=m.tags, **row_kwargs,
            )
        )
    for text, ts in chunk_session(st, chunk_chars):
        rows.append(
            MemoryRow(
                level="chunk", kind="chunk", content=text, importance=0.3,
                session_id=st.session_id, project=st.project, ts=ts or st.ended_at,
            )
        )
    return rows


def ingest(
    cfg: Config,
    store: MemoryStore,
    embedder: Embedder,
    llm: AgentCoreBackend | None,
    fallback_llm: AgentCoreBackend | None = None,
    limit: int | None = None,
    retry_extractive: bool = False,
    session_ids: set[str] | None = None,
    refresh_aggregates: bool | None = None,
    aggregates_only: bool = False,
) -> IngestStats:
    """Use a process lock only when the requested sources include Codex."""
    with ExitStack() as stack:
        from .locks import ingestion_lock as store_ingestion_lock
        stack.enter_context(store_ingestion_lock(cfg.data_dir))
        if any(ds.type == "codex" for ds in cfg.data_stores):
            from .codex import ingestion_lock

            stack.enter_context(ingestion_lock(cfg.data_dir))
        if cfg.incremental_ingestion and not aggregates_only:
            from .incremental import ingest_incremental
            return ingest_incremental(cfg, store, embedder, llm, fallback_llm, limit=limit,
                                      retry_extractive=retry_extractive, session_ids=session_ids,
                                      refresh_aggregates=refresh_aggregates)
        calls_pool = None
        if llm is not None and (cfg.segment_workers > 1 or cfg.distillation_cache):
            calls_pool = stack.enter_context(DistillationPool(
                llm, fallback_llm, workers=cfg.llm_workers, log=_log,
                cache_path=cfg.data_dir / "distillation-cache.db" if cfg.distillation_cache else None,
            ))
        return _ingest(cfg, store, embedder, llm, fallback_llm, limit, retry_extractive,
                       session_ids, refresh_aggregates, aggregates_only, calls_pool)


def _ingest(
    cfg: Config,
    store: MemoryStore,
    embedder: Embedder,
    llm: AgentCoreBackend | None,
    fallback_llm: AgentCoreBackend | None = None,
    limit: int | None = None,
    retry_extractive: bool = False,
    session_ids: set[str] | None = None,
    refresh_aggregates: bool | None = None,
    aggregates_only: bool = False,
    calls_pool: DistillationPool | None = None,
) -> IngestStats:
    """llm=None means extractive-only ingestion (no distillation calls).

    fallback_llm, when set, is retried whenever the primary refuses (safety
    classifiers) or otherwise fails to distill; only if both fail does a
    NEW session degrade to an extractive digest. Existing model distillations
    are preserved with their old file state, so normal ingestion retries.
    retry_extractive re-processes
    unchanged files previously marked extractive (e.g. past refusals).
    """
    stats = IngestStats()
    started = time.monotonic()
    sources = [] if aggregates_only else cfg.data_stores
    store.assert_embedder(embedder.descriptor)
    use_llm = llm is not None
    codex_catalog = None
    codex_policies: dict[str, str] = {}
    codex_paths: set[str] = set()
    if any(ds.type == "codex" for ds in sources):
        from .codex import CodexCatalog

        codex_catalog = CodexCatalog(cfg.data_stores, cfg.codex)
        if session_ids:
            session_ids = {sid.removeprefix("codex:") for sid in session_ids}
            missing = session_ids - set(codex_catalog.by_id)
            if missing:
                raise ValueError(f"Codex sessions were not discovered: {', '.join(sorted(missing))}")
        for path, error in codex_catalog.errors.items():
            _log(f"[codex] cannot discover {path}: {error}")
            stats.errors += 1
        for warning in codex_catalog.warnings:
            _log(f"[codex] {warning}")

    # ---- 1. scan ----
    # retry_extractive is a *targeted repair* mode: re-process ONLY files
    # previously degraded to an extractive digest (past model refusals or
    # distill failures) — new and changed files are ignored entirely, so it
    # never re-does or first-ingests anything.
    retry_mode = retry_extractive and use_llm
    pending: list[tuple] = []
    for ds in sources:
        found, parse = source_adapter(ds, codex_catalog=codex_catalog)
        for path, source_project, parent in found:
            if session_ids is not None and ds.type == "codex" and codex_catalog.sources[path].session_id not in session_ids:
                continue
            stats.scanned += 1
            project = ds.resolve_project(source_project)
            policy_changed = False
            try:
                st_info = path.stat()
                if ds.type == "codex":
                    codex_paths.add(str(path))
                    policy = codex_catalog.policy_fingerprint(path, project)
                    codex_policies[str(path)] = policy
                    policy_changed = store.get_meta(f"codex:policy:{path}") != policy
            except OSError as exc:
                if ds.type != "codex":
                    raise
                _log(f"[codex] source changed during discovery: {exc}")
                stats.errors += 1
                continue
            if retry_mode:
                if store.file_status(str(path)) == "extractive":
                    if time.time() - st_info.st_mtime < cfg.quiet_seconds:
                        stats.deferred_active += 1
                    else:
                        pending.append((parse, path, project, parent, st_info))
                else:
                    stats.unchanged += 1
                continue
            prev = store.file_state(str(path))
            if prev == (st_info.st_mtime, st_info.st_size) and not policy_changed:
                stats.unchanged += 1
                continue
            if time.time() - st_info.st_mtime < cfg.quiet_seconds:
                stats.deferred_active += 1
                continue
            pending.append((parse, path, project, parent, st_info))
    if limit is not None:
        pending = pending[:limit]
    _log(
        f"[ingest] {stats.scanned} transcripts on disk, {len(pending)} "
        + ("extractive to retry" if retry_mode else "new/changed")
    )

    # ---- 2. parse ----
    parsed: list[tuple[SessionTranscript, object]] = []
    codex_pending: list[tuple] = []
    for parse, path, project, parent, st_info in pending:
        if str(path) in codex_paths:
            codex_pending.append((parse, path, project, parent, st_info))
            continue
        st = parse(
            path, project, parent_session=parent, min_chars=cfg.min_session_chars
        )
        if st is None:
            store.mark_file(str(path), st_info.st_mtime, st_info.st_size, path.stem, "skipped")
            stats.skipped_trivial += 1
            continue
        parsed.append((st, st_info))
    store.commit()

    # ---- 3. distill (parallel) ----
    def _distill(st: SessionTranscript) -> tuple[SessionTranscript, SessionDigest, str]:
        if not use_llm:
            return st, _extractive_digest(st), "extractive"
        rendered = render_for_llm(st, 10**9, cfg.tool_result_snippet)  # uncapped
        date = (st.started_at or "")[:10]
        calls = calls_pool.session(f"{st.source_type}/{st.session_id}") if calls_pool else None

        def run(backend: AgentCoreBackend) -> SessionDigest:
            if len(rendered) <= cfg.max_transcript_chars:
                return distill_session(
                    backend, st.project, date, rendered, cfg.min_importance,
                    **({"calls": calls} if calls else {}),
                )
            _log(
                f"[ingest] {st.session_id}: {len(rendered):,} chars -> segmented map-reduce"
            )
            return distill_session_segmented(
                backend, st.project, date, rendered,
                cfg.segment_chars, cfg.segment_overlap, cfg.max_segments,
                cfg.segment_hard_cap, cfg.min_importance,
                **({"calls": calls, "workers": min(cfg.segment_workers, max(1, cfg.llm_workers))} if calls else {}),
            )

        try:
            digest = run(llm)
            return st, digest, "fallback" if calls and calls.fallback_used else "distilled"
        except Exception as e:  # noqa: BLE001 — degrade, never lose the session
            kind = "refusal" if isinstance(e, RefusalError) else "distill failure"
            _log(f"[ingest] {kind} for {st.session_id}: {e}")
            # The shared pool already attempts fallback per failed request;
            # replaying the whole session here would repeat successful maps.
            if fallback_llm is not None and calls is None:
                try:
                    digest = run(fallback_llm)
                    _log(f"[ingest] {st.session_id}: recovered via {fallback_llm.name}")
                    return st, digest, "fallback"
                except Exception as e2:  # noqa: BLE001
                    _log(f"[ingest] fallback also failed for {st.session_id}: {e2}")
            return st, _extractive_digest(st), "extractive_failed"

    # ---- 3b. store + embed each session the moment it finishes ----
    # Streaming, not batch: every distilled session is committed inside the
    # completion loop, so readers (the MCP servers) see memory grow while a
    # long ingest runs, and an interrupted run keeps everything already
    # processed — only in-flight distillations are lost.
    def _store_session(st: SessionTranscript, digest: SessionDigest, status: str, st_info) -> None:
        if status == "extractive_failed":
            stats.errors += 1
            status = "extractive"
        old = store.get_session(st.session_id)
        old_status = store.file_status(old["source_path"]) if old is not None else None
        if (status == "extractive" and old is not None
                and (old_status in {"distilled", "fallback"}
                     or (old_status != "extractive" and old["outcome"] != "(not distilled)"))):
            stats.preserved += 1
            _log(f"[ingest] preserving previous distillation for {st.session_id}; replacement remains pending")
            return
        rows = _session_rows(st, digest, cfg.chunk_chars)
        vecs = embedder.embed_passages([r.content for r in rows])
        if len(vecs) != len(rows):
            raise ValueError("embedder returned a different number of vectors than session rows")
        for r, v in zip(rows, vecs):
            r.vector = v
        # Parsing and provider work can take minutes. Do not certify a source
        # that changed during that work; the next poll will retry it.
        try:
            current = Path(st.source_path).stat()
        except OSError:
            stats.deferred_active += 1
            return
        if (current.st_mtime, current.st_size) != (st_info.st_mtime, st_info.st_size):
            stats.deferred_active += 1
            return
        with store.write_transaction():
            store.delete_session_memories(st.session_id)
            store.upsert_session(
                id=st.session_id, project=st.project, cwd=st.cwd, git_branch=st.git_branch,
                title=digest.title, summary=digest.summary, outcome=digest.outcome,
                started_at=st.started_at, ended_at=st.ended_at,
                n_user=st.n_user, n_assistant=st.n_assistant, source_path=st.source_path,
                parent_session=st.parent_session,
            )
            store.add_memories(rows)
            store.mark_file(st.source_path, st_info.st_mtime, st_info.st_size, st.session_id, status)
            if st.source_type == "codex":
                store.set_meta(f"codex:normalized:{st.session_id}", st.content_fingerprint)
                store.set_meta(f"codex:policy:{st.source_path}", codex_policies[st.source_path])
            store.mark_aggregate_dirty("project", st.project)
            # Changes can move a session between projects or weeks. Both its
            # old and new aggregates must be repaired.
            if old is not None and old["project"] and old["project"] != st.project:
                store.mark_aggregate_dirty("project", old["project"])
            for ts in {st.ended_at, old["ended_at"] if old is not None else None}:
                wk = _week_start(ts)
                if wk:
                    store.mark_aggregate_dirty("digest", wk)
        if status == "fallback":
            stats.distilled += 1
            stats.fallback += 1
        elif status == "distilled":
            stats.distilled += 1
        else:
            stats.extractive += 1

    if parsed:
        workers = cfg.llm_workers if use_llm else 8
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_distill, st): info for st, info in parsed}
            done = 0
            for fut in as_completed(futures):
                st_info = futures[fut]
                st, digest, status = fut.result()
                done += 1
                _log(f"[ingest] ({done}/{len(parsed)}) {status}: {st.project} / {digest.title[:60]}")
                _store_session(st, digest, status, st_info)

    # Codex can contain gigabytes of replay snapshots. Keep a bounded sliding
    # window of cleaned sessions; free slots refill without a batch barrier.
    if codex_pending:
        from .codex import CodexParseError

        def parsed_codex():
            for _parse, path, project, parent, st_info in codex_pending:
                try:
                    st, report = codex_catalog.inspect(path, project, cfg.min_session_chars)
                    for warning in report.warnings:
                        _log(f"[codex] {path.name}: {warning}")
                    if st is None:
                        # Do not discard an existing good session on a source
                        # that has become empty/trivial after a rewrite.
                        if store.session_for_file(str(path)) is not None:
                            raise CodexParseError(f"{path}: previously ingested session now has no usable activity")
                        store.mark_file(str(path), st_info.st_mtime, st_info.st_size, path.stem, "skipped")
                        store.set_meta(f"codex:policy:{path}", codex_policies[str(path)])
                        store.commit()
                        stats.skipped_trivial += 1
                        continue
                    old = store.get_session(st.session_id)
                    if (not retry_mode and old is not None and
                            store.get_meta(f"codex:normalized:{st.session_id}") == st.content_fingerprint):
                        status = store.file_status(old["source_path"]) or "distilled"
                        # Archiving moves the source. Preserve the paid memory
                        # while keeping get_session_transcript's path current.
                        store.update_session_source(st.session_id, st.source_path)
                        store.mark_file(st.source_path, st_info.st_mtime, st_info.st_size, st.session_id, status)
                        store.set_meta(f"codex:policy:{path}", codex_policies[str(path)])
                        store.commit()
                        stats.codex_unchanged += 1
                        stats.unchanged += 1
                        continue
                    yield st, st_info
                except (OSError, CodexParseError) as exc:
                    _log(f"[codex] not ingested: {exc}")
                    stats.errors += 1
        workers = min(cfg.codex.batch_size, max(1, cfg.llm_workers))
        _log(f"[codex] sliding window of {cfg.codex.batch_size} sessions; "
             f"{workers} session workers, {max(1, cfg.llm_workers)} shared LLM calls, "
             f"up to {min(cfg.segment_workers, max(1, cfg.llm_workers))} segments per session")
        source = iter(parsed_codex())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            exhausted = False
            try:
                while futures or not exhausted:
                    while not exhausted and len(futures) < cfg.codex.batch_size:
                        try:
                            st, info = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        futures[pool.submit(_distill, st)] = info
                    if not futures:
                        break
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        info = futures.pop(future)
                        st, digest, status = future.result()
                        _log(f"[ingest] {status}: {st.project} / {digest.title[:60]}")
                        _store_session(st, digest, status, info)
                    # Completed futures hold full transcripts; release them
                    # before parsing more sources into the bounded window.
                    done.clear()
                    del future, st, digest
            finally:
                for future in futures:
                    future.cancel()

    # ---- 4. durable aggregate refresh ----
    if use_llm and refresh_aggregates is not False:
        interval = 0 if refresh_aggregates is True else cfg.aggregate_refresh_seconds
        _refresh_aggregates(cfg, store, embedder, llm, fallback_llm, stats, interval)
    stats.aggregates_pending = len(store.pending_aggregates(float("inf")))
    stats.duration_seconds = round(time.monotonic() - started, 3)
    store.commit()
    return stats


def _refresh_aggregates(cfg, store, embedder, llm, fallback_llm, stats, interval):
    """Slow model/embedding calls outside transactions; acknowledge only success.

    A source change during generation invalidates the result and leaves dirty
    work for the next run. Empty scopes remove stale aggregates without a model.
    """
    for job in store.pending_aggregates(time.time(), interval):
        level, scope, revision = job["level"], job["scope"], job["revision"]
        try:
            if level == "project":
                sessions = store.sessions_for_project(scope, limit=cfg.profile_session_limit)
                lines = [
                    f"- {s['title']} | {(s['ended_at'] or '')[:10]} | {s['summary']} | {s['outcome']}"
                    for s in sessions
                ]
                cwd = next((s["cwd"] for s in sessions if s["cwd"]), "")

                def generate(backend):
                    return distill_profile(backend, scope, cwd, lines)
            else:
                end = (datetime.fromisoformat(scope) + timedelta(days=7)).isoformat()
                sessions = [s for s in store.sessions_since(scope) if (s["ended_at"] or "") < end]
                lines = [
                    f"- {s['project']} | {s['title']} | {s['summary']} | {s['outcome']}"
                    for s in sessions[:cfg.digest_session_limit]
                ]

                def generate(backend):
                    return distill_week(backend, scope, lines)

            rows = []
            if sessions:
                try:
                    text = generate(llm)
                except CallBudgetExceeded:
                    raise
                except Exception:
                    if fallback_llm is None:
                        raise
                    text = generate(fallback_llm)
                if not text or not text.strip():
                    raise ValueError("empty aggregate response")
                vector = embedder.embed_passages([text])[0]
                rows = [MemoryRow(
                    level=level, kind="profile" if level == "project" else "digest",
                    content=text, importance=0.9 if level == "project" else 0.5,
                    project=scope if level == "project" else None,
                    ts=sessions[0]["ended_at"] if level == "project" else scope,
                    vector=vector,
                )]
            with store.write_transaction():
                if not store.aggregate_is_current(level, scope, revision):
                    continue
                store.delete_scoped(level, project=scope if level == "project" else None,
                                    ts=scope if level == "digest" else None)
                store.add_memories(rows)
                store.acknowledge_aggregate(level, scope, revision, time.time())
            if level == "project":
                stats.profiles += 1
            else:
                stats.digests += 1
        except CallBudgetExceeded:
            stats.budget_deferred += 1
            break
        except Exception as exc:  # keep old aggregate and durable dirty job
            stats.errors += 1
            _log(f"[ingest] {level} refresh failed for {scope}: {exc}")
