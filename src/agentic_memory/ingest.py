"""Ingestion pipeline: scan transcripts → distill → embed → store.

Incremental: files are keyed by (mtime, size); unchanged files are skipped,
changed sessions are fully re-distilled (their old memories are replaced).
Streaming: each session is embedded and committed as soon as its
distillation completes, so memory is queryable while a long ingest runs.
After session ingestion, project profiles and weekly digests are regenerated
for the projects/weeks that were touched.
"""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

from .attribution import source_adapter
from .config import Config
from .distill import (
    SessionDigest,
    distill_profile,
    distill_session,
    distill_session_segmented,
    distill_week,
)
from .embed import Embedder
from .llm import AgentCoreBackend, RefusalError
from .store import MemoryRow, MemoryStore
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


def _log(msg: str) -> None:
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


def _week_start(ts: str) -> str | None:
    try:
        d = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None
    return (d - timedelta(days=d.weekday())).isoformat()


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
) -> IngestStats:
    """llm=None means extractive-only ingestion (no distillation calls).

    fallback_llm, when set, is retried whenever the primary refuses (safety
    classifiers) or otherwise fails to distill; only if both fail does a
    session degrade to an extractive digest. retry_extractive re-processes
    unchanged files previously marked extractive (e.g. past refusals).
    """
    stats = IngestStats()
    store.assert_embedder(embedder.descriptor)
    use_llm = llm is not None

    # ---- 1. scan ----
    # retry_extractive is a *targeted repair* mode: re-process ONLY files
    # previously degraded to an extractive digest (past model refusals or
    # distill failures) — new and changed files are ignored entirely, so it
    # never re-does or first-ingests anything.
    retry_mode = retry_extractive and use_llm
    pending: list[tuple] = []
    for ds in cfg.data_stores:
        found, parse = source_adapter(ds)
        for path, source_project, parent in found:
            stats.scanned += 1
            project = ds.resolve_project(source_project)
            st_info = path.stat()
            if retry_mode:
                if store.file_status(str(path)) == "extractive":
                    pending.append((parse, path, project, parent, st_info))
                else:
                    stats.unchanged += 1
                continue
            prev = store.file_state(str(path))
            if prev == (st_info.st_mtime, st_info.st_size):
                stats.unchanged += 1
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
    for parse, path, project, parent, st_info in pending:
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

        def run(backend: AgentCoreBackend) -> SessionDigest:
            if len(rendered) <= cfg.max_transcript_chars:
                return distill_session(
                    backend, st.project, date, rendered, cfg.min_importance
                )
            _log(
                f"[ingest] {st.session_id}: {len(rendered):,} chars -> segmented map-reduce"
            )
            return distill_session_segmented(
                backend, st.project, date, rendered,
                cfg.segment_chars, cfg.segment_overlap, cfg.max_segments,
                cfg.segment_hard_cap, cfg.min_importance,
            )

        try:
            return st, run(llm), "distilled"
        except Exception as e:  # noqa: BLE001 — degrade, never lose the session
            kind = "refusal" if isinstance(e, RefusalError) else "distill failure"
            _log(f"[ingest] {kind} for {st.session_id}: {e}")
            if fallback_llm is not None:
                try:
                    digest = run(fallback_llm)
                    _log(f"[ingest] {st.session_id}: recovered via {fallback_llm.name}")
                    return st, digest, "fallback"
                except Exception as e2:  # noqa: BLE001
                    _log(f"[ingest] fallback also failed for {st.session_id}: {e2}")
            return st, _extractive_digest(st), "extractive"

    # ---- 3b. store + embed each session the moment it finishes ----
    # Streaming, not batch: every distilled session is committed inside the
    # completion loop, so readers (the MCP servers) see memory grow while a
    # long ingest runs, and an interrupted run keeps everything already
    # processed — only in-flight distillations are lost.
    touched_projects: set[str] = set()
    touched_weeks: set[str] = set()

    def _store_session(st: SessionTranscript, digest: SessionDigest, status: str, st_info) -> None:
        rows = _session_rows(st, digest, cfg.chunk_chars)
        vecs = embedder.embed_passages([r.content for r in rows])
        for r, v in zip(rows, vecs):
            r.vector = v
        store.delete_session_memories(st.session_id)
        store.upsert_session(
            id=st.session_id, project=st.project, cwd=st.cwd, git_branch=st.git_branch,
            title=digest.title, summary=digest.summary, outcome=digest.outcome,
            started_at=st.started_at, ended_at=st.ended_at,
            n_user=st.n_user, n_assistant=st.n_assistant, source_path=st.source_path,
            parent_session=st.parent_session,
        )
        store.add_memories(rows)
        # "fallback" is persisted as its own status so fallback usage stays
        # queryable per file; it still counts as fully ingested, and
        # --retry-extractive sweeps only 'extractive' rows, never these.
        store.mark_file(st.source_path, st_info.st_mtime, st_info.st_size, st.session_id, status)
        store.commit()
        if status == "fallback":
            stats.distilled += 1
            stats.fallback += 1
        elif status == "distilled":
            stats.distilled += 1
        else:
            stats.extractive += 1
        touched_projects.add(st.project)
        if st.ended_at:
            wk = _week_start(st.ended_at)
            if wk:
                touched_weeks.add(wk)

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

    # ---- 4. project profiles ----
    if use_llm and touched_projects:
        _log(f"[ingest] regenerating {len(touched_projects)} project profiles")
        for project in sorted(touched_projects):
            sessions = store.sessions_for_project(project, limit=cfg.profile_session_limit)
            if not sessions:
                continue
            lines = [
                f"- {s['title']} | {(s['ended_at'] or '')[:10]} | {s['summary']} | {s['outcome']}"
                for s in sessions
            ]
            cwd = next((s["cwd"] for s in sessions if s["cwd"]), "")
            try:
                try:
                    profile = distill_profile(llm, project, cwd, lines)
                except RefusalError:
                    if fallback_llm is None:
                        raise
                    _log(f"[ingest] profile refusal for {project}; using {fallback_llm.name}")
                    profile = distill_profile(fallback_llm, project, cwd, lines)
            except Exception as e:  # noqa: BLE001
                _log(f"[ingest] profile failed for {project}: {e}")
                stats.errors += 1
                continue
            if not profile:
                continue
            vec = embedder.embed_passages([profile])[0]
            store.delete_scoped("project", project=project)
            store.add_memories([
                MemoryRow(
                    level="project", kind="profile", content=profile, importance=0.9,
                    project=project, ts=sessions[0]["ended_at"], vector=vec,
                )
            ])
            store.commit()
            stats.profiles += 1

    # ---- 5. weekly digests ----
    if use_llm and touched_weeks:
        _log(f"[ingest] regenerating {len(touched_weeks)} weekly digests")
        for wk in sorted(touched_weeks):
            wk_end = (datetime.fromisoformat(wk) + timedelta(days=7)).isoformat()
            sessions = [
                s for s in store.sessions_since(wk)
                if (s["ended_at"] or "") < wk_end
            ]
            if not sessions:
                continue
            lines = [
                f"- {s['project']} | {s['title']} | {s['summary']} | {s['outcome']}"
                for s in sessions[: cfg.digest_session_limit]
            ]
            try:
                try:
                    text = distill_week(llm, wk, lines)
                except RefusalError:
                    if fallback_llm is None:
                        raise
                    _log(f"[ingest] digest refusal for week {wk}; using {fallback_llm.name}")
                    text = distill_week(fallback_llm, wk, lines)
            except Exception as e:  # noqa: BLE001
                _log(f"[ingest] digest failed for week {wk}: {e}")
                stats.errors += 1
                continue
            if not text:
                continue
            vec = embedder.embed_passages([text])[0]
            store.delete_scoped("digest", ts=wk)
            store.add_memories([
                MemoryRow(
                    level="digest", kind="digest", content=text, importance=0.5,
                    ts=wk, vector=vec,
                )
            ])
            store.commit()
            stats.digests += 1

    store.commit()
    return stats
