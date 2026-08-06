"""Ingestion pipeline: scan transcripts → distill → embed → store.

Incremental: files are keyed by (mtime, size); unchanged files are skipped,
changed sessions are fully re-distilled (their old memories are replaced).
After session ingestion, project profiles and weekly digests are regenerated
for the projects/weeks that were touched.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import Config
from .distill import (
    SessionDigest,
    distill_profile,
    distill_session,
    distill_session_segmented,
    distill_week,
)
from .embed import Embedder
from .llm import LLMBackend, NullBackend
from .store import MemoryRow, MemoryStore
from .transcript import (
    SessionTranscript,
    chunk_session,
    discover_transcripts,
    parse_transcript,
    render_for_llm,
)


@dataclass
class IngestStats:
    scanned: int = 0
    unchanged: int = 0
    skipped_trivial: int = 0
    distilled: int = 0
    extractive: int = 0
    errors: int = 0
    profiles: int = 0
    digests: int = 0


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _extractive_digest(st: SessionTranscript) -> SessionDigest:
    """LLM-free fallback: shallow but never fails."""
    first_user = next((t.text for t in st.turns if t.role == "user"), "")
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
        rows.append(
            MemoryRow(
                level="memory", kind=m.kind, content=m.content,
                importance=m.importance, tags=m.tags, **base,
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
    llm: LLMBackend,
    limit: int | None = None,
    use_llm: bool = True,
) -> IngestStats:
    stats = IngestStats()
    store.assert_embedder(embedder.descriptor)
    if isinstance(llm, NullBackend):
        use_llm = False

    # ---- 1. scan ----
    pending: list[tuple] = []
    for path, project, parent in discover_transcripts(cfg.source_dir):
        stats.scanned += 1
        st_info = path.stat()
        prev = store.file_state(str(path))
        if prev == (st_info.st_mtime, st_info.st_size):
            stats.unchanged += 1
            continue
        pending.append((path, project, parent, st_info))
    if limit is not None:
        pending = pending[:limit]
    _log(f"[ingest] {stats.scanned} transcripts on disk, {len(pending)} new/changed")

    # ---- 2. parse ----
    parsed: list[tuple[SessionTranscript, object]] = []
    for path, project, parent, st_info in pending:
        st = parse_transcript(path, project, parent_session=parent)
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
        try:
            if len(rendered) <= cfg.max_transcript_chars:
                digest = distill_session(llm, cfg.fast_model, st.project, date, rendered)
                return st, digest, "distilled"
            _log(
                f"[ingest] {st.session_id}: {len(rendered):,} chars -> segmented map-reduce"
            )
            digest = distill_session_segmented(
                llm, cfg.fast_model, st.project, date, rendered,
                cfg.segment_chars, cfg.segment_overlap, cfg.max_segments,
            )
            return st, digest, "distilled"
        except Exception as e:  # noqa: BLE001 — degrade, never lose the session
            _log(f"[ingest] distill failed for {st.session_id}: {e}")
            return st, _extractive_digest(st), "extractive"

    touched_projects: set[str] = set()
    touched_weeks: set[str] = set()
    results = []
    if parsed:
        workers = cfg.llm_workers if use_llm else 8
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_distill, st): info for st, info in parsed}
            done = 0
            for fut in as_completed(futures):
                st_info = futures[fut]
                st, digest, status = fut.result()
                results.append((st, digest, status, st_info))
                done += 1
                _log(f"[ingest] ({done}/{len(parsed)}) {status}: {st.project} / {digest.title[:60]}")

    # ---- 4. store + embed per session ----
    for st, digest, status, st_info in results:
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
        store.mark_file(st.source_path, st_info.st_mtime, st_info.st_size, st.session_id, status)
        store.commit()
        if status == "distilled":
            stats.distilled += 1
        else:
            stats.extractive += 1
        touched_projects.add(st.project)
        if st.ended_at:
            wk = _week_start(st.ended_at)
            if wk:
                touched_weeks.add(wk)

    # ---- 5. project profiles (deep model) ----
    if use_llm and touched_projects:
        _log(f"[ingest] regenerating {len(touched_projects)} project profiles ({cfg.deep_model})")
        for project in sorted(touched_projects):
            sessions = store.sessions_for_project(project, limit=30)
            if not sessions:
                continue
            lines = [
                f"- {s['title']} | {(s['ended_at'] or '')[:10]} | {s['summary']} | {s['outcome']}"
                for s in sessions
            ]
            cwd = next((s["cwd"] for s in sessions if s["cwd"]), "")
            try:
                profile = distill_profile(llm, cfg.deep_model, project, cwd, lines)
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

    # ---- 6. weekly digests (fast model) ----
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
                for s in sessions[:40]
            ]
            try:
                text = distill_week(llm, cfg.fast_model, wk, lines)
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
