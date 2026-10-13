"""Parse agent-core runtime sessions into clean conversations.

agent-core has three runtime implementations (TypeScript, Rust, Python) that
share one session convention: a flat sessions/ directory holding a
<uuid>.json metadata document per session, plus a <uuid>/ directory with
full-transcript-with-tools.jsonl. The runtimes diverge in where message
bodies live — TypeScript and Rust inline a `messages` array in the metadata,
the Python runtime externalizes it to a <uuid>.messages.jsonl sidecar
(bodyRef) — and in which auxiliary event types they log alongside `message`
records (provider provenance, tool-execution telemetry, repairs). This module
normalizes all of it into the same SessionTranscript/Turn shape the
claude-code parser produces, so everything downstream stays source-agnostic.

It also implements type="transcript": a directory of bare per-session
<name>.jsonl files in the same message format but with no metadata document
(e.g. sessions exported from another agent), attributed to a configured
project.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from .config import to_slug
from .transcript import (
    SessionTranscript,
    Turn,
    _is_harness_text,
    _iter_records,
    _summarize_tool_use,
    _text_from_content,
)


def _load_meta(meta_path: Path) -> dict | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return meta if isinstance(meta, dict) else None


def _session_cwd(meta: dict) -> str | None:
    if meta.get("cwd"):
        return meta["cwd"]
    md = meta.get("metadata") or {}
    for key in ("originalCwd", "workspaceRoot"):
        if md.get(key):
            return md[key]
    return None


def discover_sessions(
    source_dir: Path, exclude: tuple[str, ...] | list[str] = (),
    skip_copies_of: tuple[Path, ...] | list[Path] = (),
) -> list[tuple[Path, str, str | None]]:
    """Return (meta_path, project_slug, parent_session) for every session.

    The metadata file is the change-detection key: every runtime rewrites it
    on each session update (updatedAt at minimum), so its (mtime, size) moves
    whenever the session grows. Metas that cannot be parsed or carry no cwd
    are skipped — a session that can't be attributed to a project is noise.
    So is a session whose metadata file also exists in one of
    ``skip_copies_of``: a copy of one ingested from there.
    """
    out: list[tuple[Path, str, str | None]] = []
    if not source_dir.is_dir():
        return out
    for meta_path in sorted(source_dir.glob("*.json")):
        if any((origin / meta_path.name).exists() for origin in skip_copies_of):
            continue
        meta = _load_meta(meta_path)
        if meta is None:
            continue
        cwd = _session_cwd(meta)
        if not cwd:
            continue
        slug = to_slug(cwd)
        if any(fnmatch.fnmatch(slug, pat) for pat in exclude):
            continue
        parent = (meta.get("metadata") or {}).get("parentSessionId")
        out.append((meta_path, slug, parent))
    return out


def _messages_from_transcript(path: Path) -> list[tuple[str, object, str | None]]:
    """(role, content, ts) from full-transcript-with-tools.jsonl.

    Only `message` records carry conversation; the runtimes' other event
    types (tool-execution telemetry, provider provenance, repairs, …) are
    skipped, as are unparseable lines — corrupt records exist in real stores.
    """
    out = []
    for rec in _iter_records(path):
        if rec.get("type") != "message":
            continue
        role = rec.get("role")
        if role in ("user", "assistant"):
            out.append((role, rec.get("content"), rec.get("persisted_at")))
    return out


def _messages_from_sidecar(path: Path) -> list[tuple[str, object, str | None]]:
    """(role, content, ts) from a Python-runtime .messages.jsonl sidecar.

    Message bodies nest under a `message` key. The stream can rewrite
    history (`truncate` then re-append at the same indices), so entries are
    keyed by message_index with later records winning; `commit` framing
    records carry no content and are ignored.
    """
    by_index: dict[int, tuple[str, object, str | None]] = {}
    fallback_index = 0
    for rec in _iter_records(path):
        if rec.get("type") != "message":
            continue
        msg = rec.get("message") or {}
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        idx = rec.get("message_index")
        if not isinstance(idx, int):
            idx = fallback_index
        fallback_index = idx + 1
        by_index[idx] = (role, msg.get("content"), rec.get("persisted_at"))
    return [by_index[i] for i in sorted(by_index)]


def _messages_inline(meta: dict) -> list[tuple[str, object, str | None]]:
    """(role, content, ts) from a TS/Rust metadata `messages` array."""
    out = []
    for msg in meta.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("user", "assistant"):
            out.append((role, msg.get("content"), None))
    return out


def _append_message_turns(
    st: SessionTranscript, role: str, content, ts: str | None
) -> None:
    """Normalize one message into Turn records, matching the claude-code
    parser's semantics: harness-injected user text is dropped, tool_result
    blocks ride user messages, thinking blocks are ignored."""
    if role == "user":
        text = _text_from_content(content)
        if text and not _is_harness_text(text):
            st.turns.append(Turn("user", text.strip(), ts))
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    rtext = _text_from_content(block.get("content"))
                    if rtext:
                        st.turns.append(Turn("tool_result", rtext.strip(), ts))
    else:  # assistant
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        st.turns.append(Turn("assistant", text, ts))
                elif btype == "tool_use":
                    st.turns.append(Turn("tool", _summarize_tool_use(block), ts))
        elif isinstance(content, str) and content.strip():
            st.turns.append(Turn("assistant", content.strip(), ts))


def discover_transcript_files(
    source_dir: Path, exclude: tuple[str, ...] | list[str] = (),
    project: str | None = None, skip_copies_of: tuple[Path, ...] | list[Path] = (),
) -> list[tuple[Path, str, str | None]]:
    """Discovery for type="transcript": bare per-session JSONL files.

    A directory of root-level <name>.jsonl files, each one whole session in
    the agent-core transcript message format, with no metadata document —
    e.g. sessions exported from another agent. With no cwd to derive a slug
    from, every session lands in the store's configured `project` (required).
    Sidecars (*.messages.jsonl) belong to native agent-core sessions that may
    share the directory and are skipped, as are per-session subdirectories.
    """
    if not project:
        raise ValueError(
            "transcript data_store needs an explicit 'project' (bare jsonl "
            "files carry no cwd to derive one from)"
        )
    out: list[tuple[Path, str, str | None]] = []
    if not source_dir.is_dir():
        return out
    if any(fnmatch.fnmatch(project, pat) for pat in exclude):
        return out
    for path in sorted(source_dir.glob("*.jsonl")):
        if path.name.endswith(".messages.jsonl"):
            continue
        if any((origin / path.name).exists() for origin in skip_copies_of):
            continue
        out.append((path, project, None))
    return out


def parse_transcript_file(
    path: Path, project: str, parent_session: str | None = None,
    min_chars: int = 200,
) -> SessionTranscript | None:
    """Parse one bare transcript JSONL file into a SessionTranscript.

    Same message records as full-transcript-with-tools.jsonl; session
    identity is the filename stem and the time span comes from the records'
    persisted_at (there is no metadata document to consult)."""
    st = SessionTranscript(
        session_id=path.stem,
        project=project,
        source_path=str(path),
        parent_session=parent_session,
    )
    for role, content, ts in _messages_from_transcript(path):
        if ts:
            if st.started_at is None or ts < st.started_at:
                st.started_at = ts
            if st.ended_at is None or ts > st.ended_at:
                st.ended_at = ts
        _append_message_turns(st, role, content, ts)

    if st.n_user == 0 or st.content_chars < min_chars:
        return None
    return st


def parse_session(
    meta_path: Path, project: str, parent_session: str | None = None,
    min_chars: int = 200,
) -> SessionTranscript | None:
    meta = _load_meta(meta_path)
    if meta is None:
        return None
    session_id = str(meta.get("id") or meta_path.stem)
    md = meta.get("metadata") or {}

    st = SessionTranscript(
        session_id=session_id,
        project=project,
        source_path=str(meta_path),
        parent_session=parent_session or md.get("parentSessionId"),
        cwd=_session_cwd(meta),
        git_branch=md.get("gitBranch"),
        title=meta.get("name") or (meta.get("firstUserText") or "")[:80] or None,
        started_at=meta.get("createdAt"),
        ended_at=meta.get("updatedAt"),
    )

    # Body sources, best first: the full transcript is the only one that
    # carries tool results in every runtime; the sidecar (Python) and the
    # inline array (TS/Rust) cover sessions whose transcript dir is missing.
    stem = meta_path.stem
    transcript = meta_path.parent / stem / "full-transcript-with-tools.jsonl"
    sidecar = meta_path.with_name(f"{stem}.messages.jsonl")

    if transcript.is_file():
        messages = _messages_from_transcript(transcript)
    elif sidecar.is_file():
        messages = _messages_from_sidecar(sidecar)
    else:
        messages = _messages_inline(meta)

    for role, content, ts in messages:
        _append_message_turns(st, role, content, ts)

    if st.n_user == 0 or st.content_chars < min_chars:
        return None
    return st
