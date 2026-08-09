"""Parse Claude Code session transcripts (.jsonl) into clean conversations.

A transcript file is a stream of heterogeneous JSON records: user/assistant
messages threaded by uuid/parentUuid, tool calls and results, plus harness
noise (file-history snapshots, mode changes, deferred-tool deltas). This module
extracts the actual conversation: human prompts, assistant text, and a compact
account of tool activity. Sidechains (subagent transcripts) are skipped.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Turn:
    role: str          # "user" | "assistant" | "tool" | "tool_result"
    text: str
    ts: str | None = None


@dataclass
class SessionTranscript:
    session_id: str
    project: str            # project slug (directory name)
    source_path: str
    parent_session: str | None = None   # set for subagent transcripts
    cwd: str | None = None
    git_branch: str | None = None
    title: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    turns: list[Turn] = field(default_factory=list)

    @property
    def n_user(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    @property
    def n_assistant(self) -> int:
        return sum(1 for t in self.turns if t.role == "assistant")

    @property
    def content_chars(self) -> int:
        return sum(len(t.text) for t in self.turns if t.role in ("user", "assistant"))


_TOOL_INPUT_KEYS = ("command", "file_path", "path", "pattern", "query", "url", "prompt")


def _summarize_tool_use(block: dict) -> str:
    name = block.get("name", "?")
    inp = block.get("input") or {}
    detail = ""
    for key in _TOOL_INPUT_KEYS:
        if key in inp and isinstance(inp[key], str):
            detail = inp[key]
            break
    if not detail and inp:
        try:
            detail = json.dumps(inp)[:160]
        except (TypeError, ValueError):
            detail = ""
    detail = detail.replace("\n", " ")[:200]
    return f"[tool: {name}] {detail}".rstrip()


def _text_from_content(content) -> str:
    """Extract human-readable text from a message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _clip_middle(text: str, head: int, tail: int) -> str:
    """Keep the head and tail of long text (errors/tracebacks live at the end)."""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n[…{len(text) - head - tail:,} chars elided…]\n{text[-tail:]}"


def _is_harness_text(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(
        (
            "<command-",
            "<local-command",
            "<system-reminder>",
            "<task-notification>",
            "[SYSTEM NOTIFICATION",
            "[Request interrupted",
        )
    )


def parse_transcript(
    path: Path, project: str, parent_session: str | None = None, min_chars: int = 200
) -> SessionTranscript | None:
    session_id = path.stem
    st = SessionTranscript(
        session_id=session_id, project=project, source_path=str(path),
        parent_session=parent_session,
    )
    # In a main-session file, sidechain records are interleaved subagent noise;
    # in a subagent's own file, every record is a sidechain record.
    allow_sidechain = parent_session is not None

    for rec in _iter_records(path):
        rtype = rec.get("type")

        if rtype == "ai-title":
            title = rec.get("aiTitle") or rec.get("title")
            if title:
                st.title = title
            continue
        if rtype == "summary":
            summary = rec.get("summary")
            if summary and not st.title:
                st.title = summary
            continue
        if rtype not in ("user", "assistant"):
            continue
        if rec.get("isSidechain") and not allow_sidechain:
            continue

        ts = rec.get("timestamp")
        if ts:
            if st.started_at is None or ts < st.started_at:
                st.started_at = ts
            if st.ended_at is None or ts > st.ended_at:
                st.ended_at = ts
        if rec.get("cwd") and not st.cwd:
            st.cwd = rec["cwd"]
        if rec.get("gitBranch") and not st.git_branch:
            st.git_branch = rec["gitBranch"]

        msg = rec.get("message") or {}
        content = msg.get("content")

        if rtype == "user":
            if rec.get("isMeta"):
                continue
            text = _text_from_content(content)
            if text and not _is_harness_text(text):
                st.turns.append(Turn("user", text.strip(), ts))
            # tool_result blocks ride user records
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

    if st.n_user == 0 or st.content_chars < min_chars:
        return None
    return st


def _iter_records(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def render_for_llm(st: SessionTranscript, max_chars: int, tool_result_snippet: int) -> str:
    """Render the conversation as compact text for distillation.

    Tool results are truncated hard; if the whole rendering exceeds max_chars,
    keep the head and tail with an elision marker (beginnings carry intent,
    endings carry outcomes). A [date: …] marker precedes every change of day
    so long episodic sessions (cron logs, week-spanning personal-agent chats)
    distill with their timeline intact.

    Repeated boilerplate is collapsed: a user/assistant/tool-result turn whose
    text matches an earlier same-role turn (digits normalized away, so
    timestamped cron/heartbeat prompts still match) is elided into a counter
    marker. Heartbeat sessions would otherwise repeat the same imperative
    prompt ("reply HEARTBEAT_OK") hundreds of times, drowning the signal and
    priming the distiller to obey the transcript instead of summarizing it.
    """
    lines: list[str] = []
    cur_date: str | None = None
    seen: set[tuple[str, str]] = set()
    elided = 0

    def flush_elided() -> None:
        nonlocal elided
        if elided:
            lines.append(f"[…{elided} repeated near-identical turns elided…]")
            elided = 0

    for t in st.turns:
        if t.role in ("user", "assistant", "tool_result"):
            key = (t.role, re.sub(r"\d+", "#", t.text))
            if key in seen:
                elided += 1
                continue
            seen.add(key)
        flush_elided()
        date = t.ts[:10] if t.ts and len(t.ts) >= 10 else None
        if date and date != cur_date:
            lines.append(f"[date: {date}]")
            cur_date = date
        if t.role == "user":
            lines.append(f"[user] {t.text}")
        elif t.role == "assistant":
            lines.append(f"[assistant] {t.text}")
        elif t.role == "tool":
            lines.append(t.text)
        elif t.role == "tool_result":
            head = int(tool_result_snippet * 0.6)
            clipped = _clip_middle(t.text, head, tool_result_snippet - head)
            lines.append(f"[tool-result] {clipped}")
    flush_elided()
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        head = int(max_chars * 0.6)
        tail = max_chars - head
        rendered = (
            rendered[:head]
            + "\n\n[… middle of session elided for length …]\n\n"
            + rendered[-tail:]
        )
    return rendered


# Tool results inside chunks: enough to hold an error signature plus its
# traceback tail, small enough not to crowd out the dialogue in a window.
CHUNK_TOOL_RESULT_HEAD = 350
CHUNK_TOOL_RESULT_TAIL = 250
CHUNK_SPLIT_OVERLAP = 300


def _render_turn_for_chunk(t: Turn) -> str:
    if t.role == "tool":
        return t.text  # already a compact "[tool: Name] detail" line
    if t.role == "tool_result":
        return "[tool-result] " + _clip_middle(
            t.text, CHUNK_TOOL_RESULT_HEAD, CHUNK_TOOL_RESULT_TAIL
        )
    return f"[{t.role}] {t.text}"


def chunk_session(st: SessionTranscript, chunk_chars: int) -> list[tuple[str, str | None]]:
    """Split the full activity stream into exchange-aligned retrieval chunks.

    An exchange is one semantic unit: a user prompt followed by everything the
    agent did in response (assistant text, tool calls verbatim, tool results
    clipped head+tail) up to the next user prompt. Chunks never straddle two
    exchanges; an exchange larger than chunk_chars is windowed within itself
    with a little overlap. Returns (text, timestamp) pairs.
    """
    # Group turns into exchanges, breaking at every user turn.
    exchanges: list[list[Turn]] = []
    for t in st.turns:
        if t.role == "user" or not exchanges:
            exchanges.append([])
        exchanges[-1].append(t)

    chunks: list[tuple[str, str | None]] = []
    for exchange in exchanges:
        ts = exchange[0].ts
        buf: list[str] = []
        buf_len = 0
        for t in exchange:
            piece = _render_turn_for_chunk(t)
            if buf and buf_len + len(piece) > chunk_chars:
                chunks.append(("\n".join(buf), ts))
                # carry a sliver of overlap into the next window of this exchange
                carry = buf[-1][-CHUNK_SPLIT_OVERLAP:]
                buf, buf_len = [f"[…continued] {carry}"], len(carry) + 14
            # a single turn bigger than the window gets hard-split with overlap
            while len(piece) > chunk_chars:
                chunks.append(("\n".join(buf + [piece[:chunk_chars]]), ts))
                piece = "[…continued] " + piece[chunk_chars - CHUNK_SPLIT_OVERLAP :]
                buf, buf_len = [], 0
            buf.append(piece)
            buf_len += len(piece)
        if buf:
            chunks.append(("\n".join(buf), ts))
    return chunks


def discover_transcripts(
    source_dir: Path, exclude: tuple[str, ...] | list[str] = ()
) -> list[tuple[Path, str, str | None]]:
    """Return (path, project_slug, parent_session) for every transcript on disk.

    parent_session is None for main sessions; for subagent transcripts
    (<project>/<parent-session>/subagents/agent-*.jsonl) it is the parent
    session id. `exclude` holds glob patterns matched against project slugs.
    """
    out: list[tuple[Path, str, str | None]] = []
    if not source_dir.is_dir():
        return out
    for proj_dir in sorted(source_dir.iterdir()):
        if not proj_dir.is_dir() or any(
            fnmatch.fnmatch(proj_dir.name, pat) for pat in exclude
        ):
            continue
        for f in sorted(proj_dir.glob("*.jsonl")):
            out.append((f, proj_dir.name, None))
        for f in sorted(proj_dir.glob("*/subagents/*.jsonl")):
            out.append((f, proj_dir.name, f.parent.parent.name))
    return out
