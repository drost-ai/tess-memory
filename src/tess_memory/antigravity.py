"""Read Google Antigravity conversations (the ``agy`` CLI, the IDE, the app).

Each Antigravity application keeps its data in one folder under ``~/.gemini``
(``antigravity-cli``, ``antigravity-ide``, ``antigravity``). A conversation is
a folder ``brain/<conversation-id>``, and Antigravity writes a plain log of
it to ``.system_generated/logs/transcript_full.jsonl``: one JSON object per
step, appended as the conversation runs. ``transcript.jsonl`` beside it is the
same log with long fields cut short, and is read only when the full one is
missing.

The log does not say where the conversation ran or what it is called. That is
in ``conversation_summaries.db`` (and, as a copy, in
``cache/conversation_metadata.json``), together with the parent of a
conversation a sub-agent held. Nothing here is ever written to.
"""

from __future__ import annotations

import fnmatch
import json
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import to_slug
from .transcript import SessionTranscript, Turn, _iter_records

LOGS = (".system_generated", "logs")
TRANSCRIPTS = ("transcript_full.jsonl", "transcript.jsonl")

_REQUEST = re.compile(r"<USER_REQUEST>\n?(.*)\n?</USER_REQUEST>", re.S)
_HARNESS_BLOCKS = re.compile(r"<(ADDITIONAL_METADATA|USER_SETTINGS_CHANGE)>.*?</\1>", re.S)
_MESSAGE = re.compile(r"<SYSTEM_MESSAGE>\n\[Message\] .*?sender=(\S+) .*?content=(.*?)\n?</SYSTEM_MESSAGE>\s*\Z", re.S)
_STEP_TIMES = re.compile(r"\A(?:(?:Created|Completed) At: [^\n]*(?:\n|\Z))+")
# The argument that says what a tool call was about, most telling first.
_TOOL_ARGUMENTS = ("CommandLine", "Path", "query", "Query", "Url", "Prompt", "Message", "ToolName", "Action")


def conversation_id(path: Path) -> str:
    """The conversation a transcript file belongs to: its ``brain`` folder."""
    return path.parents[len(LOGS)].name


def transcript_file(conversation: Path) -> Path | None:
    logs = conversation.joinpath(*LOGS)
    return next((logs / name for name in TRANSCRIPTS if (logs / name).is_file()), None)


def _workspace(uris) -> str | None:
    """The first workspace folder of a conversation, as a path."""
    if isinstance(uris, str):
        try:
            uris = json.loads(uris)
        except ValueError:
            return None
    for uri in uris if isinstance(uris, list) else []:
        if isinstance(uri, str) and uri:
            return unquote(urlparse(uri).path) if uri.startswith("file://") else uri
    return None


def conversation_index(root: Path) -> dict[str, dict]:
    """What Antigravity records about each conversation, by id: ``workspace``,
    ``title`` and ``parent`` (set for a sub-agent's conversation).

    The summaries database is read first. The metadata cache fills in what it
    lacks, and stands in for it when it cannot be read."""
    index: dict[str, dict] = {}
    try:
        from .audit import readonly_database

        with readonly_database(root / "conversation_summaries.db") as db:
            rows = db.execute("SELECT conversation_id, title, workspace_uris, parent_conversation_id "
                              "FROM conversation_summaries").fetchall() if db else []
        for row in rows:
            index[row[0]] = {"workspace": _workspace(row[2]), "title": row[1] or None, "parent": row[3] or None}
    except (sqlite3.Error, OSError):
        pass
    try:
        cached = json.loads((root / "cache" / "conversation_metadata.json").read_text(encoding="utf-8"))
        cached = cached.get("conversations") if isinstance(cached, dict) else None
    except (OSError, ValueError):
        cached = None
    for key, entry in (cached if isinstance(cached, dict) else {}).items():
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
        known = index.setdefault(key, {"workspace": None, "title": None, "parent": None})
        known["workspace"] = known["workspace"] or _workspace(summary.get("WorkspaceURIs"))
        known["title"] = known["title"] or summary.get("Title") or None
        # The cache does not name the parent, only that the conversation is a sub-agent's.
        known["subagent"] = bool(known["parent"] or entry.get("is_internal"))
    return index


def discover_conversations(
    root: Path, exclude: tuple[str, ...] | list[str] = (), project: str | None = None,
    skip_copies_of: tuple[Path, ...] | list[Path] = (),
) -> list[tuple[Path, str, str | None]]:
    """Return (transcript_path, project_slug, parent) for every conversation.

    The project is named after the conversation's workspace folder. A
    conversation whose workspace Antigravity did not record is kept only when
    the source binds all its sessions to one ``project``. Conversations that
    sub-agents held are left out: the conversation that started them holds
    the report each one sent back."""
    out: list[tuple[Path, str, str | None]] = []
    brain = root / "brain"
    if not brain.is_dir():
        return out
    index = conversation_index(root)
    for conversation in sorted(brain.iterdir()):
        path = transcript_file(conversation) if conversation.is_dir() else None
        if path is None:
            continue
        known = index.get(conversation.name, {})
        if known.get("parent") or known.get("subagent"):
            continue
        if any((origin / "brain" / conversation.name).exists() for origin in skip_copies_of):
            continue
        slug = to_slug(known["workspace"]) if known.get("workspace") else project
        if not slug or any(fnmatch.fnmatch(slug, pattern) for pattern in exclude):
            continue
        out.append((path, slug, None))
    return out


def user_request(content: str) -> str:
    """What the person typed. Antigravity wraps it in ``<USER_REQUEST>`` and
    adds blocks of its own (the time, settings that changed) after it."""
    match = _REQUEST.search(content)
    return (match.group(1) if match else _HARNESS_BLOCKS.sub("", content)).strip()


def _tool_call(call: dict) -> str:
    name, arguments = call.get("name") or "?", call.get("args")
    arguments = arguments if isinstance(arguments, dict) else {}
    detail = next((arguments[key] for key in _TOOL_ARGUMENTS if isinstance(arguments.get(key), str)), "")
    if name == "call_mcp_tool":
        detail = f"{arguments.get('ServerName', '?')}/{arguments.get('ToolName', '?')} " \
                 f"{json.dumps(arguments.get('Arguments', {}), ensure_ascii=False)}"
    elif not detail and arguments:
        detail = json.dumps(arguments, ensure_ascii=False)
    return f"[tool: {name}] {detail.replace(chr(10), ' ')[:200]}".rstrip()


def step_turns(step: dict) -> list[Turn]:
    """One step of the log as conversation turns.

    A person's prompt, the agent's reply and tool calls, each tool's result,
    and a message another agent sent (a sub-agent's report, a timer going
    off) are kept. The agent's private reasoning, and the steps Antigravity
    writes for itself (reminders, compaction summaries, stream errors), are
    not."""
    kind, ts = step.get("type"), step.get("created_at")
    content = step.get("content") if isinstance(step.get("content"), str) else ""
    if kind == "USER_INPUT":
        text = user_request(content)
        media = [item.get("uri") for item in step.get("media") or [] if isinstance(item, dict) and item.get("uri")]
        text = "\n".join([text, *[f"[image: {uri}]" for uri in media]]).strip()
        return [Turn("user", text, ts)] if text else []
    if kind == "PLANNER_RESPONSE":
        turns = [Turn("assistant", content.strip(), ts)] if content.strip() else []
        calls = step.get("tool_calls")
        return turns + [Turn("tool", _tool_call(call), ts) for call in calls if isinstance(call, dict)] \
            if isinstance(calls, list) else turns
    if kind == "GENERIC":
        text = _STEP_TIMES.sub("", content).strip()
        return [Turn("tool_result", text, ts)] if text else []
    if kind == "SYSTEM_MESSAGE":
        message = _MESSAGE.search(content)
        if message and message.group(2).strip():
            return [Turn("tool_result", f"[message from {message.group(1)}] {message.group(2).strip()}", ts)]
    return []


def closes_turn(step: dict) -> bool:
    """Whether a step is the agent's answer: text, with no tool call to wait for."""
    return (step.get("type") == "PLANNER_RESPONSE" and step.get("status") == "DONE"
            and isinstance(step.get("content"), str) and bool(step["content"].strip())
            and not step.get("tool_calls"))


def new_transcript(path: Path, project: str, parent_session: str | None = None) -> SessionTranscript:
    """An empty transcript for the conversation a log file belongs to, with
    what Antigravity records about it."""
    conversation = conversation_id(path)
    known = conversation_index(path.parents[len(LOGS) + 2]).get(conversation, {})
    return SessionTranscript(conversation, project, str(path), parent_session=parent_session or known.get("parent"),
                             cwd=known.get("workspace"), title=known.get("title"))


def parse_conversation(
    path: Path, project: str, parent_session: str | None = None, min_chars: int = 200,
) -> SessionTranscript | None:
    """Parse one conversation log into a SessionTranscript."""
    st = new_transcript(path, project, parent_session)
    steps: dict[int, dict] = {}
    for position, step in enumerate(_iter_records(path)):
        index = step.get("step_index")
        steps[index if isinstance(index, int) and not isinstance(index, bool) else position] = step
    for index in sorted(steps):
        st.turns.extend(step_turns(steps[index]))
    times = [turn.ts for turn in st.turns if turn.ts]
    st.started_at, st.ended_at = (min(times), max(times)) if times else (None, None)
    if st.n_user == 0 or st.content_chars < min_chars:
        return None
    return st
