"""Read Codex history without changing its rollouts or native databases.

JSON snapshots, unwrapped JSONL and wrapped JSONL all feed the same event
normalizer. Source records win over UI mirrors and compaction replays. Images
are removed before content comparison; tool results are compared in full,
then clipped once for both downstream consumers. CLI versions are provenance,
not a parser switch: one rollout can contain several generations of records.
"""

from __future__ import annotations

import hashlib
import copy
import json
import re
import sqlite3
import tomllib
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

from .config import CodexSettings, DataStore, to_slug
from .transcript import SessionTranscript, Turn, _clip_middle

NORMALIZER_VERSION = 1
UNATTRIBUTED_PROJECT = "codex-unattributed"
_IMAGE_MARKER = "[image omitted]"
_IMAGE_URI = re.compile(r"data:image/[^\s\"'<>]+", re.IGNORECASE)
_CWD = re.compile(r"<cwd>\s*([^<]+?)\s*</cwd>")
_IMAGE_TYPES = {"image", "input_image", "output_image", "image_url"}
_TEXT_TYPES = {"text", "input_text", "output_text"}
_IGNORED_ITEMS = {
    "reasoning", "compaction", "ghost_snapshot", "tool_search_call",
    "tool_search_output", "image_generation_call", "compaction_summary",
}
_IGNORED_EVENTS = {
    "token_count", "agent_reasoning", "task_started", "task_complete",
    "context_compacted", "thread_settings_applied", "thread_name_updated",
    "sub_agent_activity", "image_generation_end", "turn_aborted",
}
_IGNORED_RECORDS = {
    "world_state", "token_usage_record", "inter_agent_communication_metadata",
}
_HARNESS_PREFIXES = (
    "# AGENTS.md instructions", "<environment_context>", "<user_instructions>",
    "<permissions instructions>", "<skills_instructions>", "<recommended_plugins>",
    "<turn_aborted>", "<in-app-browser-context>", "<system-reminder>",
)


def _user_text(text: str) -> str:
    """Peel known injected blocks without dropping a following real request."""
    text = text.strip()
    while text.startswith(_HARNESS_PREFIXES):
        if text.startswith("# AGENTS.md instructions"):
            return ""
        tag = text[1:text.find(">")].split()[0]
        closing = f"</{tag}>"
        end = text.find(closing)
        if end < 0:
            return ""
        text = text[end + len(closing):].strip()
    return text


class CodexParseError(ValueError):
    """An incomplete/unsupported source must not replace existing good memory."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mirror_fingerprint(event: Event) -> str:
    # UI message text often omits the attachment block entirely. Compare the
    # remaining text while retaining the marker in the canonical source view.
    text = event.text.replace(_IMAGE_MARKER, "")
    text = re.sub(r"\n[ \t]*\n", "\n", text).strip()
    return _digest(event.role + "\0" + text)


def memory_session_id(raw_id: str) -> str:
    return "codex:" + raw_id


@contextmanager
def ingestion_lock(data_dir: Path):
    """Serialize Codex ingesters across processes on the same local store."""
    import fcntl

    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "codex-ingest.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CodexParseError(f"Codex ingestion is already running for {data_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_dedicated_database(path: Path) -> None:
    """Refuse to repurpose a populated Claude/Tess/Assistant store."""
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise CodexParseError(f"cannot open the destination read-only: {exc}") from exc
    try:
        foreign = connection.execute("SELECT 1 FROM sessions WHERE id NOT LIKE 'codex:%' LIMIT 1").fetchone()
        codex = connection.execute("SELECT 1 FROM sessions WHERE id LIKE 'codex:%' LIMIT 1").fetchone()
        plans = connection.execute("SELECT 1 FROM plan_items LIMIT 1").fetchone()
        if foreign or (plans and not codex):
            raise CodexParseError("the destination contains existing non-Codex or unattributed planning state; choose a separate data_dir")
    except sqlite3.Error as exc:
        raise CodexParseError(f"cannot verify the dedicated memory database: {exc}") from exc
    finally:
        connection.close()


def load_dedicated_config(path: Path):
    """Validate the explicit instance before either ingestion or MCP startup."""
    from .config import load_config

    path = path.expanduser()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = load_config(path)
    declared = raw.get("storage", {}).get("data_dir")
    if not declared or not raw.get("data_stores") or any(ds.type != "codex" for ds in cfg.data_stores):
        raise ValueError("Codex memory requires explicit Codex sources and storage.data_dir")
    if cfg.data_dir.resolve() != Path(declared).expanduser().resolve():
        raise ValueError("AGENTIC_MEMORY_DATA conflicts with the dedicated settings file")
    validate_dedicated_database(cfg.db_path)
    return cfg


@dataclass
class ParseReport:
    records: int = 0
    images_removed: int = 0
    duplicate_items: int = 0
    mirrored_items: int = 0
    snapshot_replays: int = 0
    snapshot_recovered: int = 0
    inherited_items: int = 0
    rollback_events: int = 0
    ignored_records: int = 0
    unknown: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {**vars(self), "unknown": dict(self.unknown)}


@dataclass
class Source:
    path: Path
    store: DataStore
    metadata: dict
    session_id: str
    cwd: str | None
    project: str
    subagent: bool
    metadata_revision: tuple | None = None

    @property
    def parent_id(self) -> str | None:
        meta = self.metadata
        base = meta.get("history_base") or {}
        if not isinstance(base, dict):
            raise CodexParseError(f"{self.path}: invalid shared-history reference")
        nested = (meta.get("source") or {}) if isinstance(meta.get("source"), dict) else {}
        subagent = nested.get("subagent")
        spawn = (subagent.get("thread_spawn") or {}) if isinstance(subagent, dict) else {}
        if not isinstance(spawn, dict):
            raise CodexParseError(f"{self.path}: invalid subagent parent metadata")
        parent = (base.get("thread_id") or base.get("rollout_id") or
                  meta.get("forked_from_id") or meta.get("parent_thread_id") or
                  spawn.get("parent_thread_id"))
        if parent is not None and not isinstance(parent, str):
            raise CodexParseError(f"{self.path}: parent identity must be a string")
        return parent


@dataclass
class Event:
    role: str
    text: str
    ts: str | None
    position: int
    offset: int
    identity: str | None = None
    turn_id: str | None = None
    ordinal: int | None = None
    channel: str = "source"
    fingerprint: str = ""

    def __post_init__(self) -> None:
        self.fingerprint = _digest(self.role + "\0" + self.text)


@dataclass
class Normalized:
    source: Source
    events: list[Event]
    report: ParseReport
    copied_metadata: bool = False
    boundaries: list[dict] = field(default_factory=list)
    revisions: list = field(default_factory=list)


@dataclass
class Resolved:
    local: list[Event]
    history: list[Event]
    report: ParseReport
    parent_id: str | None
    boundaries: list[dict] = field(default_factory=list)
    revisions: list = field(default_factory=list)


def _strip_images(value, report: ParseReport):
    """Remove images recursively, including JSON serialized inside tool text."""
    if isinstance(value, str):
        text, count = _IMAGE_URI.subn(_IMAGE_MARKER, value)
        report.images_removed += count
        # MCP results sometimes contain a serialized content array whose image
        # block has separate mimeType/data fields instead of a data URI.
        if (text.lstrip().startswith(("{", "[")) and '"data"' in text
                and ('"image"' in text or "image/" in text)):
            try:
                parsed = json.loads(text)
            except (ValueError, RecursionError):
                pass
            else:
                cleaned = _strip_images(parsed, report)
                return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
        return text
    if isinstance(value, list):
        return [_strip_images(v, report) for v in value]
    if isinstance(value, dict):
        if (value.get("type") in _IMAGE_TYPES or any(
                str(value.get(key, "")).startswith("image/")
                for key in ("mimeType", "mime_type", "media_type"))):
            report.images_removed += 1
            return _IMAGE_MARKER
        return {k: _strip_images(v, report) for k, v in value.items()}
    return value


def _text(value, report: ParseReport) -> str:
    value = _strip_images(value, report)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for v in value if (part := _text_clean(v, report)))
    return _text_clean(value, report)


def _text_clean(value, report: ParseReport) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") in _TEXT_TYPES:
            return str(value.get("text") or "")
        if "content" in value:
            parts = [_text(value["content"], report)]
            for key in ("structuredContent", "error"):
                if value.get(key) is not None:
                    parts.append(f"[{key}] " + _text(value[key], report))
            if value.get("isError"):
                parts.insert(0, "[tool error]")
            return "\n".join(part for part in parts if part)
        if value.get("type") == "resource" and isinstance(value.get("resource"), dict):
            return str(value["resource"].get("text") or "")
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "" if value is None else json.dumps(value, ensure_ascii=False)


def _records(path: Path, max_bytes: int, *, end_byte: int | None = None,
             end_ordinal: int | None = None) -> Iterator[tuple[dict, int, int]]:
    """Read a fixed-length snapshot. Never read outside the selected source."""
    size = path.stat().st_size
    boundary = min(size, end_byte) if end_byte is not None else size
    with path.open("rb") as handle:
        if path.suffix == ".json":
            if boundary > max_bytes:
                raise CodexParseError(f"{path}: JSON snapshot exceeds max_record_bytes={max_bytes}")
            try:
                value = json.loads(handle.read(boundary))
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise CodexParseError(f"{path}: invalid JSON snapshot") from exc
            if not isinstance(value, dict) or not isinstance(value.get("session"), dict) or not isinstance(value.get("items"), list):
                raise CodexParseError(f"{path}: expected a Codex {{session, items}} snapshot")
            yield {"type": "session_meta", "payload": value["session"]}, 0, 0
            for position, item in enumerate(value["items"], 1):
                if not isinstance(item, dict):
                    raise CodexParseError(f"{path}: non-object item {position}")
                yield item, position, 0
            return
        position = 0
        while handle.tell() < boundary:
            offset = handle.tell()
            raw = handle.readline(min(max_bytes + 1, boundary - offset))
            position += 1
            if len(raw) > max_bytes:
                raise CodexParseError(f"{path}:{position}: record exceeds max_record_bytes={max_bytes}")
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError) as exc:
                label = "incomplete final record" if handle.tell() == boundary else "invalid record"
                raise CodexParseError(f"{path}:{position}: {label}; retry after the writer finishes") from exc
            if not isinstance(value, dict):
                raise CodexParseError(f"{path}:{position}: expected an object")
            ordinal = value.get("ordinal")
            if end_ordinal is not None:
                if isinstance(ordinal, int):
                    if ordinal >= end_ordinal:
                        break
                elif end_byte is None:
                    raise CodexParseError(f"{path}: shared history ordinal requires ordinals or a byte boundary")
            yield value, position, offset


def _metadata(path: Path, options: CodexSettings) -> tuple[dict, str | None]:
    records = _records(path, options.max_record_bytes)
    try:
        try:
            first, _, _ = next(records)
        except StopIteration as exc:
            raise CodexParseError(f"{path}: empty rollout; retry after the writer finishes") from exc
        meta = first.get("payload") if first.get("type") == "session_meta" else first
        if not isinstance(meta, dict) or not (meta.get("id") or meta.get("session_id")):
            raise CodexParseError(f"{path}: no session identity in the first record")
        cwd = meta.get("cwd")
        if not cwd:
            # The oldest files put the workspace in an injected user message.
            # Bound discovery; absence is an explicit unattributed project.
            for item, position, offset in records:
                if position > 100 or offset > 1024 * 1024:
                    break
                p = item.get("payload") if isinstance(item.get("payload"), dict) else item
                if p.get("cwd"):
                    cwd = p["cwd"]
                    break
                if p.get("role") == "user":
                    match = _CWD.search(_text(p.get("content"), ParseReport()))
                    if match:
                        cwd = match[1].strip()
                        break
        return meta, str(cwd) if cwd else None
    finally:
        records.close()


class _Normalizer:
    def __init__(self, source: Source, options: CodexSettings):
        self.source = source
        self.options = options
        self.report = ParseReport()
        self.events: list[Event] = []
        self.mirrors: list[Event] = []
        self.snapshots: OrderedDict[str, Event] = OrderedDict()
        self.identities: dict[str, int] = {}
        self.exact: set[tuple] = set()
        self.turn_id: str | None = None
        self.copied_metadata = False
        self.calls: dict[str, str] = {}
        self.boundaries: list[dict] = []
        self.explicit_turns = False
        self.active_turn = False

    def dump_state(self) -> dict:
        return {"events": [asdict(e) for e in self.events], "mirrors": [asdict(e) for e in self.mirrors],
                "snapshots": [[k, asdict(e)] for k, e in self.snapshots.items()],
                "report": self.report.as_dict(), "identities": self.identities,
                "exact": [list(x) for x in self.exact], "turn_id": self.turn_id,
                "copied_metadata": self.copied_metadata, "calls": self.calls,
                "boundaries": self.boundaries, "explicit_turns": self.explicit_turns,
                "active_turn": self.active_turn}

    def restore_state(self, state: dict) -> None:
        self.events = [Event(**e) for e in state.get("events", [])]
        self.mirrors = [Event(**e) for e in state.get("mirrors", [])]
        self.snapshots = OrderedDict((k, Event(**e)) for k, e in state.get("snapshots", []))
        self.report = ParseReport(**state.get("report", {}))
        self.report.unknown = Counter(self.report.unknown)
        self.identities = state.get("identities", {})
        self.exact = {tuple(x) for x in state.get("exact", [])}
        self.turn_id = state.get("turn_id")
        self.copied_metadata = state.get("copied_metadata", False)
        self.calls = state.get("calls", {})
        self.boundaries = state.get("boundaries", [])
        self.explicit_turns = state.get("explicit_turns", False)
        self.active_turn = state.get("active_turn", False)

    def emit(self, role: str, text: str, record: dict, position: int, offset: int,
             identity: str | None = None, channel: str = "source") -> None:
        text = text.strip()
        if not text:
            return
        event = Event(role, text, record.get("timestamp"), position, offset,
                      identity, self.turn_id, record.get("ordinal"), channel)
        if channel == "snapshot":
            if event.fingerprint in self.snapshots:
                self.report.snapshot_replays += 1
            else:
                self.snapshots[event.fingerprint] = event
            return
        if channel != "source":
            self.mirrors.append(event)
            return
        if identity and identity in self.identities:
            # Later versions of one identified item replace the earlier view.
            index = self.identities[identity]
            previous = self.events[index]
            if previous.fingerprint == event.fingerprint:
                self.report.duplicate_items += 1
            else:
                event.position, event.offset = previous.position, previous.offset
                self.events[index] = event
            return
        # No content-only deduplication of genuine user/assistant messages.
        key = (role, event.fingerprint, event.ts, event.ordinal, identity)
        if event.ts is not None and key in self.exact:
            self.report.duplicate_items += 1
            return
        self.exact.add(key)
        if identity:
            self.identities[identity] = len(self.events)
        self.events.append(event)

    def item(self, p: dict, record: dict, position: int, offset: int,
             channel: str = "source") -> None:
        if not isinstance(p, dict):
            raise CodexParseError(f"{self.source.path}:{position}: expected an item object")
        kind = p.get("type")
        if kind in _IGNORED_ITEMS:
            return
        identity = str(p["id"]) if p.get("id") else None
        call_id = str(p["call_id"]) if p.get("call_id") else None
        if kind == "message":
            role = p.get("role")
            if role in ("system", "developer") or p.get("phase") == "analysis":
                return
            if role not in ("user", "assistant"):
                self.report.unknown[f"message role:{role}"] += 1
                return
            text = _text(p.get("content"), self.report).strip()
            if role == "user":
                text = _user_text(text)
            if role == "user" and text.startswith(("<subagent_notification", "<task-notification")):
                role = "tool_result"
            self.emit(role, text, record, position, offset,
                      f"message:{identity}" if identity else None, channel)
        elif kind in ("function_call", "custom_tool_call", "web_search_call"):
            name = str(p.get("name") or ("web_search" if kind == "web_search_call" else "unknown"))
            if call_id:
                self.calls[call_id] = name
            args = p.get("arguments", p.get("input", p.get("action", "")))
            clean = _strip_images(args, self.report)
            text = clean if isinstance(clean, str) else json.dumps(clean, ensure_ascii=False)
            self.emit("tool", f"[tool: {name}] {text}", record, position, offset,
                      f"call:{call_id or identity}" if (call_id or identity) else None, channel)
        elif kind in ("function_call_output", "custom_tool_call_output"):
            self.emit("tool_result", _text(p.get("output"), self.report), record, position, offset,
                      f"output:{call_id or identity}" if (call_id or identity) else None, channel)
        elif kind == "agent_message":
            text = _text(p.get("content"), self.report)
            self.emit("tool_result", f"[agent result] {text}", record, position, offset,
                      f"agent:{identity}" if identity else None, channel)
        else:
            self.report.unknown[f"item:{kind}"] += 1

    def ui_item(self, p: dict, record: dict, position: int, offset: int) -> None:
        if not isinstance(p, dict):
            raise CodexParseError(f"{self.source.path}:{position}: expected a UI item object")
        kind = str(p.get("type", ""))
        folded = kind.replace("_", "").lower()
        identity = p.get("id")
        if folded in ("usermessage", "agentmessage"):
            self.item({**p, "type": "message", "role": "user" if folded == "usermessage" else "assistant"},
                      record, position, offset, "ui_message")
        elif folded in ("reasoning", "contextcompaction", "imageview", "imagegeneration", "subagentactivity"):
            return
        elif folded == "commandexecution":
            command = p.get("command", "")
            if isinstance(command, list):
                command = " ".join(map(str, command))
            self.emit("tool", f"[tool: exec_command] {command}", record, position, offset,
                      f"call:{identity}" if identity else None, "ui_call")
            output = p.get("aggregated_output") or "\n".join(str(p.get(k) or "") for k in ("stdout", "stderr"))
            self.emit("tool_result", _text(output, self.report), record, position, offset,
                      f"output:{identity}" if identity else None, "ui_output")
        elif folded == "mcptoolcall":
            self.item({"type": "function_call", "call_id": identity,
                       "name": p.get("tool"), "arguments": p.get("arguments", {})},
                      record, position, offset, "ui_call")
            self.item({"type": "function_call_output", "call_id": identity,
                       "output": p.get("result", p.get("error", ""))},
                      record, position, offset, "ui_output")
        elif folded in ("filechange", "collabagenttoolcall", "extension", "websearch", "plan"):
            # These UI items can contain evidence absent from response_item.
            clean = {k: v for k, v in p.items() if k not in ("id", "type", "duration")}
            self.emit("tool_result", _text(clean, self.report), record, position, offset,
                      f"ui:{identity}" if identity else None, "ui_output")
        else:
            self.report.unknown[f"ui item:{kind}"] += 1

    def consume(self, record: dict, position: int, offset: int, end_offset: int = 0) -> None:
        before = len(self.events) + len(self.mirrors) + len(self.snapshots)
        self._consume(record, position, offset)
        after = len(self.events) + len(self.mirrors) + len(self.snapshots)
        p = record.get("payload", record)
        if not isinstance(p, dict):
            return
        kind = p.get("type")
        boundary = None
        if record.get("type") == "event_msg" and kind == "task_started":
            self.explicit_turns = True
            self.active_turn = True
        elif record.get("type") == "event_msg" and kind in ("task_complete", "turn_aborted"):
            self.explicit_turns = True
            self.active_turn = False
            boundary = "turn_complete" if kind == "task_complete" else "turn_aborted"
        elif ((kind == "message" and p.get("role") == "user") or
              (record.get("type") == "event_msg" and kind == "user_message")):
            if after > before:
                self.active_turn = True
        elif not self.explicit_turns and (kind == "message" and p.get("role") == "assistant"
                                          and p.get("phase") in (None, "final")):
            if after > before:
                self.active_turn = False
                boundary = "assistant_message"
        elif not self.explicit_turns and record.get("type") == "event_msg" and kind == "agent_message":
            if p.get("phase") in (None, "final"):
                self.active_turn = False
                boundary = "assistant_message"
        if (boundary is None and not self.active_turn and self.boundaries
                and len(self.events) + len(self.mirrors) + len(self.snapshots) > before):
            boundary = "persisted_metadata"
        if boundary:
            self.boundaries.append({"position": position, "offset": end_offset, "kind": boundary})

    def _consume(self, record: dict, position: int, offset: int) -> None:
        self.report.records += 1
        kind = record.get("type")
        if kind in {"session_meta", "response_item", "event_msg", "turn_context", "compacted"} and not isinstance(record.get("payload"), dict):
            raise CodexParseError(f"{self.source.path}:{position}: expected an object payload for {kind}")
        if record == {"record_type": "state"}:
            # Early unwrapped JSONL used an empty state framing record between
            # response items. It is neither a message nor a session header.
            self.report.ignored_records += 1
            return
        if kind == "session_meta" or (kind is None and "id" in record):
            meta = record.get("payload", record)
            if meta.get("id", meta.get("session_id")) != self.source.session_id:
                self.copied_metadata = True
            return
        if kind == "turn_context":
            self.turn_id = record.get("payload", {}).get("turn_id", self.turn_id)
            return
        if kind in _IGNORED_RECORDS:
            self.report.ignored_records += 1
            return
        if kind == "compacted":
            p = record.get("payload") or {}
            history = p.get("replacement_history") or []
            if not isinstance(history, list):
                raise CodexParseError(f"{self.source.path}:{position}: invalid replacement_history")
            for item in history:
                if isinstance(item, dict):
                    self.item(item, record, position, offset, "snapshot")
            if p.get("message"):
                self.emit("assistant", "[saved context] " + _text(p["message"], self.report),
                          record, position, offset, channel="snapshot")
            return
        if kind == "event_msg":
            p = record.get("payload") or {}
            event_type = p.get("type")
            if p.get("turn_id"):
                self.turn_id = p["turn_id"]
            if event_type in ("user_message", "agent_message"):
                self.item({"type": "message", "role": "user" if event_type == "user_message" else "assistant",
                           "content": p.get("message", ""), "phase": p.get("phase")},
                          record, position, offset, str(event_type))
            elif event_type == "task_complete" and p.get("last_agent_message"):
                self.emit("assistant", _text(p["last_agent_message"], self.report), record,
                          position, offset, channel="task_complete")
            elif event_type == "item_completed":
                self.ui_item(p.get("item") or {}, record, position, offset)
            elif event_type == "exec_command_end":
                self.ui_item({**p, "type": "CommandExecution", "id": p.get("call_id")}, record, position, offset)
            elif event_type == "mcp_tool_call_end":
                invocation = p.get("invocation") or {}
                self.ui_item({**p, **invocation, "type": "McpToolCall", "id": p.get("call_id")}, record, position, offset)
            elif event_type == "view_image_tool_call":
                self.item({"type": "function_call", "name": "view_image", "call_id": p.get("call_id"),
                           "arguments": {"path": p.get("path", "")}}, record, position, offset, "ui_call")
            elif event_type == "error":
                # Preserve runtime failures as evidence; opaque transport
                # metadata and encrypted compaction state are not dialogue.
                self.emit("tool_result", "[runtime error] " + _text(p.get("message", ""), self.report)
                          + ("\n[code] " + _text(p["codex_error_info"], self.report)
                             if p.get("codex_error_info") else ""), record, position, offset)
            elif event_type in ("web_search_end", "patch_apply_end"):
                self.ui_item({**p, "type": "Extension" if event_type == "web_search_end" else "FileChange",
                              "id": p.get("call_id")}, record, position, offset)
            elif event_type == "thread_rolled_back":
                # Retrospective memory preserves executed work, including work
                # undone through the UI; label the rollback rather than infer
                # deletion from an incomplete copy of runtime state.
                self.report.rollback_events += 1
                self.emit("tool_result", "[history rollback] " + _text(p, self.report), record, position, offset)
            elif event_type == "thread_goal_updated":
                goal = p.get("goal")
                if isinstance(goal, dict):
                    objective = _text(goal.get("objective"), self.report)
                    text = f"[goal] {objective}\nStatus: {goal.get('status', 'unknown')}"
                else:
                    text = "[goal cleared]"
                # Usage counters change every turn; objective/status changes
                # are durable history and retain their own event identity.
                self.emit("tool_result", text, record, position, offset,
                          identity="goal:" + _digest(text))
            elif event_type not in _IGNORED_EVENTS:
                self.report.unknown[f"event:{event_type}"] += 1
            return
        if kind == "response_item":
            self.item(record.get("payload") or {}, record, position, offset)
        elif kind in {"message", "function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output", "reasoning"}:
            self.item(record, record, position, offset)
        else:
            self.report.unknown[f"record:{kind}"] += 1

    def finish(self) -> Normalized:
        # Finishing reconciles/sorts a view. Reader state must retain original
        # source indices so later append identities can update the right item.
        view = copy.copy(self)
        view.events = list(self.events)
        view.report = copy.deepcopy(self.report)
        return view._finish_view()

    def _finish_view(self) -> Normalized:
        by_content: dict[str, list[Event]] = defaultdict(list)
        by_id = {e.identity: e for e in self.events if e.identity}
        for event in self.events:
            by_content[_mirror_fingerprint(event)].append(event)
        used: set[tuple[str, int]] = set()
        for event in self.mirrors:
            duplicate = bool(event.identity and event.identity in by_id)
            candidates = by_content.get(_mirror_fingerprint(event), [])
            # UI mirrors normally sit adjacent to their response item. Match
            # occurrences rather than globally deleting equal text.
            if not duplicate:
                for source in sorted(candidates, key=lambda e: abs(e.position - event.position)):
                    key = (event.channel, id(source))
                    if key in used:
                        continue
                    same_turn = bool(event.turn_id and event.turn_id == source.turn_id)
                    if same_turn or abs(source.position - event.position) <= 64:
                        used.add(key)
                        duplicate = True
                        break
            if duplicate:
                self.report.mirrored_items += 1
            else:
                self.events.append(event)
                by_content[_mirror_fingerprint(event)].append(event)
                if event.identity:
                    by_id[event.identity] = event
        for event in self.snapshots.values():
            if _mirror_fingerprint(event) in by_content or (event.identity and event.identity in by_id):
                self.report.snapshot_replays += 1
            else:
                self.events.append(event)
                self.report.snapshot_recovered += 1
        self.events.sort(key=lambda e: e.position)
        if self.report.unknown:
            kinds = ", ".join(sorted(self.report.unknown))
            raise CodexParseError(f"{self.source.path}: unsupported history records: {kinds}")
        return Normalized(self.source, self.events, self.report, self.copied_metadata, self.boundaries)


class CodexCatalog:
    """One index over explicitly configured roots, including parent lookups."""

    def __init__(self, stores: list[DataStore], options: CodexSettings | None = None, *, journal=None):
        self.options = options or CodexSettings()
        self.journal = journal
        self.sources: dict[Path, Source] = {}
        self.by_id: dict[str, Source] = {}
        self.errors: dict[str, str] = {}
        self.skipped_subagents = 0
        self.warnings: list[str] = []
        self._cache: OrderedDict[tuple, Normalized] = OrderedDict()
        for store in stores:
            if store.type != "codex":
                continue
            root = store.path.expanduser().resolve()
            if not root.is_dir():
                self.errors[str(root)] = "configured Codex root does not exist"
                continue
            directories = [root] if root.name in ("sessions", "archived_sessions") else [root / "sessions", root / "archived_sessions"]
            for directory in directories:
                if not directory.is_dir():
                    continue
                for path in sorted(directory.rglob("rollout-*")):
                    if not path.is_file():
                        continue
                    if path.suffix not in (".json", ".jsonl"):
                        self.errors[str(path)] = "unsupported rollout extension (compressed history requires a reader)"
                        continue
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root):
                        self.errors[str(path)] = "source symlink is outside the configured root"
                        continue
                    if resolved in self.sources:
                        continue
                    try:
                        meta, cwd = _metadata(resolved, self.options)
                        native_project = to_slug(cwd) if cwd else UNATTRIBUTED_PROJECT
                        if store.is_excluded(native_project):
                            continue
                        source_kind = meta.get("source")
                        subagent = isinstance(source_kind, dict) and "subagent" in source_kind
                        source = Source(resolved, store, meta, str(meta.get("id") or meta["session_id"]),
                                        cwd, native_project, subagent)
                        # Guard lookup as well as discovery: excluded subagents
                        # must not quietly re-enter via a shared-history edge.
                        if subagent and not self.options.include_subagents:
                            self.skipped_subagents += 1
                            continue
                        _ = source.parent_id  # validate references before discovery returns them
                        old = self.by_id.get(source.session_id)
                        if old:
                            self.warnings.append(f"duplicate session id {source.session_id}: selecting the most recently modified source")
                            if old.path.stat().st_mtime_ns >= resolved.stat().st_mtime_ns:
                                continue
                            self.sources.pop(old.path, None)
                        self.sources[resolved] = source
                        self.by_id[source.session_id] = source
                    except (OSError, CodexParseError) as exc:
                        self.errors[str(path)] = str(exc)

    def discover(self, store: DataStore) -> list[tuple[Path, str, str | None]]:
        return [(s.path, s.project, memory_session_id(s.parent_id) if s.parent_id else None)
                for s in self.sources.values() if s.store is store]

    def policy_fingerprint(self, path: Path, project: str) -> str:
        """Revisit unchanged files when options or parent evidence change."""
        source = self.sources[path.resolve()]
        parents = []
        visited = {source.session_id}
        parent_id = source.parent_id
        while parent_id and parent_id not in visited and len(visited) <= 32:
            visited.add(parent_id)
            parent = self.by_id.get(parent_id)
            if parent is None:
                parents.append((parent_id, "missing"))
                break
            stat = parent.path.stat()
            parents.append((parent_id, str(parent.path), stat.st_size, stat.st_mtime_ns))
            parent_id = parent.parent_id
        return _digest(json.dumps({"version": NORMALIZER_VERSION, "options": vars(self.options),
                                   "project": project, "parents": parents}, sort_keys=True))

    def _read(self, source: Source, end_byte: int | None = None, end_ordinal: int | None = None) -> Normalized:
        stat = source.path.stat()
        key = (source.path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, end_byte, end_ordinal)
        meta_revision = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.journal is not None and source.metadata_revision != meta_revision:
            meta, cwd = _metadata(source.path, self.options)
            if str(meta.get('id') or meta.get('session_id')) != source.session_id:
                raise CodexParseError('source session identity changed; rediscover before ingestion')
            source.metadata, source.cwd = meta, cwd
            source.project = to_slug(cwd) if cwd else UNATTRIBUTED_PROJECT
            source.metadata_revision = meta_revision
        if key in self._cache:
            self._cache.move_to_end(key)
            cached = self._cache[key]
            return replace(cached, revisions=[replace(r, parsed_records=0) for r in cached.revisions])
        normalizer = _Normalizer(source, self.options)
        if self.journal is not None:
            policy = "codex-reader-v1:" + _digest(json.dumps(vars(self.options), sort_keys=True))
            holder = []
            def consume(state, record, position, offset, end):
                if not holder:
                    normalizer.restore_state(state)
                    holder.append(normalizer)
                normalizer.consume(record, position, offset, end)
            def after(state, position, end, digest):
                if holder and normalizer.boundaries and normalizer.boundaries[-1]["position"] == position:
                    normalizer.boundaries[-1]["digest"] = digest
            def validate(state):
                if not holder:
                    normalizer.restore_state(state)
                normalizer.finish()
                state.clear()
                state.update(normalizer.dump_state())
            if source.path.suffix == ".json":
                def decode(record):
                    if not isinstance(record.get("session"), dict) or not isinstance(record.get("items"), list):
                        raise CodexParseError("expected a Codex {session, items} snapshot")
                    normalizer.consume({"type": "session_meta", "payload": record["session"]}, 0, 0)
                    for i, item in enumerate(record["items"], 1):
                        normalizer.consume(item, i, 0)
                    normalizer.finish()
                    return normalizer.dump_state()
                state, revision = self.journal.read_snapshot(source.path, policy, decode, self.options.max_record_bytes)
            else:
                state, revision = self.journal.read_jsonl(
                    source.path, policy, initial=normalizer.dump_state, consume=consume,
                    validate=validate, after_record=after, max_record_bytes=self.options.max_record_bytes,
                    end_byte=end_byte, end_ordinal=end_ordinal)
            normalizer.restore_state(state)
            result = normalizer.finish()
            result.revisions = [revision]
            self._cache[key] = result
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
            return result
        for record, position, offset in _records(source.path, self.options.max_record_bytes,
                                                 end_byte=end_byte, end_ordinal=end_ordinal):
            try:
                normalizer.consume(record, position, offset)
            except (UnicodeError, RecursionError) as exc:
                raise CodexParseError(f"{source.path}:{position}: invalid text or excessive nesting") from exc
        result = normalizer.finish()
        self._cache[key] = result
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return result

    def _resolve(self, source: Source, chain: tuple[str, ...] = (),
                 end_byte: int | None = None, end_ordinal: int | None = None) -> Resolved:
        if source.session_id in chain or len(chain) >= 32:
            raise CodexParseError(f"{source.path}: cyclic or excessive parent history")
        normalized = self._read(source, end_byte, end_ordinal)
        # Reports are per invocation; cached source normalization stays intact.
        report = ParseReport(**{**vars(normalized.report),
                                "unknown": normalized.report.unknown.copy(),
                                "warnings": list(normalized.report.warnings)})
        local = list(normalized.events)
        parent_id = source.parent_id
        base = source.metadata.get("history_base")
        inherited: list[Event] = []
        revisions = list(normalized.revisions)
        if parent_id:
            parent = self.by_id.get(parent_id)
            if parent is None:
                if base:
                    raise CodexParseError(f"{source.path}: shared parent {parent_id} is missing from configured sources")
                report.warnings.append(f"parent {parent_id} unavailable; retaining the self-contained rollout history")
            else:
                if base:
                    for key in ("end_byte_offset", "end_ordinal_exclusive"):
                        bound = base.get(key)
                        if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 0):
                            raise CodexParseError(f"{source.path}: invalid shared-history boundary {key}")
                    if base.get("end_byte_offset") is None and base.get("end_ordinal_exclusive") is None:
                        raise CodexParseError(f"{source.path}: shared history has no recorded boundary")
                parent_view = self._resolve(parent, (*chain, source.session_id),
                                            base.get("end_byte_offset") if base else None,
                                            base.get("end_ordinal_exclusive") if base else None)
                inherited = parent_view.history
                revisions.extend(parent_view.revisions)
                ids = {e.identity for e in inherited if e.identity}
                drop: set[int] = {i for i, e in enumerate(local) if e.identity and e.identity in ids}
                dropped_ids = {local[i].identity for i in drop}
                identified_bound = max((i + 1 for i, e in enumerate(inherited)
                                        if e.identity and e.identity in dropped_ids), default=0)
                prefix_bound = 0
                if normalized.copied_metadata and inherited and local:
                    # Legacy forks copy a prefix, including metadata from the
                    # ancestor. Compare ordered events, never a global bag of
                    # text that would erase a later repeated user request.
                    locations: dict[str, list[int]] = defaultdict(list)
                    for i, event in enumerate(inherited):
                        locations[event.fingerprint].append(i)
                    best = 0
                    for start in locations.get(local[0].fingerprint, []):
                        count = 0
                        while (count < len(local) and start + count < len(inherited)
                               and local[count].fingerprint == inherited[start + count].fingerprint):
                            count += 1
                        best = max(best, count)
                        if count and count == best:
                            prefix_bound = start + count
                    drop.update(range(best))
                if not base:
                    # The parent may have continued after an old-style fork.
                    # Only inherited items evidenced in the child's copy are
                    # valid task context for that child.
                    boundary = max(identified_bound, prefix_bound)
                    inherited = inherited[:boundary]
                    if not boundary:
                        report.warnings.append("legacy parent boundary unavailable; retaining local history only")
                report.inherited_items = len(drop)
                local = [event for i, event in enumerate(local) if i not in drop]
        elif base:
            raise CodexParseError(f"{source.path}: unsupported shared-history reference")
        return Resolved(local, inherited + local, report, parent_id, normalized.boundaries, revisions)

    def inspect(self, path: Path, project: str | None = None, min_chars: int = 200,
                *, upto_position: int | None = None) -> tuple[SessionTranscript | None, ParseReport]:
        source = self.sources.get(path.resolve())
        if source is None:
            raise CodexParseError(f"{path}: not a discovered source inside configured roots")
        resolved = self._resolve(source)
        local = resolved.local if upto_position is None else [e for e in resolved.local if e.position <= upto_position]
        # A fork can continue the parent's outstanding task without another
        # human message. Supply one explicitly labelled task for orientation.
        context: Event | None = None
        if local and not any(e.role == "user" for e in local) and resolved.parent_id:
            inherited_history = resolved.history[:-len(resolved.local)] if resolved.local else resolved.history
            context = next((e for e in reversed(inherited_history) if e.role == "user"), None)
        st = SessionTranscript(
            memory_session_id(source.session_id), project or source.store.resolve_project(source.project),
            str(source.path), parent_session=memory_session_id(resolved.parent_id) if resolved.parent_id else None,
            cwd=source.cwd, git_branch=(source.metadata.get("git") or {}).get("branch"),
            source_type="codex",
        )
        if context:
            st.turns.append(Turn("user", f"[task inherited from {memory_session_id(resolved.parent_id)}]\n{context.text}", context.ts))
        budget = self.options.tool_output_chars
        for event in local:
            text = event.text
            if event.role == "tool_result":
                head = budget * 3 // 5
                text = _clip_middle(text, head, budget - head)
            st.turns.append(Turn(event.role, text, event.ts))
        times = [e.ts for e in local if e.ts]
        st.started_at = min(times) if times else source.metadata.get("timestamp")
        st.ended_at = max(times) if times else st.started_at
        if not source.cwd:
            resolved.report.warnings.append("no reliable workspace: using the configured project or codex-unattributed")
        if local and st.n_user == 0 and resolved.parent_id:
            raise CodexParseError(f"{source.path}: continuation has no recoverable task at its parent boundary")
        if not local or st.n_user == 0 or st.content_chars < min_chars:
            return None, resolved.report
        # Compare full cleaned content before clipping. Numeric differences and
        # changes in the middle of a result still invalidate derived memory.
        st.content_fingerprint = _digest(json.dumps({
            "version": NORMALIZER_VERSION, "tool_output_chars": budget,
            "project": st.project, "cwd": st.cwd, "parent": resolved.parent_id,
            "context": context.fingerprint if context else None,
            "events": [(e.role, e.fingerprint, e.ts) for e in local],
        }, sort_keys=True))
        return st, resolved.report

    def parse(self, path: Path, project: str, parent_session: str | None = None,
              min_chars: int = 200) -> SessionTranscript | None:
        return self.inspect(path, project, min_chars)[0]
