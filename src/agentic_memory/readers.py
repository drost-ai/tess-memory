"""Incremental source adapters producing eligible, source-labelled message units."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

from .agent_core import _append_message_turns, _session_cwd
from .codex import CodexCatalog, ParseReport, _strip_images
from .config import Config, DataStore
from .source_journal import SourceJournal, SourceReadError, SourceRevision
from .transcript import SessionTranscript, _is_harness_text, _text_from_content

READER_VERSION = 1


def content_key(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class ParsedSource:
    transcript: SessionTranscript | None
    unit_ends: list[int]
    unit_kinds: list[str]
    revisions: list[SourceRevision]
    pending_tail: bool
    observed_signature: str
    parsed_records: int
    recovery_notes: list[str] = field(default_factory=list)

    def unit_keys(self) -> list[str]:
        if self.transcript is None:
            return []
        result, start = [], 0
        for end in self.unit_ends:
            result.append(content_key([asdict(t) for t in self.transcript.turns[start:end]]))
            start = end
        return result

    def boundary_records(self):
        return [{'end': end, 'key': key, 'kind': kind} for end, key, kind in
                zip(self.unit_ends, self.unit_keys(), self.unit_kinds, strict=True)]

    def retain_boundaries(self, records):
        """Reuse an incomplete unit after a later append closes that turn.

        Each earlier unit must still match exact ordered content. Edits discard
        the remaining layout; they never bless an unverified prefix.
        """
        if not self.transcript or not records:
            return self
        start, kept = 0, {}
        for row in records:
            end = row['end']
            if end <= start or end > len(self.transcript.turns):
                break
            key = content_key([asdict(t) for t in self.transcript.turns[start:end]])
            if row['key'] != key:
                break
            kept[end] = row['kind']
            start = end
        boundaries = {end: kind for end, kind in zip(self.unit_ends, self.unit_kinds)
                      if end > start}
        boundaries.update(kept)
        self.unit_ends = sorted(boundaries)
        self.unit_kinds = [boundaries[end] for end in self.unit_ends]
        return self


def source_signature(path: Path, ds: DataStore, cfg: Config, *, catalog=None) -> str:
    from .audit import source_members
    members = []
    for member in source_members(path, ds.type):
        s = member.stat()
        members.append((str(member.resolve()), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    parent_policy = catalog.policy_fingerprint(path, ds.resolve_project(catalog.sources[path.resolve()].project)) if catalog else None
    return content_key([READER_VERSION, ds.type, ds.project, members, parent_policy])


class CompletedSourceReader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.journal = SourceJournal(cfg.data_dir / "source-cache.db")
        self.catalog = (CodexCatalog(cfg.data_stores, cfg.codex, journal=self.journal)
                        if any(ds.type == "codex" for ds in cfg.data_stores) else None)

    def read(self, ds: DataStore, path: Path, project: str, parent: str | None = None) -> ParsedSource:
        if not path.resolve().is_relative_to(ds.path.resolve()):
            raise SourceReadError("source resolves outside the configured root")
        if ds.type == "codex":
            return self._codex(ds, path, project)
        return self._generic(ds, path, project, parent)

    def _codex(self, ds, path, project):
        catalog = self.catalog
        source = catalog.sources[path.resolve()]
        view = catalog._resolve(source)
        boundary = view.boundaries[-1]["position"] if view.boundaries else 0
        recovery = self.cfg.recovery_ingestion
        st, _ = catalog.inspect(path, project, min_chars=0,
                                upto_position=None if recovery else boundary,
                                allow_orphan=recovery)
        eligible = [e for e in view.local if recovery or e.position <= boundary]
        extra = len(st.turns) - len(eligible) if st else 0  # inherited task context
        ends, kinds = [], []
        positions = [e.position for e in eligible]
        for closed in view.boundaries:
            count = extra + bisect_right(positions, closed["position"])
            if count and (not ends or count > ends[-1]):
                ends.append(count)
                kinds.append(closed["kind"])
        if st and st.turns and (not ends or ends[-1] != len(st.turns)):
            ends.append(len(st.turns))
            kinds.append("incomplete_snapshot" if recovery else "assistant_message")
        pending = (any(e.position > boundary for e in view.local) or
                   any(r.partial_record for r in view.revisions))
        notes = []
        if recovery and pending:
            notes.append('Incomplete persisted history; no task completion is implied.')
        if recovery and any(r.partial_record for r in view.revisions):
            notes.append('An incomplete final record remains in the original source for later recovery.')
        if recovery and st and st.n_user == 0:
            notes.append('No user request is recoverable; available assistant/tool evidence is preserved without inventing one.')
        return ParsedSource(st, ends, kinds, view.revisions, pending,
                            source_signature(path, ds, self.cfg, catalog=catalog),
                            sum(r.parsed_records for r in view.revisions), notes)

    @staticmethod
    def _initial():
        return {"records": [], "messages": {}, "next_index": 0, "committed": None,
                "has_commit": False, "uncommitted": False}

    def _generic(self, ds, path, project, parent):
        policy = f"messages-v{READER_VERSION}:{ds.type}"
        revisions = []
        meta = {}
        if ds.type == "agent-core":
            # Persist only fields relevant to identity, attribution and body
            # selection; runtime/provider metadata is not model input.
            def metadata(value):
                result = {k: value[k] for k in ("id", "cwd", "createdAt", "updatedAt", "name", "messages") if k in value}
                md = value.get("metadata") or {}
                if not isinstance(md, dict):
                    raise SourceReadError("session metadata must be an object")
                result["metadata"] = {k: md[k] for k in ("parentSessionId", "originalCwd", "workspaceRoot", "gitBranch") if k in md}
                return _strip_images(result, ParseReport())
            meta, revision = self.journal.read_snapshot(path, policy + ":meta", metadata)
            revisions.append(revision)
            full = path.parent / path.stem / "full-transcript-with-tools.jsonl"
            sidecar = path.with_name(path.stem + ".messages.jsonl")
            body = full if full.is_file() else sidecar if sidecar.is_file() else None
            flavor = "sidecar" if body == sidecar else "full"
        else:
            body, flavor = path, "claude" if ds.type == "claude-code" else "full"
        if body is not None and not body.resolve().is_relative_to(ds.path.resolve()):
            raise SourceReadError("body resolves outside the configured source root")

        def consume(state, raw, position, start, end):
            record = _strip_images(raw, ParseReport())
            if not isinstance(record, dict):
                return
            kind = record.get("type")
            if flavor == "claude":
                if kind in {"user", "assistant", "ai-title", "summary"}:
                    state["records"].append(record)
                return
            if kind in {"truncate", "repair"}:
                count = record.get("message_count", record.get("truncated_to_message_index"))
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise SourceReadError("invalid message truncation boundary")
                state["messages"] = {k: v for k, v in state["messages"].items() if int(k) < count}
                state["next_index"] = count
                state["uncommitted"] = True
            elif kind == "message":
                message = record.get("message") if flavor == "sidecar" else record
                if not isinstance(message, dict):
                    raise SourceReadError("message must be an object")
                index = record.get("message_index", state["next_index"])
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise SourceReadError("invalid message index")
                state["messages"][str(index)] = {"role": message.get("role"), "content": message.get("content"),
                                                "stop_reason": message.get("stop_reason", message.get("stopReason")),
                                                "timestamp": record.get("persisted_at")}
                state["next_index"] = index + 1
                state["uncommitted"] = True
            elif kind == "commit" and flavor == "sidecar":
                count = record.get("message_count")
                if count is not None and (not isinstance(count, int) or isinstance(count, bool)
                                          or set(map(int, state["messages"])) != set(range(count))):
                    raise SourceReadError("sidecar commit does not match message indices")
                state["committed"] = dict(state["messages"])
                state["has_commit"] = True
                state["uncommitted"] = False

        if body is not None:
            state, revision = self.journal.read_jsonl(body, policy + ":" + flavor, initial=self._initial,
                                                      consume=consume, max_record_bytes=self.cfg.codex.max_record_bytes)
            revisions.append(revision)
            if flavor == "claude":
                records = state["records"]
            else:
                messages = state["committed"] if state["has_commit"] and not self.cfg.recovery_ingestion else state["messages"]
                records = [messages[k] for k in sorted(messages, key=int)]
        else:
            state = self._initial()
            records = _strip_images(meta.get("messages", []), ParseReport())
            if not isinstance(records, list):
                raise SourceReadError("inline messages must be an array")
        st = SessionTranscript(str(meta.get("id") or path.stem), project, str(path),
                               parent_session=parent or (meta.get("metadata") or {}).get("parentSessionId"),
                               cwd=_session_cwd(meta) if meta else None,
                               git_branch=(meta.get("metadata") or {}).get("gitBranch"),
                               title=meta.get("name"))
        ends, kinds = [], []
        def close(kind):
            if st.turns and (not ends or len(st.turns) > ends[-1]):
                ends.append(len(st.turns))
                kinds.append(kind)
        for rec in records:
            if not isinstance(rec, dict):
                raise SourceReadError("message entry must be an object")
            if flavor == "claude":
                if rec.get("type") == "ai-title":
                    st.title = rec.get("aiTitle") or rec.get("title") or st.title
                    continue
                if rec.get("type") == "summary":
                    st.title = st.title or rec.get("summary")
                    continue
                if rec.get("isSidechain") and not parent:
                    continue
                if rec.get("type") == "user" and rec.get("isMeta"):
                    continue
                role = rec.get("type")
                msg = rec.get("message") or {}
                if not isinstance(msg, dict):
                    raise SourceReadError("Claude message must be an object")
                st.cwd = st.cwd or rec.get("cwd")
                st.git_branch = st.git_branch or rec.get("gitBranch")
                ts = rec.get("timestamp")
            else:
                role, msg, ts = rec.get("role"), rec, rec.get("timestamp")
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content")
            text = _text_from_content(content)
            if (role == "user" and text and not _is_harness_text(text) and st.turns
                    and any(t.role in {"assistant", "tool", "tool_result"} for t in st.turns[(ends[-1] if ends else 0):])):
                close("interrupted_by_next_user")
            _append_message_turns(st, role, content, ts)
            has_tools = isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
            reason = msg.get("stop_reason", msg.get("stopReason"))
            if role == "assistant" and text and not has_tools and reason not in ("tool_use", "max_tokens", "length"):
                close("turn_complete" if reason in ("end_turn", "stop", "stop_sequence") else "assistant_message")
        eligible = ends[-1] if ends else 0
        pending = (eligible < len(st.turns) or any(r.partial_record for r in revisions)
                   or bool(flavor == "sidecar" and state["has_commit"] and state["uncommitted"]))
        notes = []
        if self.cfg.recovery_ingestion:
            if eligible < len(st.turns):
                ends.append(len(st.turns))
                kinds.append('incomplete_snapshot')
                eligible = len(st.turns)
            if pending:
                notes.append('Incomplete persisted history; no task completion is implied.')
            if flavor == 'sidecar' and state['uncommitted']:
                notes.append('Persisted messages lack a final commit; this is an uncommitted snapshot.')
            if any(r.partial_record for r in revisions):
                notes.append('An incomplete final record remains in the original source for later recovery.')
            if st.turns and not st.n_user:
                notes.append('No user request is recoverable; preserve available evidence without inventing one.')
        st.turns = st.turns[:eligible]
        times = [t.ts for t in st.turns if t.ts]
        st.started_at = min(times) if times else meta.get("createdAt")
        st.ended_at = max(times) if times else meta.get("updatedAt")
        if not st.turns or (not st.n_user and not self.cfg.recovery_ingestion):
            st = None
        return ParsedSource(st, ends, kinds, revisions, pending, source_signature(path, ds, self.cfg),
                            sum(r.parsed_records for r in revisions), notes)
