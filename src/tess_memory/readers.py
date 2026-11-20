"""Incremental source adapters producing eligible, source-labelled message units."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from bisect import bisect_right
import hashlib
import json
from pathlib import Path

from .engine_sessions import _append_message_turns, _session_cwd
from .codex import CodexCatalog, ParseReport, _strip_images
from .config import Config, DataStore
from .source_journal import SourceChanged, SourceJournal, SourceReadError, SourceRevision
from .transcript import SessionTranscript, _is_harness_text, _text_from_content

READER_VERSION = 1


def content_key(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def turn_key(turn):
    # Preserve historical keys when there is no explicit authorship metadata.
    value = {'role':turn.role, 'text':turn.text, 'ts':turn.ts}
    if turn.origin is not None:
        value.update(origin=turn.origin, human_evidence=turn.human_evidence)
    return value


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
    # Exposed for personal gap policy; coding parsing/eligibility is unchanged.
    message_indices: list[int] | None = None

    def unit_keys(self) -> list[str]:
        if self.transcript is None:
            return []
        result, start = [], 0
        for end in self.unit_ends:
            result.append(content_key([turn_key(t) for t in self.transcript.turns[start:end]]))
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
            key = content_key([turn_key(t) for t in self.transcript.turns[start:end]])
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


def _published(state: dict, meta: dict) -> tuple[dict, bool]:
    """The messages of the body commit the session head names, and whether
    the body holds a later commit that the head does not name.

    The engine appends messages and a commit marker to the body, then
    publishes that commit's id in the session head. If it stops in between,
    the commit was never published: the engine reads the session back as of
    the head's commit and removes the rest at its next save. Reading the
    body's last commit here would ingest history the engine discards."""
    head = meta.get("bodyCommitId")
    if not head or state["commit_id"] is None:
        return state["committed"], False  # a body or a head from before commits carried ids
    if state["commit_id"] == head:
        messages, unpublished = state["committed"], False
    else:
        previous = state["previous"]
        if not previous or previous["id"] != head:
            # The head moved between reading it and reading the body, or the
            # two do not belong together: read again later, guess nothing.
            raise SourceChanged("session head and body name different commits")
        messages = previous["messages"]
        if messages is None:
            messages = {k: v for k, v in state["committed"].items() if int(k) < previous["count"]}
        unpublished = True
    count = meta.get("messageCount")
    if count is not None and count < len(messages):
        messages = {k: v for k, v in messages.items() if int(k) < count}
    return messages, unpublished


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
                "has_commit": False, "uncommitted": False,
                "commit_id": None, "previous": None, "touched": None}

    def _generic(self, ds, path, project, parent):
        policy = f"messages-v{READER_VERSION}:origin-v2:{ds.type}"
        revisions = []
        meta = {}
        unpublished = False  # the body holds a commit the session head does not name yet
        if ds.type == "tess-core":
            # Persist only fields relevant to identity, attribution and body
            # selection; runtime/provider metadata is not model input.
            def metadata(value):
                result = {k: value[k] for k in ("id", "cwd", "createdAt", "updatedAt", "name", "messages") if k in value}
                md = value.get("metadata") or {}
                if not isinstance(md, dict):
                    raise SourceReadError("session metadata must be an object")
                result["metadata"] = {k: md[k] for k in ("parentSessionId", "originalCwd", "workspaceRoot", "gitBranch") if k in md}
                # Which body commit the engine published, and how many messages it holds.
                ref = value.get("bodyRef")
                if isinstance(ref, dict) and isinstance(ref.get("commitId"), str) and ref["commitId"]:
                    result["bodyCommitId"] = ref["commitId"]
                    count = value.get("messageCount")
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        result["messageCount"] = count
                return _strip_images(result, ParseReport())
            meta, revision = self.journal.read_snapshot(path, policy + ":meta:head", metadata)
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
                state["touched"] = count if state["touched"] is None else min(state["touched"], count)
            elif kind == "message":
                message = record.get("message") if flavor == "sidecar" else record
                if not isinstance(message, dict):
                    raise SourceReadError("message must be an object")
                index = record.get("message_index", state["next_index"])
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise SourceReadError("invalid message index")
                state["messages"][str(index)] = {"role": message.get("role"), "content": message.get("content"),
                                                "stop_reason": message.get("stop_reason", message.get("stopReason")),
                                                "timestamp": record.get("persisted_at"),
                                                "metadata": message.get("metadata", record.get("metadata", {})),
                                                "_source_message_index": index,
                                                "_source_message_id": message.get("id") or record.get("message_id"),
                                                "origin": message.get("origin", record.get("origin"))}
                state["next_index"] = index + 1
                state["uncommitted"] = True
                state["touched"] = index if state["touched"] is None else min(state["touched"], index)
            elif kind == "commit" and flavor == "sidecar":
                count = record.get("message_count")
                if count is not None and (not isinstance(count, int) or isinstance(count, bool)
                                          or set(map(int, state["messages"])) != set(range(count))):
                    raise SourceReadError("sidecar commit does not match message indices")
                if state["has_commit"]:
                    # The commit before this one stays selectable: the engine
                    # writes a commit to the body before it publishes it in the
                    # session head, and discards it if that never happens. Only
                    # its length is kept unless this commit rewrote part of it.
                    length = len(state["committed"])
                    rewrote = state["touched"] is not None and state["touched"] < length
                    state["previous"] = {"id": state["commit_id"], "count": length,
                                         "messages": state["committed"] if rewrote else None}
                commit_id = record.get("commit_id")
                state["commit_id"] = commit_id if isinstance(commit_id, str) and commit_id else None
                state["committed"] = dict(state["messages"])
                state["has_commit"] = True
                state["uncommitted"] = False
                state["touched"] = None

        if body is not None:
            # "sidecar:commits": reader state that knows each commit's id (an
            # earlier cache entry does not, and is simply not used again).
            state, revision = self.journal.read_jsonl(body, policy + ":" + ("sidecar:commits" if flavor == "sidecar" else flavor),
                                                      initial=self._initial,
                                                      consume=consume, max_record_bytes=self.cfg.codex.max_record_bytes)
            revisions.append(revision)
            if flavor == "claude":
                records = state["records"]
            else:
                if flavor == "sidecar" and state["has_commit"] and not self.cfg.recovery_ingestion:
                    messages, unpublished = _published(state, meta)
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
        for source_index, rec in enumerate(records):
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
            turn_start = len(st.turns)
            _append_message_turns(st, role, content, ts)
            metadata = msg.get('metadata') if isinstance(msg.get('metadata'), dict) else {}
            # The engine's key, then the one its earlier versions (and imported
            # transcripts from them) wrote.
            origin = metadata.get('tessCoreOrigin', metadata.get('agentCoreOrigin', msg.get('origin')))
            human = origin is None or (isinstance(origin, dict) and origin.get('actor') == 'user'
                and origin.get('intentAuthority') == 'request' and origin.get('synthetic') is False)
            for turn in st.turns[turn_start:]:
                turn.message_index = msg.get('_source_message_index', source_index)
                turn.message_id = msg.get('_source_message_id') or msg.get('id') or rec.get('uuid') or f'{st.session_id}:{turn.message_index}'
                turn.origin = origin
                turn.human_evidence = turn.role == 'user' and human
            has_tools = isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
            reason = msg.get("stop_reason", msg.get("stopReason"))
            if role == "assistant" and text and not has_tools and reason not in ("tool_use", "max_tokens", "length"):
                close("turn_complete" if reason in ("end_turn", "stop", "stop_sequence") else "assistant_message")
        eligible = ends[-1] if ends else 0
        pending = (eligible < len(st.turns) or any(r.partial_record for r in revisions) or unpublished
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
                            sum(r.parsed_records for r in revisions), notes,
                            [rec.get('_source_message_index',i) for i,rec in enumerate(records)] if ds.type=='tess-core' else None)
