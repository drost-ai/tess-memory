"""Google Antigravity conversations as a source."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tess_memory import antigravity
from tess_memory.attribution import source_adapter
from tess_memory.audit import activity_state
from tess_memory.config import Config, DataStore
from tess_memory.ingest import ingest
from tess_memory.readers import CompletedSourceReader
from tess_memory.store import MemoryStore
from tess_memory.transcript_reader import resolve_transcript_path
from test_codex import RecordingEmbedder

MAIN, CHILD, STRAY = "aaaaaaaa-0000-4000-8000-000000000001", "bbbbbbbb-0000-4000-8000-000000000002", "cccccccc-0000-4000-8000-000000000003"
ASK = "Find out why the nightly export job stops halfway and fix it. " * 4
ANSWER = "The export stopped because the cursor was closed before the last page. It now stays open. " * 4


def user(index, text, **extra):
    wrapped = (f"<USER_REQUEST>\n{text}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nThe current local time is: "
               "2026-11-21T20:23:33-07:00.\n</ADDITIONAL_METADATA>\n<USER_SETTINGS_CHANGE>\nThe user changed setting "
               "`Model Selection` from None to Some Model.\n</USER_SETTINGS_CHANGE>")
    return {"step_index": index, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
            "created_at": f"2026-11-22T03:{index:02d}:00Z", "content": wrapped, **extra}


def planner(index, text=None, calls=None):
    step = {"step_index": index, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
            "created_at": f"2026-11-22T03:{index:02d}:00Z", "model": "some-model", "thinking": "Private reasoning."}
    if text:
        step["content"] = text
    if calls:
        step["tool_calls"] = calls
    return step


def result(index, text, status="DONE"):
    return {"step_index": index, "source": "MODEL", "type": "GENERIC", "status": status, "created_at": f"2026-11-22T03:{index:02d}:00Z",
            "content": "Created At: 2026-11-21T20:23:41-07:00\nCompleted At: 2026-11-21T20:23:42-07:00\n" + text}


def system(index, text):
    return {"step_index": index, "source": "SYSTEM", "type": "SYSTEM_MESSAGE", "status": "DONE", "created_at": f"2026-11-22T03:{index:02d}:00Z",
            "content": "The following is a <SYSTEM_MESSAGE> not actually sent by the user.\n\n<SYSTEM_MESSAGE>\n" + text + "\n</SYSTEM_MESSAGE>"}


STEPS = [
    user(0, ASK),
    planner(1, calls=[{"name": "run_command", "args": {"CommandLine": "grep -n cursor\nexport.py", "WaitMsBeforeAsync": 5000}},
                      {"name": "call_mcp_tool", "args": {"ServerName": "db", "ToolName": "query", "Arguments": {"sql": "select 1"}}}]),
    result(2, "Command exited with code 0.\n\n42: cursor.close()"),
    system(3, "Reminder: work efficiently."),
    system(4, f"[Message] timestamp=2026-11-22T03:04:00Z sender={CHILD} priority=MESSAGE_PRIORITY_HIGH content=The reviewer found no other early close."),
    planner(5, ANSWER),
]


def write_log(root: Path, conversation: str, steps, name="transcript_full.jsonl") -> Path:
    logs = root / "brain" / conversation / ".system_generated" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / name
    path.write_text("".join(json.dumps(step) + "\n" for step in steps))
    return path


def append(path: Path, *steps):
    with path.open("a") as handle:
        handle.write("".join(json.dumps(step) + "\n" for step in steps))


def summaries(root: Path, rows):
    db = sqlite3.connect(root / "conversation_summaries.db")
    db.execute("CREATE TABLE IF NOT EXISTS conversation_summaries (conversation_id TEXT PRIMARY KEY, title TEXT, "
               "workspace_uris TEXT, parent_conversation_id TEXT)")
    db.executemany("INSERT OR REPLACE INTO conversation_summaries VALUES (?,?,?,?)", rows)
    db.commit()
    db.close()


@pytest.fixture
def home(tmp_path):
    """One application's data folder: a conversation in /work/shop, the
    conversation of a sub-agent it started, and one with no known workspace."""
    root = tmp_path / ".gemini" / "antigravity-cli"
    write_log(root, MAIN, STEPS)
    write_log(root, CHILD, [system(0, "[Message] sender=x content=Review the fix."), planner(1, "No other early close. " * 20)])
    write_log(root, STRAY, [user(0, ASK), planner(1, ANSWER)])
    summaries(root, [(MAIN, "Export Job Fix", json.dumps(["file:///work/shop%20floor"]), ""),
                     (CHILD, "", json.dumps(["file:///work/shop%20floor"]), MAIN)])
    return root


def test_a_conversation_belongs_to_the_project_of_its_workspace(home):
    ds = DataStore("antigravity", home, type="antigravity")
    found, parse = source_adapter(ds)
    assert [(antigravity.conversation_id(path), project, parent) for path, project, parent in found] == [
        (MAIN, "-work-shop floor", None)]  # not the sub-agent's, not the one with no workspace
    st = parse(found[0][0], found[0][1])
    assert (st.session_id, st.cwd, st.title, st.source_type) == (MAIN, "/work/shop floor", "Export Job Fix", "")
    assert (st.started_at, st.ended_at) == ("2026-11-22T03:00:00Z", "2026-11-22T03:05:00Z")
    # A source bound to one project also keeps the conversation that names no workspace.
    bound = source_adapter(DataStore("antigravity", home, type="antigravity", project="assistant"))[0]
    assert [antigravity.conversation_id(path) for path, _, _ in bound] == [MAIN, STRAY]
    assert source_adapter(DataStore("antigravity", home, type="antigravity", exclude=["-work-*"]))[0] == []


def test_the_metadata_cache_stands_in_for_an_unreadable_summaries_database(home):
    (home / "conversation_summaries.db").write_text("not a database")
    (home / "cache").mkdir()
    (home / "cache" / "conversation_metadata.json").write_text(json.dumps({"conversations": {
        MAIN: {"summary": {"Title": "Export Job Fix", "WorkspaceURIs": ["file:///work/shop"]}, "is_internal": False},
        CHILD: {"is_internal": True}}}))
    found = source_adapter(DataStore("antigravity", home, type="antigravity", project="p"))[0]
    assert [(antigravity.conversation_id(path), project) for path, project, _ in found] == [(MAIN, "-work-shop"), (STRAY, "p")]


def test_steps_become_the_conversation_and_antigravitys_own_notes_do_not(home):
    path = home / "brain" / MAIN / ".system_generated" / "logs" / "transcript_full.jsonl"
    st = antigravity.parse_conversation(path, "p")
    assert [(turn.role, turn.text) for turn in st.turns] == [
        ("user", ASK.strip()),
        ("tool", "[tool: run_command] grep -n cursor export.py"),
        ("tool", '[tool: call_mcp_tool] db/query {"sql": "select 1"}'),
        ("tool_result", "Command exited with code 0.\n\n42: cursor.close()"),
        ("tool_result", f"[message from {CHILD}] The reviewer found no other early close."),
        ("assistant", ANSWER.strip()),
    ]
    image = user(0, "What is this?", media=[{"mime_type": "image/png", "uri": "/tmp/shot.png"}])
    assert antigravity.step_turns(image)[0].text == "What is this?\n[image: /tmp/shot.png]"
    assert antigravity.step_turns({"type": "USER_INPUT", "content": "<USER_REQUEST>\nclose </USER_REQUEST> tags\n</USER_REQUEST>"})[0].text \
        == "close </USER_REQUEST> tags"
    assert antigravity.step_turns({"type": "CHECKPOINT", "content": "# Resuming from a compaction"}) == []
    assert antigravity.parse_conversation(path, "p", min_chars=10_000) is None


def test_the_shortened_log_is_read_only_when_the_full_one_is_missing(tmp_path):
    root = tmp_path / "antigravity-cli"
    short = write_log(root, MAIN, STEPS, name="transcript.jsonl")
    assert antigravity.transcript_file(root / "brain" / MAIN) == short
    full = write_log(root, MAIN, STEPS)
    assert antigravity.transcript_file(root / "brain" / MAIN) == full
    assert antigravity.transcript_file(root / "brain" / "missing") is None


def test_a_turn_is_read_once_the_agent_has_answered(home, tmp_path):
    ds = DataStore("antigravity", home, type="antigravity")
    cfg = Config(data_stores=[ds], data_dir=tmp_path / "memory")
    path, project, parent = source_adapter(ds)[0][0]
    first = CompletedSourceReader(cfg).read(ds, path, project, parent)
    assert (first.unit_kinds, first.pending_tail, first.parsed_records) == (["assistant_message"], False, len(STEPS))
    assert first.transcript.session_id == MAIN and activity_state(path, "antigravity") == "complete"
    assert all(turn.human_evidence == (turn.role == "user") for turn in first.transcript.turns)

    append(path, user(6, "Now add a test for it."), planner(7, calls=[{"name": "write_file", "args": {"Path": "/work/shop/test_export.py"}}]),
           result(8, "Tool is running as a background task", status="RUNNING"))
    running = CompletedSourceReader(cfg).read(ds, path, project, parent)
    assert (running.unit_keys(), running.pending_tail, running.parsed_records) == (first.unit_keys(), True, 3)
    assert len(running.transcript.turns) == len(first.transcript.turns) and activity_state(path, "antigravity") == "active"

    append(path, planner(9, "The test is in test_export.py and passes."))
    done = CompletedSourceReader(cfg).read(ds, path, project, parent)
    assert (done.unit_keys()[:1], len(done.unit_ends), done.pending_tail) == (first.unit_keys(), 2, False)
    assert CompletedSourceReader(cfg).read(ds, path, project, parent).parsed_records == 0

    # The person speaks again before the agent answered: what it did so far is a unit.
    append(path, user(10, "Also check the weekly export."), planner(11, calls=[{"name": "read_file", "args": {"Path": "/work/shop/weekly.py"}}]),
           user(12, "Never mind, leave the weekly one."))
    cut = CompletedSourceReader(cfg).read(ds, path, project, parent)
    assert (cut.unit_kinds[-1], len(cut.unit_ends), cut.pending_tail) == ("interrupted_by_next_user", 3, True)
    assert cut.transcript.turns[-1].text == "[tool: read_file] /work/shop/weekly.py"


class Backend:
    name = "synthetic-primary"
    cache_identity = "synthetic-model-v1"

    def complete(self, *, system, prompt):
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return json.dumps({"title": "Export fixed", "summary": "Summary " + key, "outcome": "Fixed",
                           "memories": [{"content": "The export cursor stays open until the last page", "kind": "decision", "importance": 9}]})


@pytest.mark.parametrize("incremental", [True, False])
def test_a_conversation_is_ingested_under_its_own_id(home, tmp_path, incremental):
    ds = DataStore("antigravity", home, type="antigravity")
    cfg = Config(data_stores=[ds], data_dir=tmp_path / "memory", incremental_ingestion=incremental,
                 segment_chars=3000, segment_overlap=0)
    store = MemoryStore(cfg.db_path)
    try:
        stats = ingest(cfg, store, RecordingEmbedder(), Backend(), refresh_aggregates=False)
        assert (stats.scanned, stats.errors) == (1, 0)
        session = store.get_session(MAIN)
        assert (session["project"], session["cwd"], session["n_user"]) == ("-work-shop floor", "/work/shop floor", 1)
        assert resolve_transcript_path(session["source_path"], [home]).name == "transcript_full.jsonl"
        again = ingest(cfg, store, RecordingEmbedder(), Backend(), refresh_aggregates=False)
        assert again.unchanged == 1
        if incremental:
            append(Path(session["source_path"]), user(6, "Now add a test for it. " * 12), planner(7, "The test is added and passes. " * 12))
            ingest(cfg, store, RecordingEmbedder(), Backend(), refresh_aggregates=False, session_ids={MAIN})
            assert store.get_session(MAIN)["n_user"] == 2
    finally:
        store.close()
