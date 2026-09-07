from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from agentic_memory import cli
from agentic_memory.attribution import source_adapter
from agentic_memory.codex import (
    CodexCatalog, CodexParseError, UNATTRIBUTED_PROJECT,
    ingestion_lock, validate_dedicated_database,
)
from agentic_memory.config import CodexSettings, Config, DataStore, load_config
from agentic_memory.ingest import ingest
from agentic_memory.store import MemoryStore
from agentic_memory.transcript import SessionTranscript, Turn, chunk_session, render_for_llm
from agentic_memory.transcript_reader import get_session_source_path, resolve_transcript_path


def message(role, text, **kwargs):
    return {"type": "message", "role": role,
            "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
            **kwargs}


def wrapped(kind, payload, tick=0, **kwargs):
    return {"timestamp": f"2026-08-01T10:{tick // 60:02}:{tick % 60:02}Z",
            "type": kind, "payload": payload, **kwargs}


def write_rollout(home, sid="one", items=None, *, meta=None, legacy=False, snapshot=False):
    path = home / "sessions" / (f"rollout-{sid}.json" if snapshot else f"rollout-{sid}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"id": sid, "cwd": "/work/project", "timestamp": "2026-08-01T10:00:00Z", **(meta or {})}
    if items is None:
        items = [message("user", "Remember the durable workspace decision. " * 8),
                 message("assistant", "The shared project keeps its own planning state. " * 8)]
    if snapshot:
        path.write_text(json.dumps({"session": metadata, "items": items}))
    else:
        records = [metadata if legacy else wrapped("session_meta", metadata)]
        records += items if legacy else [wrapped("response_item", item, i + 1) for i, item in enumerate(items)]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def append(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


class RecordingEmbedder:
    descriptor = "test:codex:4"

    def __init__(self):
        self.calls = []

    def embed_passages(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (len(texts), 1))


class CodexParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "codex"
        self.ds = DataStore("codex", self.home, type="codex")

    def parse(self, path, **options):
        return CodexCatalog([self.ds], CodexSettings(**options)).inspect(path, min_chars=0)

    def test_reads_all_three_file_formats(self):
        for form in ("snapshot", "legacy", "wrapped"):
            with self.subTest(form=form):
                p = write_rollout(self.home, form, snapshot=form == "snapshot", legacy=form == "legacy")
                st, _ = self.parse(p)
                self.assertEqual(st.session_id, "codex:" + form)
                self.assertEqual([t.role for t in st.turns], ["user", "assistant"])
                self.assertEqual(st.cwd, "/work/project")
                self.assertEqual(st.project, "-work-project")
                self.assertEqual(st.source_type, "codex")
                self.assertTrue(st.content_fingerprint)

    def test_early_jsonl_state_framing_is_not_content(self):
        p = write_rollout(self.home, legacy=True,
            items=[{"record_type": "state"}, message("user", "Old request"),
                   {"record_type": "state"}, message("assistant", "Old answer")])
        st, report = self.parse(p)
        self.assertEqual([t.text for t in st.turns], ["Old request", "Old answer"])
        self.assertEqual(report.ignored_records, 2)

    def test_goal_state_is_retained_without_usage_counter_replays(self):
        p = write_rollout(self.home)
        for tick, status in [(3, "active"), (4, "active"), (5, "complete")]:
            append(p, wrapped("event_msg", {"type": "thread_goal_updated", "goal": {
                "objective": "Finish the parser", "status": status, "tokensUsed": tick * 100,
            }}, tick))
        st, _ = self.parse(p)
        goals = [t.text for t in st.turns if "[goal]" in t.text]
        self.assertEqual(len(goals), 2)
        self.assertIn("Status: active", goals[0])
        self.assertIn("Status: complete", goals[1])

    def test_distinct_ids_with_identical_text_and_timestamp_are_not_deduplicated(self):
        p = write_rollout(self.home, items=[])
        append(p, wrapped("response_item", message("user", "Repeat this", id="one"), 1))
        append(p, wrapped("response_item", message("user", "Repeat this", id="two"), 1))
        st, _ = self.parse(p)
        self.assertEqual(st.n_user, 2)

    def test_missing_workspace_is_preserved_or_recovered(self):
        p = write_rollout(self.home, meta={"cwd": None}, legacy=True)
        st, report = self.parse(p)
        self.assertEqual(st.project, UNATTRIBUTED_PROJECT)
        self.assertIsNone(st.cwd)
        self.assertTrue(report.warnings)
        p = write_rollout(self.home, "recovered", meta={"cwd": None}, legacy=True,
                          items=[message("user", "<environment_context><cwd>/old/repo</cwd></environment_context>\nActual request"),
                                 message("assistant", "Preserve the old project.")])
        st, _ = self.parse(p)
        self.assertEqual(st.cwd, "/old/repo")
        self.assertEqual(st.project, "-old-repo")
        self.assertEqual(st.turns[0].text, "Actual request")

    def test_fixed_project_and_exclusions_still_apply(self):
        p = write_rollout(self.home, meta={"cwd": "/temp/excluded"})
        self.ds.project = "assistant"
        self.ds.exclude = ["-temp-*"]
        self.assertFalse(CodexCatalog([self.ds]).sources)
        p = write_rollout(self.home, meta={"cwd": "/work/repo"})
        st, _ = self.parse(p)
        self.assertEqual(st.project, "assistant")
        self.assertEqual(st.cwd, "/work/repo")

    def test_images_are_removed_recursively_before_rendering(self):
        image = {"type": "input_image", "image_url": "data:image/png;base64,VERYSECRETIMAGE"}
        m = message("user", "Look at the build result")
        m["content"].append(image)
        serialized = json.dumps({"content": [{"type": "image", "mimeType": "image/png", "data": "OTHERSECRET"},
                                               {"type": "text", "text": "diagnostic retained"}]})
        p = write_rollout(self.home, items=[m, {"type": "function_call", "name": "inspect", "call_id": "a", "arguments": {"image": image}},
                                           {"type": "function_call_output", "call_id": "a", "output": serialized},
                                           message("assistant", "The build succeeded.")])
        before = p.read_bytes()
        append(p, wrapped("compacted", {"replacement_history": [m]}, 20))
        source = p.read_bytes()
        st, report = self.parse(p)
        rendered = render_for_llm(st, 100000, 400)
        chunks = "\n".join(t for t, _ in chunk_session(st, 6000))
        for text in [rendered, chunks]:
            self.assertNotIn("VERYSECRETIMAGE", text)
            self.assertNotIn("OTHERSECRET", text)
            self.assertNotIn("data:image", text)
            self.assertIn("diagnostic retained", text)
        self.assertGreaterEqual(report.images_removed, 3)
        self.assertGreater(report.snapshot_replays, 0)
        self.assertEqual(p.read_bytes(), source)
        self.assertTrue(source.startswith(before))

    def test_serialized_mime_only_image_payload_is_removed(self):
        output = json.dumps({"result": {"mimeType": "image/png", "data": "ENCODED_IMAGE_PAYLOAD"}, "text": "keep the diagnosis"})
        p = write_rollout(self.home, items=[message("user", "Read the diagnosis"),
            {"type": "function_call_output", "call_id": "image", "output": output},
            message("assistant", "Done")])
        st, _ = self.parse(p)
        text = "\n".join(t.text for t in st.turns)
        self.assertNotIn("ENCODED_IMAGE_PAYLOAD", text)
        self.assertIn("keep the diagnosis", text)

    def test_two_thousand_char_head_tail_survives_both_consumers(self):
        output = "BEGIN" + "a" * 1100 + "HEAD_SENTINEL" + "m" * 5000 + "TAIL_SENTINEL" + "z" * 700 + "END"
        args = "full command " + "x" * 1000 + "COMMAND_END"
        p = write_rollout(self.home, items=[message("user", "Run the verification"),
            {"type": "custom_tool_call", "name": "exec", "call_id": "call", "input": args},
            {"type": "custom_tool_call_output", "call_id": "call", "output": [{"type": "input_text", "text": output}]},
            message("assistant", "The verification finished.")])
        st, _ = self.parse(p)
        result = next(t.text for t in st.turns if t.role == "tool_result")
        self.assertTrue(result.startswith("BEGIN"))
        self.assertTrue(result.endswith("END"))
        self.assertIn("HEAD_SENTINEL", result)
        self.assertIn("TAIL_SENTINEL", result)
        self.assertNotIn("m" * 1000, result)
        self.assertLess(len(result), 2100)
        for text in [render_for_llm(st, 100000, 400), "\n".join(c for c, _ in chunk_session(st, 6000))]:
            self.assertIn("HEAD_SENTINEL", text)
            self.assertIn("TAIL_SENTINEL", text)
            self.assertIn("COMMAND_END", text)

    def test_numeric_changes_and_repeated_real_requests_are_retained(self):
        p = write_rollout(self.home, items=[message("user", "Measure version 1200"),
            message("assistant", "The score is 42"), message("user", "Measure version 2000"),
            message("assistant", "The score is 43"), message("user", "Measure version 2000"),
            message("assistant", "The score is 43")])
        st, _ = self.parse(p)
        text = render_for_llm(st, 10000, 400)
        self.assertIn("score is 42", text)
        self.assertEqual(text.count("score is 43"), 2)
        self.assertEqual(text.count("Measure version 2000"), 2)

    def test_ui_and_event_mirrors_do_not_duplicate_source_messages(self):
        p = write_rollout(self.home, items=[message("user", "Do this", id="u"), message("assistant", "It is done", id="a")])
        append(p, wrapped("event_msg", {"type": "user_message", "message": "Do this"}, 3))
        append(p, wrapped("event_msg", {"type": "agent_message", "message": "It is done"}, 4))
        append(p, wrapped("event_msg", {"type": "task_complete", "last_agent_message": "It is done"}, 5))
        append(p, wrapped("event_msg", {"type": "item_completed", "item": {"type": "AgentMessage", "id": "a", "content": [{"type": "text", "text": "It is done"}]}}, 6))
        st, report = self.parse(p)
        self.assertEqual([t.text for t in st.turns], ["Do this", "It is done"])
        self.assertEqual(report.mirrored_items, 4)

    def test_user_mirror_without_attachment_matches_source_with_image(self):
        item = message("user", "Review this screenshot")
        item["content"].append({"type": "input_image", "image_url": "data:image/png;base64,AA"})
        p = write_rollout(self.home, items=[item, message("assistant", "Reviewed")])
        append(p, wrapped("event_msg", {"type": "user_message", "message": "Review this screenshot"}, 3))
        st, report = self.parse(p)
        self.assertEqual(st.n_user, 1)
        self.assertEqual(report.mirrored_items, 1)

    def test_event_only_evidence_is_retained(self):
        p = write_rollout(self.home, items=[])
        append(p, wrapped("event_msg", {"type": "user_message", "message": "Run it"}, 1))
        append(p, wrapped("event_msg", {"type": "exec_command_end", "call_id": "c", "command": ["python", "check.py"], "aggregated_output": "unique validation failure"}, 2))
        append(p, wrapped("event_msg", {"type": "agent_message", "message": "Fixed the failure"}, 3))
        st, _ = self.parse(p)
        self.assertEqual(st.n_user, 1)
        self.assertIn("unique validation failure", "\n".join(t.text for t in st.turns))

    def test_compaction_replays_removed_and_novel_text_recovered_once(self):
        original = message("user", "Original request", id="u")
        extra = message("user", "Previously saved requirement", id="recovered")
        p = write_rollout(self.home, items=[original, message("assistant", "An answer")])
        for tick in [10, 20, 30]:
            append(p, wrapped("compacted", {"replacement_history": [original, extra,
                {"type": "compaction", "encrypted_content": "OPAQUE"}]}, tick))
        st, report = self.parse(p)
        self.assertEqual([t.text for t in st.turns].count("Original request"), 1)
        self.assertEqual([t.text for t in st.turns].count("Previously saved requirement"), 1)
        self.assertEqual(report.snapshot_recovered, 1)
        self.assertNotIn("OPAQUE", "".join(t.text for t in st.turns))

    def test_guardian_and_subagent_sources_are_excluded_by_default(self):
        write_rollout(self.home, "parent")
        p = write_rollout(self.home, "guardian", meta={"source": {"subagent": {"other": "guardian"}}})
        write_rollout(self.home, "worker", meta={"source": {"subagent": {"thread_spawn": {"parent_thread_id": "parent"}}}})
        c = CodexCatalog([self.ds])
        self.assertEqual(len(c.sources), 1)
        self.assertEqual(c.skipped_subagents, 2)
        self.assertEqual(len(CodexCatalog([self.ds], CodexSettings(include_subagents=True)).sources), 3)

    def test_shared_parent_cutoff_and_task_context(self):
        parent = write_rollout(self.home, "parent", items=[])
        append(parent, wrapped("response_item", message("user", "Parent task", id="u"), 1, ordinal=1))
        append(parent, wrapped("response_item", message("assistant", "Before fork", id="a"), 2, ordinal=2))
        boundary = parent.stat().st_size
        append(parent, wrapped("response_item", message("user", "FUTURE MUST NOT LEAK", id="future"), 3, ordinal=3))
        child = write_rollout(self.home, "child", meta={"history_base": {"thread_id": "parent", "end_byte_offset": boundary, "end_ordinal_exclusive": 3}},
                              items=[message("assistant", "Child completed the parent task", id="child-answer")])
        st, _ = self.parse(child)
        self.assertEqual(st.parent_session, "codex:parent")
        self.assertIn("Parent task", st.turns[0].text)
        self.assertIn("task inherited", st.turns[0].text)
        self.assertNotIn("FUTURE", "".join(t.text for t in st.turns))
        self.assertNotIn("Before fork", "".join(t.text for t in st.turns))

    def test_legacy_fork_prefix_removed_but_later_repeat_preserved(self):
        items = [message("user", "First task"), message("assistant", "First answer")]
        write_rollout(self.home, "parent", items=items)
        child = write_rollout(self.home, "child", meta={"forked_from_id": "parent"}, items=[])
        append(child, wrapped("session_meta", {"id": "parent"}, 1))
        for tick, item in enumerate(items + [message("user", "First task"), message("assistant", "New answer")], 2):
            append(child, wrapped("response_item", item, tick))
        st, report = self.parse(child)
        self.assertEqual([t.text for t in st.turns], ["First task", "New answer"])
        self.assertEqual(report.inherited_items, 2)

    def test_parent_resolves_across_configured_homes(self):
        other = Path(self.tmp.name) / "work"
        parent = write_rollout(other, "parent")
        child = write_rollout(self.home, "child", meta={"history_base": {"thread_id": "parent", "end_byte_offset": parent.stat().st_size}}, items=[message("assistant", "Continuing work")])
        c = CodexCatalog([self.ds, DataStore("work", other, type="codex")])
        st, _ = c.inspect(child, min_chars=0)
        self.assertEqual(st.parent_session, "codex:parent")
        self.assertIn("task inherited", st.turns[0].text)

    def test_legacy_continuation_does_not_inherit_future_parent_work(self):
        old = [message("user", "Original task", id="u"), message("assistant", "Original answer", id="a")]
        write_rollout(self.home, "parent", items=old + [message("user", "FUTURE PARENT TASK", id="future")])
        child = write_rollout(self.home, "child", items=[], meta={"forked_from_id": "parent"})
        append(child, wrapped("session_meta", {"id": "parent"}))
        for i, item in enumerate(old + [message("assistant", "Finished original task", id="child")], 1):
            append(child, wrapped("response_item", item, i))
        st, _ = self.parse(child)
        self.assertIn("Original task", st.turns[0].text)
        self.assertNotIn("FUTURE", "".join(t.text for t in st.turns))

    def test_structured_mcp_result_is_retained_with_empty_content(self):
        p = write_rollout(self.home, items=[message("user", "Inspect structured result"),
            {"type": "function_call_output", "call_id": "c", "output": {
                "content": [], "structuredContent": {"diagnosis": "keep this evidence"}, "isError": True,
            }}, message("assistant", "Read the diagnosis")])
        st, _ = self.parse(p)
        result = next(t.text for t in st.turns if t.role == "tool_result")
        self.assertIn("keep this evidence", result)
        self.assertIn("tool error", result)

    def test_missing_shared_parent_and_cycles_fail_explicitly(self):
        p = write_rollout(self.home, meta={"history_base": {"thread_id": "absent"}})
        with self.assertRaisesRegex(CodexParseError, "missing"):
            self.parse(p)
        p = write_rollout(self.home, meta={"forked_from_id": "one"})
        with self.assertRaisesRegex(CodexParseError, "cyclic"):
            self.parse(p)

    def test_unknown_records_incomplete_tails_and_large_records_fail(self):
        p = write_rollout(self.home)
        append(p, wrapped("response_item", {"type": "future_message", "content": "important"}, 10))
        with self.assertRaisesRegex(CodexParseError, "unsupported"):
            self.parse(p)
        p = write_rollout(self.home)
        with p.open("a") as f: f.write('{"type":')
        with self.assertRaisesRegex(CodexParseError, "incomplete final"):
            self.parse(p)
        p = write_rollout(self.home)
        with self.assertRaisesRegex(CodexParseError, "exceeds"):
            self.parse(p, max_record_bytes=250)

    def test_discovery_does_not_read_an_escaped_symlink(self):
        other = Path(self.tmp.name) / "other"
        outside = write_rollout(other)
        (self.home / "sessions").mkdir(parents=True)
        (self.home / "sessions" / "rollout-escape.jsonl").symlink_to(outside)
        c = CodexCatalog([self.ds])
        self.assertFalse(c.sources)
        self.assertTrue(c.errors)

    def test_inspection_command_has_no_models_or_store_writes(self):
        write_rollout(self.home)
        stdout = io.StringIO()
        with mock.patch("agentic_memory.store.MemoryStore", side_effect=AssertionError("no store")), \
             mock.patch("agentic_memory.embed.make_embedder", side_effect=AssertionError("no embed")), \
             mock.patch("agentic_memory.llm.AgentCoreBackend", side_effect=AssertionError("no model")), \
             contextlib.redirect_stdout(stdout):
            result = cli.main(["inspect-codex", str(self.home)])
        self.assertEqual(result, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["discovered"], 1)
        self.assertTrue(report["sessions"][0]["usable"])


class CodexIngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "codex"
        self.cfg = Config(data_stores=[DataStore("codex", self.home, type="codex")], data_dir=self.root / "memory")
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.embedder = RecordingEmbedder()

    def ingest(self):
        return ingest(self.cfg, self.store, self.embedder, None)

    def test_incremental_ingestion_and_metadata_only_append(self):
        p = write_rollout(self.home)
        first = self.ingest()
        self.assertEqual(first.extractive, 1)
        count = len(self.embedder.calls)
        memories = [tuple(r) for r in self.store.db.execute("SELECT * FROM memories ORDER BY id")]
        self.assertEqual(self.ingest().unchanged, 1)
        append(p, wrapped("event_msg", {"type": "token_count", "info": {"tokens": 1234}}, 30))
        result = self.ingest()
        self.assertEqual(result.codex_unchanged, 1)
        self.assertEqual(len(self.embedder.calls), count)
        self.assertEqual([tuple(r) for r in self.store.db.execute("SELECT * FROM memories ORDER BY id")], memories)

    def test_pilot_selection_ingests_only_requested_sessions(self):
        write_rollout(self.home, "selected")
        write_rollout(self.home, "not-selected")
        stats = ingest(self.cfg, self.store, self.embedder, None, session_ids={"codex:selected"})
        self.assertEqual(stats.extractive, 1)
        self.assertIsNotNone(self.store.get_session("codex:selected"))
        self.assertIsNone(self.store.get_session("codex:not-selected"))

    def test_mcp_startup_uses_the_explicit_codex_config(self):
        from agentic_memory import server, retrieval

        write_rollout(self.home)
        settings = self.root / "dedicated.toml"
        settings.write_text(f'[[data_stores]]\nname="codex"\ntype="codex"\npath="{self.home}"\n[storage]\ndata_dir="{self.cfg.data_dir}"\n')
        with mock.patch.object(server, "run") as run, mock.patch.object(server, "CONFIG", server.CONFIG), \
             mock.patch.object(retrieval, "CONFIG", retrieval.CONFIG):
            self.assertEqual(cli.main(["codex-serve", "--settings", str(settings)]), 0)
            run.assert_called_once()
            self.assertEqual(server.CONFIG.db_path.resolve(), self.cfg.db_path.resolve())
            self.assertIs(server.CONFIG, retrieval.CONFIG)

    def test_mcp_missing_settings_does_not_start_a_default_store(self):
        from agentic_memory import server

        with mock.patch.object(server, "run") as run:
            self.assertEqual(cli.main(["codex-serve", "--settings", str(self.root / "missing.toml")]), 2)
            run.assert_not_called()

    def test_malformed_rewrite_preserves_last_good_memory_and_retries(self):
        p = write_rollout(self.home)
        self.ingest()
        old = dict(self.store.get_session("codex:one"))
        state = self.store.file_state(str(p))
        source = p.read_text()
        p.write_text(source + '{"type":')
        self.assertEqual(self.ingest().errors, 1)
        self.assertEqual(dict(self.store.get_session("codex:one")), old)
        self.assertEqual(self.store.file_state(str(p)), state)
        p.write_text(source)
        append(p, wrapped("response_item", message("assistant", "New validated result " * 20), 30))
        self.assertEqual(self.ingest().extractive, 1)
        self.assertIn("New validated result", self.store.get_session("codex:one")["outcome"] + self.store.get_session("codex:one")["summary"])

    def test_archive_move_updates_source_without_reembedding(self):
        p = write_rollout(self.home)
        self.ingest()
        count = len(self.embedder.calls)
        archive = self.home / "archived_sessions" / p.name
        archive.parent.mkdir()
        p.rename(archive)
        self.assertEqual(self.ingest().codex_unchanged, 1)
        self.assertEqual(len(self.embedder.calls), count)
        self.assertEqual(get_session_source_path(self.cfg.db_path, "codex:one"), str(archive.resolve()))
        self.assertEqual(resolve_transcript_path(str(archive), [self.home]), archive.resolve())

    def test_option_change_reprocesses_unchanged_source(self):
        write_rollout(self.home)
        self.ingest()
        count = len(self.embedder.calls)
        self.cfg.codex = replace(self.cfg.codex, tool_output_chars=1200)
        self.assertEqual(self.ingest().extractive, 1)
        self.assertGreater(len(self.embedder.calls), count)

    def test_changed_middle_of_tool_output_invalidates_content_fingerprint(self):
        p = write_rollout(self.home, items=[message("user", "Inspect output " * 30),
            {"type": "function_call_output", "call_id": "a", "output": "h" * 1500 + "OLD" + "t" * 1500},
            message("assistant", "Outcome " * 30)])
        self.ingest()
        before = self.store.get_meta("codex:normalized:codex:one")
        p.write_text(p.read_text().replace("OLD", "NEW"))
        self.assertEqual(self.ingest().extractive, 1)
        self.assertNotEqual(self.store.get_meta("codex:normalized:codex:one"), before)

    def test_codex_settings_do_not_change_legacy_rendering(self):
        st = SessionTranscript("legacy", "project", "/source", turns=[Turn("user", "score 12"), Turn("user", "score 13"), Turn("tool_result", "X" * 3000)])
        rendered = render_for_llm(st, 10000, 400)
        self.assertNotIn("score 13", rendered)
        self.assertIn("repeated near-identical", rendered)
        chunk = "\n".join(t for t, _ in chunk_session(st, 6000))
        self.assertLess(len(chunk), 800)

    def test_settings_are_opt_in_and_validated(self):
        p = self.root / "settings.toml"
        p.write_text('[codex]\ntool_output_chars = 1200\nbatch_size = 2\n')
        cfg = load_config(p)
        self.assertEqual(cfg.codex.tool_output_chars, 1200)
        self.assertEqual(cfg.tool_result_snippet, 400)
        self.assertEqual(cfg.data_stores[0].type, "claude-code")
        p.write_text('[codex]\ntool_output_chars = -1\n')
        with self.assertRaises(ValueError): load_config(p)

    def test_dedicated_command_rejects_missing_or_defaulting_config_before_writes(self):
        p = self.root / "settings.toml"
        with mock.patch("agentic_memory.store.MemoryStore", side_effect=AssertionError("must not open a database")):
            self.assertEqual(cli.main(["codex-ingest", "--settings", str(p), "--no-llm"]), 2)
            p.write_text('[storage]\ndata_dir = "somewhere"\n')
            self.assertEqual(cli.main(["codex-ingest", "--settings", str(p), "--no-llm"]), 2)

    def test_dedicated_database_guard_preserves_existing_non_codex_state(self):
        self.store.upsert_session(id="existing-tess-session", project="assistant", summary="existing paid memory")
        self.store.commit()
        before = dict(self.store.get_session("existing-tess-session"))
        with self.assertRaisesRegex(CodexParseError, "separate"):
            validate_dedicated_database(self.cfg.db_path)
        self.assertEqual(dict(self.store.get_session("existing-tess-session")), before)

    def test_codex_ingesters_do_not_overlap(self):
        write_rollout(self.home)
        with ingestion_lock(self.cfg.data_dir):
            with self.assertRaisesRegex(CodexParseError, "already running"):
                self.ingest()
        self.assertEqual(self.store.counts()["sessions"], 0)
        self.assertEqual(self.ingest().extractive, 1)


if __name__ == "__main__":
    unittest.main()
