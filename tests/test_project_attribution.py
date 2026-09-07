from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from agentic_memory import cli, plan
from agentic_memory.attribution import reconcile_project_attribution, source_adapter
from agentic_memory.config import Config, DataStore, ProjectAttribution, load_config
from agentic_memory.distill import SessionDigest
from agentic_memory.ingest import ingest
from agentic_memory.store import MemoryRow, MemoryStore


class FakeEmbedder:
    descriptor = "test:fixed-attribution:4"

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(i + 1), 2.0, 3.0, 4.0] for i, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


def write_claude_transcript(path: Path, cwd: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "user",
            "timestamp": "2026-08-10T10:00:00+00:00",
            "cwd": cwd,
            "gitBranch": "main",
            "message": {"content": "Explain the durable project identity design. " + "x" * 260},
        },
        {
            "type": "assistant",
            "timestamp": "2026-08-10T10:01:00+00:00",
            "cwd": cwd,
            "gitBranch": "main",
            "message": {"content": "Use a stable logical project and keep cwd as provenance. " + "y" * 260},
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def insert_session(
    store: MemoryStore,
    *,
    session_id: str,
    project: str,
    path: Path,
    cwd: str = "/Users/sam/runtime",
) -> None:
    store.upsert_session(
        id=session_id,
        project=project,
        cwd=cwd,
        git_branch="main",
        title="Preserved title",
        summary="Preserved summary",
        outcome="Preserved outcome",
        started_at="2026-08-10T10:00:00+00:00",
        ended_at="2026-08-10T10:01:00+00:00",
        n_user=1,
        n_assistant=1,
        source_path=str(path),
        parent_session=None,
    )
    stat = path.stat()
    store.mark_file(str(path), stat.st_mtime, stat.st_size, session_id, "distilled")


class ProjectAttributionTests(unittest.TestCase):
    def test_strategy_is_backward_compatible_and_fixed_when_configured(self) -> None:
        self.assertEqual(ProjectAttribution().mode, "cwd")
        self.assertEqual(ProjectAttribution().resolve("-Users-sam-repo"), "-Users-sam-repo")
        self.assertEqual(ProjectAttribution("  assistant  ").mode, "fixed")
        self.assertEqual(
            ProjectAttribution("  assistant  ").resolve("-Users-sam-repo"),
            "assistant",
        )

    def test_exclusions_apply_to_source_project_before_fixed_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_claude_transcript(
                root / "-private-tmp-claude-123" / "excluded.jsonl",
                "/private/tmp/claude-123",
            )
            included = root / "-Users-sam-runtime" / "included.jsonl"
            write_claude_transcript(included, "/Users/sam/runtime")
            ds = DataStore(
                name="runtime",
                path=root,
                type="claude-code",
                exclude=["-private-tmp-claude*"],
                project="assistant",
            )

            found, _ = source_adapter(ds)

            self.assertEqual(found, [(included, "-Users-sam-runtime", None)])
            self.assertEqual(ds.resolve_project(found[0][1]), "assistant")

    def test_config_loads_fixed_attribution_and_planning_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.toml"
            settings.write_text(
                """
[[data_stores]]
name = "runtime"
type = "agent-core"
path = "~/runtime/sessions"
project = "assistant"

[planning]
default_project = "assistant"
lock_writes_to_default = true
""".strip(),
                encoding="utf-8",
            )

            cfg = load_config(settings)

            self.assertEqual(cfg.data_stores[0].project, "assistant")
            self.assertEqual(cfg.data_stores[0].attribution.mode, "fixed")
            self.assertEqual(cfg.planning_default_project, "assistant")
            self.assertTrue(cfg.planning_lock_writes_to_default)

    def test_future_ingestion_uses_fixed_project_and_preserves_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_root = base / "transcripts"
            source_project = "-Users-sam-agents-otherhost"
            cwd = "/Users/sam/agents/otherhost"
            transcript = source_root / source_project / "future-session.jsonl"
            write_claude_transcript(transcript, cwd)
            cfg = Config(
                data_stores=[
                    DataStore(
                        name="runtime",
                        path=source_root,
                        type="claude-code",
                        project="assistant",
                    )
                ],
                data_dir=base / "memory",
            )
            store = MemoryStore(cfg.db_path)
            try:
                stats = ingest(cfg, store, FakeEmbedder(), None)
                session = store.get_session("future-session")

                self.assertEqual(stats.extractive, 1)
                self.assertIsNotNone(session)
                self.assertEqual(session["project"], "assistant")
                self.assertEqual(session["cwd"], cwd)
                self.assertEqual(session["source_path"], str(transcript))
                projects = {
                    row["project"]
                    for row in store.db.execute(
                        "SELECT project FROM memories WHERE session_id=?",
                        ("future-session",),
                    )
                }
                self.assertEqual(projects, {"assistant"})
            finally:
                store.close()


class ReconciliationTests(unittest.TestCase):
    def test_metadata_only_reconcile_preserves_rows_and_embedding_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_root = base / "transcripts"
            old_project = "-Users-sam-agents-otherhost"
            transcript = source_root / old_project / "session-1.jsonl"
            write_claude_transcript(transcript, "/Users/sam/agents/otherhost")
            cfg = Config(
                data_stores=[
                    DataStore(
                        name="runtime",
                        path=source_root,
                        type="claude-code",
                        project="assistant",
                    )
                ],
                data_dir=base / "memory",
            )
            store = MemoryStore(cfg.db_path)
            try:
                insert_session(
                    store,
                    session_id="session-1",
                    project=old_project,
                    path=transcript,
                    cwd="/Users/sam/agents/otherhost",
                )
                retained_ids = store.add_memories([
                    MemoryRow(
                        session_id="session-1",
                        project=old_project,
                        level="chunk",
                        kind="conversation",
                        content="exact source content",
                        importance=0.42,
                        ts="2026-08-10T10:00:00+00:00",
                        tags=["exact", "preserved"],
                        vector=np.asarray([1.25, 2.5, 3.75, 5.0], dtype=np.float32),
                    ),
                    MemoryRow(
                        session_id="session-1",
                        project=old_project,
                        level="memory",
                        kind="decision",
                        content="preserve the paid distillation",
                        importance=0.91,
                        ts="2026-08-10T10:01:00+00:00",
                        tags=["decision"],
                        vector=np.asarray([8.0, 6.0, 4.0, 2.0], dtype=np.float32),
                    ),
                ])
                profile_id = store.add_memories([
                    MemoryRow(
                        project=old_project,
                        level="project",
                        kind="profile",
                        content="obsolete rolling profile",
                        importance=0.9,
                        ts="2026-08-10T10:01:00+00:00",
                        vector=np.asarray([9.0, 9.0, 9.0, 9.0], dtype=np.float32),
                    )
                ])[0]
                store.commit()

                session_before = dict(store.get_session("session-1"))
                session_before.pop("project")
                memory_before = [
                    tuple(row)
                    for row in store.db.execute(
                        "SELECT id, session_id, level, kind, content, importance, ts, tags, plan_id "
                        "FROM memories WHERE id IN (?,?) ORDER BY id",
                        retained_ids,
                    )
                ]
                embeddings_before = [
                    (row["memory_id"], bytes(row["vector"]))
                    for row in store.db.execute(
                        "SELECT memory_id, vector FROM embeddings WHERE memory_id IN (?,?) "
                        "ORDER BY memory_id",
                        retained_ids,
                    )
                ]

                dry = reconcile_project_attribution(cfg, store, dry_run=True)
                self.assertEqual(dry.changed, 1)
                self.assertEqual(dry.reattributed, 0)
                self.assertEqual(store.get_session("session-1")["project"], old_project)

                stats = reconcile_project_attribution(cfg, store)

                self.assertEqual(stats.reattributed, 1)
                self.assertEqual(stats.memories_reattributed, 2)
                self.assertEqual(stats.retired_profiles, 1)
                session_after = dict(store.get_session("session-1"))
                self.assertEqual(session_after.pop("project"), "assistant")
                self.assertEqual(session_after, session_before)
                self.assertEqual(
                    [
                        tuple(row)
                        for row in store.db.execute(
                            "SELECT id, session_id, level, kind, content, importance, ts, tags, plan_id "
                            "FROM memories WHERE id IN (?,?) ORDER BY id",
                            retained_ids,
                        )
                    ],
                    memory_before,
                )
                self.assertEqual(
                    [
                        (row["memory_id"], bytes(row["vector"]))
                        for row in store.db.execute(
                            "SELECT memory_id, vector FROM embeddings WHERE memory_id IN (?,?) "
                            "ORDER BY memory_id",
                            retained_ids,
                        )
                    ],
                    embeddings_before,
                )
                self.assertEqual(
                    {
                        row["project"]
                        for row in store.db.execute(
                            "SELECT project FROM memories WHERE id IN (?,?)",
                            retained_ids,
                        )
                    },
                    {"assistant"},
                )
                self.assertIsNone(
                    store.db.execute(
                        "SELECT 1 FROM memories WHERE id=?", (profile_id,)
                    ).fetchone()
                )
                self.assertIsNone(
                    store.db.execute(
                        "SELECT 1 FROM embeddings WHERE memory_id=?", (profile_id,)
                    ).fetchone()
                )
            finally:
                store.close()

    def test_batch_reattribution_rolls_back_every_change_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            transcript = base / "source" / "old-project" / "session-1.jsonl"
            write_claude_transcript(transcript, "/Users/sam/runtime")
            store = MemoryStore(base / "memory.db")
            try:
                insert_session(
                    store,
                    session_id="session-1",
                    project="old-project",
                    path=transcript,
                )
                memory_id = store.add_memories([
                    MemoryRow(
                        session_id="session-1",
                        project="old-project",
                        level="memory",
                        kind="fact",
                        content="must roll back",
                        vector=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    )
                ])[0]
                profile_id = store.add_memories([
                    MemoryRow(
                        project="old-project",
                        level="project",
                        kind="profile",
                        content="must also roll back",
                        vector=np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
                    )
                ])[0]
                store.commit()
                embedding_before = bytes(
                    store.db.execute(
                        "SELECT vector FROM embeddings WHERE memory_id=?", (memory_id,)
                    ).fetchone()["vector"]
                )

                with self.assertRaises(ValueError):
                    store.reattribute_session_projects([
                        ("session-1", "assistant"),
                        ("missing-session", "assistant"),
                    ])

                self.assertEqual(store.get_session("session-1")["project"], "old-project")
                self.assertEqual(
                    store.db.execute(
                        "SELECT project FROM memories WHERE id=?", (memory_id,)
                    ).fetchone()["project"],
                    "old-project",
                )
                self.assertIsNotNone(
                    store.db.execute(
                        "SELECT 1 FROM memories WHERE id=?", (profile_id,)
                    ).fetchone()
                )
                self.assertEqual(
                    bytes(
                        store.db.execute(
                            "SELECT vector FROM embeddings WHERE memory_id=?", (memory_id,)
                        ).fetchone()["vector"]
                    ),
                    embedding_before,
                )
            finally:
                store.close()


class ReconciliationRefreshTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "transcripts"
        self.path = self.source / "old-project" / "moved.jsonl"
        write_claude_transcript(self.path, "/work/old-project")
        self.cfg = Config(data_stores=[DataStore("test", self.source, project="old-project")],
                          data_dir=self.root / "memory")
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.embedder = FakeEmbedder()
        self.backend = SimpleNamespace(name="offline")
        self.week = "2026-08-10"

    def seed(self):
        insert_session(self.store, session_id="moved", project="old-project", path=self.path)
        self.store.add_memories([
            MemoryRow(level="memory", kind="fact", content="Paid session memory", session_id="moved",
                      project="old-project", vector=np.ones(4, dtype=np.float32)),
            MemoryRow(level="project", kind="profile", content="Original profile",
                      project="old-project", vector=np.ones(4, dtype=np.float32)),
            MemoryRow(level="digest", kind="digest", content="Original digest", ts=self.week,
                      vector=np.ones(4, dtype=np.float32)),
        ])
        self.store.commit()

    def queued(self, store=None):
        return {(row["level"], row["scope"]) for row in
                (store or self.store).pending_aggregates(float("inf"))}

    def snapshot(self):
        return {table: [tuple(row) for row in self.store.db.execute(f"SELECT * FROM {table}")]
                for table in ("sessions", "memories", "embeddings", "files", "meta", "aggregate_refresh")}

    def exercise_refresh(self, keep_old_session):
        if keep_old_session:
            other = self.root / "unconfigured" / "remaining.jsonl"
            write_claude_transcript(other, "/work/old-project")
            insert_session(self.store, session_id="remaining", project="old-project", path=other)
            self.store.commit()
        with mock.patch("agentic_memory.ingest.distill_session", return_value=SessionDigest(
            "Moved work", "Saved summary", "Complete", []
        )) as distill, mock.patch("agentic_memory.ingest.distill_profile",
                                  side_effect=lambda _b, p, _cwd, lines: p + "\n" + "\n".join(lines)) as profiles, \
             mock.patch("agentic_memory.ingest.distill_week",
                        side_effect=lambda _b, _week, lines: "\n".join(lines)) as digests, \
             mock.patch.object(self.embedder, "embed_passages", wraps=self.embedder.embed_passages) as embed:
            ingest(self.cfg, self.store, self.embedder, self.backend, refresh_aggregates=True)
            self.assertEqual(self.queued(), set())
            file_state = self.store.file_state(str(self.path))
            source_bytes = self.path.read_bytes()
            embed_calls = embed.call_count
            distill.reset_mock()
            profiles.reset_mock()
            digests.reset_mock()
            self.cfg.data_stores[0].project = "assistant"

            self.assertEqual(reconcile_project_attribution(self.cfg, self.store).reattributed, 1)
            distill.assert_not_called()
            profiles.assert_not_called()
            digests.assert_not_called()
            self.assertEqual(embed.call_count, embed_calls)
            self.assertEqual(self.queued(), {("project", "old-project"), ("project", "assistant"),
                                             ("digest", self.week)})
            # The queue must survive another connection/process, and aggregate
            # refresh must work with no source scan or session distillation.
            reopened = MemoryStore(self.cfg.db_path)
            self.addCleanup(reopened.close)
            self.assertEqual(self.queued(reopened), self.queued())
            with mock.patch("agentic_memory.ingest.source_adapter", side_effect=AssertionError("no discovery")):
                result = ingest(self.cfg, reopened, self.embedder, self.backend,
                                aggregates_only=True, refresh_aggregates=True)
            self.assertEqual((result.scanned, result.distilled, result.aggregates_pending), (0, 0, 0))
            distill.assert_not_called()
            self.assertEqual({call.args[1] for call in profiles.call_args_list},
                             {"old-project", "assistant"} if keep_old_session else {"assistant"})
            digests.assert_called_once()
            self.assertIn("Saved summary", reopened.get_profile("assistant")["content"])
            if keep_old_session:
                old_profile = reopened.get_profile("old-project")["content"]
                self.assertIn("Preserved title", old_profile)
                self.assertNotIn("Moved work", old_profile)
            else:
                self.assertIsNone(reopened.get_profile("old-project"))
            weekly = reopened.db.execute("SELECT content FROM memories WHERE level='digest'").fetchone()[0]
            self.assertIn("assistant", weekly)
            self.assertEqual("old-project" in weekly, keep_old_session)
            self.assertEqual(reopened.file_state(str(self.path)), file_state)
            self.assertEqual(self.path.read_bytes(), source_bytes)

    def test_reconcile_then_aggregate_only_refresh_retires_empty_old_project(self):
        self.exercise_refresh(keep_old_session=False)

    def test_reconcile_then_aggregate_only_refresh_repairs_both_populated_projects(self):
        self.exercise_refresh(keep_old_session=True)

    def test_dry_run_and_unchanged_assignments_leave_queue_untouched(self):
        self.seed()
        self.cfg.data_stores[0].project = "assistant"
        before = self.snapshot()
        self.assertEqual(reconcile_project_attribution(self.cfg, self.store, dry_run=True).reattributed, 0)
        self.assertEqual(before, self.snapshot())
        self.store.reattribute_session_project("moved", "old-project")
        self.assertEqual(before, self.snapshot())
        reconcile_project_attribution(self.cfg, self.store)
        after = self.snapshot()
        self.assertEqual(reconcile_project_attribution(self.cfg, self.store).reattributed, 0)
        self.store.reattribute_session_project("moved", "assistant")
        self.assertEqual(after, self.snapshot())

    def test_queue_failure_rolls_back_single_and_batch_reassignment(self):
        self.seed()
        with self.store.write_transaction():
            for level, scope in (("project", "old-project"), ("digest", self.week)):
                self.store.mark_aggregate_dirty(level, scope)
                self.store.acknowledge_aggregate(level, scope, 1, 1)
        before = self.snapshot()
        mark = self.store.mark_aggregate_dirty

        def fail_after_writing(level, scope):
            mark(level, scope)
            if level == "digest":
                raise RuntimeError("injected queue failure")

        for batch in (False, True):
            with self.subTest(batch=batch), mock.patch.object(self.store, "mark_aggregate_dirty",
                                                             side_effect=fail_after_writing):
                with self.assertRaisesRegex(RuntimeError, "queue failure"):
                    if batch:
                        self.store.reattribute_session_projects([("moved", "assistant")])
                    else:
                        self.store.reattribute_session_project("moved", "assistant")
            self.assertFalse(self.store.db.in_transaction)
            self.store.commit()  # close/commit cannot expose a partial move
            self.assertEqual(before, self.snapshot())

    def test_later_batch_failure_rolls_back_earlier_refresh_jobs(self):
        self.seed()
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.store.reattribute_session_projects([("moved", "assistant"), ("missing", "assistant")])
        self.assertEqual(before, self.snapshot())

    def test_end_week_matches_ingestion_across_year_boundary_and_missing_dates(self):
        for i, (ended_at, expected_week) in enumerate((
            ("2027-01-03T23:59:00Z", "2026-12-28"),
            ("2027-01-04T00:01:00-07:00", "2027-01-04"),
            (None, None), ("not-a-date", None),
        )):
            with self.subTest(ended_at=ended_at):
                # A legacy null project still queues its destination. Digest
                # ownership follows ended_at, not every week the session spans.
                with self.store.write_transaction():
                    self.store.db.execute("DELETE FROM aggregate_refresh")
                    self.store.upsert_session(id=str(i), project=None,
                                              started_at="2026-01-01", ended_at=ended_at)
                self.store.reattribute_session_project(str(i), "assistant")
                expected = {("project", "assistant")}
                if expected_week:
                    expected.add(("digest", expected_week))
                self.assertEqual(self.queued(), expected)

    def test_reassignment_invalidates_an_inflight_digest_refresh(self):
        self.seed()
        self.store.assert_embedder(self.embedder.descriptor)
        with self.store.write_transaction():
            self.store.mark_aggregate_dirty("digest", self.week)
        other = MemoryStore(self.cfg.db_path)
        self.addCleanup(other.close)

        def reassign_during_generation(*args):
            other.reattribute_session_project("moved", "assistant")
            return "Stale digest describing old-project"

        with mock.patch("agentic_memory.ingest.distill_week", side_effect=reassign_during_generation):
            result = ingest(self.cfg, self.store, self.embedder, self.backend,
                            aggregates_only=True, refresh_aggregates=True)
        self.assertEqual(result.aggregates_pending, 3)
        row = self.store.db.execute("SELECT content FROM memories WHERE level='digest'").fetchone()
        self.assertEqual(row[0], "Original digest")
        self.assertEqual(self.store.get_session("moved")["project"], "assistant")


class PlanningAndCliPolicyTests(unittest.TestCase):
    def test_plan_policy_preserves_legacy_behavior_and_locks_assistant_writes(self) -> None:
        self.assertEqual(
            plan.resolve_project_policy("-Users-sam-repo", write=True),
            "-Users-sam-repo",
        )
        self.assertEqual(
            plan.resolve_project_policy(
                "", default_project="assistant", lock_writes_to_default=True, write=True
            ),
            "assistant",
        )
        self.assertEqual(
            plan.resolve_project_policy(
                "other", default_project="assistant", lock_writes_to_default=True,
                write=False,
            ),
            "other",
        )
        with self.assertRaisesRegex(ValueError, "locked to project 'assistant'"):
            plan.resolve_project_policy(
                "other", default_project="assistant", lock_writes_to_default=True,
                write=True,
            )

    def test_reconcile_cli_has_no_model_or_embedder_construction(self) -> None:
        source = inspect.getsource(cli.cmd_reconcile_attribution)
        self.assertNotIn("make_embedder", source)
        self.assertNotIn("AgentCoreBackend", source)
        self.assertNotIn(".llm", source)
        self.assertNotIn(".embed", source)
        parser_args = argparse.Namespace(dry_run=True)
        self.assertTrue(hasattr(parser_args, "dry_run"))


if __name__ == "__main__":
    unittest.main()
