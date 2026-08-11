from __future__ import annotations

import argparse
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentic_memory import cli, plan
from agentic_memory.attribution import reconcile_project_attribution, source_adapter
from agentic_memory.config import Config, DataStore, ProjectAttribution, load_config
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
