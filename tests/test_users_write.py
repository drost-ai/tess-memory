"""Writes carry their owner and author; team profiles never include private
material."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tess_memory import plan
from tess_memory.config import Config, DataStore, load_config
from tess_memory.ingest import ingest
from tess_memory.principals import Principal
from tess_memory.store import MemoryStore
from test_codex import RecordingEmbedder, write_rollout
from test_incremental_ingest import Backend


class ProfileBackend(Backend):
    """Answers distillation like Backend, and records what profiles were built from."""

    def __init__(self):
        super().__init__()
        self.profile_prompts = []

    def complete(self, *, system, prompt):
        if "profile" in system.lower() and "project" in system.lower():
            self.profile_prompts.append(prompt)
            return '{"profile": "Built from what the team may read."}'
        if "weekly" in system.lower():
            return '{"digest": "A week."}'
        return super().complete(system=system, prompt=prompt)


class IngestStampingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.home = self.root / "codex"
        write_rollout(self.home)
        self.embedder = RecordingEmbedder()

    def run_ingest(self, incremental, **ds_options):
        cfg = Config(data_stores=[DataStore("codex", self.home, type="codex", project="p", **ds_options)],
                     data_dir=self.root / f"memory-{incremental}-{len(ds_options)}", incremental_ingestion=incremental,
                     segment_chars=3000, segment_overlap=0)
        store = MemoryStore(cfg.db_path)
        self.addCleanup(store.close)
        backend = ProfileBackend()
        ingest(cfg, store, self.embedder, backend, refresh_aggregates=True)
        return store, backend

    def test_a_data_store_user_and_team_are_stamped_on_everything_ingested(self):
        for incremental in (False, True):
            store, backend = self.run_ingest(incremental, user="ana", team="core")
            owners = {r[0] for r in store.db.execute("SELECT owner_user FROM memories WHERE level IN ('chunk','memory','session')")}
            self.assertEqual(owners, {"ana"}, incremental)
            self.assertEqual({r[0] for r in store.db.execute("SELECT owner_user FROM sessions")}, {"ana"})
            self.assertEqual(tuple(store.db.execute("SELECT key, owner_user, team_id FROM projects").fetchone()), ("p", "ana", "core"))
            # A team project's profile is built from team-visible memories, not session summaries.
            self.assertTrue(backend.profile_prompts, incremental)
            prompt = backend.profile_prompts[-1]
            self.assertIn("A durable unchanged decision", prompt)
            self.assertNotIn("Summary ", prompt)
            profile = store.db.execute("SELECT owner_user, visibility, content FROM memories WHERE level='project'").fetchone()
            self.assertEqual(tuple(profile), (None, None, "Built from what the team may read."))

    def test_the_default_owner_stays_unstamped_and_profiles_are_built_as_before(self):
        store, backend = self.run_ingest(False)
        self.assertEqual({r[0] for r in store.db.execute("SELECT owner_user FROM memories")}, {None})
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM projects").fetchone()[0], 0)
        self.assertIn("Summary ", backend.profile_prompts[-1])


class PlanAuthorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = MemoryStore(Path(tmp.name) / "memory.db")
        self.addCleanup(self.store.close)

    def test_plan_writes_record_their_author_and_the_project_owner(self):
        embedder = RecordingEmbedder()
        self.store.principal = Principal("ana")
        item = plan.add_item(self.store, embedder, project="p", kind="epic", title="Ship it")
        with self.store.write_transaction():  # the project becomes the team's
            self.store.db.execute("INSERT INTO teams(id) VALUES('core')")
            self.store.db.execute("UPDATE projects SET team_id='core' WHERE key='p'")
        self.store.principal = Principal("ben", teams=frozenset({"core"}))
        plan.update_item(self.store, embedder, item["id"], status="in_progress")
        row = self.store.plan_get(item["id"])
        self.assertEqual((row["created_by"], row["updated_by"]), ("ana", "ben"))
        self.assertEqual([e["actor"] for e in self.store.plan_history(item["id"])], ["ana", "ben"])
        self.assertEqual(tuple(self.store.db.execute("SELECT owner_user FROM projects WHERE key='p'").fetchone()), ("ana",))
        mirror = self.store.db.execute("SELECT owner_user FROM memories WHERE plan_id=?", (item["id"],)).fetchone()
        self.assertEqual(mirror["owner_user"], "ben")  # the mirror is rewritten by the last editor

    def test_the_default_owner_writes_unstamped(self):
        item = plan.add_item(self.store, RecordingEmbedder(), project="p", kind="epic", title="Ship it")
        row = self.store.plan_get(item["id"])
        self.assertEqual((row["created_by"], row["updated_by"]), (None, None))
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM projects").fetchone()[0], 0)

    def test_replacing_a_session_keeps_its_owner(self):
        with self.store.write_transaction():
            self.store.upsert_session(id="s", project="p", title="one", owner_user="ana")
            self.store.upsert_session(id="s", project="p", title="two")
        self.assertEqual(self.store.get_session("s")["owner_user"], "ana")


class VisibilityConfigTests(unittest.TestCase):
    def test_the_policy_can_be_tuned_per_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            path.write_text('[visibility]\nfallback = "private"\n[visibility.memory_kinds]\npreference = "team"\n[visibility.levels]\nsession = "team"\n')
            cfg = load_config(path)
        self.assertEqual(cfg.visibility.default_for("memory", "preference"), "team")
        self.assertEqual(cfg.visibility.default_for("memory", "decision"), "team")
        self.assertEqual(cfg.visibility.default_for("session", "summary"), "team")
        self.assertEqual(cfg.visibility.default_for("chunk", "chunk"), "private")


if __name__ == "__main__":
    unittest.main()
