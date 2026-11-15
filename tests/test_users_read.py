"""Every tool reads only what its caller may (Epic U, U4).

The compatibility fixture (test_users_compat) is given owners: project alpha
belongs to team core (ana and ben); s-alpha-1 is ana's, s-alpha-2 ben's;
beta is ana's alone (no team); plan items are on alpha (team) and gamma
(ana's). cy is a stranger; the default owner (admin) sees everything.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tess_memory import plan, retrieval, server
from tess_memory.config import Config
from tess_memory.store import MemoryStore
from test_users_compat import Embedder, build_fixture


class ReadAccessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = Config(data_stores=[], data_dir=Path(tmp.name))
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.embedder = Embedder()
        build_fixture(self.store, self.embedder)
        db = self.store.db
        with self.store.write_transaction():
            db.execute("INSERT INTO teams(id) VALUES('core')")
            db.executemany("INSERT INTO team_members(team_id, user_id) VALUES('core', ?)", [("ana",), ("ben",)])
            db.execute("INSERT INTO projects(key, owner_user, team_id) VALUES('alpha', 'ana', 'core')")
            db.execute("INSERT INTO projects(key, owner_user) VALUES('beta', 'ana')")
            db.execute("INSERT INTO projects(key, owner_user) VALUES('gamma', 'ana')")
            for session_id, owner in (("s-alpha-1", "ana"), ("s-alpha-2", "ben"), ("s-beta-1", "ana")):
                db.execute("UPDATE sessions SET owner_user=? WHERE id=?", (owner, session_id))
                db.execute("UPDATE memories SET owner_user=? WHERE session_id=?", (owner, session_id))
            db.execute("UPDATE memories SET owner_user='ana' WHERE level='plan'")

    def call(self, user, tool, *args, **kwargs):
        self.store.principal = self.store.resolve_principal(user)
        with mock.patch.object(server, "_get_store", return_value=self.store), \
             mock.patch.object(server, "CONFIG", self.cfg), \
             mock.patch.object(retrieval, "CONFIG", self.cfg), \
             mock.patch.object(server, "_get_embedder", return_value=self.embedder), \
             mock.patch.object(server, "_get_search_llm", return_value=None):
            self.cfg.identity_user = user
            return getattr(server, tool)(*args, **kwargs)

    def test_search_shows_a_team_member_shared_knowledge_but_not_private_material(self):
        ben = self.call("ben", "search_memory", "sqlite triggers rebuild index commits pytest", mode="static", limit=50)
        self.assertIn("Decided in alpha: Fix FTS trigger corruption", ben)       # ana's decision, team project
        self.assertIn("Solution for Fix FTS trigger corruption", ben)            # ana's solution, team project
        self.assertIn("Prefers small commits while working on Add retry", ben)  # his own preference
        self.assertNotIn("Prefers small commits while working on Fix FTS", ben)  # ana's preference: private
        self.assertNotIn("$ pytest -k fts", ben)                                 # ana's raw chunk: private
        self.assertNotIn("billing webhook", ben)                                 # beta is ana's alone
        self.assertNotIn("Fix FTS trigger corruption)", ben)                     # her session titles stay hers
        self.assertIn("Profile of alpha", self.call("ben", "search_memory", "profile careful migrations", mode="static", limit=50))
        for user in ("cy",):
            self.assertIn("No matching memories", self.call(user, "search_memory", "sqlite", mode="static"))

    def test_context_sessions_and_transcripts_are_their_authors(self):
        ben = self.call("ben", "get_project_context", "alpha")
        self.assertIn("Profile of alpha", ben)
        self.assertIn("Add retry budget to ingest", ben)          # his session
        self.assertNotIn("Fix FTS trigger corruption", ben)       # ana's session
        self.assertIn("Unknown session", self.call("ben", "get_session", "s-alpha-1"))
        self.assertIn("Summary of Fix FTS", self.call("ana", "get_session", "s-alpha-1"))
        self.assertIn("Unknown session", self.call("ben", "get_session_transcript", "s-alpha-1"))
        self.assertIn("No memory for project 'beta'", self.call("ben", "get_project_context", "beta"))

    def test_projects_boards_activity_and_status(self):
        self.assertEqual(self.call("ben", "list_projects").splitlines(), ["- alpha (1 sessions, last active 2026-03-09; 2 plan items)"])
        self.assertEqual(self.call("cy", "list_projects"), "No projects yet.")
        self.assertIn("Retry budget", self.call("ben", "plan_board", "alpha", include_done=True))
        self.assertIn("No plan items", self.call("ben", "plan_board", "gamma"))
        self.assertIn("No plan item #3", self.call("ben", "plan_get", 3))
        recent = self.call("ben", "recent_activity", days=100000)
        self.assertNotIn("Weekly digests", recent)                 # digests are the owner's
        self.assertIn("Add retry budget", recent)
        self.assertNotIn("billing webhook", recent)
        counts = self.call("ben", "memory_status")
        self.assertIn('"sessions": 1', counts)
        self.assertNotIn("ingestion", counts)

    def test_plan_writes_need_project_access(self):
        self.store.principal = self.store.resolve_principal("cy")
        with self.assertRaises(PermissionError):
            plan.add_item(self.store, self.embedder, project="alpha", kind="task", title="Sneak in")
        item = plan.add_item(self.store, self.embedder, project="cy-new", kind="task", title="My own board")
        self.assertEqual(item["project"], "cy-new")
        self.assertIn("cy-new", self.call("cy", "list_projects"))
        self.store.principal = self.store.resolve_principal("ben")
        plan.add_item(self.store, self.embedder, project="alpha", kind="task", title="Team task")  # team access

    def test_the_owner_still_sees_everything(self):
        owner = self.call(None, "search_memory", "billing webhook pytest commits", mode="static", limit=50)
        self.assertIn("billing webhook", owner)
        self.assertEqual(self.call(None, "search_memory", "pytest", level="chunk", mode="static").count("$ pytest"), 3)
        self.assertEqual(self.call("ben", "search_memory", "pytest", level="chunk", mode="static").count("$ pytest"), 1)
        self.assertIn("Weekly digests", self.call(None, "recent_activity", days=100000))


if __name__ == "__main__":
    unittest.main()
