"""Users, teams and visibility: who may read which row."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tess_memory.principals import Principal, VisibilityPolicy, memory_access_sql, session_access_sql
from tess_memory.store import USERS_SCHEMA_VERSION, MemoryRow, MemoryStore


class AccessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = MemoryStore(Path(tmp.name) / "memory.db")
        self.addCleanup(self.store.close)
        db = self.store.db
        with self.store.write_transaction():
            db.execute("INSERT INTO teams(id, name) VALUES('core', 'Core')")
            db.executemany("INSERT INTO team_members(team_id, user_id) VALUES('core', ?)", [("ana",), ("ben",)])
            db.execute("INSERT INTO projects(key, owner_user, team_id) VALUES('shared', 'ana', 'core')")
            db.execute("INSERT INTO projects(key, owner_user) VALUES('solo', 'ana')")
            db.execute("INSERT INTO principals(id, admin) VALUES('root', 1)")
            db.execute("INSERT INTO principals(id, disabled) VALUES('gone', 1)")
            rows = [
                # (owner, project, level, kind, visibility, label)
                ("ana", "shared", "memory", "decision", None, "ana-shared-decision"),
                ("ana", "shared", "memory", "preference", None, "ana-shared-preference"),
                ("ana", "shared", "chunk", "chunk", None, "ana-shared-chunk"),
                ("ana", "shared", "chunk", "chunk", "team", "ana-shared-chunk-shared"),
                ("ana", "solo", "memory", "decision", None, "ana-solo-decision"),
                ("ana", "solo", "memory", "fact", "org", "ana-solo-org-fact"),
                ("ben", "shared", "memory", "solution", "private", "ben-unshared-solution"),
                (None, "legacy", "memory", "decision", None, "legacy-owner-decision"),
                (None, "shared", "project", "profile", None, "shared-profile"),
            ]
            for owner, project, level, kind, visibility, label in rows:
                ids = self.store.add_memories([MemoryRow(level, kind, label, project=project)])
                db.execute("UPDATE memories SET owner_user=?, visibility=? WHERE id=?", (owner, visibility, ids[0]))

    def readable(self, principal):
        sql, args = memory_access_sql(principal, VisibilityPolicy(), self.store.default_owner())
        return {r[0] for r in self.store.db.execute(f"SELECT m.content FROM memories m WHERE {sql}", args)}

    def test_each_reader_sees_exactly_what_the_rules_allow(self):
        everything = {r[0] for r in self.store.db.execute("SELECT content FROM memories")}
        self.assertTrue(self.store.resolve_principal(None).admin)  # admins skip the predicate entirely
        self.assertEqual(self.readable(Principal("owner")), everything - {"ana-shared-decision", "ana-shared-preference", "ana-shared-chunk", "ana-shared-chunk-shared", "ana-solo-decision", "ben-unshared-solution"} | {"ana-solo-org-fact"})  # NULL owners are the default owner
        ana = self.store.resolve_principal("ana")
        self.assertEqual(self.readable(ana), {
            "ana-shared-decision", "ana-shared-preference", "ana-shared-chunk", "ana-shared-chunk-shared",
            "ana-solo-decision", "ana-solo-org-fact", "shared-profile",
        })
        ben = self.store.resolve_principal("ben")
        self.assertEqual(self.readable(ben), {
            # team-visible in a team project, an explicit share, an org-wide fact, the team profile, his own
            "ana-shared-decision", "ana-shared-chunk-shared", "ana-solo-org-fact", "shared-profile", "ben-unshared-solution",
        })
        stranger = self.store.resolve_principal("cy")
        self.assertEqual(self.readable(stranger), {"ana-solo-org-fact"})

    def test_principal_resolution(self):
        owner = self.store.resolve_principal(None)
        self.assertEqual((owner.user, owner.admin), ("owner", True))
        self.assertTrue(self.store.resolve_principal("owner").admin)
        self.assertTrue(self.store.resolve_principal("root").admin)
        ana = self.store.resolve_principal("ana")
        self.assertEqual((ana.admin, ana.teams), (False, frozenset({"core"})))
        with self.assertRaises(PermissionError):
            self.store.resolve_principal("gone")
        with self.assertRaises(ValueError):
            self.store.resolve_principal("bad user")
        with self.store.write_transaction():
            self.store.set_meta("default_owner", "sam")
        self.assertTrue(self.store.resolve_principal("sam").admin)
        self.assertFalse(self.store.resolve_principal("owner").admin)

    def test_sessions_are_their_authors(self):
        with self.store.write_transaction():
            self.store.upsert_session(id="a1", project="shared", title="ana's")
            self.store.upsert_session(id="o1", project="legacy", title="legacy")
            self.store.db.execute("UPDATE sessions SET owner_user='ana' WHERE id='a1'")
        sql, args = session_access_sql(Principal("ana"), self.store.default_owner())
        self.assertEqual([r[0] for r in self.store.db.execute(f"SELECT s.id FROM sessions s WHERE {sql}", args)], ["a1"])
        sql, args = session_access_sql(Principal("owner"), self.store.default_owner())
        self.assertEqual([r[0] for r in self.store.db.execute(f"SELECT s.id FROM sessions s WHERE {sql}", args)], ["o1"])

    def test_the_migration_is_recorded_and_policy_defaults_are_the_owners(self):
        self.assertEqual(self.store.db.execute("PRAGMA user_version").fetchone()[0], USERS_SCHEMA_VERSION)
        policy = VisibilityPolicy()
        self.assertEqual([policy.default_for(level, kind) for level, kind in (
            ("memory", "decision"), ("memory", "preference"), ("memory", "task_state"), ("chunk", "chunk"),
            ("session", "summary"), ("project", "profile"), ("plan", "epic"), ("digest", "digest"))],
            ["team", "private", "private", "private", "private", "team", "team", "private"])
        with self.assertRaises(ValueError):
            VisibilityPolicy(fallback="public")


if __name__ == "__main__":
    unittest.main()
