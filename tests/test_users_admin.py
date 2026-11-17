"""Administration and sharing."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from tess_memory import admin, cli, server
from tess_memory.config import Config
from tess_memory.principals import Principal
from tess_memory.store import MemoryRow, MemoryStore


class AdminTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cfg = Config(data_stores=[], data_dir=Path(tmp.name))
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        with self.store.write_transaction():
            self.store.upsert_session(id="s1", project="alpha", title="ana's work", owner_user="ana")
            self.ids = self.store.add_memories([
                MemoryRow("memory", "decision", "use triggers", session_id="s1", project="alpha", owner_user="ana"),
                MemoryRow("memory", "preference", "likes tabs", session_id="s1", project="alpha", owner_user="ana"),
                MemoryRow("session", "summary", "ana's session summary", session_id="s1", project="alpha", owner_user="ana"),
                MemoryRow("chunk", "chunk", "$ raw command", session_id="s1", project="alpha", owner_user="ana"),
            ])

    def visible_to(self, user):
        self.store.principal = self.store.resolve_principal(user)
        try:
            return {r["content"] for r in self.store.get_memories(self.ids)}
        finally:
            self.store.principal = None

    def test_administration_needs_an_administrator(self):
        self.store.principal = Principal("ana")
        for call in (lambda: admin.add_user(self.store, "x"), lambda: admin.add_team(self.store, "t"),
                     lambda: admin.map_project(self.store, "alpha", team="t"), lambda: admin.list_users(self.store),
                     lambda: admin.set_default_owner(self.store, "ana"), lambda: admin.who_can_see(self.store, self.ids[0])):
            with self.assertRaises(PermissionError):
                call()

    def test_mapping_a_project_shares_its_knowledge_and_authors_control_the_rest(self):
        self.assertEqual(self.visible_to("ben"), set())
        admin.add_user(self.store, "ana", display_name="Ana")
        admin.add_team(self.store, "core")
        admin.add_member(self.store, "core", "ana")
        admin.add_member(self.store, "core", "ben")
        admin.map_project(self.store, "alpha", team="core", owner="ana")
        self.assertEqual(self.visible_to("ben"), {"use triggers"})  # the decision; not preference, summary, chunk

        self.store.principal = self.store.resolve_principal("ana")
        admin.share(self.store, f"memory:{self.ids[0]}", "private")
        admin.share(self.store, f"memory:{self.ids[1]}", "team")
        self.assertEqual(self.visible_to("ben"), {"likes tabs"})
        self.store.principal = self.store.resolve_principal("ana")
        result = admin.share(self.store, "session:s1", "team")
        self.assertEqual(result["rows"], 3)  # memories and summary; never the raw chunk
        self.assertEqual(self.visible_to("ben"), {"use triggers", "likes tabs", "ana's session summary"})
        self.store.principal = self.store.resolve_principal("ana")
        admin.share(self.store, "session:s1", None)
        self.assertEqual(self.visible_to("ben"), {"use triggers"})

        self.store.principal = self.store.resolve_principal("ben")
        with self.assertRaises(PermissionError):
            admin.share(self.store, f"memory:{self.ids[1]}", "org")
        self.store.principal = None
        seen = admin.who_can_see(self.store, self.ids[0])
        self.assertEqual((seen["visibility"], seen["team"], seen["readers"]), ("team", "core", ["ana", "ben", "owner"]))
        events = [tuple(r) for r in self.store.db.execute("SELECT actor, target, visibility FROM visibility_events ORDER BY id")]
        self.assertEqual(events[0], ("ana", f"memory:{self.ids[0]}", "private"))
        self.assertEqual(len(events), 4)

    def test_users_teams_and_the_default_owner(self):
        admin.add_user(self.store, "ana", admin=True)
        admin.add_team(self.store, "core")
        admin.add_member(self.store, "core", "ben")
        admin.disable_user(self.store, "cy")
        users = {u["id"]: u for u in admin.list_users(self.store)}
        self.assertTrue(users["owner"]["default_owner"])
        self.assertTrue(users["ana"]["admin"])
        self.assertEqual(users["ben"]["teams"], ["core"])
        self.assertTrue(users["cy"]["disabled"])
        with self.assertRaises(ValueError):
            admin.disable_user(self.store, "owner")
        with self.assertRaises(ValueError):
            admin.add_member(self.store, "nope", "ben")
        admin.set_default_owner(self.store, "sam")
        self.assertTrue(self.store.resolve_principal("sam").admin)

    def test_cli_and_mcp_tool(self):
        with mock.patch.object(cli, "CONFIG", self.cfg), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["users", "team-add", "core"]), 0)
            self.assertEqual(cli.main(["users", "member-add", "core", "ben"]), 0)
            self.assertEqual(cli.main(["users", "project-map", "alpha", "--team", "core"]), 0)
            self.assertEqual(cli.main(["users", "list"]), 0)
        self.assertIn('"ben"', out.getvalue())
        self.store.principal = self.store.resolve_principal("ana")
        with mock.patch.object(server, "_get_store", return_value=self.store):
            self.assertIn("1 rows now org", server.memory_share(f"memory:{self.ids[1]}", "org"))
        self.assertIn("likes tabs", self.visible_to("cy"))
        self.cfg.identity_user = "ben"
        with mock.patch.object(cli, "CONFIG", self.cfg), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["users", "add", "eve"]), 2)  # not an administrator


if __name__ == "__main__":
    unittest.main()
