"""Caller identity is bound when a server or command starts."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tess_memory import cli, server
from tess_memory.config import Config, load_config


class IdentityConfigTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def settings(self, text):
        path = self.root / "settings.toml"
        path.write_text(text)
        return path

    def test_identity_and_data_store_owners_are_read_and_env_wins(self):
        path = self.settings('[identity]\nuser = "ana"\n[[data_stores]]\nname = "c"\npath = "/tmp/x"\nuser = "ana"\nteam = "core"\n')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TESS_MEMORY_USER", None)
            cfg = load_config(path)
        self.assertEqual((cfg.identity_user, cfg.data_stores[0].user, cfg.data_stores[0].team), ("ana", "ana", "core"))
        with mock.patch.dict(os.environ, {"TESS_MEMORY_USER": "ben"}):
            self.assertEqual(load_config(path).identity_user, "ben")

    def test_no_identity_means_the_unrestricted_default_owner(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TESS_MEMORY_USER", None)
            cfg = load_config(self.settings(""))
        self.assertIsNone(cfg.identity_user)
        with self.assertRaises(ValueError):
            load_config(self.settings('[identity]\nuser = "not a user"\n'))

    def test_the_server_binds_its_store_to_the_configured_user_once(self):
        cfg = Config(data_stores=[], data_dir=self.root / "memory", identity_user="ana")
        with mock.patch.object(server, "CONFIG", cfg), mock.patch.object(server, "_store", None):
            store = server._get_store()
            try:
                self.assertEqual((store.principal.user, store.principal.admin), ("ana", False))
                self.assertIs(server._get_store(), store)
            finally:
                store.close()
        cfg = Config(data_stores=[], data_dir=self.root / "memory2")
        with mock.patch.object(server, "CONFIG", cfg), mock.patch.object(server, "_store", None):
            store = server._get_store()
            try:
                self.assertIsNone(store.principal)
            finally:
                store.close()

    def test_serve_takes_a_per_process_user(self):
        cfg = Config(data_stores=[], data_dir=self.root / "memory")
        with mock.patch.object(server, "CONFIG", cfg), mock.patch.object(server, "run") as run:
            self.assertEqual(cli.main(["serve", "--user", "tess:sam"]), 0)
        run.assert_called_once()
        self.assertEqual(cfg.identity_user, "tess:sam")

    def test_no_tool_takes_a_user_argument(self):
        import inspect
        for name in ("search_memory", "get_project_context", "get_session", "get_session_transcript", "list_projects",
                     "recent_activity", "plan_add", "plan_update", "plan_board", "plan_get", "memory_status"):
            parameters = inspect.signature(getattr(server, name)).parameters
            self.assertFalse({"user", "owner", "principal", "as_user", "team"} & set(parameters), name)


if __name__ == "__main__":
    unittest.main()
