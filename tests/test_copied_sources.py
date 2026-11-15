"""A store's sessions copied from another source directory are skipped
(skip_copies_of), so a copied runtime home is not distilled a second time."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tess_memory.attribution import source_adapter
from tess_memory.config import DataStore, load_config


def write_meta(directory: Path, session_id: str, cwd: str = "/work/tess") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.json"
    path.write_text(json.dumps({"id": session_id, "cwd": cwd, "messages": []}), encoding="utf-8")
    return path


class CopiedSourcesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.origin = self.root / "otherhost" / "sessions"
        self.home = self.root / "tess" / "sessions"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def names(self, ds: DataStore) -> list[str]:
        return [path.name for path, _project, _parent in source_adapter(ds)[0]]

    def test_agent_core_copies_are_skipped_and_native_sessions_kept(self) -> None:
        write_meta(self.origin, "copied")
        write_meta(self.origin, "resumed")
        write_meta(self.home, "copied")
        write_meta(self.home, "resumed", cwd="/work/tess/continued")  # changed after the copy: still a copy
        write_meta(self.home, "native")
        ds = DataStore(name="tess", path=self.home, type="agent-core", project="assistant", skip_copies_of=[self.origin])
        self.assertEqual(self.names(ds), ["native.json"])

    def test_without_the_option_every_session_is_discovered(self) -> None:
        write_meta(self.origin, "copied")
        write_meta(self.home, "copied")
        write_meta(self.home, "native")
        ds = DataStore(name="tess", path=self.home, type="agent-core", project="assistant")
        self.assertEqual(self.names(ds), ["copied.json", "native.json"])

    def test_transcript_copies_are_skipped(self) -> None:
        for directory in (self.origin, self.home):
            directory.mkdir(parents=True)
            (directory / "old.jsonl").write_text("{}\n", encoding="utf-8")
        (self.home / "new.jsonl").write_text("{}\n", encoding="utf-8")
        ds = DataStore(name="archive", path=self.home, type="transcript", project="assistant", skip_copies_of=[self.origin])
        self.assertEqual(self.names(ds), ["new.jsonl"])

    def test_settings_parse_and_expand_the_origins(self) -> None:
        settings = self.root / "settings.toml"
        settings.write_text(
            '[[data_stores]]\nname = "tess"\ntype = "agent-core"\npath = "~/tess/sessions"\n'
            'project = "assistant"\nskip_copies_of = ["~/otherhost/sessions"]\n\n'
            '[[data_stores]]\nname = "plain"\ntype = "agent-core"\npath = "~/plain"\n',
            encoding="utf-8",
        )
        with mock.patch.dict(os.environ, {"HOME": str(self.root), "TESS_MEMORY_SETTINGS": str(settings)}):
            cfg = load_config()
        tess, plain = cfg.data_stores
        self.assertEqual(tess.skip_copies_of, [self.root / "otherhost" / "sessions"])
        self.assertEqual(plain.skip_copies_of, [])


if __name__ == "__main__":
    unittest.main()
