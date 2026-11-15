"""Epic U compatibility harness: users and visibility must not change what an
instance with no identity configured sees, and older code must keep working
against a store the new code has opened.

- Golden: every MCP tool's output over a fixture store built through the
  public APIs, compared with outputs recorded from the code before Epic U
  (tests/fixtures/users_compat_golden.json; AGENTIC_MEMORY_UPDATE_GOLDEN=1
  rewrites it, which only ever happens on purpose).
- Concurrency: many processes open a store created by the old code at the
  same instant; every open succeeds and the store ends up fully migrated.
- Old code: the code at BASE (the commit Epic U started from) reads and
  writes a store the new code has migrated.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from agentic_memory import plan, retrieval, server
from agentic_memory.config import Config
from agentic_memory.store import MemoryRow, MemoryStore

BASE = "910d5c0"  # main when Epic U began (2026-10-07)
GOLDEN = Path(__file__).parent / "fixtures" / "users_compat_golden.json"
REPO = Path(__file__).resolve().parents[1]
FROZEN = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)  # "now" for the golden run


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN if tz is not None else FROZEN.replace(tzinfo=None)


_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?")


class Embedder:
    descriptor = "test:users:4"

    def embed_passages(self, texts):
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        # Deterministic, content-dependent: a word's letters steer the vector.
        values = np.zeros(4, dtype=np.float32)
        for index, char in enumerate(text.lower()):
            values[index % 4] += (ord(char) % 13) / 13
        norm = np.linalg.norm(values)
        return values / norm if norm else values


def build_fixture(store: MemoryStore, embedder: Embedder) -> None:
    """Two projects, every memory level, a plan board: through public APIs."""
    sessions = [
        ("s-alpha-1", "alpha", "Fix FTS trigger corruption", "2026-03-02T10:00:00+00:00"),
        ("s-alpha-2", "alpha", "Add retry budget to ingest", "2026-03-09T10:00:00+00:00"),
        ("s-beta-1", "beta", "Design the billing webhook", "2026-03-05T10:00:00+00:00"),
    ]
    with store.write_transaction():
        for session_id, project, title, ended in sessions:
            store.upsert_session(id=session_id, project=project, cwd=f"/work/{project}", git_branch="main",
                                 title=title, summary=f"Summary of {title}.", outcome=f"Outcome of {title}.",
                                 started_at=ended, ended_at=ended, n_user=3, n_assistant=3,
                                 source_path=f"/missing/{session_id}.jsonl")
            rows = [
                MemoryRow("chunk", "chunk", f"$ pytest -k {title.split()[1].lower()} # raw activity in {project}", 0.2, session_id, project, ended),
                MemoryRow("memory", "decision", f"Decided in {project}: {title} uses sqlite triggers", 0.7, session_id, project, ended),
                MemoryRow("memory", "preference", f"Prefers small commits while working on {title}", 0.5, session_id, project, ended),
                MemoryRow("memory", "solution", f"Solution for {title}: rebuild the index", 0.8, session_id, project, ended),
                MemoryRow("session", "summary", f"{title}: Summary of {title}.", 0.6, session_id, project, ended),
            ]
            for row in rows:
                row.vector = embedder.embed_passages([row.content])[0]
            store.add_memories(rows)
        for project in ("alpha", "beta"):
            text = f"# {project}\nProfile of {project}: sqlite store, careful migrations."
            store.add_memories([MemoryRow("project", "profile", text, 0.9, None, project, "2026-03-09T10:00:00+00:00",
                                          vector=embedder.embed_passages([text])[0])])
        digest = "Week of work on sqlite triggers and the billing webhook."
        store.add_memories([MemoryRow("digest", "digest", digest, 0.5, None, None, "2026-03-02",
                                      vector=embedder.embed_passages([digest])[0])])
    epic = plan.add_item(store, embedder, project="alpha", kind="epic", title="Harden the store", body="Make it durable.")
    slice_ = plan.add_item(store, embedder, project="alpha", kind="slice", title="Retry budget", parent_id=epic["id"])
    plan.update_item(store, embedder, slice_["id"], status="in_progress")
    plan.add_item(store, embedder, project="gamma", kind="vision", title="A board before any session")


def tool_outputs(store: MemoryStore, cfg: Config, embedder: Embedder) -> dict[str, str]:
    calls = {
        "search_static": lambda: server.search_memory("sqlite triggers index", mode="static"),
        "search_alpha": lambda: server.search_memory("retry budget", project="alpha", mode="static"),
        "search_chunks": lambda: server.search_memory("pytest", level="chunk", mode="static"),
        "search_preference": lambda: server.search_memory("commits", kind="preference", mode="static"),
        "search_plan": lambda: server.search_memory("harden store", level="plan", mode="static"),
        "context_alpha": lambda: server.get_project_context("alpha"),
        "context_gamma": lambda: server.get_project_context("gamma"),
        "session": lambda: server.get_session("s-alpha-1"),
        "projects": lambda: server.list_projects(),
        "recent": lambda: server.recent_activity(days=100000),
        "board": lambda: server.plan_board("alpha", include_done=True),
        "plan_item": lambda: server.plan_get(1),
    }
    with mock.patch.object(server, "_get_store", return_value=store), \
         mock.patch.object(server, "CONFIG", cfg), \
         mock.patch.object(retrieval, "CONFIG", cfg), \
         mock.patch.object(server, "_get_embedder", return_value=embedder), \
         mock.patch.object(server, "_get_search_llm", return_value=None):
        return {name: _TS.sub("<TS>", call()) for name, call in calls.items()}


class GoldenTests(unittest.TestCase):
    def test_no_identity_returns_exactly_what_the_code_before_epic_u_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(data_stores=[], data_dir=Path(tmp))
            store = MemoryStore(cfg.db_path)
            try:
                embedder = Embedder()
                # Plan timestamps and the recency boost read the clock: freeze it.
                # Each plan write a minute apart: list_projects orders ties by hash.
                ticks = ((FROZEN + timedelta(minutes=n)).isoformat(timespec="seconds") for n in range(1000))
                with mock.patch.object(plan, "_now", side_effect=lambda: next(ticks)), \
                     mock.patch.object(retrieval, "datetime", _FrozenDatetime):
                    build_fixture(store, embedder)
                    outputs = tool_outputs(store, cfg, embedder)
            finally:
                store.close()
        if os.environ.get("AGENTIC_MEMORY_UPDATE_GOLDEN") == "1":
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
        expected = json.loads(GOLDEN.read_text())
        self.assertEqual(sorted(outputs), sorted(expected))
        for name in expected:
            self.assertEqual(outputs[name], expected[name], name)


def _base_code(target: Path) -> Path | None:
    """The package as it was at BASE, extracted with git archive."""
    archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE, "src"], capture_output=True)
    if archive.returncode != 0:
        return None
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(target)], input=archive.stdout, check=True)
    return target / "src"


def _run(python_path: Path, code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(python_path)}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True, timeout=120)


class OldCodeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.old = _base_code(self.root / "base")
        if self.old is None:
            self.skipTest(f"git archive of {BASE} unavailable")
        self.db = self.root / "memory.db"
        made = _run(self.old, f"""
            from pathlib import Path
            from agentic_memory.store import MemoryStore, MemoryRow
            s = MemoryStore(Path({str(self.db)!r}))
            with s.write_transaction():
                s.upsert_session(id='old-1', project='alpha', title='old session', ended_at='2026-01-01')
                s.add_memories([MemoryRow('memory', 'decision', 'old decision about sqlite', session_id='old-1', project='alpha')])
            s.close()
        """)
        self.assertEqual(made.returncode, 0, made.stderr)

    def test_many_processes_open_an_old_store_at_once(self):
        current = REPO / "src"
        code = f"""
            import time
            from pathlib import Path
            from agentic_memory.store import MemoryStore
            time.sleep(max(0, {__import__('time').time() + 1.5} - time.time()))  # a common start instant
            s = MemoryStore(Path({str(self.db)!r}))
            s.close()
        """
        env = {**os.environ, "PYTHONPATH": str(current)}
        procs = [subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], env=env, stderr=subprocess.PIPE, text=True) for _ in range(8)]
        errors = [p.communicate(timeout=120)[1] for p in procs]
        self.assertEqual([p.returncode for p in procs], [0] * 8, "\n".join(errors))
        store = MemoryStore(self.db)
        try:
            self.assertEqual(store.get_session("old-1")["title"], "old session")
        finally:
            store.close()

    def test_old_code_reads_and_writes_a_store_the_new_code_opened(self):
        MemoryStore(self.db).close()  # the new code opens (and migrates) it
        result = _run(self.old, f"""
            from pathlib import Path
            from agentic_memory.store import MemoryStore, MemoryRow
            s = MemoryStore(Path({str(self.db)!r}))
            with s.write_transaction():
                s.add_memories([MemoryRow('memory', 'solution', 'old code still writes', session_id='old-1', project='alpha')])
            hits = s.fts_search('sqlite writes')
            print(len(hits), s.get_session('old-1')['title'])
            s.close()
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["2", "old", "session"])
        store = MemoryStore(self.db)
        try:
            rows = store.db.execute("SELECT content FROM memories ORDER BY id").fetchall()
            self.assertEqual([r["content"] for r in rows], ["old decision about sqlite", "old code still writes"])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
