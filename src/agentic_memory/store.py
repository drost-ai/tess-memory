"""SQLite-backed memory store: sessions, multi-timescale memories, FTS5, vectors.

Memory levels (timescales, finest → coarsest):
    chunk    raw conversation windows            (exact recall of code/errors)
    memory   LLM-extracted atomic facts          (decisions, solutions, prefs)
    session  one summary per session
    digest   one summary per ISO week
    project  one rolling profile per project

Vectors are stored as float32 blobs and searched brute-force with numpy — at
this corpus size (tens of thousands of rows) that is faster than maintaining
an index, and it keeps the store dependency-free.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .time_utils import week_start

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
    path TEXT PRIMARY KEY,
    mtime REAL,
    size INTEGER,
    session_id TEXT,
    status TEXT,               -- 'distilled' | 'fallback' | 'extractive' | 'skipped'
    ingested_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY,
    project TEXT,
    cwd TEXT,
    git_branch TEXT,
    title TEXT,
    summary TEXT,
    outcome TEXT,
    started_at TEXT,
    ended_at TEXT,
    n_user INTEGER,
    n_assistant INTEGER,
    source_path TEXT
);
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    project TEXT,
    level TEXT NOT NULL,       -- chunk | memory | session | digest | project
    kind TEXT,                 -- decision|fact|preference|solution|problem|task_state|summary|profile|chunk|digest
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.3,
    ts TEXT,
    tags TEXT DEFAULT '[]',
    plan_id INTEGER            -- links a level='plan' mirror row to plan_items.id
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(level);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, content='memories', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES('delete', old.id, old.content, old.tags);
END;
CREATE TABLE IF NOT EXISTS embeddings(
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS plan_items(
    id INTEGER PRIMARY KEY,
    project TEXT NOT NULL,
    kind TEXT NOT NULL,                   -- vision | epic | slice | task | note
    parent_id INTEGER REFERENCES plan_items(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'planned',  -- planned|in_progress|blocked|complete|dropped
    priority INTEGER NOT NULL DEFAULT 3,  -- 1 highest .. 5 lowest
    tags TEXT NOT NULL DEFAULT '[]',
    status_at TEXT,                       -- when the item entered its current status
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_plan_project ON plan_items(project);
CREATE TABLE IF NOT EXISTS plan_events(
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES plan_items(id),
    ts TEXT,
    event TEXT,                           -- created | status | edit
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_plan_events_item ON plan_events(item_id);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS aggregate_refresh(
    level TEXT NOT NULL,
    scope TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    applied_revision INTEGER NOT NULL DEFAULT 0,
    refreshed_at REAL,
    PRIMARY KEY(level, scope)
);
"""


@dataclass
class MemoryRow:
    level: str
    kind: str
    content: str
    importance: float = 0.3
    session_id: str | None = None
    project: str | None = None
    ts: str | None = None
    tags: list[str] = field(default_factory=list)
    vector: np.ndarray | None = None
    plan_id: int | None = None  # set on level='plan' mirror rows


class MemoryStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: MCP tool dispatch and the search agent may
        # touch the store from worker threads; CPython's sqlite3 serializes
        # statement execution, and plan writes additionally hold self.lock.
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.lock = threading.RLock()  # serializes multi-statement write groups
        self._emb_cache: tuple[np.ndarray, np.ndarray] | None = None  # (ids, matrix)
        self._emb_cache_data_version: int | None = None

    @contextmanager
    def write_transaction(self):
        """Serialize a short write group locally and against other processes."""
        with self.lock:
            if not self.db.in_transaction:
                self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                self._emb_cache = None
                raise

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(sessions)")}
        if "parent_session" not in cols:
            self.db.execute("ALTER TABLE sessions ADD COLUMN parent_session TEXT")
        mcols = {r["name"] for r in self.db.execute("PRAGMA table_info(memories)")}
        if "plan_id" not in mcols:
            self.db.execute("ALTER TABLE memories ADD COLUMN plan_id INTEGER")
        # Created here, not in SCHEMA: on pre-plan databases the column only
        # exists after the ALTER above.
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_plan ON memories(plan_id)")
        pcols = {r["name"] for r in self.db.execute("PRAGMA table_info(plan_items)")}
        if "status_at" not in pcols:
            self.db.execute("ALTER TABLE plan_items ADD COLUMN status_at TEXT")
        self.db.execute(
            "UPDATE plan_items SET status_at=updated_at WHERE status_at IS NULL"
        )
        # Status vocabulary migration (the first plan release used open/done).
        self.db.execute("UPDATE plan_items SET status='planned' WHERE status='open'")
        self.db.execute("UPDATE plan_items SET status='complete' WHERE status='done'")
        self.db.commit()

    # ---------------- file tracking ----------------

    def file_state(self, path: str) -> tuple[float, int] | None:
        row = self.db.execute(
            "SELECT mtime, size FROM files WHERE path=?", (path,)
        ).fetchone()
        return (row["mtime"], row["size"]) if row else None

    def file_status(self, path: str) -> str | None:
        row = self.db.execute(
            "SELECT status FROM files WHERE path=?", (path,)
        ).fetchone()
        return row["status"] if row else None

    def mark_file(self, path: str, mtime: float, size: int, session_id: str, status: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO files(path, mtime, size, session_id, status) "
            "VALUES(?,?,?,?,?)",
            (path, mtime, size, session_id, status),
        )

    # ---------------- sessions ----------------

    def upsert_session(self, **kw) -> None:
        cols = ["id", "project", "cwd", "git_branch", "title", "summary", "outcome",
                "started_at", "ended_at", "n_user", "n_assistant", "source_path",
                "parent_session"]
        vals = [kw.get(c) for c in cols]
        placeholders = ",".join("?" for _ in cols)
        self.db.execute(
            f"INSERT OR REPLACE INTO sessions({','.join(cols)}) VALUES({placeholders})",
            vals,
        )

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()

    def update_session_source(self, session_id: str, source_path: str) -> None:
        """Relocate source evidence without replacing distilled session state."""
        self.db.execute("UPDATE sessions SET source_path=? WHERE id=?", (source_path, session_id))

    def session_for_file(self, path: str) -> sqlite3.Row | None:
        """Return the stored session tracked by one ingestion source file."""
        return self.db.execute(
            "SELECT s.* FROM files f JOIN sessions s ON s.id=f.session_id "
            "WHERE f.path=?",
            (path,),
        ).fetchone()

    def reattribute_session_project(
        self, session_id: str, project: str, *, commit: bool = True
    ) -> dict:
        """Move one session and its indexed artifacts to a logical project.

        This is deliberately metadata-only: memory ids, content, tags,
        timestamps, importance, and embedding rows/blobs are untouched. If the
        old project has no sessions left, its rolling profile is retired rather
        than left behind as a phantom project. Both projects and the session's
        end week are queued for later summary refresh in the same transaction.
        commit=False leaves transaction ownership to the batch caller.
        """
        project = (project or "").strip()
        if not project:
            raise ValueError("project is required")
        with self.write_transaction() if commit else self.lock:
            session = self.get_session(session_id)
            if session is None:
                raise ValueError(f"session '{session_id}' does not exist")
            old_project = session["project"]
            if old_project == project:
                return {
                    "session_id": session_id,
                    "old_project": old_project,
                    "new_project": project,
                    "memories": 0,
                    "retired_profiles": 0,
                }
            memory_count = self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            self.db.execute(
                "UPDATE sessions SET project=? WHERE id=?", (project, session_id)
            )
            self.db.execute(
                "UPDATE memories SET project=? WHERE session_id=?",
                (project, session_id),
            )
            retired_profiles = 0
            if old_project and not self.db.execute(
                "SELECT 1 FROM sessions WHERE project=? LIMIT 1", (old_project,)
            ).fetchone():
                ids = [r["id"] for r in self.db.execute(
                    "SELECT id FROM memories WHERE level='project' AND project=?",
                    (old_project,),
                )]
                self._delete_memory_ids(ids)
                retired_profiles = len(ids)
            for scope in (old_project, project):
                if scope:
                    self.mark_aggregate_dirty("project", scope)
            week = week_start(session["ended_at"])
            if week:
                self.mark_aggregate_dirty("digest", week)
            return {
                "session_id": session_id,
                "old_project": old_project,
                "new_project": project,
                "memories": memory_count,
                "retired_profiles": retired_profiles,
            }

    def reattribute_session_projects(
        self, assignments: list[tuple[str, str]]
    ) -> list[dict]:
        """Atomically apply a batch of metadata-only project assignments."""
        with self.write_transaction():
            return [
                self.reattribute_session_project(session_id, project, commit=False)
                for session_id, project in assignments
            ]

    def sessions_for_project(self, project: str, limit: int = 50) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM sessions WHERE project=? ORDER BY ended_at DESC LIMIT ?",
            (project, limit),
        ).fetchall()

    def sessions_since(self, iso_ts: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM sessions WHERE ended_at >= ? ORDER BY ended_at DESC",
            (iso_ts,),
        ).fetchall()

    def list_projects(self) -> list[dict]:
        """All known projects: with ingested sessions, plan boards, or both."""
        sess = {r["project"]: r for r in self.db.execute(
            "SELECT project, COUNT(*) AS n_sessions, MAX(ended_at) AS last_active "
            "FROM sessions GROUP BY project"
        )}
        plans = {r["project"]: r for r in self.db.execute(
            "SELECT project, COUNT(*) AS n_plan, MAX(updated_at) AS plan_updated "
            "FROM plan_items GROUP BY project"
        )}
        out = []
        for project in set(sess) | set(plans):
            s, p = sess.get(project), plans.get(project)
            out.append({
                "project": project,
                "n_sessions": s["n_sessions"] if s else 0,
                "last_active": s["last_active"] if s else None,
                "n_plan": p["n_plan"] if p else 0,
                "plan_updated": p["plan_updated"] if p else None,
            })
        out.sort(key=lambda r: max(r["last_active"] or "", r["plan_updated"] or ""),
                 reverse=True)
        return out

    def known_projects(self) -> set[str]:
        rows = self.db.execute(
            "SELECT DISTINCT project FROM sessions WHERE project IS NOT NULL "
            "UNION SELECT DISTINCT project FROM plan_items"
        )
        return {r["project"] for r in rows}

    # ---------------- memories ----------------

    def delete_session_memories(self, session_id: str) -> None:
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM memories WHERE session_id=?", (session_id,)
        )]
        self._delete_memory_ids(ids)

    def delete_scoped(self, level: str, project: str | None = None, ts: str | None = None) -> None:
        """Delete e.g. an old project profile or a week digest before regeneration."""
        q = "SELECT id FROM memories WHERE level=?"
        args: list = [level]
        if project is not None:
            q += " AND project=?"
            args.append(project)
        if ts is not None:
            q += " AND ts=?"
            args.append(ts)
        ids = [r["id"] for r in self.db.execute(q, args)]
        self._delete_memory_ids(ids)

    def _delete_memory_ids(self, ids: list[int]) -> None:
        for mid in ids:
            self.db.execute("DELETE FROM memories WHERE id=?", (mid,))
            self.db.execute("DELETE FROM embeddings WHERE memory_id=?", (mid,))
        if ids:
            self._emb_cache = None

    def add_memories(self, rows: list[MemoryRow]) -> list[int]:
        ids = []
        for r in rows:
            cur = self.db.execute(
                "INSERT INTO memories(session_id, project, level, kind, content, "
                "importance, ts, tags, plan_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (r.session_id, r.project, r.level, r.kind, r.content,
                 r.importance, r.ts, json.dumps(r.tags), r.plan_id),
            )
            mid = cur.lastrowid
            ids.append(mid)
            if r.vector is not None:
                self.db.execute(
                    "INSERT OR REPLACE INTO embeddings(memory_id, vector) VALUES(?,?)",
                    (mid, r.vector.astype(np.float32).tobytes()),
                )
        if rows:
            self._emb_cache = None
        return ids

    def get_memories(self, ids: list[int]) -> list[sqlite3.Row]:
        if not ids:
            return []
        qmarks = ",".join("?" for _ in ids)
        rows = self.db.execute(
            f"SELECT m.*, s.title AS session_title FROM memories m "
            f"LEFT JOIN sessions s ON s.id = m.session_id WHERE m.id IN ({qmarks})",
            ids,
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def memories_for_session(self, session_id: str, levels: tuple[str, ...] = ("memory",)) -> list[sqlite3.Row]:
        qmarks = ",".join("?" for _ in levels)
        return self.db.execute(
            f"SELECT * FROM memories WHERE session_id=? AND level IN ({qmarks}) "
            "ORDER BY importance DESC",
            (session_id, *levels),
        ).fetchall()

    def expand_chunk(
        self, memory_id: int, session_id: str | None, neighbor_chars: int = 1_000
    ) -> str | None:
        """Return a chunk's content with bounded slices of its ±1 neighbors.

        Chunks for a session are inserted consecutively, so id-adjacency within
        the same session and level is conversation-adjacency. The hit chunk is
        returned in full; the previous neighbor contributes its tail and the
        next neighbor its head, each capped at neighbor_chars — so an expanded
        hit stays bounded (~chunk + 2*neighbor_chars) no matter the chunk size.
        """
        if session_id is None:
            return None
        rows = self.db.execute(
            "SELECT id, content FROM memories WHERE session_id=? AND level='chunk' "
            "AND id BETWEEN ? AND ? ORDER BY id",
            (session_id, memory_id - 1, memory_id + 1),
        ).fetchall()
        if not rows:
            return None
        parts = []
        for r in rows:
            if r["id"] < memory_id:
                parts.append("[…earlier] " + r["content"][-neighbor_chars:])
            elif r["id"] > memory_id:
                parts.append("[…later] " + r["content"][:neighbor_chars])
            else:
                parts.append(r["content"])
        return "\n".join(parts)

    def get_profile(self, project: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM memories WHERE level='project' AND project=?", (project,)
        ).fetchone()

    def get_digests(self, limit: int = 8) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM memories WHERE level='digest' ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()

    # ---------------- plan items (authored planning plane) ----------------

    def plan_insert(
        self, *, project: str, kind: str, title: str, body: str,
        parent_id: int | None, priority: int, tags: list[str], now: str,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO plan_items(project, kind, parent_id, title, body, "
            "priority, tags, status, status_at, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,'planned',?,?,?)",
            (project, kind, parent_id, title, body, priority,
             json.dumps(tags), now, now, now),
        )
        return cur.lastrowid

    def plan_get(self, item_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM plan_items WHERE id=?", (item_id,)
        ).fetchone()

    def plan_children(self, item_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM plan_items WHERE parent_id=? ORDER BY priority, id",
            (item_id,),
        ).fetchall()

    def plan_items_for(self, project: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM plan_items WHERE project=? ORDER BY id", (project,)
        ).fetchall()

    def plan_set(self, item_id: int, fields: dict, now: str) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)  # keys are internal, never user input
        self.db.execute(
            f"UPDATE plan_items SET {sets}, updated_at=? WHERE id=?",
            (*fields.values(), now, item_id),
        )

    def plan_log(self, item_id: int, event: str, detail: str, ts: str) -> None:
        self.db.execute(
            "INSERT INTO plan_events(item_id, ts, event, detail) VALUES(?,?,?,?)",
            (item_id, ts, event, detail),
        )

    def plan_history(self, item_id: int) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM plan_events WHERE item_id=? ORDER BY id", (item_id,)
        ).fetchall()

    def replace_plan_mirror(self, item_id: int, row: MemoryRow) -> None:
        """Keep exactly one retrieval-index row (level='plan') per plan item."""
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM memories WHERE plan_id=?", (item_id,)
        )]
        self._delete_memory_ids(ids)
        row.plan_id = item_id
        self.add_memories([row])

    def attach_plan_embedding(self, item_id: int, expected: MemoryRow,
                              vector: np.ndarray, descriptor: str) -> bool:
        """Attach an embedding only if the indexed plan text is still current.

        The API call happens before this short transaction. A concurrent edit
        can replace the mirror while it runs; an old response must never label
        the new text with the old vector, even if SQLite reuses the row id.
        """
        with self.write_transaction():
            row = self.db.execute(
                "SELECT id, content, importance, ts, tags FROM memories WHERE plan_id=? AND level='plan'",
                (item_id,),
            ).fetchone()
            if row is None or (row["content"], row["importance"], row["ts"], row["tags"]) != (
                expected.content, expected.importance, expected.ts, json.dumps(expected.tags),
            ):
                return False
            current = self.get_meta("embedder")
            if current is not None and current != descriptor:
                raise ValueError("plan embedding model differs from the memory store")
            if current is None:
                self.set_meta("embedder", descriptor)
            self.db.execute("INSERT OR REPLACE INTO embeddings(memory_id, vector) VALUES(?,?)",
                            (row["id"], vector.astype(np.float32).tobytes()))
            self._emb_cache = None
            return True

    # ---------------- search primitives ----------------

    @staticmethod
    def _memory_filters(project=None, levels=None, kinds=None, after=None, before=None):
        clauses, args = [], []
        if project:
            clauses.append("m.project=?")
            args.append(project)
        for column, values in (("level", levels), ("kind", kinds)):
            if values:
                clauses.append(f"m.{column} IN ({','.join('?' for _ in values)})")
                args.extend(values)
        for op, value in ((">=", after), ("<=", before)):
            if value:
                clauses.append(f"m.ts {op} ?")
                args.append(value)
        return (" AND " + " AND ".join(clauses) if clauses else ""), args

    def has_memories(self, project=None, levels=None, kinds=None, after=None, before=None) -> bool:
        """Structural scope check; no embeddings, relevance inference, or model calls."""
        where, args = self._memory_filters(project, levels, kinds, after, before)
        return self.db.execute(
            "SELECT 1 FROM memories m WHERE 1=1" + where + " LIMIT 1", args
        ).fetchone() is not None

    def fts_search(self, query: str, limit: int = 100, *, project=None,
                   levels=None, kinds=None, after=None, before=None) -> dict[int, float]:
        """BM25 matches with constraints applied BEFORE top-N. Higher is better."""
        tokens = [t.replace('"', "") for t in query.split() if t.strip('"')]
        if not tokens:
            return {}
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        where, args = self._memory_filters(project, levels, kinds, after, before)
        try:
            rows = self.db.execute(
                "SELECT m.id AS rowid, memories_fts.rank AS rank FROM memories_fts "
                "JOIN memories m ON m.id=memories_fts.rowid WHERE memories_fts MATCH ?"
                + where + " ORDER BY rank, m.id LIMIT ?",
                (fts_query, *args, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        # fts5 rank: smaller (more negative) = better match
        return {r["rowid"]: -float(r["rank"]) for r in rows}

    def embedding_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Cache embeddings until either this connection or another writes."""
        with self.lock:
            version = self.db.execute("PRAGMA data_version").fetchone()[0]
            if self._emb_cache is not None and self._emb_cache_data_version == version:
                return self._emb_cache
            rows = self.db.execute("SELECT memory_id, vector FROM embeddings").fetchall()
            if not rows:
                self._emb_cache = (np.empty(0, dtype=np.int64), np.empty((0, 1), dtype=np.float32))
            else:
                ids = np.array([r["memory_id"] for r in rows], dtype=np.int64)
                mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
                self._emb_cache = (ids, mat)
            # Stamp the version from BEFORE the read. A concurrent commit
            # during materialization will invalidate this cache next time.
            self._emb_cache_data_version = version
            return self._emb_cache

    def filter_memory_ids(
        self,
        project: str | None = None,
        levels: tuple[str, ...] | None = None,
        kinds: tuple[str, ...] | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> set[int] | None:
        """Return the id-set matching the filters, or None if no filters given."""
        if not any([project, levels, kinds, after, before]):
            return None
        where, args = self._memory_filters(project, levels, kinds, after, before)
        return {r["id"] for r in self.db.execute("SELECT m.id FROM memories m WHERE 1=1" + where, args)}

    # ---------------- durable aggregate refresh work ----------------

    def mark_aggregate_dirty(self, level: str, scope: str) -> None:
        """Call in the transaction that changed the aggregate's source sessions."""
        self.db.execute(
            "INSERT INTO aggregate_refresh(level, scope) VALUES(?,?) "
            "ON CONFLICT(level, scope) DO UPDATE SET revision=revision+1",
            (level, scope),
        )

    def pending_aggregates(self, now: float, interval: float = 0) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM aggregate_refresh WHERE revision>applied_revision "
            "AND (refreshed_at IS NULL OR refreshed_at<=?) ORDER BY level, scope",
            (now - interval,),
        ).fetchall()

    def acknowledge_aggregate(self, level: str, scope: str, revision: int, now: float) -> None:
        """Acknowledge only the version read; concurrent newer work remains dirty."""
        self.db.execute(
            "UPDATE aggregate_refresh SET applied_revision=?, refreshed_at=? "
            "WHERE level=? AND scope=? AND applied_revision<?",
            (revision, now, level, scope, revision),
        )

    def aggregate_is_current(self, level: str, scope: str, revision: int) -> bool:
        """Check inside the replacement transaction after slow provider work."""
        return self.db.execute(
            "SELECT 1 FROM aggregate_refresh WHERE level=? AND scope=? "
            "AND revision=? AND applied_revision<revision", (level, scope, revision)
        ).fetchone() is not None

    # ---------------- meta / vector-space guard ----------------

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, value)
        )

    def assert_embedder(self, descriptor: str) -> None:
        """Vectors from different models/dims are not comparable — refuse to mix."""
        current = self.get_meta("embedder")
        if current is None:
            self.set_meta("embedder", descriptor)
            self.db.commit()
        elif current != descriptor:
            raise RuntimeError(
                f"store was embedded with '{current}' but the active embedder is "
                f"'{descriptor}'. Set AGENTIC_MEMORY_EMBED_PROVIDER to match, or "
                f"delete the db and re-ingest to switch vector spaces."
            )

    # ---------------- misc ----------------

    def counts(self) -> dict[str, int]:
        out = {}
        for level, n in self.db.execute(
            "SELECT level, COUNT(*) FROM memories GROUP BY level"
        ).fetchall():
            out[level] = n
        out["sessions"] = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        out["files"] = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        for status, n in self.db.execute(
            "SELECT status, COUNT(*) FROM files GROUP BY status"
        ).fetchall():
            out[f"files_{status}"] = n
        out["embeddings"] = self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        out["plan_items"] = self.db.execute("SELECT COUNT(*) FROM plan_items").fetchone()[0]
        return out

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
