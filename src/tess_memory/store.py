"""SQLite-backed memory store: sessions, multi-timescale memories, FTS5, vectors.

Memory levels (timescales, finest → coarsest):
    chunk    raw conversation windows            (exact recall of code/errors)
    memory   LLM-extracted atomic facts          (decisions, solutions, prefs)
    session  one summary per session
    digest   one summary per ISO week
    project  one rolling profile per project

Vectors are stored as float32 blobs. Retrieval uses exact scoped cosine search
with bounded caching/batches; the whole-matrix reader remains for compatibility.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .principals import (
    DEFAULT_OWNER,
    Principal,
    VisibilityPolicy,
    memory_access_sql,
    principal_id,
    project_access_sql,
    session_access_sql,
)
from .time_utils import week_start

USERS_SCHEMA_VERSION = 1
USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS principals(
    id TEXT PRIMARY KEY, display_name TEXT, kind TEXT NOT NULL DEFAULT 'user',
    admin INTEGER NOT NULL DEFAULT 0, disabled INTEGER NOT NULL DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS teams(id TEXT PRIMARY KEY, name TEXT, org TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS team_members(
    team_id TEXT NOT NULL REFERENCES teams(id), user_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member', PRIMARY KEY(team_id, user_id));
CREATE TABLE IF NOT EXISTS projects(key TEXT PRIMARY KEY, owner_user TEXT, team_id TEXT REFERENCES teams(id), created_at TEXT);
CREATE TABLE IF NOT EXISTS visibility_events(
    id INTEGER PRIMARY KEY, ts TEXT, actor TEXT, target TEXT NOT NULL, visibility TEXT, detail TEXT);
"""
# Additive and nullable: NULL is the instance default owner / the policy default.
USERS_COLUMNS = (
    ("sessions", "owner_user", "TEXT"),
    ("memories", "owner_user", "TEXT"),
    ("memories", "visibility", "TEXT"),
    ("plan_items", "created_by", "TEXT"),
    ("plan_items", "updated_by", "TEXT"),
    ("plan_events", "actor", "TEXT"),
    ("ingest_sources", "owner_user", "TEXT"),
)

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
CREATE TABLE IF NOT EXISTS ingest_sources(
    path TEXT PRIMARY KEY,
    session_id TEXT,
    project TEXT,
    source_type TEXT NOT NULL,
    observed_signature TEXT,
    status TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT '{}',
    observed_at REAL,
    indexed_at REAL,
    last_error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS vector_scope_versions(scope TEXT PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TRIGGER IF NOT EXISTS vector_scope_ai AFTER INSERT ON embeddings BEGIN
    INSERT INTO vector_scope_versions VALUES('G',1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
    INSERT INTO vector_scope_versions SELECT 'P:'||COALESCE(project,''),1 FROM memories WHERE id=new.memory_id
        ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS vector_scope_au AFTER UPDATE ON embeddings BEGIN
    INSERT INTO vector_scope_versions VALUES('G',1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
    INSERT INTO vector_scope_versions SELECT 'P:'||COALESCE(project,''),1 FROM memories WHERE id=old.memory_id OR id=new.memory_id
        ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS vector_scope_ad AFTER DELETE ON embeddings BEGIN
    INSERT INTO vector_scope_versions VALUES('G',1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
    INSERT INTO vector_scope_versions SELECT 'P:'||COALESCE(project,''),1 FROM memories WHERE id=old.memory_id
        ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS vector_scope_md BEFORE DELETE ON memories WHEN EXISTS(SELECT 1 FROM embeddings WHERE memory_id=old.id) BEGIN
    INSERT INTO vector_scope_versions VALUES('G',1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
    INSERT INTO vector_scope_versions VALUES('P:'||COALESCE(old.project,''),1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS vector_scope_mu AFTER UPDATE OF project,level,kind,ts ON memories
WHEN EXISTS(SELECT 1 FROM embeddings WHERE memory_id=new.id)
 AND (old.project IS NOT new.project OR old.level IS NOT new.level OR old.kind IS NOT new.kind OR old.ts IS NOT new.ts) BEGIN
    INSERT INTO vector_scope_versions VALUES('G',1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
    INSERT INTO vector_scope_versions VALUES('P:'||COALESCE(old.project,''),1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
    INSERT INTO vector_scope_versions VALUES('P:'||COALESCE(new.project,''),1) ON CONFLICT(scope) DO UPDATE SET revision=revision+1;
END;
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
    owner_user: str | None = None   # None = the instance default owner
    visibility: str | None = None   # None = the policy default for level/kind


class MemoryStore:
    def __init__(self, db_path: Path, *, readonly: bool = False,
                 principal: Principal | None = None, policy: VisibilityPolicy | None = None):
        if not readonly:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: MCP tool dispatch and the search agent may
        # touch the store from worker threads; CPython's sqlite3 serializes
        # statement execution, and plan writes additionally hold self.lock.
        if readonly:
            uri = db_path.resolve().as_uri() + "?mode=ro"
            if not Path(str(db_path) + "-wal").exists():
                uri += "&immutable=1"
            self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            self.db.executescript(SCHEMA)
            self._migrate()
        self.lock = threading.RLock()  # serializes multi-statement write groups
        # Who reads and writes through this store; None = the default owner,
        # unrestricted (every instance created before users existed, and any without identity).
        self.principal = principal
        self.policy = policy or VisibilityPolicy()
        self._emb_cache: tuple[np.ndarray, np.ndarray] | None = None  # (ids, matrix)
        self._emb_cache_data_version: int | None = None
        self._scoped_vectors = None

    @contextmanager
    def read_snapshot(self):
        """Keep vector/filter/FTS reads consistent; never commit a caller's write."""
        with self.lock:
            owns = not self.db.in_transaction
            if owns:
                self.db.execute('BEGIN')
            try:
                yield owns
            finally:
                if owns:
                    self.db.rollback()  # read-only snapshot

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
        """Additive, idempotent, and safe when many processes open the store at
        once: every check is repeated inside one IMMEDIATE transaction."""
        self.db.commit()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._migrate_locked()
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def _migrate_locked(self) -> None:
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
        source_columns = {r['name'] for r in self.db.execute('PRAGMA table_info(ingest_sources)')}
        if 'retry_at' not in source_columns:
            self.db.execute('ALTER TABLE ingest_sources ADD COLUMN retry_at REAL NOT NULL DEFAULT 0')
        # Users, teams, projects and per-row owner/visibility.
        for statement in USERS_SCHEMA.split(";"):
            if statement.strip():
                self.db.execute(statement)
        for table, column, kind in USERS_COLUMNS:
            if column not in {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_user)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_user)")
        if self.db.execute("PRAGMA user_version").fetchone()[0] < USERS_SCHEMA_VERSION:
            self.db.execute(f"PRAGMA user_version={USERS_SCHEMA_VERSION}")
        # Engine sessions were recorded as 'agent-core' sources before the
        # engine was named tess-core. Only the label changes.
        if self.db.execute("SELECT 1 FROM ingest_sources WHERE source_type='agent-core' LIMIT 1").fetchone():
            self.db.execute("UPDATE ingest_sources SET source_type='tess-core' WHERE source_type='agent-core'")

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
            """INSERT INTO files(path,mtime,size,session_id,status) VALUES(?,?,?,?,?)
            ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime,size=excluded.size,
                session_id=excluded.session_id,status=excluded.status,ingested_at=datetime('now')
            WHERE NOT (files.mtime IS excluded.mtime AND files.size IS excluded.size
                AND files.session_id IS excluded.session_id AND files.status IS excluded.status)""",
            (path, mtime, size, session_id, status),
        )

    def source_progress(self, path: str):
        return self.db.execute("SELECT * FROM ingest_sources WHERE path=?", (path,)).fetchone()

    def supersede_source_paths(self, session_id: str, path: str) -> None:
        """Retain old progress as provenance after a verified relocation.

        Called in the publication transaction; no source, memory, embedding or
        checkpoint is deleted. The original path's indexes remain available.
        """
        plain = path[:-4] if path.endswith('.jsonl.zst') else path
        rows = self.db.execute("SELECT path,state FROM ingest_sources WHERE path!=? AND status!='held' AND (session_id=? OR path=?)",
                               (path, session_id, plain)).fetchall()
        for row in rows:
            state = json.loads(row['state'])
            state['superseded_by'] = path
            self.db.execute("UPDATE ingest_sources SET status='superseded',state=?,last_error=NULL,attempts=0,retry_at=0 WHERE path=?",
                            (json.dumps(state), row['path']))

    def record_source(self, *, path: str, session_id: str | None, project: str, source_type: str,
                      signature: str | None, status: str, state: dict, now: float,
                      indexed: bool = False, error: str | None = None) -> None:
        old = self.source_progress(path)
        attempts = (old['attempts'] + 1 if old and old['status'] == 'error' else 1) if status == 'error' else 0
        retry_at = now + min(1800, 30 * 2 ** min(attempts - 1, 6)) if attempts else 0
        self.db.execute("""INSERT INTO ingest_sources
            (path,session_id,project,source_type,observed_signature,status,state,observed_at,indexed_at,last_error,attempts,retry_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
            session_id=COALESCE(excluded.session_id,ingest_sources.session_id), project=excluded.project,
            source_type=excluded.source_type,observed_signature=excluded.observed_signature,status=excluded.status,
            state=excluded.state,observed_at=excluded.observed_at,
            indexed_at=COALESCE(excluded.indexed_at,ingest_sources.indexed_at),last_error=excluded.last_error,
            attempts=excluded.attempts,retry_at=excluded.retry_at""",
            (path, session_id, project, source_type, signature, status, json.dumps(state), now,
             now if indexed else None, error, attempts, retry_at))

    def session_rows_with_vectors(self, session_id: str):
        return self.db.execute("SELECT m.*,e.vector FROM memories m LEFT JOIN embeddings e ON e.memory_id=m.id WHERE m.session_id=? ORDER BY m.id",
                               (session_id,)).fetchall()

    def replace_session_rows(self, session_id: str, rows: list[MemoryRow]) -> None:
        """Preserve matching IDs/vectors; caller owns the publication transaction."""
        from collections import defaultdict, deque
        def identity(row):
            # Session/atomic timestamps may advance with the session; raw
            # chunks retain their source-event timestamps and occurrences.
            return (row['level'], row['kind'], row['content'], row['tags'],
                    row['ts'] if row['level'] == 'chunk' else None)
        available = defaultdict(deque)
        for old in self.session_rows_with_vectors(session_id):
            available[identity(old)].append(old)
        additions = []
        for row in rows:
            key = (row.level, row.kind, row.content, json.dumps(row.tags), row.ts if row.level == 'chunk' else None)
            if available[key]:
                old = available[key].popleft()
                self.db.execute("UPDATE memories SET project=?,importance=?,ts=?,owner_user=COALESCE(?,owner_user) WHERE id=?",
                                (row.project, row.importance, row.ts, row.owner_user, old['id']))
                if old['vector'] is None and row.vector is not None:
                    self.db.execute("INSERT INTO embeddings VALUES (?,?)", (old['id'], row.vector.astype(np.float32).tobytes()))
                    self._emb_cache = None
            else:
                additions.append(row)
        self._delete_memory_ids([row['id'] for group in available.values() for row in group])
        self.add_memories(additions)

    # ---------------- sessions ----------------

    def upsert_session(self, **kw) -> None:
        cols = ["id", "project", "cwd", "git_branch", "title", "summary", "outcome",
                "started_at", "ended_at", "n_user", "n_assistant", "source_path",
                "parent_session", "owner_user"]
        if "owner_user" not in kw:
            # INSERT OR REPLACE rewrites the row: keep the owner it had.
            old = self.db.execute("SELECT owner_user FROM sessions WHERE id=?", (kw.get("id"),)).fetchone()
            kw = {**kw, "owner_user": old["owner_user"] if old else None}
        vals = [kw.get(c) for c in cols]
        placeholders = ",".join("?" for _ in cols)
        self.db.execute(
            f"INSERT OR REPLACE INTO sessions({','.join(cols)}) VALUES({placeholders})",
            vals,
        )

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        access, args = self._session_where("s")
        return self.db.execute(
            f"SELECT s.* FROM sessions s WHERE s.id=?{access}", (session_id, *args)
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
        access, args = self._session_where("s")
        return self.db.execute(
            f"SELECT s.* FROM sessions s WHERE s.project=?{access} ORDER BY s.ended_at DESC LIMIT ?",
            (project, *args, limit),
        ).fetchall()

    def sessions_since(self, iso_ts: str) -> list[sqlite3.Row]:
        access, args = self._session_where("s")
        return self.db.execute(
            f"SELECT s.* FROM sessions s WHERE s.ended_at >= ?{access} ORDER BY s.ended_at DESC",
            (iso_ts, *args),
        ).fetchall()

    def list_projects(self) -> list[dict]:
        """All known projects: with ingested sessions, plan boards, or both."""
        session_access, session_args = self._session_where("s")
        plan_access, plan_args = self._plan_where("i")
        sess = {r["project"]: r for r in self.db.execute(
            "SELECT s.project, COUNT(*) AS n_sessions, MAX(s.ended_at) AS last_active "
            f"FROM sessions s WHERE 1=1{session_access} GROUP BY s.project", session_args
        )}
        plans = {r["project"]: r for r in self.db.execute(
            "SELECT i.project, COUNT(*) AS n_plan, MAX(i.updated_at) AS plan_updated "
            f"FROM plan_items i WHERE 1=1{plan_access} GROUP BY i.project", plan_args
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
        session_access, session_args = self._session_where("s")
        plan_access, plan_args = self._plan_where("i")
        rows = self.db.execute(
            f"SELECT DISTINCT s.project FROM sessions s WHERE s.project IS NOT NULL{session_access} "
            f"UNION SELECT DISTINCT i.project FROM plan_items i WHERE 1=1{plan_access}", (*session_args, *plan_args)
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
                "importance, ts, tags, plan_id, owner_user, visibility) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (r.session_id, r.project, r.level, r.kind, r.content,
                 r.importance, r.ts, json.dumps(r.tags), r.plan_id, r.owner_user, r.visibility),
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
        if self.restricted:
            # A session's title is its author's: shown only where the session is.
            access, args = self.access_sql("m")
            session, session_args = session_access_sql(self.principal, self.default_owner(), "s")
            rows = self.db.execute(
                f"SELECT m.*, CASE WHEN {session} THEN s.title END AS session_title FROM memories m "
                f"LEFT JOIN sessions s ON s.id = m.session_id WHERE m.id IN ({qmarks}) AND {access}",
                (*session_args, *ids, *args),
            ).fetchall()
        else:
            rows = self.db.execute(
                f"SELECT m.*, s.title AS session_title FROM memories m "
                f"LEFT JOIN sessions s ON s.id = m.session_id WHERE m.id IN ({qmarks})",
                ids,
            ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def memories_for_session(self, session_id: str, levels: tuple[str, ...] = ("memory",)) -> list[sqlite3.Row]:
        qmarks = ",".join("?" for _ in levels)
        access, args = self._access_where("m")
        return self.db.execute(
            f"SELECT m.* FROM memories m WHERE m.session_id=? AND m.level IN ({qmarks}){access} "
            "ORDER BY m.importance DESC",
            (session_id, *levels, *args),
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
        access, args = self._access_where("m")
        rows = self.db.execute(
            "SELECT m.id, m.content FROM memories m WHERE m.session_id=? AND m.level='chunk' "
            f"AND m.id BETWEEN ? AND ?{access} ORDER BY m.id",
            (session_id, memory_id - 1, memory_id + 1, *args),
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
        access, args = self._access_where("m")
        return self.db.execute(
            f"SELECT m.* FROM memories m WHERE m.level='project' AND m.project=?{access}", (project, *args)
        ).fetchone()

    def get_digests(self, limit: int = 8) -> list[sqlite3.Row]:
        access, args = self._access_where("m")
        return self.db.execute(
            f"SELECT m.* FROM memories m WHERE m.level='digest'{access} ORDER BY m.ts DESC LIMIT ?",
            (*args, limit),
        ).fetchall()

    # ---------------- plan items (authored planning plane) ----------------

    def plan_insert(
        self, *, project: str, kind: str, title: str, body: str,
        parent_id: int | None, priority: int, tags: list[str], now: str,
    ) -> int:
        writer = self.writer()
        cur = self.db.execute(
            "INSERT INTO plan_items(project, kind, parent_id, title, body, "
            "priority, tags, status, status_at, created_at, updated_at, created_by, updated_by) "
            "VALUES(?,?,?,?,?,?,?,'planned',?,?,?,?,?)",
            (project, kind, parent_id, title, body, priority,
             json.dumps(tags), now, now, now, writer, writer),
        )
        self.ensure_project(project, owner=writer)
        return cur.lastrowid

    def plan_get(self, item_id: int) -> sqlite3.Row | None:
        access, args = self._plan_where("i")
        return self.db.execute(
            f"SELECT i.* FROM plan_items i WHERE i.id=?{access}", (item_id, *args)
        ).fetchone()

    def plan_children(self, item_id: int) -> list[sqlite3.Row]:
        access, args = self._plan_where("i")
        return self.db.execute(
            f"SELECT i.* FROM plan_items i WHERE i.parent_id=?{access} ORDER BY i.priority, i.id",
            (item_id, *args),
        ).fetchall()

    def plan_items_for(self, project: str) -> list[sqlite3.Row]:
        access, args = self._plan_where("i")
        return self.db.execute(
            f"SELECT i.* FROM plan_items i WHERE i.project=?{access} ORDER BY i.id", (project, *args)
        ).fetchall()

    def plan_set(self, item_id: int, fields: dict, now: str) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)  # keys are internal, never user input
        self.db.execute(
            f"UPDATE plan_items SET {sets}, updated_at=?, updated_by=? WHERE id=?",
            (*fields.values(), now, self.writer(), item_id),
        )

    def plan_log(self, item_id: int, event: str, detail: str, ts: str) -> None:
        self.db.execute(
            "INSERT INTO plan_events(item_id, ts, event, detail, actor) VALUES(?,?,?,?,?)",
            (item_id, ts, event, detail, self.writer()),
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
        if row.owner_user is None:
            row.owner_user = self.writer()
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

    def _memory_filters(self, project=None, levels=None, kinds=None, after=None, before=None):
        clauses, args = [], []
        if self.restricted:
            access, access_args = self.access_sql("m")
            clauses.append(access)
            args.extend(access_args)
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

    def has_vectors(self, project=None, levels=None, kinds=None, after=None, before=None):
        where, args = self._memory_filters(project, levels, kinds, after, before)
        if not where:  # unrestricted with no filters: today's fast path
            return self.db.execute('SELECT 1 FROM embeddings LIMIT 1').fetchone() is not None
        return self.db.execute('SELECT 1 FROM memories m JOIN embeddings e ON e.memory_id=m.id WHERE 1=1' + where + ' LIMIT 1', args).fetchone() is not None

    def semantic_candidates(self, vector, **kwargs):
        from .vector_search import ScopedVectorSearch
        if self._scoped_vectors is None:
            self._scoped_vectors = ScopedVectorSearch()
        return self._scoped_vectors.search(self, vector, **kwargs)

    def filter_memory_ids(
        self,
        project: str | None = None,
        levels: tuple[str, ...] | None = None,
        kinds: tuple[str, ...] | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> set[int] | None:
        """Return the id-set matching the filters, or None if no filters given."""
        if not any([project, levels, kinds, after, before]) and not self.restricted:
            return None
        where, args = self._memory_filters(project, levels, kinds, after, before)
        return {r["id"] for r in self.db.execute("SELECT m.id FROM memories m WHERE 1=1" + where, args)}

    # ---------------- durable aggregate refresh work ----------------

    def bootstrap_aggregate_tracking(self, levels: tuple[str, ...]) -> int:
        """Schedule legacy scopes that predate the dirty queue, once per scope."""
        added = 0
        with self.write_transaction():
            scopes = set()
            if 'project' in levels:
                scopes.update(('project', r[0]) for r in self.db.execute('SELECT DISTINCT project FROM sessions WHERE project IS NOT NULL'))
            if 'digest' in levels:
                scopes.update(('digest', week) for r in self.db.execute('SELECT DISTINCT ended_at FROM sessions') if (week := week_start(r[0])))
            for level, scope in scopes:
                if self.db.execute('SELECT 1 FROM aggregate_refresh WHERE level=? AND scope=?', (level, scope)).fetchone() is None:
                    self.mark_aggregate_dirty(level, scope)
                    added += 1
        return added

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
                f"'{descriptor}'. Set TESS_MEMORY_EMBED_PROVIDER to match, or "
                f"delete the db and re-ingest to switch vector spaces."
            )

    # ---------------- principals ----------------

    def default_owner(self) -> str:
        """The user NULL owners stand for, and the instance's first admin."""
        if not self._has_users_schema():
            return DEFAULT_OWNER
        return self.get_meta("default_owner") or DEFAULT_OWNER

    def _has_users_schema(self) -> bool:
        if not hasattr(self, "_users_schema"):
            self._users_schema = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='principals'").fetchone() is not None
        return self._users_schema

    def resolve_principal(self, user: str | None) -> Principal:
        """The principal for a configured user id; none or the default owner
        is the unrestricted default owner."""
        owner = self.default_owner()
        if user is None or user == owner:
            return Principal(owner, admin=True)
        principal_id(user)
        if not self._has_users_schema():
            raise RuntimeError("this store predates users; open it once with a writable client to migrate it")
        row = self.db.execute("SELECT admin, disabled FROM principals WHERE id=?", (user,)).fetchone()
        if row is not None and row["disabled"]:
            raise PermissionError(f"user {user!r} is disabled")
        teams = frozenset(r[0] for r in self.db.execute("SELECT team_id FROM team_members WHERE user_id=?", (user,)))
        return Principal(user, teams=teams, admin=bool(row and row["admin"]))

    def owner_for(self, user: str | None) -> str | None:
        """How a user is stamped on a row: None for the default owner."""
        return None if not user or user == self.default_owner() else principal_id(user)

    def ensure_project(self, project: str | None, *, owner: str | None = None, team: str | None = None) -> None:
        """Record who a project belongs to (its first named writer) and its
        team. Nothing is recorded for the default owner with no team, so a
        single-user store gains no rows."""
        if not project or (owner is None and team is None):
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if team:
            principal_id(team, "team")
            self.db.execute("INSERT OR IGNORE INTO teams(id, name, created_at) VALUES(?,?,?)", (team, team, now))
        self.db.execute(
            "INSERT INTO projects(key, owner_user, team_id, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET team_id=COALESCE(projects.team_id, excluded.team_id)",
            (project, owner, team, now))

    def project_team(self, project: str | None) -> str | None:
        if not project or not self._has_users_schema():
            return None
        row = self.db.execute("SELECT team_id FROM projects WHERE key=?", (project,)).fetchone()
        return row["team_id"] if row else None

    def shared_memories(self, project: str, limit: int = 60) -> list[sqlite3.Row]:
        """A team project's distilled knowledge its team may read: what a
        team-visible profile is built from (never private material)."""
        visibility, args = self.policy.sql("m")
        return self.db.execute(
            f"SELECT m.* FROM memories m WHERE m.project=? AND m.level='memory' AND {visibility} IN ('team','org') "
            "ORDER BY m.importance DESC, m.ts DESC, m.id LIMIT ?", (project, *args, limit)).fetchall()

    @property
    def restricted(self) -> bool:
        return self.principal is not None and self.principal.restricted

    def access_sql(self, alias: str = "m") -> tuple[str, list]:
        """The predicate a memory row must pass for this store's principal."""
        return memory_access_sql(self.principal, self.policy, self.default_owner(), alias)

    def access_key(self) -> tuple:
        return self.principal.key() if self.restricted else ("*",)

    def _access_where(self, alias: str) -> tuple[str, list]:
        if not self.restricted:
            return "", []
        sql, args = self.access_sql(alias)
        return " AND " + sql, args

    def _session_where(self, alias: str) -> tuple[str, list]:
        if not self.restricted:
            return "", []
        sql, args = session_access_sql(self.principal, self.default_owner(), alias)
        return " AND " + sql, args

    def _plan_where(self, alias: str) -> tuple[str, list]:
        """Plan boards are shared per project: readable with project access."""
        if not self.restricted:
            return "", []
        sql, args = project_access_sql(self.principal, f"{alias}.project", self.default_owner())
        return " AND " + sql, args

    def can_write_project(self, project: str) -> bool:
        """A restricted writer may write a project it has access to, or a new
        one (which becomes theirs). An existing project nobody mapped is the
        default owner's."""
        if not self.restricted:
            return True
        sql, args = project_access_sql(self.principal, "?", self.default_owner())
        if self.db.execute(f"SELECT {sql}", (project, *args)).fetchone()[0]:
            return True
        exists = self.db.execute(
            "SELECT 1 FROM projects WHERE key=? UNION SELECT 1 FROM plan_items WHERE project=? "
            "UNION SELECT 1 FROM sessions WHERE project=? LIMIT 1", (project, project, project)).fetchone()
        return exists is None

    def _restricted_counts(self) -> dict[str, int]:
        access, args = self._access_where("m")
        out = {level: n for level, n in self.db.execute(
            f"SELECT m.level, COUNT(*) FROM memories m WHERE 1=1{access} GROUP BY m.level", args)}
        session_access, session_args = self._session_where("s")
        out["sessions"] = self.db.execute(f"SELECT COUNT(*) FROM sessions s WHERE 1=1{session_access}", session_args).fetchone()[0]
        plan_access, plan_args = self._plan_where("i")
        out["plan_items"] = self.db.execute(f"SELECT COUNT(*) FROM plan_items i WHERE 1=1{plan_access}", plan_args).fetchone()[0]
        return out

    def writer(self) -> str | None:
        """The user a write is stamped with; None (= default owner) when the
        writer is the default owner, so single-user stores stay unstamped."""
        if self.principal is None or self.principal.user == self.default_owner():
            return None
        return self.principal.user

    # ---------------- misc ----------------

    def counts(self) -> dict[str, int]:
        if self.restricted:
            return self._restricted_counts()
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
