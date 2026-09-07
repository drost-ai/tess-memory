"""Private, per-store checkpoints for successfully validated model requests.

This is exact-request reuse, not an append-aware transcript/event cache. The
namespace, provider identity, system prompt and full prompt all enter the key.
Connections are short-lived so worker threads never share the memory database
connection or keep a write transaction open during provider work.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class DistillationCheckpoints:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create with private permissions before SQLite opens the file.
        path.touch(mode=0o600, exist_ok=True)
        db = self._connect()
        try:
            with db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("""CREATE TABLE IF NOT EXISTS checkpoints (
                    key TEXT PRIMARY KEY,
                    result TEXT NOT NULL
                )""")
                # Keep the original two-column table compatible with running
                # pre-upgrade backfills that use positional INSERT statements.
                db.execute("CREATE TABLE IF NOT EXISTS checkpoint_access (key TEXT PRIMARY KEY, touched REAL NOT NULL)")
                db.execute("INSERT OR IGNORE INTO checkpoint_access SELECT key,? FROM checkpoints", (time.time(),))
                db.execute("CREATE TABLE IF NOT EXISTS pins (namespace TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(namespace,key))")
                db.execute("CREATE TABLE IF NOT EXISTS passages (key TEXT PRIMARY KEY, vector BLOB NOT NULL, touched REAL NOT NULL)")
        finally:
            db.close()

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def key(namespace: str, provider: str, system: str, prompt: str) -> str:
        request = json.dumps([1, namespace, provider, system, prompt], ensure_ascii=False)
        return hashlib.sha256(request.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        db = self._connect()
        try:
            row = db.execute("SELECT result FROM checkpoints WHERE key=?", (key,)).fetchone()
        finally:
            db.close()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, result: dict) -> None:
        db = self._connect()
        try:
            with db:
                db.execute("INSERT OR REPLACE INTO checkpoints(key,result) VALUES (?, ?)",
                           (key, json.dumps(result, ensure_ascii=False)))
                db.execute("INSERT OR REPLACE INTO checkpoint_access VALUES (?,?)", (key, time.time()))
        finally:
            db.close()

    def pin(self, namespace: str, keys: set[str]) -> None:
        db = self._connect()
        try:
            with db:
                db.execute("DELETE FROM pins WHERE namespace=?", (namespace,))
                db.executemany("INSERT INTO pins VALUES (?,?)", [(namespace, k) for k in keys])
        finally:
            db.close()

    def prune(self, before: float) -> dict:
        db = self._connect()
        try:
            with db:
                requests = db.execute("DELETE FROM checkpoints WHERE key IN (SELECT key FROM checkpoint_access WHERE touched<?) AND key NOT IN (SELECT key FROM pins)", (before,)).rowcount
                db.execute("DELETE FROM checkpoint_access WHERE key NOT IN (SELECT key FROM checkpoints)")
                passages = db.execute("DELETE FROM passages WHERE touched<?", (before,)).rowcount
            return {"requests": requests, "passages": passages}
        finally:
            db.close()

    def passage(self, key: str) -> bytes | None:
        db = self._connect()
        try:
            row = db.execute("SELECT vector FROM passages WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            db.close()

    def save_passages(self, values: list[tuple[str, bytes]]) -> None:
        db = self._connect()
        try:
            with db:
                db.executemany("INSERT OR REPLACE INTO passages VALUES (?,?,?)",
                               [(key, vector, time.time()) for key, vector in values])
        finally:
            db.close()
