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
                db.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?)",
                           (key, json.dumps(result, ensure_ascii=False)))
        finally:
            db.close()
