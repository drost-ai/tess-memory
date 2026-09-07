"""Persistent normalized reader state with verified source-prefix revisions."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Callable
import zlib


class SourceChanged(ValueError):
    pass


class SourceReadError(ValueError):
    pass


def prefix_digest(path: Path, length: int):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = length
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise SourceChanged("source was truncated")
            digest.update(block)
            remaining -= len(block)
    return digest


@dataclass
class SourceRevision:
    path: str
    device: int
    inode: int
    offset: int
    digest: str
    observed_size: int
    mtime_ns: int
    ctime_ns: int
    partial_record: bool = False
    records: int = 0
    parsed_records: int = 0

    def unchanged(self) -> bool:
        try:
            s = Path(self.path).stat()
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) == (
                self.device, self.inode, self.observed_size, self.mtime_ns, self.ctime_ns)
        except OSError:
            return False

    def verify(self) -> bool:
        try:
            path = Path(self.path)
            before = path.stat()
            if (before.st_dev, before.st_ino) != (self.device, self.inode) or before.st_size < self.offset:
                return False
            if self.unchanged():
                return True
            actual = prefix_digest(path, self.offset).hexdigest()
            after = path.stat()
            return (actual == self.digest and (after.st_dev, after.st_ino) == (self.device, self.inode)
                    and after.st_size >= self.offset and before.st_mtime_ns == after.st_mtime_ns
                    and before.st_ctime_ns == after.st_ctime_ns)
        except (OSError, SourceChanged):
            return False


class SourceJournal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        db = self.connect()
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS readers (key TEXT PRIMARY KEY, revision TEXT NOT NULL, state BLOB NOT NULL, touched REAL NOT NULL)")
            db.commit()
        finally:
            db.close()

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def _load(self, key):
        db = self.connect()
        try:
            row = db.execute("SELECT revision,state FROM readers WHERE key=?", (key,)).fetchone()
        finally:
            db.close()
        if row is None:
            return None
        try:
            state = json.loads(zlib.decompress(row[1]))
            return (SourceRevision(**json.loads(row[0])), state) if isinstance(state, dict) else None
        except (ValueError, TypeError, zlib.error):
            return None  # derived cache can always be rebuilt from source

    def _save(self, key, revision, state):
        packed = zlib.compress(json.dumps(state, ensure_ascii=False).encode(), level=1)
        db = self.connect()
        try:
            with db:
                db.execute("INSERT OR REPLACE INTO readers VALUES(?,?,?,?)",
                           (key, json.dumps(asdict(revision)), packed, time.time()))
        finally:
            db.close()

    def prune(self, older_than: float) -> int:
        """Call under the store ingestion lock; removes derived reader state only."""
        db = self.connect()
        try:
            with db:
                return db.execute("DELETE FROM readers WHERE touched<?", (older_than,)).rowcount
        finally:
            db.close()

    def read_jsonl(self, path: Path, policy: str, *, initial: Callable[[], dict],
                   consume: Callable[[dict, dict, int, int, int], None],
                   validate: Callable[[dict], None] | None = None,
                   after_record: Callable[[dict, int, int, str], None] | None = None,
                   max_record_bytes: int = 64 * 1024 * 1024,
                   end_byte: int | None = None, end_ordinal: int | None = None) -> tuple[dict, SourceRevision]:
        path = path.resolve()
        key = hashlib.sha256(json.dumps([str(path), policy, end_byte, end_ordinal]).encode()).hexdigest()
        info = path.stat()
        limit = min(info.st_size, end_byte) if end_byte is not None else info.st_size
        old = self._load(key)
        state = initial()
        offset = position = 0
        digest = hashlib.sha256()
        if old:
            revision, previous = old
            if revision.unchanged() and revision.offset <= limit:
                revision.parsed_records = 0
                return previous, revision
            if ((info.st_dev, info.st_ino) == (revision.device, revision.inode)
                    and revision.offset <= limit):
                digest = prefix_digest(path, revision.offset)
                if digest.hexdigest() == revision.digest:
                    state, offset, position = previous, revision.offset, revision.records
                else:
                    digest = hashlib.sha256()
        partial = False
        parsed = 0
        with path.open("rb") as handle:
            handle.seek(offset)
            while offset < limit:
                raw = handle.readline(min(max_record_bytes + 1, limit - offset))
                if len(raw) > max_record_bytes:
                    raise SourceReadError("record exceeds configured maximum size")
                if not raw.endswith(b"\n"):
                    partial = True
                    break
                if raw.strip():
                    try:
                        record = json.loads(raw)
                    except (ValueError, UnicodeError) as exc:
                        raise SourceReadError(f"invalid JSON record at byte {offset}") from exc
                    if not isinstance(record, dict):
                        raise SourceReadError("source record must be an object")
                    if end_ordinal is not None:
                        ordinal = record.get("ordinal")
                        if isinstance(ordinal, int) and not isinstance(ordinal, bool):
                            if ordinal >= end_ordinal:
                                break
                        elif end_byte is None:
                            raise SourceReadError("ordinal history requires ordinals or a byte boundary")
                    consume(state, record, position + 1, offset, offset + len(raw))
                    parsed += 1
                position += 1
                digest.update(raw)
                offset += len(raw)
                if after_record:
                    after_record(state, position, offset, digest.hexdigest())
        revision = SourceRevision(str(path), info.st_dev, info.st_ino, offset, digest.hexdigest(),
                                  info.st_size, info.st_mtime_ns, info.st_ctime_ns, partial, position, parsed)
        if validate:
            validate(state)
        if not revision.verify():
            raise SourceChanged("source prefix changed while parsing")
        self._save(key, revision, state)
        return state, revision

    def read_snapshot(self, path: Path, policy: str, decode: Callable[[dict], dict],
                      max_bytes: int = 64 * 1024 * 1024) -> tuple[dict, SourceRevision]:
        path = path.resolve()
        key = hashlib.sha256(json.dumps([str(path), policy, "snapshot"]).encode()).hexdigest()
        old = self._load(key)
        if old and old[0].unchanged():
            old[0].parsed_records = 0
            return old[1], old[0]
        info = path.stat()
        if info.st_size > max_bytes:
            raise SourceReadError("snapshot exceeds configured maximum size")
        with path.open("rb") as handle:
            raw = handle.read(info.st_size)
        try:
            record = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise SourceReadError("incomplete or invalid JSON snapshot") from exc
        if not isinstance(record, dict):
            raise SourceReadError("snapshot must be an object")
        state = decode(record)
        revision = SourceRevision(str(path), info.st_dev, info.st_ino, len(raw), hashlib.sha256(raw).hexdigest(),
                                  info.st_size, info.st_mtime_ns, info.st_ctime_ns, False, 1, 1)
        if not revision.verify():
            raise SourceChanged("snapshot changed while parsing")
        self._save(key, revision, state)
        return state, revision
