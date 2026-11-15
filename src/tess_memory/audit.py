"""Read-only corpus inventory. No transcripts, credentials or model calls in reports."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import sqlite3
import time
from pathlib import Path

from .attribution import source_adapter
from .config import Config


@contextmanager
def readonly_database(path: Path):
    if not path.is_file():
        yield None
        return
    # immutable is safe only without a WAL; an active WAL must remain visible.
    uri = path.resolve().as_uri() + "?mode=ro"
    if not Path(str(path) + "-wal").exists():
        uri += "&immutable=1"
    db = sqlite3.connect(uri, uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def source_members(path: Path, source_type: str) -> list[Path]:
    """Include physical body files; tess-core metadata is not the whole source."""
    members = [path]
    if source_type == "tess-core":
        members += [p for p in (path.parent / path.stem / "full-transcript-with-tools.jsonl",
                               path.with_name(path.stem + ".messages.jsonl")) if p.is_file()]
    return members


def activity_state(path: Path, source_type: str) -> str:
    """Classify only explicit bounded-tail evidence; never infer completion from age."""
    members = source_members(path, source_type)
    body = members[1] if source_type == "tess-core" and len(members) > 1 else path
    with body.open("rb") as handle:
        size = body.stat().st_size
        start = max(0, size - 64 * 1024)
        handle.seek(start)
        if start:
            handle.readline()  # incomplete first tail record
        raw = handle.read(64 * 1024)
    state = "unknown"
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, UnicodeError):
            state = "unknown"
            continue
        if not isinstance(record, dict):
            continue
        payload = record.get("payload", record)
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if record.get("type") == "event_msg" and isinstance(kind, str):
            if kind == "task_complete":
                state = "complete"
            elif kind == "turn_aborted":
                state = "aborted"
            elif kind in {"task_started", "user_message"}:
                state = "active"
        else:
            message = payload.get("message", payload)
            if not isinstance(message, dict):
                continue
            role = message.get("role", record.get("type") if record.get("type") in {"user", "assistant"} else None)
            reason = message.get("stop_reason", message.get("stopReason"))
            if role == "user":
                state = "active"
            elif role == "assistant" and isinstance(reason, str) and reason in {"end_turn", "stop", "stop_sequence"}:
                state = "complete"
            elif role == "assistant":
                state = "active" if reason == "tool_use" else "unknown"
    return state


def inventory(cfg: Config, *, recent_seconds: float = 900, now: float | None = None) -> dict:
    started = time.monotonic()
    now = time.time() if now is None else now
    catalog = None
    if any(ds.type == "codex" for ds in cfg.data_stores):
        from .codex import CodexCatalog
        catalog = CodexCatalog(cfg.data_stores, cfg.codex)
    with readonly_database(cfg.db_path) as db:
        tracked = {r["path"]: r for r in db.execute("SELECT * FROM files")} if db else {}
        counts = {
            "sessions": db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] if db else 0,
            "memory_levels": dict(db.execute("SELECT level,COUNT(*) FROM memories GROUP BY level")) if db else {},
            "file_statuses": dict(db.execute("SELECT status,COUNT(*) FROM files GROUP BY status")) if db else {},
            "embedding_rows": db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] if db else 0,
            "embedding_payload_bytes": db.execute("SELECT COALESCE(SUM(LENGTH(vector)),0) FROM embeddings").fetchone()[0] if db else 0,
        }
    sources = []
    for ds in cfg.data_stores:
        found, _ = source_adapter(ds, codex_catalog=catalog)
        counters = Counter()
        total_bytes = largest = 0
        seen = set()
        for path, _project, _parent in found:
            counters["eligible_sources"] += 1
            try:
                info = path.stat()
                members = [p for p in source_members(path, ds.type) if p not in seen]
                stats = [p.stat() for p in members]
                seen.update(members)
                physical_bytes = sum(s.st_size for s in stats)
                total_bytes += physical_bytes
                largest = max(largest, physical_bytes)
                counters["recently_modified_sources"] += any(now - s.st_mtime < recent_seconds for s in stats)
                counters["activity_" + activity_state(path, ds.type)] += 1
                old = tracked.get(str(path))
                current = old is not None and (old["mtime"], old["size"]) == (info.st_mtime, info.st_size)
                counters["tracked_sources"] += old is not None
                counters["unchanged_metadata_sources"] += current
                counters["new_or_changed_sources"] += not current
            except OSError:
                counters["stat_errors"] += 1
        sources.append({"name": ds.name, "type": ds.type, "root_exists": ds.path.is_dir(),
                        "eligible_sources": 0, "recently_modified_sources": 0, "new_or_changed_sources": 0,
                        **dict(counters), "physical_files": len(seen), "source_bytes": total_bytes,
                        "largest_source_bytes": largest, "fixed_project": ds.project})
    return {"schema_version": 1, "observed_at_epoch": now, "database_exists": cfg.db_path.is_file(),
            "database_bytes": cfg.db_path.stat().st_size if cfg.db_path.is_file() else 0,
            **counts, "sources": sources,
            "codex_discovery_errors": len(catalog.errors) if catalog else 0,
            "codex_excluded_subagents": catalog.skipped_subagents if catalog else 0,
            "codex_discovery_warnings": len(catalog.warnings) if catalog else 0,
            "source_activity_note": "Recent mtime is an activity heuristic, not proof of turn completion. Metadata equality does not certify body-file freshness.",
            "duration_seconds": round(time.monotonic() - started, 3)}
