"""Image records: what an agent's host knows about the images people sent it.

A host that keeps images (Tess Agent does) has a model describe each one and
records the description here, so that images are found by the same search as
everything else the agent remembers, and with the same embedding model. This
package never sees an image: it stores the description, where the host keeps
the file, the session the image was sent in, and whatever facts the host read
out of it.

Like a plan item, an image record is authored, not derived from a transcript:
nothing here can rebuild it, and re-ingesting a session never touches it.
Every write re-mirrors the record into the retrieval index (level='image'),
so ``search_memory`` finds it; the mirror says the image's id, its path (an
agent with a file-reading tool can look at it) and its session.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

from .embed import Embedder
from .store import MemoryRow, MemoryStore

MAX_DESCRIPTION = 20_000   # characters kept of one description
_INDEX_CONTENT_CAP = 12_000  # mirror content cap: a detailed description, whole
MAX_FACTS_BYTES = 64 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(value: Any, limit: int = 2_000) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text[:limit] or None


def render_for_index(record) -> str:
    """Self-contained text mirrored into FTS + vectors for one image record."""
    head = [f"image {record['ref']}"]
    if record["name"]:
        head.append(record["name"])
    if record["sent_at"]:
        head.append(f"sent {record['sent_at'][:10]}")
    if record["taken_at"]:
        head.append(f"taken {record['taken_at'][:16]}")
    lines = ["[" + " | ".join(head) + "]"]
    if record["path"]:
        lines.append(f"File: {record['path']}")
    if record["session_id"]:
        lines.append(f"Sent in session {record['session_id']}")
    lines.append(record["description"])
    return "\n".join(lines)[:_INDEX_CONTENT_CAP]


def _mirror(store: MemoryStore, record) -> MemoryRow:
    row = MemoryRow(level="image", kind="image", content=render_for_index(record), importance=0.5,
                    project=record["project"], ts=record["sent_at"] or record["updated_at"])
    store.replace_image_mirror(record["ref"], row)
    return row


def _embed(store: MemoryStore, embedder: Embedder, record, row: MemoryRow) -> bool:
    if not store.embedder_matches(embedder.descriptor):
        return False  # keyword-only until `tess-memory reembed`; the record itself is saved
    try:
        vector = embedder.embed_passages([row.content])[0]
        return store.attach_image_embedding(record["ref"], row, vector, embedder.descriptor)
    except Exception as e:  # noqa: BLE001 — a failed embed must not block the write
        print(f"[images] embedding failed for {record['ref']} ({e}); keyword-only until it is recorded again", file=sys.stderr)
        return False


def record_image(
    store: MemoryStore,
    embedder: Embedder,
    *,
    ref: str,
    description: str,
    project: str | None = None,
    session_id: str | None = None,
    path: str | None = None,
    name: str | None = None,
    sent_at: str | None = None,
    taken_at: str | None = None,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add an image record, or replace the one with this ``ref``.

    ``project`` is where it is filed; when the session it was sent in is
    already in the store, the session's project is used instead, so an image
    is found with its conversation.
    """
    ref = (ref or "").strip()
    if not ref or len(ref) > 200 or any(c.isspace() for c in ref):
        raise ValueError("ref is the host's id for the image: 1 to 200 characters, no spaces")
    description = (description or "").strip()[:MAX_DESCRIPTION]
    if not description:
        raise ValueError("description is required")
    facts = dict(facts or {})
    if len(json.dumps(facts, ensure_ascii=False).encode()) > MAX_FACTS_BYTES:
        raise ValueError("facts is too large")
    session_id = _text(session_id, 200)
    with store.write_transaction():
        session = store.db.execute("SELECT project FROM sessions WHERE id=?", (session_id,)).fetchone() if session_id else None
        filed = (session["project"] if session and session["project"] else None) or _text(project, 200)
        if filed and not store.can_write_project(filed):
            raise PermissionError(f"no access to project '{filed}'")
        store.image_upsert(ref=ref, project=filed, session_id=session_id, path=_text(path, 4_000), name=_text(name, 400),
                           description=description, sent_at=_text(sent_at, 64), taken_at=_text(taken_at, 64), facts=facts, now=_now())
        record = store.image_get(ref)
        row = _mirror(store, record)
    embedded = _embed(store, embedder, record, row)
    return {**as_dict(record), "embedded": embedded}


def remove_image(store: MemoryStore, ref: str) -> bool:
    """Forget an image record (the host no longer keeps the image). True if there was one."""
    with store.write_transaction():
        return store.image_delete((ref or "").strip())


def get_image(store: MemoryStore, ref: str) -> dict[str, Any] | None:
    record = store.image_get((ref or "").strip())
    return as_dict(record) if record is not None else None


def as_dict(record) -> dict[str, Any]:
    return {**{key: record[key] for key in ("ref", "project", "session_id", "path", "name", "description", "sent_at", "taken_at", "updated_at")},
            "facts": json.loads(record["facts"] or "{}")}


def apply(store: MemoryStore, embedder: Embedder, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply a host's operations in order: ``{"op": "record", ...fields}`` or
    ``{"op": "remove", "ref": ...}``. One result each; a refused operation
    says why and does not stop the rest."""
    results = []
    for operation in operations:
        ref = operation.get("ref") if isinstance(operation, dict) else None
        try:
            if not isinstance(operation, dict) or operation.get("op") not in ("record", "remove"):
                raise ValueError("op is 'record' or 'remove'")
            if operation["op"] == "remove":
                results.append({"ref": ref, "ok": True, "removed": remove_image(store, str(ref or ""))})
                continue
            fields = {key: operation.get(key) for key in ("ref", "description", "project", "session_id", "path", "name", "sent_at", "taken_at", "facts")}
            if fields["facts"] is not None and not isinstance(fields["facts"], dict):
                raise ValueError("facts is an object")
            saved = record_image(store, embedder, **fields)
            results.append({"ref": saved["ref"], "ok": True, "project": saved["project"], "embedded": saved["embedded"]})
        except (ValueError, PermissionError, TypeError) as refused:
            results.append({"ref": ref, "ok": False, "error": str(refused)})
    return results
