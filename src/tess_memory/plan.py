"""Authored planning plane: a durable scratchpad and plan per project.

Everything else in the store is retrospective — distilled after the fact from
transcripts of what already happened. Plan items are prospective and authored:
an agent mid-session creates and maintains them deliberately — product vision,
epics, slices, tasks, and freeform notes — with stable ids, parent links, and
a status lifecycle (planned / in_progress / blocked / complete / dropped).
Every status change stamps status_at on the item and lands, timestamped, in
plan_events.

Two rules keep an autonomous writer safe:
- No hard deletes: items retire by status (complete/dropped) and every
  transition is recorded in plan_events.
- Every write re-mirrors the item into the retrieval index (level='plan'), so
  search_memory finds plan state exactly like distilled memories; a failed
  embedding degrades the mirror to keyword-only, never blocks the write.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from .embed import Embedder
from .store import MemoryRow, MemoryStore

PLAN_KINDS = ("vision", "epic", "slice", "task", "note")
PLAN_STATUSES = ("planned", "in_progress", "blocked", "complete", "dropped")
OPEN_STATUSES = ("planned", "in_progress", "blocked")

_KIND_ORDER = {k: i for i, k in enumerate(PLAN_KINDS)}
_STATUS_ORDER = {
    s: i for i, s in enumerate(("in_progress", "blocked", "planned", "complete", "dropped"))
}
_UNSET: Any = object()
_INDEX_CONTENT_CAP = 6_000  # mirror content cap; matches the chunk_chars scale


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- validation ----------------


def resolve_project_policy(
    project: str | None,
    *,
    default_project: str = "",
    lock_writes_to_default: bool = False,
    write: bool = False,
) -> str:
    """Apply an instance's default/locked logical planning namespace.

    Inputs are expected to be normalized project slugs. Legacy instances leave
    both policy fields unset, so explicit project behavior is unchanged.
    """
    requested = (project or "").strip()
    default = (default_project or "").strip()
    resolved = requested or default
    if not resolved:
        raise ValueError("project is required")
    if write and lock_writes_to_default and resolved != default:
        raise ValueError(
            f"plan writes are locked to project '{default}', not '{resolved}'"
        )
    return resolved


def _validate_priority(priority) -> int:
    try:
        p = int(priority)
    except (TypeError, ValueError):
        raise ValueError("priority must be an integer 1-5") from None
    if not 1 <= p <= 5:
        raise ValueError("priority must be 1 (highest) to 5 (lowest)")
    return p


def _validate_parent(
    store: MemoryStore, project: str, parent_id: int, item_id: int | None = None
) -> None:
    parent = store.plan_get(parent_id)
    if parent is None:
        raise ValueError(f"parent #{parent_id} does not exist")
    if parent["project"] != project:
        raise ValueError(
            f"parent #{parent_id} is in project '{parent['project']}', not '{project}'"
        )
    seen: set[int] = set()
    cur = parent
    while cur is not None:
        if item_id is not None and cur["id"] == item_id:
            raise ValueError(
                f"#{item_id} cannot be nested under its own descendant #{parent_id}"
            )
        if cur["id"] in seen:  # defensive: pre-existing cycle must not hang us
            break
        seen.add(cur["id"])
        cur = store.plan_get(cur["parent_id"]) if cur["parent_id"] else None


def unknown_project_note(store: MemoryStore, project: str) -> str | None:
    """Warning text when a plan write targets a project the store has never
    seen (no sessions, no plan items) — the typo guard. Call BEFORE the write;
    returns None for known projects. Creating a board for a genuinely new
    project stays legal; the note just makes a misfiled slug visible to the
    agent at the moment it can still self-correct."""
    known = store.known_projects()
    if project in known:
        return None
    import difflib

    close = difflib.get_close_matches(project, sorted(known), n=1, cutoff=0.6)
    note = (
        f"Note: '{project}' has no prior sessions or plan items, so this "
        "starts a new board. If you meant an existing project, verify the slug"
    )
    if close:
        note += f" (closest known: {close[0]})"
    return note + "."


# ---------------- retrieval-index mirror ----------------


def render_for_index(item) -> str:
    """Self-contained text mirrored into FTS + vectors for one plan item."""
    head = (
        f"[plan {item['kind']} #{item['id']} | {item['status']} | "
        f"p{item['priority']} | {item['project']}] {item['title']}"
    )
    body = (item["body"] or "").strip()
    text = f"{head}\n{body}" if body else head
    return text[:_INDEX_CONTENT_CAP]


def index_importance(item) -> float:
    imp = 1.0 - 0.1 * item["priority"]  # p1 -> 0.9 … p5 -> 0.5
    if item["status"] in ("complete", "dropped"):
        imp -= 0.2  # open work outranks finished history
    return max(0.3, min(0.9, imp))


def _write_keyword_mirror(store: MemoryStore, item) -> MemoryRow:
    content = render_for_index(item)
    row = MemoryRow(
        level="plan", kind=item["kind"], content=content,
        importance=index_importance(item), project=item["project"],
        ts=item["updated_at"], tags=json.loads(item["tags"] or "[]"),
    )
    store.replace_plan_mirror(item["id"], row)
    return row


def _embed_mirror(store: MemoryStore, embedder: Embedder, item, row: MemoryRow) -> None:
    if not store.embedder_matches(embedder.descriptor):
        return  # keyword-only until `tess-memory reembed`; the plan item itself is saved
    try:
        vector = embedder.embed_passages([row.content])[0]
        store.attach_plan_embedding(item["id"], row, vector, embedder.descriptor)
    except Exception as e:  # noqa: BLE001 — a failed embed must not block the write
        print(
            f"[plan] embedding failed for #{item['id']} ({e}); "
            "keyword-only until the next update",
            file=sys.stderr,
        )


# ---------------- write operations ----------------


def add_item(
    store: MemoryStore,
    embedder: Embedder,
    *,
    project: str,
    kind: str,
    title: str,
    body: str = "",
    parent_id: int | None = None,
    priority: int = 3,
    tags: list[str] | None = None,
):
    kind = (kind or "").strip().lower()
    if kind not in PLAN_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(PLAN_KINDS)}")
    project = (project or "").strip()
    if not project:
        raise ValueError("project is required")
    title = " ".join((title or "").split())
    if not title:
        raise ValueError("title is required")
    priority = _validate_priority(priority)
    if parent_id is not None:
        parent_id = int(parent_id)
    clean_tags = [str(t).lower() for t in (tags or [])][:8]
    now = _now()
    with store.write_transaction():
        if not store.can_write_project(project):
            raise PermissionError(f"no access to project '{project}'")
        if parent_id is not None:
            _validate_parent(store, project, parent_id)
        item_id = store.plan_insert(
            project=project, kind=kind, title=title, body=(body or "").strip(),
            parent_id=parent_id, priority=priority, tags=clean_tags, now=now,
        )
        store.plan_log(item_id, "created", f"created as {kind} '{title[:80]}'", now)
        item = store.plan_get(item_id)
        row = _write_keyword_mirror(store, item)
    _embed_mirror(store, embedder, item, row)
    return item


def update_item(
    store: MemoryStore,
    embedder: Embedder,
    item_id: int,
    *,
    status: str | None = None,
    title: str | None = None,
    body: str | None = None,
    priority=None,
    parent_id=_UNSET,
    tags: list[str] | None = None,
    append: str | None = None,
):
    """Commit the edit and keyword index before requesting its embedding.

    ``append`` adds a paragraph to the end of the body as it stands inside the
    write transaction, so two writers adding to one item both keep their text;
    ``body`` replaces it and is for a writer that has just read the item."""
    if body is not None and append is not None:
        raise ValueError("pass body to replace the text or append to add to it, not both")
    with store.write_transaction():
        item, changes = _update_item_fields(
            store, item_id, status=status, title=title, body=body,
            priority=priority, parent_id=parent_id, tags=tags, append=append,
        )
        if not changes:
            return item, changes
        row = _write_keyword_mirror(store, item)
    _embed_mirror(store, embedder, item, row)
    return item, changes


def _update_item_fields(
    store: MemoryStore, item_id: int, *, status: str | None, title: str | None,
    body: str | None, priority, parent_id, tags: list[str] | None, append: str | None = None,
):
    """Read, validate and edit under the caller's cross-process transaction."""
    item = store.plan_get(int(item_id))
    if item is None:
        raise ValueError(f"no plan item #{item_id}")
    item_id = item["id"]
    fields: dict = {}
    changes: list[str] = []
    if status is not None:
        status = status.strip().lower().replace("-", "_")  # accept "in-progress"
        if status not in PLAN_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(PLAN_STATUSES)}")
        if status != item["status"]:
            fields["status"] = status
            changes.append(f"status {item['status']} -> {status}")
    if title is not None:
        title = " ".join(title.split())
        if title and title != item["title"]:
            fields["title"] = title
            changes.append("title edited")
    if body is not None and body.strip() != (item["body"] or ""):
        fields["body"] = body.strip()
        changes.append("body edited")
    if append is not None and append.strip():
        fields["body"] = "\n\n".join(part for part in ((item["body"] or "").strip(), append.strip()) if part)
        changes.append("body appended")
    if priority is not None:
        p = _validate_priority(priority)
        if p != item["priority"]:
            fields["priority"] = p
            changes.append(f"priority p{item['priority']} -> p{p}")
    if tags is not None:
        clean = [str(t).lower() for t in tags][:8]
        if json.dumps(clean) != (item["tags"] or "[]"):
            fields["tags"] = json.dumps(clean)
            changes.append("tags edited")
    if parent_id is not _UNSET:
        new_parent = int(parent_id) if parent_id else None  # 0/None -> top level
        if new_parent != item["parent_id"]:
            if new_parent is not None:
                _validate_parent(store, item["project"], new_parent, item_id=item_id)
            fields["parent_id"] = new_parent
            changes.append(
                f"parent -> #{new_parent}" if new_parent else "parent -> top level"
            )
    if not fields:
        return item, []
    now = _now()
    if "status" in fields:
        fields["status_at"] = now  # when the item entered its current status
    store.plan_set(item_id, fields, now)
    store.plan_log(
        item_id, "status" if "status" in fields else "edit", "; ".join(changes), now
    )
    item = store.plan_get(item_id)
    return item, changes


# ---------------- rendering ----------------


def format_line(item) -> str:
    return (
        f"#{item['id']} [{item['kind']} | {item['status']} | "
        f"p{item['priority']}] {item['title']}"
    )


def render_board(
    store: MemoryStore,
    project: str,
    include_done: bool = False,
    max_items: int | None = None,
) -> str:
    """The project's plan as an indented tree; empty string if no items."""
    items = store.plan_items_for(project)
    if not items:
        return ""
    children: dict[int | None, list] = defaultdict(list)
    for i in items:
        children[i["parent_id"]].append(i)
    for siblings in children.values():
        siblings.sort(key=lambda i: (
            _KIND_ORDER.get(i["kind"], 9), _STATUS_ORDER.get(i["status"], 9),
            i["priority"], i["id"],
        ))

    show_cache: dict[int, bool] = {}

    def show(i) -> bool:
        # A retired item stays visible while it still shelters open work.
        if include_done:
            return True
        cached = show_cache.get(i["id"])
        if cached is None:
            cached = i["status"] in OPEN_STATUSES or any(
                show(c) for c in children[i["id"]]
            )
            show_cache[i["id"]] = cached
        return cached

    lines: list[str] = []

    def walk(parent_id: int | None, depth: int) -> None:
        for i in children[parent_id]:
            if not show(i):
                continue
            lines.append("  " * depth + format_line(i))
            walk(i["id"], depth + 1)

    walk(None, 0)

    counts = Counter(i["status"] for i in items)
    summary = ", ".join(f"{counts[s]} {s}" for s in PLAN_STATUSES if counts[s])
    out = [f"{project} — {summary}"]
    if max_items is not None and len(lines) > max_items:
        out.extend(lines[:max_items])
        out.append(f"… {len(lines) - max_items} more — plan_board shows the full tree")
    else:
        out.extend(lines)
    hidden = len(items) - len(lines)
    if hidden:
        out.append(f"({hidden} complete/dropped hidden — plan_board include_done=true)")
    return "\n".join(out)


def render_item(store: MemoryStore, item) -> str:
    """One item in full: metadata, body, children, and change history."""
    lines = [format_line(item)]
    lines.append(
        f"Project: {item['project']} | created {(item['created_at'] or '')[:10]}"
        f" | updated {(item['updated_at'] or '')[:10]}"
        f" | {item['status']} since {(item['status_at'] or '')[:16] or '?'}"
    )
    if item["parent_id"]:
        parent = store.plan_get(item["parent_id"])
        if parent:
            lines.append(f"Parent: {format_line(parent)}")
    tags = json.loads(item["tags"] or "[]")
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    if item["body"]:
        lines.append("\n" + item["body"])
    kids = store.plan_children(item["id"])
    if kids:
        lines.append("\nChildren:\n" + "\n".join("  " + format_line(k) for k in kids))
    events = store.plan_history(item["id"])
    if events:
        lines.append("\nHistory:\n" + "\n".join(
            f"  {(e['ts'] or '')[:16]} [{e['event']}] {e['detail']}" for e in events
        ))
    return "\n".join(lines)
