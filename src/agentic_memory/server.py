"""MCP server exposing the memory store to coding agents (stdio transport).

Tools are designed for an agent mid-conversation: compact, self-describing
text results with session/project anchors so the agent can drill down.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import CONFIG, to_slug
from .embed import Embedder, make_embedder
from .store import MemoryStore
from .transcript_reader import (
    DEFAULT_PAGE_BYTES,
    MAX_PAGE_BYTES,
    MIN_PAGE_BYTES,
    get_session_source_path,
    read_transcript_page,
    resolve_transcript_path,
)
from . import plan, retrieval

INSTRUCTIONS = """\
Long-term memory over the user's past AI coding sessions (ingested transcripts,
distilled into decisions, solutions, preferences, and project state, plus the
raw activity stream: exact commands, file paths, and error output) — and the
user's durable PLAN: an authored board of vision/epics/slices/tasks/notes per
project that persists across sessions and harnesses.

When to reach for it:
- Starting work in a project -> get_project_context (orient before exploring).
- The user references past work ("like we did before", "the bug from last
  week", "how did we set that up") -> search_memory.
- You hit an error that may have been solved before -> search_memory with the
  error text verbatim.
- You need the user's established preferences or past decisions -> search_memory
  with kind='preference' or kind='decision'.
- You need the exact source transcript after finding a session ->
  get_session_transcript, paging with its byte cursor.
- "What was I working on?" -> recent_activity.
- The user asks to PLAN work ("let's plan this", "break this down", "add it to
  the plan", "note this down") -> write the plan as plan_add items, not just
  prose in your reply.
- You start, finish, or get blocked on planned work -> plan_update, at that
  moment, not at session end.

Planning plane contract:
- Decompose: vision = durable product direction; epic = a multi-session
  feature; slice = a shippable increment; task = one session-sized unit of
  work; note = durable scratchpad. Nest vision -> epic -> slice -> task
  (convention, not enforced); create parents first — each plan_add returns
  the new item's id. Write bodies self-contained (context, acceptance
  criteria): future sessions read them without today's conversation.
- Statuses (all timestamped): planned | in_progress | blocked | complete |
  dropped. Mark in_progress when you begin, complete when done AND verified,
  blocked/dropped when reality changes (put the reason in the body). Items
  are never deleted. Never let the board drift from reality — and roll
  status UP the tree: starting an item's first child makes the parent
  in_progress; completing its last open child completes the parent. A parent
  must never read staler than its children.
- Harness todo lists are ephemeral; this board is durable. Mirror anything
  that must survive the session into it.
- A project is the durable logical memory/planning scope. By default it is the
  session cwd slug (`/Users/x/proj -> -Users-x-proj`), but an instance may bind
  sessions from many workspaces to one fixed project (for example `assistant`);
  cwd then remains provenance only. Treat list_projects as the authority and
  reuse its exact project key. Absolute cwd paths are accepted for legacy
  cwd-derived projects. Plan items are searchable (search_memory level='plan');
  the open board rides along in get_project_context, full tree via plan_board.

Workflow: list_projects to discover slugs; get_project_context for orientation;
search_memory for specifics (write SELF-CONTAINED queries — you have the
conversation context, the search does not); get_session to drill into any
session id a result mentions.

For routine/direct recall, use search_memory(mode='static'): one hybrid
lookup with no retrieval LLM. Use mode='agentic' for multi-step historical
research: the retrieval agent reformulates, fans out sub-queries, follows
session links, and returns curated evidence. mode='auto' preserves the
configured default. State the FULL information need in one call rather than
firing keyword fragments; if something is missing, refine the request.
"""

mcp = MCPServer("agentic-memory", instructions=INSTRUCTIONS)

_store: MemoryStore | None = None
_embedder: Embedder | None = None
_search_llm: object = None  # None = not tried; False = unavailable


def _get_search_llm(force: bool = False):
    """Lazily construct the search-agent backend; False if unusable."""
    global _search_llm
    if not force and not CONFIG.search_agent:
        return None
    if _search_llm is None:
        try:
            from .llm import AgentCoreBackend

            _search_llm = AgentCoreBackend.for_role(CONFIG, "search")
        except Exception as e:  # noqa: BLE001 — degrade to static search
            import sys

            print(
                f"[agentic-memory] search agent unavailable ({e}); "
                "falling back to static retrieval",
                file=sys.stderr,
            )
            _search_llm = False
    return _search_llm or None


def _outside_event_loop(fn, *args, **kwargs):
    """Run fn in a fresh thread if this thread holds a running event loop.

    The agent backend calls asyncio.run(), which raises inside a running
    loop; MCP tool dispatch is expected to use worker threads, but this
    guard keeps retrieval working even if it doesn't.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result()


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(CONFIG.db_path)
    return _store


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = make_embedder(CONFIG)
    return _embedder


def _fmt_hit(h: dict) -> str:
    where = h["project"] or "-"
    when = (h["ts"] or "")[:10]
    anchor = f" (session {h['session_id']}: {h['session_title']})" if h["session_id"] else ""
    score = f" | score {h['score']}" if h.get("score") is not None else ""
    from .freshness import describe
    freshness = '\n  ' + describe(h['freshness']) if h.get('freshness') else ''
    return (
        f"- [{h['level']}/{h['kind']} | {where} | {when}{score}]"
        f"{anchor}\n  {h['content']}{freshness}"
    )


@mcp.tool()
def search_memory(
    query: Annotated[str, Field(description=(
        "A SELF-CONTAINED, context-rich query — NOT the user's literal words. "
        "You have the conversation context; the search does not. Rewrite the "
        "ask into a standalone query naming the project, technologies, file "
        "names, and any error text verbatim (error strings retrieve extremely "
        "well). Bad: 'the bug from before'. Good: 'fix for FTS5 trigger "
        "corruption during incremental ingest in agentic-memory sqlite store'."
    ))],
    project: Annotated[str | None, Field(description=(
        "Logical project key to scope the search (get keys from list_projects), "
        "or an absolute cwd path for a cwd-derived project. Omit to search across "
        "all projects."
    ))] = None,
    level: Annotated[str | None, Field(description=(
        "Timescale filter: 'chunk' = raw activity (exact commands, error "
        "output, code snippets); 'memory' = distilled atomic facts; "
        "'session' = whole-session summaries; 'digest' = weekly summaries; "
        "'project' = rolling project profiles; 'plan' = authored plan items "
        "(vision/epics/slices/tasks/notes). Omit to search all levels."
    ))] = None,
    kind: Annotated[str | None, Field(description=(
        "Memory-kind filter (applies to level='memory'): decision | fact | "
        "preference | solution | problem | task_state. Use 'solution' for "
        "past fixes, 'preference' for how the user likes things done."
    ))] = None,
    after: Annotated[str | None, Field(description=(
        "ISO date lower bound, e.g. '2026-07-01'."
    ))] = None,
    before: Annotated[str | None, Field(description=(
        "ISO date upper bound, e.g. '2026-07-31'."
    ))] = None,
    limit: Annotated[int | None, Field(description=(
        "Max evidence items returned (default: search.evidence_limit in "
        "settings.toml, 12)."
    ))] = None,
    mode: Annotated[str, Field(description=(
        "'static' runs one hybrid lookup without the retrieval LLM (routine recall); "
        "'agentic' enables multi-step search; 'auto' follows [search] agent."
    ))] = "auto",
) -> str:
    """Search past-session memory and authored plans.

    Use mode='static' for routine/direct recall (one hybrid lookup, no
    retrieval LLM), or mode='agentic' for multi-step historical research.
    mode='auto' follows the configured default. Structurally empty scopes
    return immediately without constructing embedding or LLM providers.

    Use it for: "have we solved this before", past decisions and their
    rationale, exact commands/flags/paths from earlier work, error messages
    seen before, and the user's preferences. In agentic mode, an agent inside the
    server reformulates the query, fans out sub-searches, follows session
    links across projects, and returns curated evidence (never a synthesized
    answer) — so state the FULL information need in one call instead of
    firing keyword fragments. Results carry session ids — drill down with
    get_session. If something is missing, refine the request and call again.
    """
    project = to_slug(project) if project else None
    if mode not in {"auto", "static", "agentic"}:
        raise ValueError("mode must be auto, static, or agentic")
    store = _get_store()
    if (limit is not None and limit <= 0) or not store.has_memories(
        project=project, levels=(level,) if level else None,
        kinds=(kind,) if kind else None, after=after, before=before,
    ):
        from .freshness import project_coverage
        coverage = project_coverage(store, project)
        return "No matching memories found (requested scope is empty)." + ('\n' + coverage if coverage else '')
    embedder = _get_embedder()
    llm = _get_search_llm(force=mode == "agentic") if mode != "static" else None
    if llm is not None:
        try:
            from .search_agent import agentic_search

            hits, trace = _outside_event_loop(
                agentic_search,
                CONFIG, store, embedder, llm, query,
                project=project, level=level, kind=kind,
                after=after, before=before, limit=limit,
            )
            header = (
                f"[retrieval agent: {trace.rounds} rounds, "
                f"{len(trace.queries)} queries"
                + (f", {len(trace.sessions)} sessions expanded" if trace.sessions else "")
                + "]"
            )
            if trace.note:
                header += f"\n[agent note: {trace.note}]"
            if trace.fallback_used:
                header += '\n[selection fallback: results were not confirmed by a valid final model selection]'
            if not hits:
                return f"{header}\nNo matching memories found."
            return header + "\n" + "\n".join(_fmt_hit(h) for h in hits)
        except Exception as e:  # noqa: BLE001 — degrade to static search
            import sys

            print(f"[agentic-memory] agentic search failed ({e}); "
                  "falling back to static retrieval", file=sys.stderr)
    hits = retrieval.search(
        store,
        embedder,
        query,
        project=project,
        levels=(level,) if level else None,
        kinds=(kind,) if kind else None,
        after=after,
        before=before,
        limit=limit,
    )
    if not hits:
        return "No matching memories found."
    return "\n".join(_fmt_hit(h) for h in hits)


@mcp.tool()
def get_project_context(
    project: Annotated[str, Field(description=(
        "Logical project key from list_projects, e.g. 'assistant' or "
        "'-Users-sam-agents-assistant'. Absolute cwd paths are also accepted "
        "for cwd-derived projects."
    ))],
) -> str:
    """Orient in a project: rolling profile, open plan, last 10 session summaries.

    Call this when starting work in a project the user has history with —
    it covers architecture, key decisions, current state, known problems, the
    user's preferences for that project, and the open plan board, cheaper than
    re-exploring.
    """
    project = to_slug(project)
    store = _get_store()
    parts: list[str] = []
    from .freshness import describe, project_coverage, rollup_freshness, session_freshness
    coverage = project_coverage(store, project)
    if coverage:
        parts.append('## Ingestion coverage\n' + coverage)
    profile = store.get_profile(project)
    if profile:
        parts.append(f"## Project profile\n{profile['content']}\n{describe(rollup_freshness(store, 'project', project))}")
    board = plan.render_board(store, project, max_items=12)
    if board:
        parts.append(
            f"## Open plan\n{board}\n"
            "(keep this board live: plan_update the moment work starts or "
            "finishes; plan_add for new scope; full tree via plan_board)"
        )
    sessions = store.sessions_for_project(project, limit=10)
    if sessions:
        lines = []
        for s in sessions:
            when = (s["ended_at"] or "")[:10]
            lines.append(
                f"- [{when}] {s['title']} (id {s['id']})\n"
                f"  {s['summary']}\n  Outcome: {s['outcome']}"
                f"\n  {describe(session_freshness(store, s['id'], CONFIG))}"
            )
        parts.append("## Recent sessions (newest first)\n" + "\n".join(lines))
    if not parts:
        return f"No memory for project '{project}'. Use list_projects to see known projects."
    return "\n\n".join(parts)


@mcp.tool()
def get_session(
    session_id: Annotated[str, Field(description=(
        "Session id exactly as shown in search results or project context "
        "(a UUID, 'codex:<native-id>' for Codex, or 'agent-...' for subagent tasks)."
    ))],
) -> str:
    """Drill into one past session: full summary, outcome, and its memories.

    Use after search_memory or get_project_context surfaces a session id you
    want the complete picture of.
    """
    store = _get_store()
    s = store.get_session(session_id)
    if not s:
        return f"Unknown session id '{session_id}'."
    mems = store.memories_for_session(session_id)
    from .freshness import describe, session_freshness
    freshness = describe(session_freshness(store, session_id, CONFIG))
    mem_lines = "\n".join(
        f"- [{m['kind']} | importance {m['importance']:.1f}] {m['content']}" for m in mems
    ) or "(none extracted)"
    parent = ""
    if "parent_session" in s.keys() and s["parent_session"]:
        label = "Parent history" if session_id.startswith("codex:") else "Subagent task within session"
        parent = f"\n{label} {s['parent_session']}."
    return (
        f"# {s['title']}\n"
        f"Project: {s['project']} | {(s['started_at'] or '')[:16]} → {(s['ended_at'] or '')[:16]}"
        f" | branch: {s['git_branch'] or '-'}{parent}\n{freshness}\n\n"
        f"## Summary\n{s['summary']}\n\n"
        f"## Outcome\n{s['outcome']}\n\n"
        f"## Memories\n{mem_lines}"
    )


@mcp.tool(annotations=ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
))
def get_session_transcript(
    session_id: Annotated[str, Field(description=(
        "Session id exactly as shown by search_memory, get_project_context, or "
        "get_session. The source path is resolved internally; arbitrary paths "
        "cannot be read."
    ))],
    cursor: Annotated[int, Field(description=(
        "Zero-based BYTE offset returned as next_cursor by the previous page. "
        "Use 0 for the first page."
    ), ge=0)] = 0,
    max_bytes: Annotated[int, Field(description=(
        f"Maximum source bytes in this page ({MIN_PAGE_BYTES}-{MAX_PAGE_BYTES}; "
        "UTF-8 boundaries may make the page slightly smaller)."
    ), ge=MIN_PAGE_BYTES, le=MAX_PAGE_BYTES)] = DEFAULT_PAGE_BYTES,
) -> str:
    """Read one bounded page from a session's existing source transcript.

    This tool is strictly read-only: it does not ingest, distill, embed, index,
    or modify the transcript or memory database. Continue with `next_cursor`
    until `complete` is true.
    """
    try:
        source_path = get_session_source_path(CONFIG.db_path, session_id)
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"Unable to open the memory store read-only: {exc}"
    if not source_path:
        return f"Unknown session id '{session_id}' or no source transcript path."

    try:
        path = resolve_transcript_path(
            source_path,
            (data_store.path for data_store in CONFIG.data_stores),
        )
        page = read_transcript_page(path, cursor=cursor, max_bytes=max_bytes)
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        return f"Unable to read transcript for session '{session_id}': {exc}"

    envelope = {
        "session_id": session_id,
        "source_format": "jsonl" if path.suffix == ".jsonl" else "json",
        "cursor": page.cursor,
        "next_cursor": page.next_cursor,
        "page_bytes": page.page_bytes,
        "source_size_bytes": page.source_size_bytes,
        "complete": page.complete,
        "content": page.content,
    }
    return json.dumps(envelope, ensure_ascii=False)


@mcp.tool()
def list_projects() -> str:
    """List all known projects, newest activity first.

    Covers both planes: projects with ingested session memory and projects
    that so far exist only as a plan board (planned before their first
    session). This is the authority for logical project keys.
    """
    rows = _get_store().list_projects()
    if not rows:
        return "No projects yet."
    lines = []
    for r in rows:
        if r["n_sessions"]:
            line = (f"- {r['project']} ({r['n_sessions']} sessions, "
                    f"last active {(r['last_active'] or '?')[:10]}")
            if r["n_plan"]:
                line += f"; {r['n_plan']} plan items"
            line += ")"
        else:
            line = (f"- {r['project']} (no sessions yet; {r['n_plan']} plan "
                    f"items, updated {(r['plan_updated'] or '?')[:10]})")
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def recent_activity(
    days: Annotated[int, Field(description="Lookback window in days (default 7).")] = 7,
) -> str:
    """What the user worked on recently, across all projects.

    Answers "what was I doing last week / where did I leave off" — weekly
    digests plus per-session outcomes, newest first.
    """
    from datetime import datetime, timedelta, timezone

    store = _get_store()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sessions = store.sessions_since(cutoff)
    parts = []
    digests = store.get_digests(limit=max(1, days // 7 + 1))
    if digests:
        parts.append(
            "## Weekly digests\n"
            + "\n".join(f"- week of {(d['ts'] or '')[:10]}: {d['content']}" for d in digests)
        )
    if sessions:
        lines = [
            f"- [{(s['ended_at'] or '')[:10]}] {s['project']}: {s['title']} — {s['outcome']}"
            for s in sessions[:25]
        ]
        parts.append(f"## Sessions in the last {days} days\n" + "\n".join(lines))
    return "\n\n".join(parts) if parts else f"No activity recorded in the last {days} days."


@mcp.tool()
def plan_add(
    project: Annotated[str, Field(description=(
        "Logical project key the item belongs to — use the exact key from "
        "list_projects. Absolute cwd paths are also accepted for cwd-derived "
        "projects."
    ))],
    kind: Annotated[str, Field(description=(
        "vision | epic | slice | task | note. Typical nesting: vision -> epic "
        "-> slice -> task (a convention, not enforced). 'note' is freeform "
        "scratchpad — attach it to any item or leave it top-level."
    ))],
    title: Annotated[str, Field(description=(
        "Short imperative title; for notes, a one-line gist."
    ))],
    body: Annotated[str, Field(description=(
        "Full text: context, rationale, acceptance criteria, scratch content. "
        "Write it SELF-CONTAINED — future sessions read this without today's "
        "conversation."
    ))] = "",
    parent_id: Annotated[int | None, Field(description=(
        "id of the parent item (find ids with plan_board). Omit for top-level."
    ))] = None,
    priority: Annotated[int, Field(description=(
        "1 (highest) to 5 (lowest); default 3."
    ))] = 3,
    tags: Annotated[list[str] | None, Field(description=(
        "Lowercase keyword tags for retrieval."
    ))] = None,
) -> str:
    """Create a plan item: vision, epic, slice, task, or freeform note.

    When the user asks to plan work, capture the plan HERE — one call per
    item, parents before children (each call returns the new item's id) —
    not just as prose in your reply. New items start as 'planned'. The plan
    is durable working memory: it persists across sessions and harnesses,
    and items are searchable via search_memory (level='plan'). Check
    plan_board first to pick the right parent and avoid duplicates.
    """
    try:
        project = plan.resolve_project_policy(
            to_slug(project),
            default_project=CONFIG.planning_default_project,
            lock_writes_to_default=CONFIG.planning_lock_writes_to_default,
            write=True,
        )
    except ValueError as e:
        return f"Cannot create item: {e}"
    store = _get_store()
    note = plan.unknown_project_note(store, project)  # before the write
    try:
        item = plan.add_item(
            store, _get_embedder(), project=project, kind=kind,
            title=title, body=body, parent_id=parent_id, priority=priority,
            tags=tags or [],
        )
    except ValueError as e:
        return f"Cannot create item: {e}"
    out = f"Created {plan.format_line(item)} in {project}"
    return f"{out}\n{note}" if note else out


@mcp.tool()
def plan_update(
    item_id: Annotated[int, Field(description="Plan item id (see plan_board).")],
    status: Annotated[str | None, Field(description=(
        "New status: planned | in_progress | blocked | complete | dropped. "
        "Items are never deleted — 'complete' finishes, 'dropped' retires. "
        "Every change is timestamped in the item's history. Omit to keep."
    ))] = None,
    title: Annotated[str | None, Field(description="New title. Omit to keep.")] = None,
    body: Annotated[str | None, Field(description=(
        "Replacement body text (full replacement, not appended). Omit to keep."
    ))] = None,
    priority: Annotated[int | None, Field(description=(
        "New priority 1-5. Omit to keep."
    ))] = None,
    parent_id: Annotated[int | None, Field(description=(
        "New parent id; 0 makes the item top-level. Omit to keep."
    ))] = None,
    tags: Annotated[list[str] | None, Field(description=(
        "Replacement tag list. Omit to keep."
    ))] = None,
) -> str:
    """Update a plan item: move status, edit, reprioritize, or re-parent.

    Maintain status LIVE, at the moment it changes: in_progress when you
    begin an item, complete when the work is done AND verified, blocked or
    dropped when reality changes (put the reason in the body). Roll status up
    the tree too: a parent epic/slice goes in_progress with its first started
    child and complete with its last finished one. Do not batch updates for
    session end — a stale board misleads every future session. Every
    transition is timestamped and recorded in the item's history.
    """
    store = _get_store()
    existing = store.plan_get(item_id)
    if existing is not None:
        try:
            plan.resolve_project_policy(
                existing["project"],
                default_project=CONFIG.planning_default_project,
                lock_writes_to_default=CONFIG.planning_lock_writes_to_default,
                write=True,
            )
        except ValueError as e:
            return f"Cannot update item: {e}"
    kwargs: dict = dict(status=status, title=title, body=body,
                        priority=priority, tags=tags)
    if parent_id is not None:
        kwargs["parent_id"] = parent_id
    try:
        item, changes = plan.update_item(
            store, _get_embedder(), item_id, **kwargs
        )
    except ValueError as e:
        return f"Cannot update item: {e}"
    if not changes:
        return f"No changes applied to {plan.format_line(item)}"
    return f"Updated {plan.format_line(item)} — " + "; ".join(changes)


@mcp.tool()
def plan_board(
    project: Annotated[str, Field(description=(
        "Logical project key — use the exact key from list_projects. Absolute "
        "cwd paths are also accepted for cwd-derived projects."
    ))],
    include_done: Annotated[bool, Field(description=(
        "Include complete/dropped items (default: open work only)."
    ))] = False,
) -> str:
    """The project's authored plan as a tree: vision/epics/slices/tasks/notes.

    Shows ids, status, and priority for every item. Call it when starting
    work, and before plan_add to find parent ids and avoid duplicates.
    """
    project = to_slug(project)
    board = plan.render_board(_get_store(), project, include_done=include_done)
    return board or (
        f"No plan items for '{project}' yet — plan_add creates the first "
        "(check the slug against list_projects if that's unexpected)."
    )


@mcp.tool()
def plan_get(
    item_id: Annotated[int, Field(description=(
        "Plan item id, as shown by plan_board or search results."
    ))],
) -> str:
    """One plan item in full: body, parent, children, and change history."""
    store = _get_store()
    item = store.plan_get(item_id)
    if not item:
        return f"No plan item #{item_id}."
    return plan.render_item(store, item)


@mcp.tool()
def memory_status() -> str:
    """Report stored counts, source progress, failures and pending rollups.

    Existing count fields remain at the top level. The ingestion section
    distinguishes published source coverage from worker/supervisor liveness.
    """
    from .freshness import store_status
    return json.dumps(store_status(_get_store(), CONFIG), indent=2)


def run() -> None:
    # Keep third-party log chatter off stdout (it carries JSON-RPC) and quiet.
    import logging
    import sys

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
    for name in ("sentence_transformers", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    mcp.run()  # stdio transport


if CONFIG.personal.enabled:
    from .autobiography_server import register_personal_tools
    register_personal_tools(mcp, CONFIG)


if __name__ == "__main__":
    run()
