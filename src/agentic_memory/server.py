"""MCP server exposing the memory store to coding agents (stdio transport).

Tools are designed for an agent mid-conversation: compact, self-describing
text results with session/project anchors so the agent can drill down.
"""

from __future__ import annotations

import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from .config import CONFIG
from .embed import Embedder, make_embedder
from .store import MemoryStore
from . import retrieval

INSTRUCTIONS = """\
Long-term memory over the user's past AI coding sessions (ingested transcripts,
distilled into decisions, solutions, preferences, and project state, plus the
raw activity stream: exact commands, file paths, and error output).

When to reach for it:
- Starting work in a project -> get_project_context (orient before exploring).
- The user references past work ("like we did before", "the bug from last
  week", "how did we set that up") -> search_memory.
- You hit an error that may have been solved before -> search_memory with the
  error text verbatim.
- You need the user's established preferences or past decisions -> search_memory
  with kind='preference' or kind='decision'.
- "What was I working on?" -> recent_activity.

Workflow: list_projects to discover slugs; get_project_context for orientation;
search_memory for specifics (write SELF-CONTAINED queries — you have the
conversation context, the search does not); get_session to drill into any
session id a result mentions.

search_memory runs a retrieval agent inside the server: it reformulates,
fans out sub-queries, follows session links, and returns curated evidence.
State the FULL information need in one call rather than firing keyword
fragments; if something is missing, refine the request and call again.
"""

mcp = MCPServer("agentic-memory", instructions=INSTRUCTIONS)

_store: MemoryStore | None = None
_embedder: Embedder | None = None
_search_llm: object = None  # None = not tried; False = unavailable


def _get_search_llm():
    """Lazily construct the search-agent backend; False if unusable."""
    global _search_llm
    if _search_llm is None:
        if not CONFIG.search_agent:
            _search_llm = False
        else:
            try:
                from .llm import AgentCoreBackend

                _search_llm = AgentCoreBackend(
                    CONFIG, probe=False, profile=CONFIG.search_profile or None
                )
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
    return (
        f"- [{h['level']}/{h['kind']} | {where} | {when}{score}]"
        f"{anchor}\n  {h['content']}"
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
        "Project slug to scope the search (get slugs from list_projects). "
        "Omit to search across all projects."
    ))] = None,
    level: Annotated[str | None, Field(description=(
        "Timescale filter: 'chunk' = raw activity (exact commands, error "
        "output, code snippets); 'memory' = distilled atomic facts; "
        "'session' = whole-session summaries; 'digest' = weekly summaries; "
        "'project' = rolling project profiles. Omit to search all levels."
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
) -> str:
    """Agentic search over everything in past-session memory.

    Use it for: "have we solved this before", past decisions and their
    rationale, exact commands/flags/paths from earlier work, error messages
    seen before, and the user's preferences. A retrieval agent inside the
    server reformulates the query, fans out sub-searches, follows session
    links across projects, and returns curated evidence (never a synthesized
    answer) — so state the FULL information need in one call instead of
    firing keyword fragments. Results carry session ids — drill down with
    get_session. If something is missing, refine the request and call again.
    """
    store, embedder = _get_store(), _get_embedder()
    llm = _get_search_llm()
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
        "Project slug from list_projects, e.g. '-Users-sam-agents-assistant' "
        "(slugs are the project cwd with slashes replaced by dashes)."
    ))],
) -> str:
    """Orient in a project: rolling profile + the last 10 session summaries.

    Call this when starting work in a project the user has history with —
    it covers architecture, key decisions, current state, known problems, and
    the user's preferences for that project, cheaper than re-exploring.
    """
    store = _get_store()
    parts: list[str] = []
    profile = store.get_profile(project)
    if profile:
        parts.append(f"## Project profile\n{profile['content']}")
    sessions = store.sessions_for_project(project, limit=10)
    if sessions:
        lines = []
        for s in sessions:
            when = (s["ended_at"] or "")[:10]
            lines.append(
                f"- [{when}] {s['title']} (id {s['id']})\n"
                f"  {s['summary']}\n  Outcome: {s['outcome']}"
            )
        parts.append("## Recent sessions (newest first)\n" + "\n".join(lines))
    if not parts:
        return f"No memory for project '{project}'. Use list_projects to see known projects."
    return "\n\n".join(parts)


@mcp.tool()
def get_session(
    session_id: Annotated[str, Field(description=(
        "Session id exactly as shown in search results or project context "
        "(a UUID for main sessions, 'agent-...' for subagent tasks)."
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
    mem_lines = "\n".join(
        f"- [{m['kind']} | importance {m['importance']:.1f}] {m['content']}" for m in mems
    ) or "(none extracted)"
    parent = ""
    if "parent_session" in s.keys() and s["parent_session"]:
        parent = f"\nSubagent task within session {s['parent_session']}."
    return (
        f"# {s['title']}\n"
        f"Project: {s['project']} | {(s['started_at'] or '')[:16]} → {(s['ended_at'] or '')[:16]}"
        f" | branch: {s['git_branch'] or '-'}{parent}\n\n"
        f"## Summary\n{s['summary']}\n\n"
        f"## Outcome\n{s['outcome']}\n\n"
        f"## Memories\n{mem_lines}"
    )


@mcp.tool()
def list_projects() -> str:
    """List all projects with ingested memory, newest activity first."""
    rows = _get_store().list_projects()
    if not rows:
        return "No projects ingested yet."
    return "\n".join(
        f"- {r['project']} ({r['n_sessions']} sessions, last active {(r['last_active'] or '?')[:10]})"
        for r in rows
    )


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
def memory_status() -> str:
    """Report ingestion statistics for the memory store (diagnostics)."""
    return json.dumps(_get_store().counts(), indent=2)


def run() -> None:
    # Keep third-party log chatter off stdout (it carries JSON-RPC) and quiet.
    import logging
    import sys

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
    for name in ("sentence_transformers", "httpx", "huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    mcp.run()  # stdio transport


if __name__ == "__main__":
    run()
