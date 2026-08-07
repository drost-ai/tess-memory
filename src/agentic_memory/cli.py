"""CLI: agentic-memory {ingest, search, status, serve}."""

from __future__ import annotations

import argparse
import json
import sys

from .config import CONFIG


def cmd_ingest(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .ingest import ingest
    from .llm import AgentCoreBackend
    from .store import MemoryStore

    CONFIG.ensure_dirs()
    store = MemoryStore(CONFIG.db_path)
    embedder = make_embedder(CONFIG)
    llm = None if args.no_llm else AgentCoreBackend(CONFIG)
    fallback = None
    if llm is not None and CONFIG.agent_core_fallback_profile:
        # probe=False: no API call unless a refusal actually needs it
        fallback = AgentCoreBackend(
            CONFIG, probe=False, profile=CONFIG.agent_core_fallback_profile
        )
    for ds in CONFIG.data_stores:
        print(f"[ingest] data_store {ds.name} ({ds.type}): {ds.path}", file=sys.stderr)
    print(f"[ingest] db={CONFIG.db_path}", file=sys.stderr)
    print(f"[ingest] llm={'none (extractive-only)' if llm is None else llm.name}", file=sys.stderr)
    if fallback is not None:
        print(f"[ingest] refusal fallback={fallback.name}", file=sys.stderr)
    stats = ingest(
        CONFIG, store, embedder, llm, fallback_llm=fallback,
        limit=args.limit, retry_extractive=args.retry_extractive,
    )
    print(json.dumps(vars(stats), indent=2))
    store.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .retrieval import search
    from .store import MemoryStore

    project = _to_slug(args.project) if args.project else None
    store = MemoryStore(CONFIG.db_path)
    embedder = make_embedder(CONFIG)
    hits = None
    if CONFIG.search_agent and not args.static:
        from .llm import AgentCoreBackend
        from .search_agent import agentic_search

        llm = AgentCoreBackend(
            CONFIG, probe=False, profile=CONFIG.search_profile or None
        )
        print(f"[search] retrieval agent: {llm.name}", file=sys.stderr)
        hits, trace = agentic_search(
            CONFIG, store, embedder, llm, args.query,
            project=project, level=args.level, limit=args.limit,
        )
        print(
            f"[search] {trace.rounds} rounds, queries: {trace.queries}"
            + (f", sessions expanded: {trace.sessions}" if trace.sessions else ""),
            file=sys.stderr,
        )
        if trace.note:
            print(f"[search] agent note: {trace.note}", file=sys.stderr)
    if hits is None:
        hits = search(
            store, embedder, args.query,
            project=project,
            levels=(args.level,) if args.level else None,
            limit=args.limit,
        )
    for h in hits:
        score = f"{h['score']:.3f}" if h["score"] is not None else "  -  "
        print(f"[{score}] ({h['level']}/{h['kind']}) {h['project'] or '-'} "
              f"{(h['ts'] or '')[:10]}\n    {h['content'][:300]}\n")
    if not hits:
        print("no results")
    store.close()
    return 0


def _to_slug(project: str) -> str:
    """Accept a project slug, or a filesystem path (resolved and slugified).

    Slugs start with '-' (they are cwds with slashes replaced by dashes),
    which argparse rejects as a positional — paths like '.' sidestep that.
    """
    if "/" in project or project in (".", ".."):
        from pathlib import Path

        return str(Path(project).expanduser().resolve()).replace("/", "-")
    return project


def cmd_plan(args: argparse.Namespace) -> int:
    from . import plan
    from .store import MemoryStore

    project = _to_slug(args.project)
    store = MemoryStore(CONFIG.db_path)
    board = plan.render_board(store, project, include_done=args.all)
    print(board or f"no plan items for '{project}'")
    store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .store import MemoryStore

    store = MemoryStore(CONFIG.db_path)
    print(json.dumps(store.counts(), indent=2))
    store.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import run

    run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-memory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="Ingest new/changed session transcripts")
    p.add_argument("--limit", type=int, default=None, help="max sessions to process")
    p.add_argument("--no-llm", action="store_true", help="extractive-only (no LLM calls)")
    p.add_argument("--retry-extractive", action="store_true",
                   help="re-distill sessions previously degraded to extractive "
                        "digests (e.g. past model refusals)")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("search", help="Search the memory store")
    p.add_argument("query")
    p.add_argument("--project", default=None)
    p.add_argument("--level", default=None,
                   choices=["chunk", "memory", "session", "digest", "project", "plan"])
    p.add_argument("--limit", type=int, default=None,
                   help="max results (default: search.evidence_limit for agentic, "
                        "retrieval.default_limit for static)")
    p.add_argument("--static", action="store_true",
                   help="skip the retrieval agent; plain one-shot hybrid search")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("plan", help="Show a project's plan board")
    p.add_argument("project",
                   help="project slug or path — e.g. '.' for the current project "
                        "(slugs start with '-'; pass them after '--')")
    p.add_argument("--all", action="store_true", help="include done/dropped items")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("status", help="Show store statistics")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("serve", help="Run the MCP server (stdio)")
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
