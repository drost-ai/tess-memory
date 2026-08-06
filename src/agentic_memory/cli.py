"""CLI: agentic-memory {ingest, search, status, serve}."""

from __future__ import annotations

import argparse
import json
import sys

from .config import CONFIG


def cmd_ingest(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .ingest import ingest
    from .llm import NullBackend, make_backend
    from .store import MemoryStore

    CONFIG.ensure_dirs()
    store = MemoryStore(CONFIG.db_path)
    embedder = make_embedder(CONFIG)
    llm = NullBackend() if args.no_llm else make_backend(CONFIG)
    print(f"[ingest] source={CONFIG.source_dir}", file=sys.stderr)
    print(f"[ingest] db={CONFIG.db_path}", file=sys.stderr)
    print(f"[ingest] llm backend={llm.name}", file=sys.stderr)
    stats = ingest(
        CONFIG, store, embedder, llm,
        limit=args.limit, use_llm=not args.no_llm,
    )
    print(json.dumps(vars(stats), indent=2))
    store.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .retrieval import search
    from .store import MemoryStore

    store = MemoryStore(CONFIG.db_path)
    hits = search(
        store, make_embedder(CONFIG), args.query,
        project=args.project,
        levels=(args.level,) if args.level else None,
        limit=args.limit,
    )
    for h in hits:
        print(f"[{h['score']:.3f}] ({h['level']}/{h['kind']}) {h['project'] or '-'} "
              f"{(h['ts'] or '')[:10]}\n    {h['content'][:300]}\n")
    if not hits:
        print("no results")
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
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("search", help="Search the memory store")
    p.add_argument("query")
    p.add_argument("--project", default=None)
    p.add_argument("--level", default=None,
                   choices=["chunk", "memory", "session", "digest", "project"])
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("status", help="Show store statistics")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("serve", help="Run the MCP server (stdio)")
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
