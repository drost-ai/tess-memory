"""CLI: agentic-memory {ingest, reconcile-attribution, search, status, serve}."""

from __future__ import annotations

import argparse
import json
import sys

from .config import CONFIG


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _aggregate_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-aggregates", dest="refresh_aggregates", action="store_false",
                       help="queue dirty aggregates for a later refresh")
    group.add_argument("--refresh-aggregates", dest="refresh_aggregates", action="store_true",
                       help="refresh pending aggregates regardless of cadence")
    parser.set_defaults(refresh_aggregates=None)


def cmd_ingest(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .ingest import ingest
    from .llm import AgentCoreBackend
    from .store import MemoryStore

    cfg = CONFIG
    if getattr(args, "codex_only", False):
        from pathlib import Path
        from .codex import load_dedicated_config

        path = Path(args.settings).expanduser()
        try:
            cfg = load_dedicated_config(path)
        except (OSError, ValueError, TypeError) as exc:
            print(f"[codex] invalid dedicated settings: {exc}", file=sys.stderr)
            return 2
    cfg.ensure_dirs()
    store = MemoryStore(cfg.db_path)
    embedder = make_embedder(cfg)
    llm = None if args.no_llm else AgentCoreBackend(cfg, probe=not cfg.incremental_ingestion)
    fallback = None
    if llm is not None and cfg.agent_core_fallback_profile:
        # probe=False: no API call unless a refusal actually needs it
        fallback = AgentCoreBackend.for_role(cfg, "fallback")
    for ds in cfg.data_stores:
        print(f"[ingest] data_store {ds.name} ({ds.type}): {ds.path}", file=sys.stderr)
    print(f"[ingest] db={cfg.db_path}", file=sys.stderr)
    print(f"[ingest] llm={'none (extractive-only)' if llm is None else llm.name}", file=sys.stderr)
    if fallback is not None:
        print(f"[ingest] refusal fallback={fallback.name}", file=sys.stderr)
    try:
        stats = ingest(
            cfg, store, embedder, llm, fallback_llm=fallback,
            limit=args.limit, retry_extractive=args.retry_extractive,
            session_ids=set(args.session_id) if getattr(args, "session_id", None) else None,
            refresh_aggregates=getattr(args, "refresh_aggregates", None),
            aggregates_only=getattr(args, "aggregates_only", False),
            aggregate_levels=tuple(args.aggregate_levels) if getattr(args, 'aggregate_levels', None) else None,
        )
    except ValueError as exc:
        if not any(ds.type == "codex" for ds in cfg.data_stores):
            raise
        print(f"[codex] ingestion stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    counts = dict(vars(stats))
    if not any(ds.type == "codex" for ds in cfg.data_stores):
        counts.pop("codex_unchanged", None)
    print(json.dumps(counts, indent=2))
    return 1 if stats.errors else 0


def cmd_inspect_codex(args: argparse.Namespace) -> int:
    """Read and normalize locally; never construct a memory store or model."""
    from dataclasses import replace
    from pathlib import Path
    from .codex import CodexCatalog, CodexParseError
    from .config import DataStore

    stores = ([DataStore(name=f"codex-{i}", path=Path(p).expanduser(), type="codex")
               for i, p in enumerate(args.roots)] if args.roots else
              [ds for ds in CONFIG.data_stores if ds.type == "codex"])
    if not stores:
        print("Provide Codex home paths, or configure type='codex' data stores.", file=sys.stderr)
        return 2
    options = CONFIG.codex
    if args.tool_output_chars is not None:
        options = replace(options, tool_output_chars=args.tool_output_chars)
    catalog = CodexCatalog(stores, options)
    selected = list(catalog.sources.values())
    if args.session_id:
        raw_id = args.session_id.removeprefix("codex:")
        selected = [source for source in selected if source.session_id == raw_id]
        if not selected:
            catalog.errors[args.session_id] = "session was not discovered"
    if args.limit is not None:
        selected = selected[:args.limit]
    output = {"discovered": len(catalog.sources), "subagents_skipped": catalog.skipped_subagents,
              "warnings": catalog.warnings, "errors": dict(catalog.errors), "sessions": []}
    for source in selected:
        try:
            st, report = catalog.inspect(source.path, min_chars=0)
            output["sessions"].append({
                "source": str(source.path), "session_id": "codex:" + source.session_id,
                "project": st.project if st else source.store.resolve_project(source.project),
                "usable": st is not None, "turns": len(st.turns) if st else 0,
                "text_chars": sum(len(t.text) for t in st.turns) if st else 0,
                "user_turns": st.n_user if st else 0,
                "assistant_turns": st.n_assistant if st else 0,
                "normalization": report.as_dict(),
            })
        except (OSError, CodexParseError) as exc:
            output["errors"][str(source.path)] = str(exc)
    print(json.dumps(output, indent=2))
    return 1 if output["errors"] else 0


def cmd_search(args: argparse.Namespace) -> int:
    from .embed import make_embedder
    from .retrieval import search
    from .store import MemoryStore

    project = _to_slug(args.project) if args.project else None
    store = MemoryStore(CONFIG.db_path)
    if (args.limit is not None and args.limit <= 0) or not store.has_memories(
        project=project, levels=(args.level,) if args.level else None,
    ):
        print("no results (requested scope is empty)")
        store.close()
        return 0
    embedder = make_embedder(CONFIG)
    hits = None
    if CONFIG.search_agent and not args.static:
        from .llm import AgentCoreBackend
        from .search_agent import agentic_search

        llm = AgentCoreBackend.for_role(CONFIG, "search", probe=True)
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


def cmd_reconcile_attribution(args: argparse.Namespace) -> int:
    """Repair stored project metadata without constructing models."""
    from .attribution import reconcile_project_attribution
    from .store import MemoryStore

    CONFIG.ensure_dirs()
    store = MemoryStore(CONFIG.db_path)
    try:
        stats = reconcile_project_attribution(CONFIG, store, dry_run=args.dry_run)
        print(json.dumps(vars(stats), indent=2))
    finally:
        store.close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .store import MemoryStore

    store = MemoryStore(CONFIG.db_path)
    from .freshness import store_status
    print(json.dumps(store_status(store, CONFIG), indent=2))
    store.close()
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .audit import inventory
    from .config import load_config
    path = Path(args.settings).expanduser()
    if not path.is_file():
        raise ValueError("audit settings must exist")
    print(json.dumps(inventory(load_config(path)), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .evaluation import evaluate
    result = evaluate(Path(args.cases), vectors_path=Path(args.vectors) if args.vectors else None,
                      keyword_only=args.keyword)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0 if result["passed"] == result["cases"] else 1


def cmd_scheduler(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .scheduler import serve, run_once, worker, stop_run, service_status
    from .scheduler_config import load_service
    from .scheduler_queue import WorkQueue
    path = Path(args.config).expanduser()
    if args.action in ('install', 'start', 'stop', 'restart'):
        from .scheduler_install import install, service_control
        result = install(path, label=args.label, start=not args.no_start) if args.action == 'install' else service_control(args.action, label=args.label)
        print(json.dumps(result, indent=2))
        return 0
    if args.action == 'ingest-once':
        from .scheduler_once import ingest_once
        return ingest_once(path, until_caught_up=args.until_caught_up)
    if args.action == 'serve':
        serve(path)
        return 0
    if args.action == 'worker':
        return worker(path, args.run_id)
    if args.action == 'run':
        return run_once(path, args.instance, args.kind)
    config = load_service(path)
    if args.action == 'usage':
        from datetime import datetime, timezone
        from .scheduler import _gate
        def timestamp(value):
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                raise ValueError('usage timestamps must include a timezone, e.g. 2026-09-08T00:00:00Z')
            return parsed.timestamp()
        since = timestamp(args.since) if args.since else datetime.now(timezone.utc).timestamp() - 86400
        until = timestamp(args.until) if args.until else None
        if until is not None and until <= since:
            raise ValueError('usage --until must be later than --since')
        print(json.dumps(_gate(config, readonly=True).usage(since=since, until=until, instance=args.instance), indent=2))
        return 0
    if args.action == 'status':
        print(json.dumps(service_status(config), indent=2))
        return 0
    queue = WorkQueue(config)
    if args.action in ('enable', 'disable'):
        queue.set_enabled(args.instance, args.action == 'enable')
    elif args.action == 'retry':
        queue.request(args.instance, args.kind)
    elif args.action == 'stop-run':
        print(json.dumps({'stopped': stop_run(config, args.run_id, force=args.force)}))
        return 0
    print(json.dumps(service_status(config), indent=2))
    return 0


def cmd_evaluate_public(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .config import load_config
    from .public_evaluation import evaluate_public
    result = evaluate_public(load_config(Path(args.settings).expanduser()), Path(args.cases), workers=args.workers, limit=args.limit)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, indent=2))
    return 0 if result['passed'] == result['cases'] else 1


def cmd_observe(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .observation import observe
    report = observe(Path(args.config), Path(args.output), seconds=args.seconds, interval=args.interval)
    print(json.dumps({k: v for k, v in report.items() if k != 'latest'}, indent=2))
    return 0 if report['passed'] else 1


def _bind_user(cfg, user):
    """A per-process identity (e.g. the host's per-session server)."""
    if user:
        from .principals import principal_id
        cfg.identity_user = principal_id(user)


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    _bind_user(server.CONFIG, getattr(args, "user", None))
    server.run()
    return 0


def cmd_codex_serve(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .codex import load_dedicated_config

    try:
        cfg = load_dedicated_config(Path(args.settings))
    except (OSError, ValueError, TypeError) as exc:
        print(f"[codex] invalid dedicated settings: {exc}", file=sys.stderr)
        return 2
    from . import server, retrieval

    _bind_user(cfg, getattr(args, "user", None))
    server.CONFIG = cfg
    retrieval.CONFIG = cfg
    server.run()
    return 0


def cmd_users(args: argparse.Namespace) -> int:
    """Users, teams, project mappings and sharing, as the configured identity."""
    from . import admin
    from .store import MemoryStore

    store = MemoryStore(CONFIG.db_path)
    try:
        if CONFIG.identity_user:
            store.principal = store.resolve_principal(CONFIG.identity_user)
        action = args.action
        if action == "list":
            result = admin.list_users(store)
        elif action == "set-owner":
            admin.set_default_owner(store, args.user)
            result = {"default_owner": args.user}
        elif action == "add":
            admin.add_user(store, args.user, display_name=args.name, admin=args.admin)
            result = {"added": args.user}
        elif action == "disable":
            admin.disable_user(store, args.user)
            result = {"disabled": args.user}
        elif action == "team-add":
            admin.add_team(store, args.team, name=args.name)
            result = {"team": args.team}
        elif action == "member-add":
            admin.add_member(store, args.team, args.user, role=args.role)
            result = {"team": args.team, "member": args.user}
        elif action == "member-remove":
            admin.remove_member(store, args.team, args.user)
            result = {"team": args.team, "removed": args.user}
        elif action == "project-map":
            admin.map_project(store, _to_slug(args.project), team=args.team, owner=args.owner)
            result = {"project": _to_slug(args.project), "team": args.team, "owner": args.owner}
        elif action == "share":
            result = admin.share(store, args.target, None if args.visibility == "default" else args.visibility)
        else:  # who-can-see
            result = admin.who_can_see(store, args.memory_id)
    except (PermissionError, LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    print(json.dumps(result, indent=2))
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
    _aggregate_flags(p)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("codex-ingest", help="Ingest Codex into an explicitly configured separate store")
    p.add_argument("--settings", required=True, help="existing Codex-only settings.toml with an explicit data directory")
    p.add_argument("--limit", type=_positive_int, default=None)
    p.add_argument("--no-llm", action="store_true", help="skip distillation (embedding calls still run)")
    p.add_argument("--retry-extractive", action="store_true")
    p.add_argument("--session-id", action="append", help="ingest only this session; repeat to select a pilot batch")
    _aggregate_flags(p)
    p.set_defaults(fn=cmd_ingest, codex_only=True)

    p = sub.add_parser("refresh-aggregates", help="Refresh pending profiles/digests without scanning transcripts")
    p.add_argument('--aggregate-levels', nargs='+', choices=['project', 'digest'])
    p.set_defaults(fn=cmd_ingest, aggregates_only=True, refresh_aggregates=True,
                   no_llm=False, limit=None, retry_extractive=False)

    p = sub.add_parser("inspect-codex", help="Inspect cleaned Codex history without model calls or database writes")
    p.add_argument("roots", nargs="*", help="Codex homes, or their sessions/archived_sessions directories")
    p.add_argument("--session-id", help="inspect one session, with or without the codex: prefix")
    p.add_argument("--limit", type=_positive_int, default=None)
    p.add_argument("--tool-output-chars", type=_positive_int, default=None)
    p.set_defaults(fn=cmd_inspect_codex)

    p = sub.add_parser(
        "reconcile-attribution",
        help="Repair stored project metadata from config (no LLM or embeddings)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="report project drift without writing the database",
    )
    p.set_defaults(fn=cmd_reconcile_attribution)

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

    p = sub.add_parser("users", help="Users, teams, project mappings and sharing (runs as [identity] user)")
    actions = p.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="users, admins and their teams")
    a = actions.add_parser("set-owner", help="name the default owner (the user unowned rows belong to)")
    a.add_argument("user")
    a = actions.add_parser("add", help="add or update a user")
    a.add_argument("user")
    a.add_argument("--name", default=None)
    a.add_argument("--admin", action="store_true")
    a = actions.add_parser("disable", help="disable a user")
    a.add_argument("user")
    a = actions.add_parser("team-add", help="create a team")
    a.add_argument("team")
    a.add_argument("--name", default=None)
    a = actions.add_parser("member-add", help="add a user to a team")
    a.add_argument("team")
    a.add_argument("user")
    a.add_argument("--role", choices=["member", "admin"], default="member")
    a = actions.add_parser("member-remove", help="remove a user from a team")
    a.add_argument("team")
    a.add_argument("user")
    a = actions.add_parser("project-map", help="give a project to a team and/or name its owner")
    a.add_argument("project")
    a.add_argument("--team", default=None)
    a.add_argument("--owner", default=None)
    a = actions.add_parser("share", help="set visibility of memory:<id> or session:<id> (private|team|org|default)")
    a.add_argument("target")
    a.add_argument("visibility", choices=["private", "team", "org", "default"])
    a = actions.add_parser("who-can-see", help="who may read one memory")
    a.add_argument("memory_id", type=int)
    p.set_defaults(fn=cmd_users)

    p = sub.add_parser("serve", help="Run the MCP server (stdio)")
    p.add_argument("--user", default=None, help="read and write as this user (default: [identity] user, else the instance owner)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("audit-sources", help="Read-only source inventory and stored coverage; no model calls")
    p.add_argument("--settings", required=True)
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("evaluate", help="Evaluate private source-labelled retrieval cases read-only")
    p.add_argument("--cases", required=True)
    p.add_argument("--vectors", help="precomputed query vectors keyed by case ID (.npz); avoids model calls")
    p.add_argument("--keyword", action="store_true", help="offline keyword-only baseline; no embeddings or LLM calls")
    p.add_argument("--output", help="metrics report path (query/transcript text is omitted)")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser('evaluate-public', help='Send labelled public synthetic evidence fixtures to the configured retrieval model')
    p.add_argument('--settings', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=_positive_int, default=4)
    p.add_argument('--limit', type=_positive_int)
    p.set_defaults(fn=cmd_evaluate_public)

    p = sub.add_parser('observe', help='Record local rollout health without model calls')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', type=_positive_int, default=86400)
    p.add_argument('--interval', type=_positive_int, default=60)
    p.set_defaults(fn=cmd_observe)

    p = sub.add_parser("codex-serve", help="Serve an explicitly configured, dedicated Codex memory store")
    p.add_argument("--settings", required=True)
    p.add_argument("--user", default=None, help="read and write as this user (default: the instance owner)")
    p.set_defaults(fn=cmd_codex_serve)

    p = sub.add_parser('scheduler', help='Manage persistent ingestion and aggregate queues')
    p.add_argument('--config', required=True)
    commands = p.add_subparsers(dest='action', required=True)
    for action in ('serve', 'status', 'ingest-once'):
        command = commands.add_parser(action)
        if action == 'ingest-once':
            command.add_argument('--until-caught-up', action='store_true',
                                 help='keep running passes while sources wait, until every store is caught up')
        command.set_defaults(fn=cmd_scheduler)
    command = commands.add_parser('usage', help='Observed model tokens, retries and accounting coverage')
    command.add_argument('--instance')
    command.add_argument('--since', help='ISO timestamp with timezone; defaults to 24 hours ago')
    command.add_argument('--until', help='ISO timestamp with timezone; defaults to now')
    command.set_defaults(fn=cmd_scheduler)
    for action in ('install', 'start', 'stop', 'restart'):
        command = commands.add_parser(action)
        command.add_argument('--label', default='com.agentic-memory.scheduler')
        if action == 'install':
            command.add_argument('--no-start', action='store_true')
        command.set_defaults(fn=cmd_scheduler)
    for action in ('enable', 'disable', 'retry', 'run'):
        command = commands.add_parser(action)
        command.add_argument('--instance', required=True)
        if action in ('retry', 'run'):
            command.add_argument('--kind', choices=['ingest', 'profiles', 'digests', 'detect', 'recovery'], default='ingest')
        command.set_defaults(fn=cmd_scheduler)
    command = commands.add_parser('stop-run')
    command.add_argument('--run-id', required=True)
    command.add_argument('--force', action='store_true')
    command.set_defaults(fn=cmd_scheduler)

    p = sub.add_parser('scheduler-worker', help=argparse.SUPPRESS)
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.set_defaults(fn=cmd_scheduler, action='worker')

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
