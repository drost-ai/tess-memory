# Command line

Every command reads the settings file named by `TESS_MEMORY_SETTINGS`.

```sh
tess-memory <command> [options]
```

Inside the checkout, run it as `uv run tess-memory ...`.

## Everyday commands

| Command | What it does |
|---|---|
| `ingest` | Read new and changed sessions into memory |
| `search "<query>"` | Search memory |
| `plan <project-or-path>` | Print a project's plan board |
| `status` | What is stored, pending and failing |
| `serve` | Run the MCP server on stdio |

### `ingest`

```sh
tess-memory ingest                    # everything new or changed
tess-memory ingest --no-llm           # no distillation
tess-memory ingest --retry-extractive # distil sessions that have only an extractive summary
tess-memory ingest --limit 20
tess-memory ingest --skip-aggregates  # leave profiles and digests for later
```

### `search`

```sh
tess-memory search "decision on idempotency keys for the billing cutover"
tess-memory search "..." --project -work-billing --level memory --limit 5
tess-memory search "..." --static     # plain search, no retrieval agent
```

### `plan`

```sh
tess-memory plan .         # the project for the current directory
tess-memory plan . --all   # include complete and dropped items
```

### `serve`

```sh
tess-memory serve
tess-memory serve --user ana    # bind this server to one user
```

## Maintenance

| Command | What it does |
|---|---|
| `refresh-aggregates` | Refresh pending project profiles and weekly digests, without reading sources |
| `audit-sources` | A read-only inventory of sources against what is in memory. No model calls. |
| `reconcile-attribution` | Move sessions to the right project after a configuration change. No model calls. |

`reconcile-attribution --dry-run` reports what would move.

## Codex

| Command | What it does |
|---|---|
| `inspect-codex <home>...` | Show what would be read from Codex homes. No model calls, no writes. |
| `codex-ingest --settings <file>` | Ingest into a store dedicated to Codex |
| `codex-serve --settings <file>` | Serve a store dedicated to Codex |

The dedicated commands refuse to run against a store that holds anything other than Codex
sessions. A store that mixes Codex with other sources uses plain `ingest` and `serve`.

## Users and teams

```sh
tess-memory users list
tess-memory users set-owner <user>
tess-memory users add <user> [--name "..."] [--admin]
tess-memory users disable <user>
tess-memory users team-add <team>
tess-memory users member-add <team> <user>
tess-memory users member-remove <team> <user>
tess-memory users project-map <project> [--team <team>] [--owner <user>]
tess-memory users share memory:<id>|session:<id> private|team|org|default
tess-memory users who-can-see <memory-id>
```

See [Teams and visibility](teams.md).

## Scheduler

```sh
tess-memory scheduler --config <file> <action>
```

| Action | What it does |
|---|---|
| `serve` | Run the scheduler in the foreground |
| `install`, `start`, `stop`, `restart` | Manage the background service (macOS) |
| `status` | Running, queued and failing work |
| `run --instance <store> [--kind ...]` | One pass now |
| `retry --instance <store> [--kind ...]` | Retry failed work |
| `enable`, `disable` `--instance <store>` | Turn scheduling for a store on or off |
| `ingest-once [--until-caught-up]` | One pass over every store, or passes until nothing is left |
| `usage` | Observed model usage |
| `stop-run --run-id <id>` | Stop one running job |

`--kind` is one of `ingest`, `profiles`, `digests`, `detect`, `recovery`.

See [Scheduler](scheduler.md).

## Evaluation

| Command | What it does |
|---|---|
| `evaluate --cases <file>` | Score retrieval against your own labelled cases, read-only |
| `evaluate-public` | Run public synthetic fixtures through the configured retrieval model |
| `observe` | Record service health over time, without model calls |

## Exit codes

Commands exit nonzero when errors remain, for example when a session could not be
distilled by either model. A run that preserved existing memory after a failure still
reports the error.
