# Getting started

This guide takes you from nothing to an agent that remembers. The first half runs entirely
on your machine and takes a few minutes. The second half adds a model, which is what turns
an index of transcripts into distilled memory.

## Requirements

- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- macOS or Linux
- An agent that writes session transcripts. This guide uses Claude Code; see
  [Sources](sources.md) for Codex and others.

## 1. Install

```sh
git clone https://github.com/drost-ai/tess-memory.git
cd tess-memory
uv sync
```

## 2. Write a settings file

Tess Memory reads one settings file. Keep it outside the checkout, for example at
`~/.tess-memory/settings.toml`:

```toml
[[data_stores]]
name = "claude-code"
type = "claude-code"
path = "~/.claude/projects"

[storage]
data_dir = "~/.tess-memory"

[embedding]
provider = "local"
allow_download = true

[search]
agent = false
```

This says: read Claude Code's sessions, keep the memory in `~/.tess-memory`, compute
embeddings on this machine, and use plain search. `allow_download` lets the small local
embedding model be fetched the first time; after that it runs offline.

Tell Tess Memory where the file is:

```sh
export TESS_MEMORY_SETTINGS=~/.tess-memory/settings.toml
```

## 3. Ingest your history

```sh
uv run tess-memory ingest --no-llm
```

`--no-llm` skips distillation. Every session is still read, split into searchable chunks,
embedded and summarised extractively, so you have a working memory that never left your
machine.

Check what it found:

```sh
uv run tess-memory status
```

## 4. Search it

```sh
uv run tess-memory search "why does checkout fail for an empty cart"
```

Results come back ranked, each with its level, project and date:

```
[0.883] (chunk/chunk) -work-shop-api 2026-11-10
    [user] The checkout endpoint returns 500 when the cart is empty. Find the cause...
    [assistant] I found it: cart_total() divides by the item count...
```

## 5. Give it to your agent

Register the MCP server with Claude Code:

```sh
claude mcp add --scope user tess-memory \
  --env TESS_MEMORY_SETTINGS=$HOME/.tess-memory/settings.toml \
  -- uv run --directory /path/to/tess-memory tess-memory serve
```

Start a new session and ask about past work. The agent now has
[twelve tools](reference/tools.md) for recalling history and keeping a plan.

To make it a habit, add a few lines to your agent's instruction file (`CLAUDE.md`,
`AGENTS.md`):

```markdown
## Memory
- At the start of work in a project, call `get_project_context`.
- When I refer to past work, call `search_memory` before investigating again.
- When I ask you to plan, write the plan with `plan_add` and keep status live with `plan_update`.
```

See [Connecting agents](connecting-agents.md) for Codex and other clients.

## 6. Add a model

So far the memory holds what was said. A model is what turns that into what was *learned*:
decisions with their reasons, solutions, problems, your preferences, a profile of each
project and a digest of each week.

Tess Memory uses the model providers of [Tess Core](https://github.com/drost-ai/tess-core).
Declare them once in a provider file, for example `~/.tess-memory/providers.yaml`:

```yaml
schema_version: 1
providerProfiles:
  distil:
    kind: anthropic
    defaultModel: <a capable model>
    auth: { kind: anthropic_api_key, envVar: ANTHROPIC_API_KEY }
```

Point the settings file at it:

```toml
[llm]
settings = "~/.tess-memory/providers.yaml"
profile = "distil"

[search]
agent = true
profile = "distil"
```

Then ingest again, this time with distillation:

```sh
uv run tess-memory ingest
```

Sessions already indexed without a model are upgraded with `ingest --retry-extractive`.
See [Models and embeddings](models.md) for fallbacks, cheaper models and local models.

## 7. Keep it current

Run `ingest` whenever you like: it only processes sessions that are new or have changed.
To have it happen without you, turn on the [scheduler](scheduler.md).

## Next

- [Concepts](concepts.md): the vocabulary
- [The plan board](plan-board.md): the feature that changes how long projects go
- [Privacy and data](privacy.md): exactly what is stored and what leaves your machine
