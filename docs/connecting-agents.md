# Connecting agents

Tess Memory is served over the [Model Context Protocol](https://modelcontextprotocol.io),
so any agent that speaks MCP can use it. Connecting an agent has two halves:

1. **Read its sessions**: add the agent's transcript folder as a [source](sources.md), so
   what it does becomes memory.
2. **Give it the tools**: register the MCP server, so it can recall and plan.

An agent can have either half without the other. An agent with only the tools can use
memory other agents produced, and can keep a plan.

[`tess-memory setup`](setup.md) does both for Claude Code, Codex and Antigravity from its
page, and also adds the usage instructions to each agent's instruction file. The rest of
this page is for doing it by hand, and for other MCP clients.

One store can hold several agents' sessions at once. Claude Code and Codex in the same
store is the usual way to give them one shared memory, and the scheduler keeps such a store
current like any other.

## Claude Code

Source:

```toml
[[data_stores]]
name = "claude-code"
type = "claude-code"
path = "~/.claude/projects"
```

Server:

```sh
claude mcp add --scope user tess-memory \
  --env TESS_MEMORY_SETTINGS=$HOME/.tess-memory/settings.toml \
  -- uv run --directory /path/to/tess-memory tess-memory serve
```

## Codex

Source:

```toml
[[data_stores]]
name = "codex"
type = "codex"
path = "~/.codex"
```

Server, in `~/.codex/config.toml`:

```toml
[mcp_servers.tess-memory]
command = "uv"
args = ["run", "--directory", "/path/to/tess-memory", "tess-memory", "serve"]
env = { TESS_MEMORY_SETTINGS = "/home/you/.tess-memory/settings.toml" }
```

## Antigravity

Source, one for each Antigravity data folder that holds conversations (`antigravity-cli`
is the `agy` command's):

```toml
[[data_stores]]
name = "antigravity-cli"
type = "antigravity"
path = "~/.gemini/antigravity-cli"
```

Server. Flags go before the server's name:

```sh
agy mcp add --env TESS_MEMORY_SETTINGS=$HOME/.tess-memory/settings.toml tess-memory \
  -- uv run --directory /path/to/tess-memory tess-memory serve
```

This writes the server to `~/.gemini/config/mcp_config.json`, Antigravity's configuration
for every session.

## Any other MCP client

```json
{
  "mcpServers": {
    "tess-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/tess-memory", "tess-memory", "serve"],
      "env": { "TESS_MEMORY_SETTINGS": "/home/you/.tess-memory/settings.toml" }
    }
  }
}
```

## Agents built on Tess Core

Add the server to the session's MCP servers, and add the runtime home's `sessions` folder
as a `tess-core` source. See [Sources](sources.md).

## One memory for all your agents

This is the configuration that gives you continuity *between* agents. List every agent's
transcripts as sources of one store, and register the same server, with the same settings
file, in every agent:

```toml
[[data_stores]]
name = "claude-code"
type = "claude-code"
path = "~/.claude/projects"

[[data_stores]]
name = "codex"
type = "codex"
path = "~/.codex"

[storage]
data_dir = "~/.tess-memory"
```

Now a search from Codex finds what was decided in Claude Code, `get_project_context`
describes the project the same way to both, and both read and write one plan board.

Two things make this work well:

- **Projects line up by themselves.** A project is named after the directory the session
  ran in, so two agents working in the same repository share a project without any
  configuration.
- **The plan is shared immediately.** Plan items are written straight to the store. One
  agent can leave a task, and the next agent, in a different tool, sees it at once.

Memory from sessions appears after the next ingestion. Run `ingest` by hand, or let the
[scheduler](scheduler.md) do it every few minutes.

## A memory per agent

Sometimes you want separation: a work store and a personal one, or an assistant that
should not see your coding history. Give each its own settings file and data directory,
and point each agent at its own:

```toml
# ~/.assistant/memory.toml
[[data_stores]]
name = "assistant"
type = "tess-core"
path = "~/.assistant/sessions"
project = "assistant"

[planning]
default_project = "assistant"
lock_writes_to_default = true

[storage]
data_dir = "~/.assistant/memory"
```

Here `project = "assistant"` gives the assistant one memory wherever it runs, and the
`[planning]` section keeps its plan on one board.

If you would rather keep Codex in a store of its own, there is a dedicated pair of
commands for that, `codex-ingest` and `codex-serve`. They refuse to start against a store
that contains anything else, which protects you from pointing them at the wrong database.
You do not need them to use Codex in a shared store.

## Teaching the agent to use it

MCP servers describe their tools, but agents follow their instruction files more reliably.

`tess-memory setup` does this when it connects an agent: it adds a "Memory and planning"
section to the agent's instruction file for every project: `CLAUDE.md` in a Claude Code
home, `AGENTS.md` in a Codex home and `~/.gemini/GEMINI.md` for Antigravity. The section is
added once at the end of the file. A
file that already has it is left exactly as it is, and a file that is changed is first
copied beside itself. The full text is in
[`src/tess_memory/setup/guidance.md`](../src/tess_memory/setup/guidance.md), and setup
shows it under "The instructions agents are given" for pasting into any other agent.

If you connect an agent by hand, a few lines are enough:

```markdown
## Memory
- At the start of work in a project, call `get_project_context` to recover its profile,
  recent work and open plan. Continue planned work when it matches my request.
- When I refer to past work, decisions or an error we may have seen, call `search_memory`
  before investigating again. Write a self-contained query.
- When I ask you to plan or to remember work, record it with `plan_add`.
- Keep plan status current with `plan_update` at the moment it changes.
```

## Several people

To have a team share one store, each person's server is bound to their user, and each
source says whose sessions it holds. See [Teams and visibility](teams.md).
