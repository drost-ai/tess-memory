<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tess-memory-dark.svg">
    <img src="docs/assets/tess-memory-light.svg" alt="Tess Memory" height="72">
  </picture>
</h1>

**The context engine for agents.**

Your agents forget everything the moment a session ends. Tess Memory gives them one
long-term memory: what was decided, what was built, what is planned and who they are working
with. Shared across every agent you use, served over MCP, stored on your own machine.

[Quickstart](#quickstart) · [What you get](#what-you-get) · [How it works](#how-it-works) · [Built to be trusted](#built-to-be-trusted) · [Documentation](https://continuityresearch.ai/tess-memory/docs)

Works with Claude Code, Codex and any MCP client · Local SQLite · Python 3.12+ · Apache-2.0

</div>

---

Every session with a coding agent starts the same way. You explain the project again. The
agent re-reads the codebase. It rediscovers the decision you made last Tuesday, or worse,
makes the opposite one. The work you did together yesterday is gone, and the plan you agreed
on lives in a chat window nobody will open again.

Tess Memory ends that. It reads the transcripts your agents already write, distils them into
memories at several timescales, and hands the right ones back when they are needed:

```
You:    Pick up the billing migration where we left off.

Agent:  [get_project_context]  [plan_board]
        We finished the schema change and the backfill on Thursday. Two tasks are open:
        the dual-write cutover (in progress, blocked on the idempotency key decision) and
        the reconciliation report. You chose UUIDv7 keys on Friday, so the cutover is
        unblocked. Starting there.
```

No re-reading. No re-explaining. The agent already knows.

## What you get

### 1. Continuity between agents

Point Claude Code, Codex and any other MCP-capable agent at the same store and they share
one memory. What you debugged in one tool this morning is known to the other this afternoon.
Switch agents, switch models, switch machines: the context comes with you instead of staying
locked inside one vendor's chat history.

### 2. Continuity within an agent

Sessions end; projects do not. Tess Memory keeps the whole history of a project (decisions
and their reasons, solutions, problems, your preferences, the exact commands that worked) and
a **plan board** the agent maintains itself: vision, epics, slices, tasks and notes, each
with a status and a dated history. An agent can plan a month of work, be interrupted a
hundred times, and start every session knowing what is done, what is in progress and what
comes next.

### 3. Memory for a personal AI

For an assistant that is meant to know *you*, Tess Memory adds a separate personal memory:
authored foundation documents (its soul, its identity, its person, the memories essential to
remaining itself) and living records of people, relationships, experiences and reflections.
It is what lets a personal AI keep its voice and its relationship with you across months,
instead of meeting you for the first time every morning.

## Quickstart

It runs on your machine, and you set it up in the browser. No settings files to write.

```sh
git clone https://github.com/drost-ai/tess-memory.git
cd tess-memory
uv sync
uv run tess-memory setup
```

`setup` opens a page on your own machine and walks you through five steps:

1. **Sources.** It finds the Claude Code and Codex sessions already on your disk. Tick the
   ones to remember.
2. **Model.** Sign in with your ChatGPT account, paste an Anthropic or OpenAI API key, or
   point it at any OpenAI-compatible endpoint, including a model on your own hardware.
   It is tested with a real request before you move on.
3. **Embeddings.** Use a small model on this machine (downloaded once, then offline),
   Google, or any OpenAI-compatible embeddings service.
4. **Connect agents.** One click registers the memory server with each agent.
5. **First run.** It ingests a first batch of your past sessions while you watch, then
   keeps ingesting in the background: the rest of your history, and every new session.

Keys go into your system keychain, not into a file. The page is served on `127.0.0.1`
only and needs the one-time link printed in your terminal.

Then start a new session in a connected agent and ask it to pick up where you left off.

Setup writes ordinary files you can read and edit: a settings file and a provider file in
`~/.tess-memory`. To write them by hand instead, see
[Getting started](docs/getting-started.md) and
[Configuration](docs/configuration.md). To keep the store current without thinking about
it, [turn on the scheduler](https://continuityresearch.ai/tess-memory/docs/scheduler).

## What an agent can do with it

Twelve tools, served over MCP:

| Tool | What it is for |
|---|---|
| `get_project_context` | Orient in a project: its rolling profile, the open plan and recent sessions |
| `search_memory` | "Have we solved this before?" Hybrid search, or a retrieval agent for harder questions |
| `recent_activity` | "Where did I leave off?" Recent work across every project |
| `get_session` | One past session in full: summary, outcome and its memories |
| `get_session_transcript` | The original transcript, page by page, when exact evidence matters |
| `list_projects` | Every project the memory knows |
| `plan_add`, `plan_update` | Write the plan and keep its status live |
| `plan_board`, `plan_get` | Read the plan as a tree, or one item with its history |
| `memory_share` | Share a memory with your team, or take it back |
| `memory_status` | What is stored, what is pending and what failed |

## How it works

```
 Claude Code      Codex       Tess Core agents      any transcript
      └─────────────┴──────┬────────┴────────────────────┘
                           ▼
                 read      only new or changed sessions, verified by content
                           ▼
                 distil    a model extracts decisions, facts, solutions,
                           problems, preferences and task state, each rated
                           for importance
                           ▼
                 embed     hosted or local embeddings
                           ▼
        ┌──────────────────────────────────────────────────────────┐
        │  one SQLite file on your machine                          │
        │                                                           │
        │  chunk     what was said and run, verbatim                │
        │  memory    atomic facts, decisions and solutions          │
        │  session   what one session achieved                      │
        │  digest    what a week achieved                           │
        │  project   a rolling profile of each project              │
        │  plan      the board your agents write and maintain       │
        └──────────────────────────────────────────────────────────┘
                           ▼
                 retrieve  meaning + keywords + importance + recency
                           ▼
                 MCP       any agent, any model
```

**Memory at several timescales.** A question about an error message needs the exact command
from three weeks ago. A question about direction needs the summary of the quarter. Tess
Memory keeps both and everything between, and retrieval draws on all of them.

**Retrieval that thinks.** `search_memory` can run a bounded retrieval agent: it searches,
judges what is missing, reformulates, follows links between sessions and returns curated
evidence. It never invents an answer; it returns what was actually recorded, with sources.
A plain one-shot hybrid search is always available and needs no model.

**A plan that survives.** The plan board is not extracted from transcripts. It is authored:
the agent writes it, updates it as work moves, and reads it at the start of the next
session. Items are never deleted, every status change is dated, and the whole board is
searchable like any other memory.

## Built to be trusted

A memory you cannot rely on is worse than none. These are properties of the system, not
aspirations:

- **Your data stays yours.** One SQLite file in a directory you choose. No hosted service,
  no account, no telemetry. With local embeddings and `--no-llm`, nothing leaves the machine.
- **Evidence, not hearsay.** Every memory traces to the session it came from, and
  `get_session_transcript` reads the original transcript back, unmodified.
- **It never loses what it has.** If a model fails, refuses or returns garbage, the
  last good memory is kept and the work is retried. Writes are transactional.
- **You never pay twice.** Ingestion is incremental and content-verified. An unchanged
  session costs nothing to re-check, and no change of model, settings or version
  re-processes your history. Even the embedding model can be changed in place.
- **The plan cannot be wiped.** It is authored, not derived, so nothing could rebuild it.
  No command deletes it, and the database itself refuses to.
- **Bounded cost.** Per-run limits on sources and model calls, shared rate limits across
  workers, quiet periods so an active conversation is not distilled mid-thought.
- **Private by default, shareable by choice.** In a team store, distilled knowledge and the
  plan are visible to the project's team; raw transcripts, session summaries and personal
  preferences stay with their author.
- **Model-independent.** Distillation, retrieval and embeddings each take any provider you
  configure, with a fallback. Run it on a frontier model, a cheap one, or one on your own
  hardware.
- **Tested.** More than 600 tests cover failure recovery, concurrent writers, incremental
  correctness and retrieval behaviour, all runnable offline.

## How it compares

| | Tess Memory | Notes files in the repo | A vector database | Built-in chat memory |
|---|---|---|---|---|
| Works across agents and vendors | Yes, over MCP | Per tool | You build it | No |
| Captures what happened without being asked | Yes, from transcripts | No, someone must write it | You build it | Partly |
| Decisions, plans and raw evidence together | Yes | Whatever was written down | Chunks only | Opaque |
| A plan the agent maintains | Yes | No | No | No |
| Where the data lives | Your disk | Your repo | Your infrastructure | The vendor |
| Inspectable and correctable | Yes, one SQLite file | Yes | Yes | Rarely |

## In daily use

Tess Memory is not a prototype. It is the memory behind the agents that build it. Claude
Code, Codex and Tess Code feed stores holding more than 6,000 sessions and 540,000 memories
across 2,000 projects, with 2,400 plan items written by the agents themselves, kept current
by more than 128,000 unattended scheduler runs.

It is also the memory of a personal AI that has kept one continuous identity and one
relationship with its person across model changes and months of conversation.

## Documentation

The full documentation, with search, is at
**[continuityresearch.ai/tess-memory/docs](https://continuityresearch.ai/tess-memory/docs)**.
The same pages live in this repository:

- [Getting started](docs/getting-started.md): install, ingest, connect an agent
- [Concepts](docs/concepts.md): stores, sources, projects, memory levels, the plan
- [How it works](docs/how-it-works.md): the pipeline from transcript to answer
- [Connecting agents](docs/connecting-agents.md): Claude Code, Codex, any MCP client, shared stores
- [Searching memory](docs/searching.md): the tools, search modes, writing good queries
- [The plan board](docs/plan-board.md): planning and executing long projects
- [Sources](docs/sources.md): Claude Code, Codex, Tess Core and plain transcripts
- [Personal memory](docs/personal-memory.md): memory for a personal AI
- [Teams and visibility](docs/teams.md): several people, one store
- [Models and embeddings](docs/models.md): choosing providers, running offline
- [Ingestion](docs/ingestion.md): incremental processing, failure handling, cost
- [Scheduler](docs/scheduler.md): keeping memory current automatically
- [Configuration](docs/configuration.md): the settings file and environment overrides
- [Command line](docs/cli.md)
- [Privacy and data](docs/privacy.md): what is stored, where, and what leaves your machine
- [MCP tool reference](docs/reference/tools.md)

## Requirements

- Python 3.12 or 3.13, and [uv](https://docs.astral.sh/uv/)
- macOS or Linux
- For distilled memories: a model from any provider supported by
  [Tess Core](https://github.com/drost-ai/tess-core) (Anthropic, OpenAI, any
  OpenAI-compatible endpoint, or a local model)
- For embeddings: a hosted embedding model, or the built-in local one

## Status

Version 0.1. The storage format is migrated forward automatically, and the MCP tool names
and arguments are stable. Configuration keys may still change before 1.0; changes are noted
in the release notes.

## License

[Apache License 2.0](LICENSE).

Created by Migel Tissera at [Continuity Research](https://continuityresearch.ai).
