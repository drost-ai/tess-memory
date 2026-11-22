# Concepts

A short vocabulary. Everything else in the documentation builds on these.

## Store

A **store** is one memory: a directory holding a SQLite database (`memory.db`) and its
caches. A store is described by one settings file. Everything in a store can be searched
together, so what you put in one store is what your agents share.

You can run one store for everything, or several: one per agent, one per team, one for a
personal AI. Stores never mix unless you configure them to.

## Source

A **source** is a place where an agent writes its session transcripts. Tess Memory reads
sources; it never writes to them. Each source has a type that says how to read it:

| Type | Reads |
|---|---|
| `claude-code` | Claude Code session files |
| `codex` | Codex session rollouts, including archived ones |
| `antigravity` | Google Antigravity conversations (the `agy` command) |
| `tess-core` | Sessions of any agent built on Tess Core |
| `transcript` | A folder of plain transcript files |

See [Sources](sources.md).

## Session

A **session** is one conversation between a person and an agent, from one source. Sessions
are the unit of ingestion: a session is read, distilled and published as a whole, and
replaced as a whole when it changes.

## Project

A **project** is the scope that memory and plans belong to. By default it is derived from
the directory a session ran in, so work in `/work/shop-api` belongs to the project
`-work-shop-api`. A source can instead be bound to one fixed project, which suits an
assistant that should have a single memory wherever it runs.

`list_projects` is the authority on project names.

## Memory levels

One session produces memory at several levels, and time adds more:

| Level | What it holds | Answers |
|---|---|---|
| `chunk` | What was said and run, verbatim, in windows | "What was the exact command?" |
| `memory` | One fact, decision, solution, problem, preference or task state | "What did we decide, and why?" |
| `session` | The summary and outcome of one session | "What happened on Tuesday?" |
| `digest` | A summary of one week | "What did we get done last week?" |
| `project` | A rolling profile of a project | "What is this project and where is it?" |
| `plan` | Items on the plan board | "What is next?" |
| `image` | A description of an image someone sent an agent | "Which screenshot showed that error?" |

Each `memory` has a **kind** (`decision`, `fact`, `preference`, `solution`, `problem`,
`task_state`) and an **importance** from 1 to 10 that the distilling model assigns.
Importance is part of ranking, so a decision that changes how future work must be done
outranks a passing remark.

## Distillation

**Distillation** is the step where a model reads a session and writes its summary and
memories. Long sessions are split into segments, distilled separately and merged. If no
model is available, the session still gets an **extractive** summary and its chunks, and
can be distilled later.

## Retrieval

**Retrieval** scores candidates by four signals: meaning (embedding similarity), keywords,
importance and recency. Two modes use it:

- **static**: one search, no model.
- **agentic**: a bounded retrieval agent that searches, judges coverage, reformulates and
  follows links between sessions, then returns curated evidence.

Neither mode writes an answer. They return evidence; the calling agent, which has the
conversation, does the reasoning. See [Searching memory](searching.md).

## The plan board

Everything above is a record of the past. The **plan board** is the opposite: state the
agent writes on purpose, about the future. Items are a `vision`, `epic`, `slice`, `task`
or `note`, nested as a tree, each with a status and a dated history. See
[The plan board](plan-board.md).

## Personal memory

An optional, separate memory for a personal AI: who it is, who its person is, and what it
has come to know about the people in its life. It is kept apart from coding memory and is
off unless you enable it. See [Personal memory](personal-memory.md).

## User, team, visibility

A store can serve several people. Every entry has an owner and a visibility (`private`,
`team` or `org`), and every reader is bound to a user when the server starts. With no
identity configured, a store behaves as a single-user store. See
[Teams and visibility](teams.md).
