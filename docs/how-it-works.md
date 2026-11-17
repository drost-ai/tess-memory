# How it works

This page follows one session from a transcript on disk to an answer in a later
conversation. You do not need it to use Tess Memory; it is here so you can judge it.

## The pipeline

```
sources ──▶ read ──▶ distil ──▶ embed ──▶ publish ──▶ retrieve ──▶ MCP
```

## 1. Read

Each run looks at the configured sources and decides which sessions need work. A session
that has not changed is skipped without reading its contents again, and without any model
or embedding call.

Reading cleans a transcript before anything else sees it: harness noise is stripped, image
payloads are replaced by a small marker, long tool output is clipped to its head and tail,
and sub-agent side conversations are left out because the parent session already carries
their results. The original file is never modified.

With incremental ingestion on, a **source journal** records which byte ranges of a file
have been read and verified. When a session grows, only the new, completed part is
processed. An exchange that is still in progress waits until it finishes.

## 2. Distil

A model reads the cleaned session and returns:

- a title, a summary and an outcome for the session,
- a list of atomic memories, each with a kind, an importance from 1 to 10 and tags.

A session too long for one request is split into overlapping segments. Segments are
distilled independently, in parallel if you allow it, and merged in order. Results are
checkpointed as they complete, so an interrupted run resumes instead of starting again,
and identical requests are never paid for twice.

If the primary model refuses or fails, the fallback model is tried for that request. If
both fail, an existing distilled session is left exactly as it was, and a new session gets
an extractive summary so it is still searchable.

## 3. Embed

Chunks, memories, summaries and plan items are embedded, by a hosted embedding model or a
local one. A store holds one vector space: it records which model produced its vectors and
never mixes in another. Moving a store to another embedding model is done in place, with
`tess-memory reembed`.

## 4. Publish

A session is replaced in one transaction: its summary, memories, chunks and vectors, the
record of how far its source has been read, and the markers that say which project profile
and weekly digest are now stale. The transaction runs only after embedding has succeeded.
If anything fails, nothing changes.

Before committing, the source is checked again. If the file changed while the model was
working, the result is not published and the session is picked up on the next run.

## 5. Roll up

Project profiles and weekly digests are built from session summaries. They refresh on
their own cadence, separately from ingestion, because a profile does not need rewriting
every time a session ends. A failed refresh keeps the previous profile and stays queued.

## 6. Retrieve

A search first narrows the scope in SQL (project, level, kind, dates), then scores
candidates:

```
score = 0.55 × meaning + 0.20 × keywords + 0.15 × importance + 0.10 × recency
```

Recency decays with a 30-day half-life. At most three results come from any one session,
so one long conversation cannot crowd out the rest. The weights are settings.

Vector search is exact, not approximate: small scopes are cached in memory up to a byte
budget and large scopes are streamed in batches, so results do not depend on an index
being rebuilt.

In **agentic** mode a retrieval agent runs inside the server. It starts from a static
search, judges whether the evidence covers the question, issues reformulated sub-queries,
follows links between sessions and returns the evidence it kept. It has a fixed budget of
rounds and queries. If its model fails, the static results are returned.

## 7. Serve

The MCP server is a process your agent starts over stdio. It opens the store, answers tool
calls and exits with the session. Several servers, from several agents, can use one store
at once: SQLite handles the concurrency, writers take short transactions, and each process
notices what the others commit.

Plan writes are immediate. They do not wait for ingestion, and an agent that writes a plan
item can read it back in the same turn.

## What makes it dependable

- **Idempotent.** Running ingestion twice does the work once.
- **Resumable.** Checkpoints survive a crash, a restart or a killed process.
- **Last-good.** A failure never replaces good memory with worse memory.
- **Verified.** Progress is recorded against the content that was read, not a timestamp.
- **Bounded.** Every run has limits on sources, model calls and time.
- **One writer per store.** Ingesters hold a lock; servers read and write plans alongside.
