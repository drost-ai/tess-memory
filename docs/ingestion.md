# Ingestion

Ingestion is how sessions become memory. It is designed around one promise: **run it as
often as you like, and it does only the work that is new.**

```sh
tess-memory ingest
```

## What a run does

1. Looks at every source and finds sessions that are new or have changed.
2. Reads and cleans them.
3. Distils them with a model (unless `--no-llm`).
4. Embeds the results.
5. Publishes each session in one transaction.
6. Marks the affected project profiles and weekly digests for refresh.

A session that has not changed is skipped. An unchanged store costs no model calls and no
embedding calls to re-check.

## Useful flags

| Flag | Effect |
|---|---|
| `--no-llm` | Skip distillation. Sessions get chunks and an extractive summary. |
| `--retry-extractive` | Distil sessions that only have an extractive summary |
| `--limit N` | Process at most N sessions |
| `--skip-aggregates` | Leave profiles and digests for later |
| `--refresh-aggregates` | Refresh every pending profile and digest now |

`tess-memory refresh-aggregates` refreshes profiles and digests without looking at sources
at all.

## Waiting for a conversation to finish

Distilling a session while you are still in it wastes a model call on half a thought:

```toml
[ingest]
quiet_seconds = 900        # process a session after 15 minutes without a write
```

A recently written session is noticed and left pending. It is processed once it goes
quiet.

## Incremental ingestion

For long-lived sessions, turn on completed-prefix ingestion:

```toml
[ingest]
incremental = true
segment_workers = 16            # parallel segments per session
max_model_calls_per_run = 128   # 0 = unlimited
cache_retention_days = 30
```

A journal records which parts of each source file have been read and verified. When a
session grows, only the new, finished part is distilled, and the earlier work is reused.
An exchange still in progress stays pending until it completes. If a source is rewritten,
truncated or compacted, the affected state is invalidated and rebuilt.

With a call budget set, a run that reaches it stops cleanly and the next run continues
where it left off.

## When things go wrong

Models fail, refuse, time out and return malformed output. The rules are simple:

- **A failure never destroys memory.** If distillation fails on both the primary and the
  fallback model, a session that was already distilled keeps its memories, its vectors and
  its record of what was read. The next run tries again.
- **A new session is never lost.** If no model is available, it gets an extractive summary
  and is fully searchable. Upgrade it later with `--retry-extractive`.
- **Writes are all-or-nothing.** A session is replaced in one transaction, after embedding
  has succeeded. A failed write changes nothing.
- **A change during processing is noticed.** If the source file changes while a model is
  working, the result is not published and the session is picked up next time.
- **Completed work is kept.** Segment results are checkpointed as they finish. A crash or
  a killed process loses at most the requests that were in flight.
- **One bad file does not block the rest.** A malformed source is reported, backed off and
  retried; other sources continue.

The run's report counts what was scanned, unchanged, distilled, served by the fallback,
preserved and deferred, and how many errors remain. The command exits nonzero when errors
remain, so a script can tell.

## Long sessions

A session too large for one request is split into overlapping segments, distilled in
parallel and merged in order:

```toml
[ingest]
max_transcript_chars = 150000   # above this, split
segment_chars = 150000
segment_overlap = 7500
max_segments = 100
```

## What gets kept

```toml
[ingest]
min_session_chars = 200     # skip sessions with almost no dialogue
min_importance = 0.25       # drop memories the model rated below this (0 to 1)
chunk_chars = 6000
```

## Profiles and digests

Project profiles and weekly digests are built from session summaries, not from
transcripts. They are queued when a session changes and refreshed on their own interval:

```toml
[ingest]
aggregate_refresh_seconds = 86400    # at most once a day per profile or digest
profile_session_limit = 30
digest_session_limit = 40
```

A failed refresh keeps the previous version and stays queued.

## Cost, honestly

- You pay for distillation **once per session**, and again only for the part of a session
  that is new.
- You pay for retrieval only in agentic mode, a few short calls per search.
- Re-checking unchanged sources is free.
- Changing models, settings or the version of Tess Memory does not re-process history.

## What never redoes your memory

A session that has been distilled stays distilled. As long as its content has not changed,
none of these sends it back to a model or changes what is stored for it:

- another distillation, fallback or retrieval model, or another provider;
- any `[ingest]` setting: segment sizes, the importance threshold, chunk size, how much
  tool output is kept;
- a new release of Tess Memory, including one that rewords its prompts.

New settings and prompts apply to new sessions, and to what is added to an old session
later. The first run after such a change reads each source once to confirm it still
matches, with no model or embedding call.

If you *want* old sessions distilled again, for example to benefit from a better model, ask
for it. Nothing else does this:

```sh
tess-memory redistil --project ~/work/shop-api       # shows how many sessions
tess-memory redistil --project ~/work/shop-api --yes
tess-memory redistil --since 2026-10-01 --yes
tess-memory redistil --all --yes
```

A session's current memory is replaced only when its new distillation succeeds.

Changing the embedding model is the one change that needs a step from you, and it does not
redo any distillation either: see
[Changing the embedding model](models.md#embeddings).

The plan board is never rebuilt, replaced or deleted by any of this. It is authored, not
derived, and the database itself refuses to delete a plan item or its history.

Run `tess-memory status` to see what is stored and what is pending, or
`audit-sources` for a read-only inventory of sources against memory.

## Running it on a schedule

Running `ingest` by hand works. For anything long-lived, use the
[scheduler](scheduler.md), which adds queues, retries, shared rate limits and recovery.
