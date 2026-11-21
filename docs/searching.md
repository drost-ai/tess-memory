# Searching memory

Memory is only useful if the right piece comes back at the right moment. This page covers
the tools an agent uses to recall, the two search modes, and how to ask well.

## Orient first: `get_project_context`

At the start of work in a project, one call returns:

- the project's **rolling profile**: what it is, how it is built, key decisions, current
  state, known problems and your preferences for it,
- the **open plan board**,
- the last ten **session summaries**.

This is the call that replaces "let me read the codebase to get my bearings". It costs one
lookup and no model.

## Ask a question: `search_memory`

```
search_memory(
  query   = "fix for duplicate webhook deliveries in the billing service retry queue",
  project = "-work-billing",     # optional
  level   = "memory",            # optional: chunk | memory | session | digest | project | plan | image
  kind    = "decision",          # optional, for level=memory
  mode    = "static",            # static | agentic | auto
)
```

### The two modes

**Static** runs one hybrid search and returns ranked results. It uses no model, is fast,
and is right for direct recall: an error message, a file name, "what did we decide about
X".

**Agentic** runs a retrieval agent inside the server. It starts from a static search, then
judges whether the evidence answers the question, issues reformulated sub-queries, follows
links between sessions (including across projects) and returns the evidence it decided to
keep. Use it for questions that need several hops: "how did the auth design change over
the last month, and why".

`auto`, the default, follows the `[search] agent` setting.

Both modes return **evidence, never a written answer**. The retrieval agent may also
return nothing, when nothing recorded supports the question. That is deliberate: a memory
that guesses is worse than one that says it does not know.

### Writing a good query

The search does not have your conversation. The agent does. So the query must stand alone:

| Instead of | Write |
|---|---|
| "the bug from before" | "fix for FTS5 trigger corruption during incremental ingest in the SQLite store" |
| "what did we decide" | "decision on idempotency key format for the billing dual-write cutover" |
| "that error" | the error text itself, verbatim |

Error strings retrieve extremely well. Paste them.

### Narrowing

- `project` scopes to one project. Omit it to search everything.
- `level` picks a timescale. `chunk` for exact commands and output; `memory` for facts and
  decisions; `session` and `digest` for what happened when; `plan` for planned work.
- `kind` filters memories: `decision`, `fact`, `preference`, `solution`, `problem`,
  `task_state`. `kind="preference"` is the way to ask "how do I like this done?".
- Date filters bound the search in time.

Filters apply before ranking, so a narrow search is not crowded out by other projects.

## Go deeper

- **`get_session(session_id)`**: one session in full, with its summary, outcome and every
  memory it produced. Use it when a search result points at a session worth understanding.
- **`get_session_transcript(session_id, cursor)`**: the original transcript, a page at a
  time. Use it when the exact wording, the full command or the complete tool output
  matters. It reads the source file as it is; it never changes memory.
- **`recent_activity(days)`**: what was worked on lately, across all projects.
- **`list_projects()`**: every project, newest first.

## Freshness

Results say when a source has changed since it was last ingested, or when a project
profile is waiting to be refreshed, so an agent knows when memory is behind the present.
`memory_status` gives the full picture.

## From the command line

```sh
tess-memory search "idempotency key decision billing cutover"
tess-memory search "..." --static        # force plain search
```

## How results are ranked

```
score = 0.55 × meaning + 0.20 × keywords + 0.15 × importance + 0.10 × recency
```

with a 30-day half-life on recency and at most three results per session. All of these are
settings under `[retrieval]`; see [Configuration](configuration.md).

## Finding images

An agent whose host keeps the images people send it (Tess Agent does) records a detailed
description of each one here. See [Image records](images.md). They are found by the same
search:

```
search_memory(query="a woman in a red dress on a rooftop at night", level="image")
```

Describe what would be visible. Descriptions never name people or places, so "who" and
"where" come from the conversation the image was sent in: a result names that session, and
`get_session` lists every image sent in it. A result also gives the image's id and the path
of its file, which an agent with a file-reading tool can open to look at the image.
