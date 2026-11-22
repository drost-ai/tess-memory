# The plan board

Most of what an agent knows about a project is in the past: what was said, what was done.
The plan board is the other half. It is where an agent writes down what is *going* to
happen, keeps it current as work moves, and reads it back at the start of the next session.

It is the difference between an agent that helps with a task and an agent that can carry a
project.

## Why it matters

A large piece of work does not fit in one session. Context windows fill, sessions end,
people switch tools, and the plan that existed in the conversation goes with it. An agent's
built-in to-do list has the same problem: it lasts as long as the session.

The plan board is durable. It lives in the memory store, not in the conversation. A new
session, a different agent, or the same agent three weeks later, calls `plan_board` and
sees exactly what is done, what is in progress, what is blocked and what is next.

## Items

| Kind | Use it for |
|---|---|
| `vision` | The lasting direction of a product |
| `epic` | A feature that takes several sessions |
| `slice` | An increment that can ship |
| `task` | One session-sized unit of work |
| `note` | Anything worth keeping: a finding, a decision, a scratchpad |

Items nest as a tree: vision, then epic, slice and task, by convention. Notes attach
anywhere. Each item has a title, a body, a priority from 1 to 5, tags and a stable id.

A note is a record, not work. One filed under an item is shown on the board while that
item is, and leaves with it when the work is finished, whatever status the note was left
in; it never keeps a finished item on the board. A note at the top of the board stays
until you complete or drop it. Nothing is deleted or changed: `include_done=true` and
search show every note.

## Lifecycle

```
planned ──▶ in_progress ──▶ complete
                │
                ├──▶ blocked ──▶ in_progress
                └──▶ dropped
```

Nothing is ever deleted. Finished work is `complete`; abandoned work is `dropped`, with
the reason in its body. Every status change is timestamped, and the full history of an
item is kept, so the board is also a record of how the project actually went.

## The four tools

**`plan_add`** creates an item and returns its id. Create parents before children.

```
plan_add(project="-work-billing", kind="epic",
         title="Migrate billing to the new ledger",
         body="Goal, constraints, what done looks like...")          → #41
plan_add(project="-work-billing", kind="task", parent_id=41,
         title="Dual-write cutover",
         body="Write to both ledgers behind a flag. Done when...")   → #42
```

**`plan_update`** changes status, title, body, priority, parent or tags. `append` adds a
paragraph to the end of the body without replacing it.

```
plan_update(item_id=42, status="in_progress")
plan_update(item_id=42, status="complete", append="Shipped in 3f2a91c; both ledgers agree.")
```

**`plan_board`** shows the project's open work as a tree. Pass `include_done=true` to see
everything.

**`plan_get`** returns one item in full: its body, parent, children and history.

## Working rules that make it pay off

These are the habits that separate a board that helps from one that misleads.

- **Write bodies that stand alone.** The reader is a future session with none of today's
  conversation. State the objective, the relevant decisions and what done means.
- **Update status when it changes, not at the end.** `in_progress` when work starts,
  `complete` when it is done and verified, `blocked` with the reason when reality gets in
  the way. A stale board is worse than no board.
- **Roll status up.** Starting a child starts its parent. Finishing the last open child
  finishes the parent.
- **Check before adding.** Call `plan_board` first, to find the right parent and avoid
  duplicates.
- **Append, or read before replacing.** `plan_update(append=...)` adds a paragraph to the
  end of the body and keeps what is there, so a result or a blocker can be recorded without
  reading the item first, and two agents adding to one item both keep their text.
  `plan_update(body=...)` replaces the whole body: read the item with `plan_get` first and
  keep what still matters.

Put these in your agent's instruction file and it will follow them.

## The plan is searchable

Each write is mirrored into the search index at the `plan` level. `search_memory` finds
plan items alongside memories, and `get_project_context` leads with the open board. An
agent asking "what do we know about the ledger migration" gets the history *and* the plan.

## Shared across agents

Plan writes are immediate and independent of ingestion. If Claude Code and Codex use the
same store, one can leave a task for the other, and it is there on the next call.

## Keeping a board in one place

A project is normally named after the working directory. An assistant that moves between
directories should have one board, not one per folder:

```toml
[planning]
default_project = "assistant"
lock_writes_to_default = true
```

With the lock on, a write to any other project is refused, so a board cannot be scattered
by accident. `plan_add` also warns when a project name matches nothing known and suggests
the closest one.

## From the command line

```sh
tess-memory plan .            # the board for the current directory's project
tess-memory plan . --all      # including complete and dropped items
```
