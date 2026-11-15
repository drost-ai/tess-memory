# Teams and visibility

One store can serve several people: a team that wants its agents to share what the team
has learned, without sharing everything each person has typed.

A store with none of this configured behaves as a single-user store. Nothing changes until
you opt in.

## The model

- Every entry (session, memory, plan item) has an **owner**: the user who produced it.
- Every entry has a **visibility**: `private` (the owner), `team` (the project's team) or
  `org` (everyone on the store).
- Every reader is a **user**, fixed when the server process starts.
- A **project** can be given to a **team**.
- **Administrators** can read everything and manage users, teams and projects.

An entry is readable when the reader is an administrator, owns it, or it is `org`, or it
is `team` and the reader belongs to the project's team.

## The default policy

| What | Who can read it |
|---|---|
| Decisions, solutions, facts, problems | The project's team |
| Project profiles and the plan board | The project's team |
| Raw chunks, session summaries, weekly digests | Only their author |
| Preferences, task state | Only their author |

In words: what the team has *learned* is shared; what each person *said* is not. A
teammate searching memory finds your decision and its reasoning, not your transcript.

A team project's profile is built only from material the team may read, so it cannot leak
a private summary.

Change the policy per store:

```toml
[visibility]
fallback = "private"

[visibility.memory_kinds]
preference = "team"

[visibility.levels]
session = "team"
```

A policy change applies to existing entries too, except those someone explicitly shared or
unshared.

## Who is reading

Identity is bound when a process starts. A model can never choose it.

```toml
[identity]
user = "ana"
```

or the `TESS_MEMORY_USER` environment variable, or `tess-memory serve --user ana`. Each
person registers the server with their own user.

## Whose sessions are whose

A source says whose transcripts it holds, so ingestion records the right owner and team:

```toml
[[data_stores]]
name = "ana-claude"
type = "claude-code"
path = "/workspace/ana/claude-projects"
user = "ana"
team = "core"
```

## Administration

```sh
tess-memory users set-owner sam                  # the default owner of existing entries
tess-memory users add ana --name "Ana"           # add --admin for an administrator
tess-memory users team-add core
tess-memory users member-add core ana
tess-memory users project-map -work-shop-api --team core
tess-memory users list
```

## Sharing one thing

An author can widen or narrow a single entry:

```sh
tess-memory users share memory:123 team          # private | team | org | default
tess-memory users share session:<id> org
tess-memory users who-can-see 123
```

An agent can do the same through the `memory_share` tool, when its user asks it to share
something or take it back. Sharing a session shares its distilled output, never its raw
chunks. Every visibility change is recorded.

## Adopting it on an existing store

The migration is additive and runs once, in one transaction. Everything already in the
store belongs to the default owner. Tool signatures do not change, and no tool takes a
user argument.

## Scope

Today a shared store is a database that each person's server process can reach, for
example on a shared machine or volume. A hosted, multi-user service over HTTP is planned
and will use this same model.
