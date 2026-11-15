# Sources

A source is a folder where an agent writes its session transcripts. You list sources in the
settings file under `[[data_stores]]`. Tess Memory only ever reads them.

```toml
[[data_stores]]
name = "claude-code"         # your name for it
type = "claude-code"         # how to read it
path = "~/.claude/projects"  # where it is
```

## Options common to every source

| Key | Meaning |
|---|---|
| `name` | A label, shown in status output |
| `type` | `claude-code`, `codex`, `tess-core` or `transcript` |
| `path` | The folder to read |
| `project` | Bind every session from this source to one project, instead of naming projects after working directories |
| `exclude` | Glob patterns of project names to skip |
| `skip_copies_of` | Folders whose sessions should not be ingested again from this source |
| `user`, `team` | Whose sessions these are, in a [multi-user store](teams.md) |

## Claude Code

```toml
[[data_stores]]
name = "claude-code"
type = "claude-code"
path = "~/.claude/projects"
exclude = ["-private-var-folders*", "-private-tmp-*"]
```

Reads `<path>/<project>/<session>.jsonl`. The project folder name is the project. Sub-agent
transcripts are skipped, because the parent session already contains their results. If you
use more than one Claude Code configuration directory, add one source for each.

The `exclude` patterns above drop sessions that ran in temporary directories, which are
usually noise.

## Codex

```toml
[[data_stores]]
name = "codex"
type = "codex"
path = "~/.codex"

[codex]
tool_output_chars = 2000      # keep the head and tail of long tool output
include_subagents = false
```

Reads session rollouts under `sessions/` and `archived_sessions/`, in every format Codex
has used: JSON snapshots, JSONL, and Zstandard-compressed archives, which are streamed
without being unpacked on disk.

Codex histories need more cleaning than most, and Tess Memory does it for you:

- Image payloads are removed, leaving a small marker.
- Tool inputs are kept whole; long tool outputs keep their beginning and end.
- Duplicated records (response items mirrored as UI events, compaction snapshots) are
  reconciled, so the same text is not remembered twice.
- A forked session's copied history is recognised and attributed to the original, not
  counted again.
- A change that only touches metadata, or an archive move, does not trigger any model or
  embedding work.

Look at what would be read, without any model call or database write:

```sh
tess-memory inspect-codex ~/.codex --limit 5
```

Sessions are identified as `codex:<native-session-id>`.

## Tess Core

```toml
[[data_stores]]
name = "my-agent"
type = "tess-core"
path = "~/.my-agent/sessions"
project = "assistant"
```

Reads the sessions of any agent built on [Tess Core](https://github.com/drost-ai/tess-core):
the `sessions` folder of its runtime home. Sub-agent lineage is taken from session
metadata.

## Plain transcripts

```toml
[[data_stores]]
name = "imported"
type = "transcript"
path = "~/exports/other-agent"
project = "imported"          # required
```

Reads a folder of `<name>.jsonl` files in the Tess Core transcript message format. Use it
for sessions exported or converted from an agent with no native reader. The file name is
the session id. Because these files carry no working directory, `project` is required.

## Naming projects

By default a session belongs to the project named after the directory it ran in. That is
right for coding work: the repository is the project.

Set `project` on a source when every session from it should share one memory. An assistant
that runs from many directories is the usual case.

## Avoiding double ingestion

If one agent's home was copied from another's, the copy holds the same sessions.
`skip_copies_of` tells a source to ignore sessions that also exist elsewhere:

```toml
skip_copies_of = ["~/.original-agent/sessions"]
```

## What is skipped

- Sessions with no human prompt, or less than about 200 characters of dialogue.
- Sub-agent side conversations.
- Anything matched by `exclude`.

## Auditing

```sh
tess-memory audit-sources
```

lists what each source contains and how much of it is in memory, without calling a model.
