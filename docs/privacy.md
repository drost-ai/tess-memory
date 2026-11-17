# Privacy and data

Agent transcripts contain your code, your prompts, command output, and sometimes secrets
that scrolled past. A memory built from them deserves a plain account of what it keeps and
where it goes.

## Where your data lives

In one directory on your machine: the `data_dir` of the store.

| File | Holds |
|---|---|
| `memory.db` | Sessions, memories, chunks, vectors, profiles, digests, the plan board |
| `distillation-cache.db` | Model outputs kept so interrupted work can resume |
| Reader and request caches | Progress records and checkpoints |

It is SQLite. You can open it, query it, back it up by copying it, and delete it.

There is no Tess Memory service, no account and no telemetry. The software does not
contact its authors.

## What is stored

- **Chunks**: windows of the conversation, verbatim, including tool calls. Long tool
  output is clipped to its beginning and end. Images are replaced by a marker.
- **Distilled memories**: what a model extracted from the session.
- **Summaries, profiles and digests**.
- **Plan items**: what agents wrote on the board.
- **Vectors** for all of the above.

Original transcripts are **not copied** into the store, and are never modified.
`get_session_transcript` reads them from where your agent wrote them.

## What leaves your machine

Only what you configure a provider for.

| Step | Sent to | Avoid it with |
|---|---|---|
| Distillation | Your distillation provider: the cleaned session text | `ingest --no-llm`, or a local model |
| Agentic search | Your retrieval provider: the query and candidate results | `[search] agent = false`, or a local model |
| Embeddings | Your embedding provider: the text being embedded | `[embedding] provider = "local"` |

With a local model server and local embeddings, or with `--no-llm` and plain search,
**nothing leaves the machine**.

The providers you choose see this content under their own terms. Choose them as you would
choose where to send your source code.

## Secrets

Tess Memory does not scan for or redact secrets. If a key was printed in a session, it is
in that session's chunks. Practical measures:

- Keep the store directory readable only by you. It is as sensitive as your shell history.
- Use `exclude` to keep whole projects out of memory.
- In a scheduled setup, a store can name a private list of literal strings; any source
  containing one is held back before any model or embedding call.
- Rotate anything that leaked, as you would anyway.

## Credentials

Provider keys are never written to the store, and provider errors are reported without
them. A provider file names where each key lives: an environment variable, or an entry in
your system keychain. `tess-memory setup` puts the keys you enter in the keychain (macOS
Keychain, or the Secret Service on Linux) and the provider file holds only a reference.
With `--no-keychain` it keeps them in the provider file's own `env` map instead, in a file
only your user can read. A ChatGPT sign-in is stored in the keychain and can be revoked
from ChatGPT's settings.

## Sharing with a team

In a multi-user store, the default is that distilled knowledge and plans are visible to a
project's team, while raw chunks, session summaries and preferences stay with their
author. See [Teams and visibility](teams.md).

## Personal memory

Personal memory is off by default, stored in separate tables, and never read by the coding
tools or coding search. See [Personal memory](personal-memory.md).

## Deleting

- **Everything**: delete the store directory.
- **A source**: remove it from the settings file. Its existing memories stay until you
  delete the store or the sessions.
- **The plan**: items are retired with a status, not deleted, by design. To remove the
  history entirely, delete the store.

There is no command to delete a single session today.

## What Tess Memory is not

It is not an access-control boundary between processes on your machine: anything that can
read the store directory can read the memory. It is not a compliance product. It is a
local database with careful engineering, and you remain in control of it.
