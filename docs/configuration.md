# Configuration

A store is configured by one TOML settings file. Select it with the `TESS_MEMORY_SETTINGS`
environment variable, in your shell and in each MCP registration:

```sh
export TESS_MEMORY_SETTINGS=~/.tess-memory/settings.toml
```

Precedence is: built-in defaults, then the settings file, then environment variables.

A path to a settings file that does not exist is an error. Relative paths inside the file
are resolved against the file's own directory, so a store behaves the same wherever a
command is run from.

## A complete example

```toml
# ---- sources: where sessions are read from ---------------------------------
[[data_stores]]
name = "claude-code"
type = "claude-code"
path = "~/.claude/projects"
exclude = ["-private-var-folders*", "-private-tmp-*"]

[[data_stores]]
name = "codex"
type = "codex"
path = "~/.codex"

# ---- storage ----------------------------------------------------------------
[storage]
data_dir = "~/.tess-memory"

# ---- distillation -----------------------------------------------------------
[llm]
settings = "~/.tess-memory/providers.yaml"
profile = "strong"
fallback_profile = "cheap"
workers = 4

# ---- retrieval agent --------------------------------------------------------
[search]
agent = true
profile = "cheap"
max_rounds = 4
queries_per_round = 3
per_query_limit = 8
evidence_limit = 12

# ---- embeddings -------------------------------------------------------------
[embedding]
provider = "local"
local_model = "BAAI/bge-small-en-v1.5"
allow_download = true
batch_size = 64

# ---- ingestion --------------------------------------------------------------
[ingest]
quiet_seconds = 900
incremental = true
max_model_calls_per_run = 128

# ---- ranking ----------------------------------------------------------------
[retrieval]
half_life_days = 30
weights = { semantic = 0.55, bm25 = 0.20, importance = 0.15, recency = 0.10 }
per_session_cap = 3
default_limit = 8
```

## Sections

### `[[data_stores]]`

One entry per source. See [Sources](sources.md).

### `[storage]`

| Key | Meaning |
|---|---|
| `data_dir` | The store's directory: `memory.db` and its caches |

### `[llm]`

| Key | Meaning |
|---|---|
| `settings` | Path to the provider file |
| `profile` | The profile used for distillation |
| `model`, `reasoning_effort` | Optional overrides for this store |
| `fallback_profile` | Tried when the primary fails or refuses. Omit to disable. |
| `fallback_model`, `fallback_reasoning_effort` | Optional overrides for the fallback |
| `workers` | Distillation calls in parallel |

See [Models and embeddings](models.md).

### `[search]`

| Key | Default | Meaning |
|---|---|---|
| `agent` | `true` | Run the retrieval agent. `false` gives plain hybrid search. |
| `profile` | | The profile the retrieval agent uses |
| `max_rounds` | 4 | Model calls per search |
| `queries_per_round` | 3 | Sub-queries per round |
| `per_query_limit` | 8 | Results taken from each sub-query |
| `evidence_limit` | 12 | Most evidence returned |

### `[embedding]`

| Key | Meaning |
|---|---|
| `provider` | `local`, or a profile name |
| `dims` | Vector size, for hosted models |
| `local_model` | The local model to use |
| `allow_download` | Allow fetching the local model. Off by default. |
| `batch_size` | Texts per embedding request |

### `[ingest]`

| Key | Default | Meaning |
|---|---|---|
| `quiet_seconds` | 0 | Wait this long after a session's last write before processing it |
| `incremental` | `false` | Process only the new, completed part of a grown session |
| `segment_workers` | 1 | Parallel segments per session |
| `max_model_calls_per_run` | 0 | Cap on model calls per run. 0 is unlimited. |
| `aggregate_refresh_seconds` | 0 | Minimum interval between refreshes of a profile or digest |
| `cache_retention_days` | 30 | How long unused checkpoints are kept |
| `max_transcript_chars` | 150000 | Above this, a session is split into segments |
| `segment_chars`, `segment_overlap`, `max_segments` | | Segment size, overlap and count |
| `chunk_chars` | 6000 | Size of a searchable chunk |
| `min_session_chars` | 200 | Skip sessions shorter than this |
| `min_importance` | 0.25 | Drop memories rated below this |
| `profile_session_limit` | 30 | Sessions read to build a project profile |
| `digest_session_limit` | 40 | Sessions read to build a weekly digest |

See [Ingestion](ingestion.md).

### `[retrieval]`

| Key | Default | Meaning |
|---|---|---|
| `weights` | 0.55 / 0.20 / 0.15 / 0.10 | Meaning, keywords, importance, recency |
| `half_life_days` | 30 | How fast recency fades |
| `candidate_pool` | 100 | Candidates per signal before they are combined |
| `per_session_cap` | 3 | Most results from one session |
| `default_limit` | 8 | Results when the caller does not say |

### `[planning]`

| Key | Meaning |
|---|---|
| `default_project` | The project plan calls use when none is given |
| `lock_writes_to_default` | Refuse plan writes to any other project |

See [The plan board](plan-board.md).

### `[identity]` and `[visibility]`

Who is reading, and who may read what. See [Teams and visibility](teams.md).

### `[codex]`

Options for Codex sources. See [Sources](sources.md).

### `[personal]`

Personal memory. See [Personal memory](personal-memory.md).

## Environment variables

For MCP registrations and scheduled jobs that should not depend on a file:

| Variable | Overrides |
|---|---|
| `TESS_MEMORY_SETTINGS` | Which settings file to use |
| `TESS_MEMORY_DATA` | `storage.data_dir` |
| `TESS_MEMORY_SOURCE` | Replaces the sources with one Claude Code source at this path |
| `TESS_MEMORY_USER` | `identity.user` |
| `TESS_MEMORY_ENGINE_SETTINGS` | `llm.settings` |
| `TESS_MEMORY_ENGINE_PROFILE` | `llm.profile` |
| `TESS_MEMORY_SEARCH_AGENT` | `search.agent` (`0` turns the retrieval agent off) |
| `TESS_MEMORY_SEARCH_PROFILE` | `search.profile` |
| `TESS_MEMORY_EMBED_PROVIDER` | `embedding.provider` |
| `TESS_MEMORY_EMBED_DIMS` | `embedding.dims` |
| `TESS_MEMORY_EMBED_MODEL` | `embedding.local_model` |

## Credentials

Keys are never written in the settings file or the provider file. A provider profile names
an environment variable, and the key is read from the environment of the process. For a
background service with no shell environment, a provider file written as JSON may carry an
`env` map whose values are exported to the process; keep that file readable only by you.

## Embedding Tess Memory in an application

An application that hosts Tess Memory, rather than a person running it, passes a JSON
manifest instead of a TOML file. The manifest names the runtime home and the provider
registry explicitly, and nothing is discovered from the environment.
