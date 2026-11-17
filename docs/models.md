# Models and embeddings

Tess Memory uses models for three jobs, and you choose each one separately:

| Job | Setting | Needed for |
|---|---|---|
| **Distillation** | `[llm]` | Turning sessions into memories, profiles and digests |
| **Retrieval** | `[search]` | The agentic search mode |
| **Embeddings** | `[embedding]` | Search by meaning |

None of them is required to start. With local embeddings, `ingest --no-llm` and
`[search] agent = false`, Tess Memory runs with no model provider at all.

## Declaring providers

Models are reached through [Tess Core](https://github.com/drost-ai/tess-core), so any
provider it supports works: Anthropic, OpenAI, any OpenAI-compatible endpoint, or a model
on your own hardware.

`tess-memory setup` does this for you: it connects a ChatGPT account, an API key or an
endpoint, lists the models it offers, tests the ones you pick and writes the provider
file. A ChatGPT account can only be connected that way, because signing in happens in the
browser.

To declare providers by hand, write them once in a provider file:

```yaml
# ~/.tess-memory/providers.yaml
schema_version: 1
providerProfiles:
  strong:
    kind: anthropic
    defaultModel: <a capable model>
    auth: { kind: anthropic_api_key, envVar: ANTHROPIC_API_KEY }

  cheap:
    kind: openai
    apiMode: chat_completions
    baseUrl: https://api.example.com/v1
    defaultModel: <an inexpensive model>
    auth: { kind: custom_bearer, envVar: EXAMPLE_API_KEY }

  local:
    kind: openai
    apiMode: chat_completions
    baseUrl: http://localhost:11434/v1
    defaultModel: <a local model>
    auth: { kind: none }
```

Credentials are never written in this file. A profile names the environment variable that
holds its key.

## Distillation

```toml
[llm]
settings = "~/.tess-memory/providers.yaml"
profile = "strong"
fallback_profile = "cheap"     # tried when the primary fails or refuses
workers = 4                    # distillation calls in parallel
```

Optional overrides pin a model or reasoning effort for this store only, without changing
the profile: `model`, `reasoning_effort`, `fallback_model`, `fallback_reasoning_effort`.

**Which model?** Distillation reads every session once and decides what was worth
remembering and how important it is, so it sets the quality of the memory. It is also where
nearly all the usage goes. A stronger model writes better memories and costs more per
session; a cheaper one makes a long history affordable. Because ingestion is incremental,
you pay once per session, not once per search, and you can change the model later without
re-reading anything.

**The fallback.** Some models refuse some transcripts, for example security work. A
fallback from a different provider turns a refusal into a completed distillation. If both
fail, the existing memory for that session is kept and the session is retried later.

## Retrieval

```toml
[search]
agent = true           # false: plain hybrid search, no model
profile = "cheap"
max_rounds = 4         # model calls per search
queries_per_round = 3
evidence_limit = 12
```

The retrieval agent makes a few short calls per search, while an agent waits for the
answer. It uses far fewer tokens than distillation, so a strong model is affordable here,
and a fast one keeps searches quick. If its model fails, the search returns the plain
hybrid results.

## Embeddings

```toml
[embedding]
provider = "local"                       # or a profile name
local_model = "BAAI/bge-small-en-v1.5"
allow_download = true                    # fetch the model the first time
```

`provider = "local"` runs a sentence-transformers model on your machine. It is free and
private, and good enough for most stores. A hosted embedding model gives better recall on
large stores:

```toml
[embedding]
provider = "embeddings"    # a profile in your provider file
dims = 4096
batch_size = 64
```

**One store, one vector space.** A store records which model produced its vectors and
never mixes in another, because vectors from different models cannot be compared. The
provider profile's name is not part of that: renaming a profile, or reaching the same model
through another one, changes nothing.

**Changing the embedding model.** You can, and you lose nothing. Point `[embedding]` at the
new model and run:

```sh
tess-memory reembed          # shows what it would do
tess-memory reembed --yes
```

Every stored row is embedded again with the new model. No session is read or distilled
again, and the plan is not touched. The new vectors are staged beside the old ones, so it
can be stopped and continued, and one transaction switches the store over when every row
is done. If it is interrupted before that, the store is exactly as it was.

Between changing the setting and finishing `reembed`, ingestion waits and search uses
keywords, importance and recency without the meaning signal. To avoid that gap on a large
store, leave your settings as they are and give `reembed` the new ones in a copy
(`tess-memory reembed --to new-settings.toml --yes`): the store keeps working on its
current vectors throughout, and you switch your settings file when it finishes.

A failure of a hosted embedding provider never silently falls back to local vectors.

## Changing models later

Changing the distillation or retrieval model is free. Existing memories stay as they are,
and only new or changed sessions use the new model. Tess Memory does not re-distil your
history because a model, a setting, or a prompt in a newer release changed. See
[what never redoes your memory](ingestion.md#what-never-redoes-your-memory).

## Running fully offline

```toml
[llm]
settings = "~/.tess-memory/providers.yaml"
profile = "local"

[search]
agent = true
profile = "local"

[embedding]
provider = "local"
allow_download = false      # the model is already cached
```

With a local model server and the local embedding model, nothing leaves the machine.

## Cost control

- `[ingest] max_model_calls_per_run` caps model calls in one ingestion pass.
- `[llm] workers` caps parallel calls.
- The [scheduler](scheduler.md) applies shared limits per provider across all its workers.
- `tess-memory scheduler --config ... usage` reports observed model usage.
