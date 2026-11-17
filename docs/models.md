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

**Which model?** Distillation is where quality matters most: the model decides what was
worth remembering and how important it is. Use the best model you are comfortable paying
for. Because ingestion is incremental, you pay once per session, not once per search.

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

The retrieval agent makes a few short calls per search, so a fast, inexpensive model is a
good choice. If its model fails, the search returns the plain hybrid results.

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
refuses to mix in another, because vectors from different models cannot be compared.
Choose before you ingest a large history: changing the embedding model later means
embedding the store again.

A failure of a hosted embedding provider never silently falls back to local vectors.

## Changing models later

Changing the distillation or retrieval model is free. Existing memories stay as they are,
and only new or changed sessions use the new model. Tess Memory does not re-distil your
history because a setting changed.

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
