# Scheduler

The scheduler keeps memory current without you thinking about it. It is a small service
that watches your stores, ingests what has changed every few minutes, refreshes profiles
and digests on their own cadence, and recovers by itself when something goes wrong.

One scheduler can look after several stores.

## Configure

Write a scheduler file, outside the checkout:

```toml
# ~/.tess-memory-service/config.toml
[service]
data_dir = "~/.tess-memory-service"
parallel_stores = 4
model_limit = 16            # model calls in flight, across all stores
embedding_limit = 4

[provider_limits]           # optional ceilings per provider
anthropic = 4
openai = 16

[[stores]]
name = "main"
settings = "~/.tess-memory/settings.toml"
enabled = true
poll_seconds = 300          # look for changes every five minutes
profile_seconds = 3600      # refresh project profiles hourly
digest_seconds = 86400      # refresh weekly digests daily
sources_per_run = 16
model_calls_per_run = 128
```

Each store names an existing settings file. The scheduler does not change those files, the
sources they list or the models they choose.

## Run it

On macOS, install it as a background service that starts at login:

```sh
tess-memory scheduler --config ~/.tess-memory-service/config.toml install
```

Elsewhere, run it in the foreground under your service manager:

```sh
tess-memory scheduler --config ~/.tess-memory-service/config.toml serve
```

## Operate it

```sh
C="--config $HOME/.tess-memory-service/config.toml"

tess-memory scheduler $C status                      # what is running, queued and failing
tess-memory scheduler $C run --instance main         # one ingestion pass now
tess-memory scheduler $C run --instance main --kind profiles
tess-memory scheduler $C retry --instance main --kind ingest
tess-memory scheduler $C disable --instance main     # stop scheduling this store
tess-memory scheduler $C enable --instance main
tess-memory scheduler $C ingest-once --until-caught-up   # work through a backlog, then exit
tess-memory scheduler $C usage                       # observed model usage
tess-memory scheduler $C stop | start | restart      # the installed service
```

`enabled` in the file is the starting state. Enabling or disabling from the command line
is remembered across restarts.

`ingest-once --until-caught-up` is the command for a first run on a large history: it
keeps making passes until every store has nothing left, however long that takes, and then
stops.

## How it behaves

**Bounded passes.** Each pass handles a limited number of sources and model calls, so no
single run can monopolise a provider. When there is a backlog, the next pass starts
sooner.

**Fair scanning.** Recent changes are prioritised, and older sources are rotated through,
so one huge unfinished file cannot occupy every pass.

**Shared limits.** Model and embedding limits are shared across all the scheduler's worker
processes. A provider that is rate-limited or cooling down is skipped, and work for other
providers carries on.

**Durable jobs.** Jobs and their results are recorded in a coordinator database. At most
one job works on a store at a time, and a manual `ingest` takes the same lock.

**Recovery.** If a worker dies, the scheduler notices, keeps its completed checkpoints and
requeues the job. A worker that runs past its timeout is stopped and recovered. A source
that keeps failing is backed off, and tried again at once if it changes.

**Nightly recovery.** Optionally, at an hour you choose, a pass revisits sources that
could not be completed during the day:

```toml
[[stores]]
recovery_hour = 22
recovery_timezone = "Europe/London"
```

## Health

`status` separates three questions that are easy to confuse:

- Is the scheduler alive?
- Is work queued or failing?
- Is memory actually up to date with the sources?

The third is what matters, and it is reported per source. The same information is
available to agents through the `memory_status` tool, and search results flag sources that
have changed since they were last ingested.

## Safety

The scheduler's commands never delete a memory database or an original transcript.
Stopping the service can interrupt a pass; the next start continues from durable state.
