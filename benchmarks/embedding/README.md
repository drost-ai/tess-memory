# Choosing the local embedding default (Epic #92 C1)

Measured October 22, 2026 PT on two hosts with the files in this directory.
Raw receipts are in `results/` (one JSON per model, device and host, plus
`downloads.json` with revisions, licences, sizes and weight checksums).

## Method

- **Corpus** (`corpus.py`): 596 invented passages and 79 queries with one
  labelled answer each. Categories: personal facts, relationships, technical
  notes, exact error text, file paths, multilingual (six languages, English
  questions), and long passages that bury the answer after ~400 words, plus
  480 plausible distractors about the same people and components.
- **Models**, pinned by revision, downloaded once with `run.py download`
  (sentence-transformers files only, no remote code): bge-small-en-v1.5
  `5c38ec7c` (MIT), all-MiniLM-L6-v2 `1110a243` (Apache-2.0),
  multilingual-e5-small `614241f6` (MIT), Qwen3-Embedding-0.6B `97b0c614`
  (Apache-2.0). Each model's documented query/passage prefixes are applied;
  Qwen3 uses its documented left padding (without it, batched passages are
  pooled from a pad token).
- **Measurement** (`run.py all`): one fresh offline process per model and
  device; weights checked against the recorded SHA-256 first. Recall@5/10 and
  MRR over all queries and per category; single-query latency p50/p95 after
  warm-up; corpus throughput at batch 32 and docs/s at batch 8/32/128; load
  time with files in the OS cache; process peak RSS.
- **Hosts**: Apple M4 Max, macOS 26.6.2, Python 3.12.11, torch 2.14.1,
  sentence-transformers 6.1.0 (CPU and MPS); Intel i9-13900KF, Ubuntu 22.04,
  Python 3.10.12, torch 2.9.1 CPU, sentence-transformers 6.2.0 (CPU).

## Results

Quality is identical on every host and device, so it is shown once.

| model | R@5 | R@10 | MRR | personal R@5 | relationship R@5 | multilingual R@5 | long passage R@5 | dims | max tokens | download |
|---|---|---|---|---|---|---|---|---|---|---|
| multilingual-e5-small | **0.924** | 0.937 | **0.912** | 1.00 | 1.00 | **0.83** | 0.17 | 384 | 512 | 515 MB |
| bge-small-en-v1.5 | 0.911 | 0.937 | 0.793 | 1.00 | 1.00 | 0.50 | 0.33 | 384 | 512 | 134 MB |
| all-MiniLM-L6-v2 | 0.861 | 0.861 | 0.760 | 1.00 | 1.00 | 0.17 | 0.00 | 384 | 256 | 92 MB |
| Qwen3-Embedding-0.6B | 0.810 | 0.861 | 0.709 | 0.97 | 0.42 | 0.50 | 0.50 | 1024 | 32768 | 1.2 GB |

Speed and memory (query latency p50/p95 in ms; corpus docs/s; process peak
RSS; load seconds):

| model | Mac CPU | Mac MPS | Linux CPU | peak RSS Mac CPU / Linux CPU | load s (Mac / Linux) |
|---|---|---|---|---|---|
| multilingual-e5-small | 6.1/7.6, 598/s | 5.3/6.2, 1304/s | 5.2/5.3, 304/s | 2.8 GB / 2.1 GB | 0.7 / 1.1 |
| bge-small-en-v1.5 | 6.7/7.4, 598/s | 9.4/9.9, 636/s | 5.5/5.8, 314/s | 2.4 GB / 1.9 GB | 0.1 / 0.1 |
| all-MiniLM-L6-v2 | 3.5/3.9, 1730/s | 4.7/101.9, 2396/s | 4.1/4.3, 871/s | 1.4 GB / 1.2 GB | 0.1 / 0.1 |
| Qwen3-Embedding-0.6B | 96/131, 5.5/s | 19.5/22.1, 152/s | 204/257, 3.5/s | 16.1 GB / 4.0 GB | 0.3 / 0.4 |

## Recommendation (needs the owner's approval)

- **Default: multilingual-e5-small.** Best ranking (MRR 0.91), the only
  model that handles non-English memories (0.83), perfect on personal,
  relationship, technical, error and path queries, ~5-6 ms per query on
  either CPU. Costs: a 515 MB download and the `query: `/`passage: `
  prefixes, which C3 must record in the index fingerprint (the current
  `LocalEmbedder` only knows the BGE prefix).
- **Light alternative: bge-small-en-v1.5** for small disks or English-only
  owners: 134 MB, close on English recall, weaker ranking and multilingual.
- **Not recommended:** all-MiniLM-L6-v2 (256-token limit, no multilingual),
  Qwen3-Embedding-0.6B (lowest quality on this corpus under its documented
  settings, 16 GB peak on macOS CPU with uncapped sequence length, under 6
  docs/s on CPU).
- **Long passages fail for every model** (best 0.50): ingestion must chunk
  long text rather than rely on the embedder's window. This is a C3/memory
  task, not a model choice.

Proposed acceptance thresholds for any local default, to be confirmed before
adoption: overall R@5 >= 0.90 and MRR >= 0.85 on this corpus; multilingual
R@5 >= 0.75; CPU query p95 <= 50 ms on both receipt hosts; CPU peak RSS
<= 3 GB; download <= 600 MB; permissive licence. Only multilingual-e5-small
meets all of them.

## Limits

- 79 queries: one query is 1.3 points of R@5. The corpus is synthetic and
  tuned to separate models, not a leaderboard.
- Peak RSS is the whole process (torch included, batch-128 throughput runs).
  On MPS it excludes GPU allocations, so MPS memory is not comparable.
- Load times are with files in the OS cache; a cold disk needs root to
  measure and was not.
- Library versions differ between hosts (Python 3.10/torch 2.9.1 on Linux);
  each receipt records its own. WSL was not measured (no WSL host).

## Reproduce

```sh
python run.py download --cache CACHE   # explicit, network
python run.py all --cache CACHE --out results --device cpu [--device mps]
python run.py report --out results
```
