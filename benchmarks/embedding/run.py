"""Measure local embedding candidates on the synthetic corpus.

    python run.py download --cache DIR            # explicit, records revisions/checksums/licenses
    python run.py all --cache DIR --out results/   # one fresh offline process per model and device
    python run.py report --out results/            # rebuild report.md from the raw JSON

Downloads happen only in `download`, from the pinned repository list below,
into the cache you name; measuring runs with HF_HUB_OFFLINE=1 and never
loads remote code. Weights are verified against the recorded SHA-256 before
each measurement. Load times are with the files in the OS cache (a truly
cold disk needs root to drop caches and is not measured).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    "bge-small-en-v1.5": {"repo": "BAAI/bge-small-en-v1.5",
                          "query_prefix": "Represent this sentence for searching relevant passages: ", "passage_prefix": ""},
    "all-MiniLM-L6-v2": {"repo": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "passage_prefix": ""},
    "multilingual-e5-small": {"repo": "intfloat/multilingual-e5-small", "query_prefix": "query: ", "passage_prefix": "passage: "},
    # Last-token pooling: the model card requires left padding, or batched
    # passages shorter than the longest are pooled from a pad token.
    "Qwen3-Embedding-0.6B": {"repo": "Qwen/Qwen3-Embedding-0.6B",
                             "query_prefix": "Instruct: Given a question, retrieve the passages that answer it\nQuery:",
                             "passage_prefix": "", "tokenizer_kwargs": {"padding_side": "left"}},
}
# Only what sentence-transformers needs: no ONNX/OpenVINO exports, no code.
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "1_Pooling/*", "2_Normalize/*"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(cache: Path, names: list[str]) -> None:
    from huggingface_hub import HfApi, snapshot_download
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    api = HfApi()
    record = json.loads((cache / "downloads.json").read_text()) if (cache / "downloads.json").exists() else {}
    for name in names:
        repo = MODELS[name]["repo"]
        info = api.model_info(repo)
        revision = info.sha
        path = Path(snapshot_download(repo, revision=revision, cache_dir=cache, allow_patterns=ALLOW))
        files = sorted(p for p in path.rglob("*") if p.is_file())
        record[name] = {
            "repo": repo, "revision": revision, "snapshot": str(path.relative_to(cache)),
            "license": (info.card_data or {}).get("license") if info.card_data else None,
            "bytes": sum(p.stat().st_size for p in files),
            "weights": {str(p.relative_to(path)): _sha256(p) for p in files if p.suffix == ".safetensors"},
        }
        print(f"{name}: {revision} {record[name]['bytes'] / 1e6:.1f} MB license={record[name]['license']}")
        (cache / "downloads.json").write_text(json.dumps(record, indent=2))


def measure(cache: Path, name: str, device: str) -> dict:
    """In a fresh process: load, encode, query; returns the raw receipt."""
    started = time.perf_counter()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, str(HERE))
    from corpus import build
    downloaded = json.loads((cache / "downloads.json").read_text())[name]
    snapshot = cache / downloaded["snapshot"] if "snapshot" in downloaded else Path(downloaded["path"])
    for relative, expected in downloaded["weights"].items():
        if _sha256(snapshot / relative) != expected:
            raise SystemExit(f"{name}: {relative} does not match its recorded checksum")
    import numpy as np
    import torch
    import sentence_transformers
    import transformers
    from sentence_transformers import SentenceTransformer
    imported = time.perf_counter()
    spec = MODELS[name]
    model = SentenceTransformer(str(snapshot), device=device, trust_remote_code=False, local_files_only=True,
                                tokenizer_kwargs=spec.get("tokenizer_kwargs"))
    loaded = time.perf_counter()
    passages, queries = build()
    texts = [spec["passage_prefix"] + p.text for p in passages]
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0 = time.perf_counter()
    corpus = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    wall1 = time.perf_counter()
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    throughput = {}
    sample = texts[:128]
    for batch in (8, 32, 128):
        t0 = time.perf_counter()
        model.encode(sample, batch_size=batch, normalize_embeddings=True)
        throughput[str(batch)] = round(len(sample) / (time.perf_counter() - t0), 1)
    for warm in queries[:5]:
        model.encode([spec["query_prefix"] + warm.text], normalize_embeddings=True)
    latencies, ranks = [], []
    index = {p.id: i for i, p in enumerate(passages)}
    per_category: dict[str, list[int]] = {}
    for query in queries:
        t0 = time.perf_counter()
        vector = model.encode([spec["query_prefix"] + query.text], normalize_embeddings=True, convert_to_numpy=True)[0]
        latencies.append((time.perf_counter() - t0) * 1000)
        order = np.argsort(-(corpus @ vector))
        rank = int(np.where(order == index[query.relevant])[0][0]) + 1
        ranks.append(rank)
        per_category.setdefault(query.category, []).append(rank)

    def scores(values: list[int]) -> dict:
        return {"recall@5": round(sum(r <= 5 for r in values) / len(values), 3),
                "recall@10": round(sum(r <= 10 for r in values) / len(values), 3),
                "mrr": round(sum(1 / r for r in values) / len(values), 3), "queries": len(values)}

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    latencies.sort()
    return {
        "model": name, "repo": spec["repo"], "revision": downloaded["revision"], "license": downloaded["license"],
        "download_mb": round(downloaded["bytes"] / 1e6, 1), "device": device,
        "dims": int(corpus.shape[1]), "max_seq_length": int(model.max_seq_length),
        "query_prefix": spec["query_prefix"], "passage_prefix": spec["passage_prefix"],
        "tokenizer_kwargs": spec.get("tokenizer_kwargs") or {},
        "import_s": round(imported - started, 2), "load_s": round(loaded - imported, 2),
        "corpus_encode_s": round(wall1 - wall0, 2), "corpus_docs_per_s": round(len(texts) / (wall1 - wall0), 1),
        "cpu_utilization": round(((cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime)) / (wall1 - wall0), 2),
        "docs_per_s_by_batch": throughput,
        "query_ms_p50": round(statistics.median(latencies), 1),
        "query_ms_p95": round(latencies[max(0, int(len(latencies) * 0.95) - 1)], 1),
        "peak_rss_mb": round(peak_bytes / 1e6, 1),
        "overall": scores(ranks), "by_category": {k: scores(v) for k, v in sorted(per_category.items())},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "processor": _cpu_name(),
                        "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
                        "transformers": transformers.__version__, "torch_threads": torch.get_num_threads()},
    }


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def run_all(cache: Path, out: Path, names: list[str], devices: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        for device in devices:
            target = out / f"{name}--{device}--{platform.system().lower()}.json"
            proc = subprocess.run([sys.executable, "-B", str(Path(__file__)), "measure", "--cache", str(cache),
                                   "--model", name, "--device", device], capture_output=True, text=True)
            if proc.returncode:
                target.write_text(json.dumps({"model": name, "device": device, "error": proc.stderr.strip()[-2000:]}, indent=2))
                print(f"{name} on {device}: FAILED")
                continue
            target.write_text(proc.stdout)
            result = json.loads(proc.stdout)
            print(f"{name} on {device}: R@5={result['overall']['recall@5']} MRR={result['overall']['mrr']} "
                  f"p50={result['query_ms_p50']}ms load={result['load_s']}s rss={result['peak_rss_mb']}MB")
    report(out)


def report(out: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*--*.json"))]
    lines = ["# Embedding candidates: raw results", "",
             "| model | device | host | R@5 | R@10 | MRR | long-ctx R@5 | multilingual R@5 | query p50/p95 ms | corpus docs/s | load s | peak RSS MB | download MB | dims | max tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['model']} | {row['device']} | | failed: see JSON | | | | | | | | | | | |")
            continue
        category = row["by_category"]
        host = f"{row['environment']['processor']} ({row['environment']['platform'].split('-')[0]})"
        lines.append(f"| {row['model']} | {row['device']} | {host} | {row['overall']['recall@5']} | {row['overall']['recall@10']} | "
                     f"{row['overall']['mrr']} | {category.get('long_context', {}).get('recall@5', '-')} | "
                     f"{category.get('multilingual', {}).get('recall@5', '-')} | {row['query_ms_p50']}/{row['query_ms_p95']} | "
                     f"{row['corpus_docs_per_s']} | {row['load_s']} | {row['peak_rss_mb']} | {row['download_mb']} | {row['dims']} | {row['max_seq_length']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "measure", "all", "report"])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--device", action="append", dest="devices")
    options = parser.parse_args()
    names = options.models or list(MODELS)
    if options.command == "download":
        download(options.cache, names)
    elif options.command == "measure":
        print(json.dumps(measure(options.cache, names[0], (options.devices or ["cpu"])[0]), indent=2))
    elif options.command == "all":
        run_all(options.cache, options.out, names, options.devices or ["cpu"])
    else:
        report(options.out)


if __name__ == "__main__":
    main()
