"""Explicit, consented downloads of the local embedding model.

Nothing downloads at import, at start or during a check: the owner sees the
model's size, licence and destination first, then asks. Each model in the
catalogue is pinned to a repository revision and its weights' SHA-256 (from
the benchmark receipts), fetched without remote code into a staging folder
under the runtime home, verified, and only then moved into place, so a
half-finished or tampered download is never used. An interrupted download
resumes from its staging folder; cancelling leaves the staging folder for
the next attempt and nothing in place.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

MARKER = ".tess-model.json"
# Only what sentence-transformers needs: no ONNX/OpenVINO exports, no code.
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "1_Pooling/*", "2_Normalize/*"]
CATALOG: dict[str, dict[str, Any]] = {
    "multilingual-e5-small": {
        "repo": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "license": "mit", "bytes": 514948334, "dims": 384, "query_prefix": "query: ", "passage_prefix": "passage: ",
        "weights": {"model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"},
        "recommended": True,
    },
    "bge-small-en-v1.5": {
        "repo": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "license": "mit", "bytes": 134411157, "dims": 384,
        "query_prefix": "Represent this sentence for searching relevant passages: ", "passage_prefix": "",
        "weights": {"model.safetensors": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"},
        "recommended": False,
    },
}


class DownloadError(Exception):
    """A download that did not produce a usable model; ``code`` is static."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Cancelled(Exception):
    pass


def target(home: str | os.PathLike[str], name: str) -> Path:
    if name not in CATALOG:
        raise DownloadError("unknown_model")
    return Path(home) / "models" / name


def describe(home: str | os.PathLike[str], name: str) -> dict[str, Any]:
    """What a download would fetch, where, and whether it is already here."""
    entry = CATALOG[name] if name in CATALOG else None
    if entry is None:
        raise DownloadError("unknown_model")
    path = target(home, name)
    return {"name": name, "repo": entry["repo"], "revision": entry["revision"], "license": entry["license"],
            "bytes": entry["bytes"], "dims": entry["dims"], "path": str(path), "installed": installed(home, name),
            "recommended": entry["recommended"]}


def installed(home: str | os.PathLike[str], name: str) -> bool:
    path = target(home, name)
    try:
        marker = json.loads((path / MARKER).read_text())
    except (OSError, ValueError):
        return False
    return marker.get("revision") == CATALOG[name]["revision"] and all((path / weights).is_file() for weights in CATALOG[name]["weights"])


def embedding_settings(home: str | os.PathLike[str], name: str) -> dict[str, Any]:
    """The [embedding] settings for a downloaded catalogue model."""
    entry = CATALOG[name]
    return {"provider": "local", "local_model": f"./models/{name}", "dims": entry["dims"],
            "query_prefix": entry["query_prefix"], "passage_prefix": entry["passage_prefix"]}


def download(home: str | os.PathLike[str], name: str, *, progress: Callable[[int, int], None] | None = None,
             cancel: threading.Event | None = None, fetch: Callable[..., Any] | None = None) -> Path:
    """Fetch, verify and install one catalogue model; returns its path."""
    entry = CATALOG[name] if name in CATALOG else None
    if entry is None:
        raise DownloadError("unknown_model")
    path = target(home, name)
    if installed(home, name):
        return path
    staging = path.parent / f".{name}.partial"
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    needed = entry["bytes"] - _size(staging)
    if shutil.disk_usage(staging).free < needed + 256 * 1024 * 1024:
        raise DownloadError("insufficient_space")
    cancel = cancel or threading.Event()
    report = progress or (lambda done, total: None)
    total = entry["bytes"]

    from tqdm.auto import tqdm

    class Progress(tqdm):  # huggingface_hub reports transfers through a tqdm class
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # silent: progress goes to the owner, not a console
            super().__init__(*args, **kwargs)

        def update(self, amount=1):
            if cancel.is_set():
                raise Cancelled()
            report(min(_size(staging), total), total)
            return super().update(amount)

    if fetch is None:
        from huggingface_hub import snapshot_download as fetch
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        fetch(entry["repo"], revision=entry["revision"], local_dir=str(staging), allow_patterns=ALLOW, tqdm_class=Progress)
    except Cancelled:
        raise DownloadError("cancelled") from None
    except DownloadError:
        raise
    except OSError as error:
        raise DownloadError("insufficient_space" if getattr(error, "errno", None) == 28 else "download_failed") from None
    except Exception:
        raise DownloadError("download_failed") from None
    for relative, expected in entry["weights"].items():
        if not (staging / relative).is_file() or _sha256(staging / relative) != expected:
            shutil.rmtree(staging, ignore_errors=True)  # never keep bytes that failed verification
            raise DownloadError("checksum_mismatch")
    (staging / MARKER).write_text(json.dumps({"repo": entry["repo"], "revision": entry["revision"], "weights": entry["weights"]}))
    shutil.rmtree(staging / ".cache", ignore_errors=True)
    if path.exists():
        shutil.rmtree(path)  # a stale or partial earlier install, never a verified one
    os.replace(staging, path)
    report(total, total)
    return path


def _size(directory: Path) -> int:
    return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file()) if directory.exists() else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
