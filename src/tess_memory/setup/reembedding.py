"""Changing the embedding model of a store that already holds memory.

Vectors from two models cannot be compared, so a store built with one model
cannot simply be pointed at another. Setup notices when the model chosen on
the embeddings step is not the one the store was built with, and offers to
re-embed: every stored row is embedded again with the new model, nothing is
read or distilled again, and the plan is not touched (see ``tess_memory.reembed``).

The store's current settings stay in force while that runs, so search and
ingestion carry on with the old model. The new settings are written only once
the store has switched; if the work is stopped or fails, nothing has changed.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from . import apply
from .providers import SetupError
from .vault import Vault, write_private

NEXT = "settings.next.toml"


def _stored(home: Path) -> tuple[str | None, int]:
    """The model the store was built with and how many rows it holds, read
    without opening the store for writing."""
    database = Path(home) / "memory.db"
    if not database.exists():
        return None, 0
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
        try:
            row = connection.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
            rows = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        finally:
            connection.close()
        return (row[0] if row else None), rows
    except sqlite3.Error:
        return None, 0


def stored_sessions(home: Path) -> int:
    """How many sessions the store holds, read without opening it for writing."""
    database = Path(home) / "memory.db"
    if not database.exists():
        return 0
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
        try:
            return connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error:
        return 0


def _target(home: Path, registry: Path, draft: dict[str, Any]):
    """The configuration the chosen settings describe, and its embedder's name."""
    from ..config import load_config
    path = Path(home) / NEXT
    write_private(path, apply.render(Path(home), registry, draft, settings=Path(home) / apply.SETTINGS))
    try:
        cfg = load_config(path)
    except (ValueError, TypeError) as error:
        path.unlink(missing_ok=True)
        raise SetupError("invalid_settings", str(error)[:240]) from None
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if cfg.embed_provider == "local":
        from ..embed import LocalEmbedder
        descriptor = LocalEmbedder(cfg.embed_model, query_prefix=cfg.embed_query_prefix,
                                   passage_prefix=cfg.embed_passage_prefix).descriptor
    else:
        descriptor = f"{cfg.embed_provider}:{draft['embedding']['model']}:{cfg.embed_dims}d"
    return cfg, descriptor


def pending(home: Path, registry: Path, draft: dict[str, Any]) -> dict[str, Any] | None:
    """What a re-embed would involve, or None when the chosen model is the
    store's own (or the store holds nothing yet). Leaves the chosen settings
    in ``settings.next.toml`` when one is needed."""
    from ..store import same_vector_space
    current, rows = _stored(home)
    if current is None or rows == 0:
        return None
    _, descriptor = _target(home, registry, draft)
    if same_vector_space(current, descriptor):
        (Path(home) / NEXT).unlink(missing_ok=True)
        return None
    choice = draft["embedding"]
    return {"current": current, "target": descriptor, "rows": rows, "hosted": choice["mode"] != "local",
            "label": choice["name"] if choice["mode"] == "local" else choice["model"]}


def current_embedding(home: Path, profiles: dict[str, Any]) -> dict[str, Any] | None:
    """The ``[embedding]`` table the store is running on, when the rest of the
    settings can be rewritten around it: it must still describe the model the
    store was built with, and be reachable from setup's own provider file (a
    local model, or a hosted one setup connected). Otherwise None, and the
    whole settings change waits for the re-embed."""
    import tomllib
    from ..store import same_vector_space
    try:
        table = tomllib.loads((Path(home) / apply.SETTINGS).read_text(encoding="utf-8")).get("embedding")
    except (OSError, ValueError):
        return None
    if not isinstance(table, dict) or not table:
        return None
    provider = table.get("provider", "local")
    if provider == "local":
        from ..embed import LocalEmbedder
        model = str(table.get("local_model") or "")
        if not model:
            return None
        descriptor = LocalEmbedder(model, query_prefix=table.get("query_prefix"), passage_prefix=table.get("passage_prefix")).descriptor
    elif provider in profiles and isinstance(table.get("dims"), int):
        descriptor = f"{provider}:{profiles[provider].get('defaultModel')}:{table['dims']}d"
    else:
        return None
    stored, _ = _stored(home)
    try:
        apply.embedding_table(table)
    except SetupError:
        return None
    return table if same_vector_space(stored, descriptor) else None


class Reembedding:
    """The one re-embed that may be running, and what happens when it ends."""

    def __init__(self, vault: Vault) -> None:
        self.vault = vault
        self.info: dict[str, Any] | None = None
        self.job: dict[str, Any] = {"state": "idle"}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def offer(self, info: dict[str, Any] | None) -> None:
        if not self.running():
            self.info = info
            self.job = {"state": "idle"}

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        """Drop the offer. Anything already staged in the store is left for a
        later run to use; the store itself has not changed."""
        if self.running():
            raise SetupError("reembed_running")
        self.info = None
        self.job = {"state": "idle"}
        (self.vault.home / NEXT).unlink(missing_ok=True)

    def stop(self) -> None:
        self._stop.set()

    def start(self, registry: Path, *, after: Any, embedder: Any = None) -> dict[str, Any]:
        """Re-embed in the background; ``after()`` runs once the store has
        switched, to write the settings that go with it."""
        if self.info is None:
            raise SetupError("nothing_to_reembed")
        if self.running():
            raise SetupError("reembed_running")
        self._stop = threading.Event()
        self.job = {"state": "running", "done": 0, "total": self.info["rows"]}
        home, draft = self.vault.home, self.vault.draft

        def shown(error: Exception) -> str:
            return error.detail if isinstance(error, SetupError) and error.detail else f"{type(error).__name__}: {str(error)[:240]}"

        def run() -> None:
            from .. import reembed
            from ..embed import make_embedder
            from ..store import MemoryStore
            store = None
            try:
                cfg, _ = _target(home, registry, draft)
                store = MemoryStore(cfg.db_path)
                result = reembed.reembed(store, embedder or make_embedder(cfg), should_stop=self._stop.is_set,
                                         progress=lambda done, total: self.job.update(done=done, total=total))
                if result["state"] == "stopped":
                    self.job.update(state="stopped")
                    return
                try:
                    after()
                except Exception as error:  # noqa: BLE001
                    # The store is on the new model and its settings are not.
                    # Say so: starting again finds nothing left to embed and
                    # only writes the settings.
                    self.job.update(state="failed", switched=True, detail=shown(error))
                    return
                (home / NEXT).unlink(missing_ok=True)
                self.info = None
                self.job.update(state="switched")
            except Exception as error:  # noqa: BLE001 (shown to the person, shortened)
                self.job.update(state="failed", detail=shown(error))
            finally:
                if store is not None:
                    store.close()

        self._thread = threading.Thread(target=run, name="tess-memory-reembed", daemon=True)
        self._thread.start()
        return self.job
