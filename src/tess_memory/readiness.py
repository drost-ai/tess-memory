"""Can this memory configuration remember, before the first conversation?

Each check exercises the real thing and leaves nothing behind: the memory
store opens (creating its directory and schema on a new install, which is
what the first conversation would do anyway); personal memory accepts a
write and returns it, inside a transaction that is always rolled back; the
embedding model loads from local files only and returns a finite vector of
the configured size; the processing role's profile and credential resolve.
Nothing here downloads a model or calls a provider: a model that is not on
disk is reported missing, and whether a provider answers is the host's
connection check. Results are ``{capability: (ready, code)}``; codes are
short and never carry stored content.
"""
from __future__ import annotations

import math
from typing import Any, Callable

from .config import Config, validate_config

CAPABILITIES = ("memory_store", "personal_memory_write", "personal_memory_read", "embedding", "memory_processing")
PROBE_KEY = "readiness-probe"


class _Rollback(Exception):
    pass


def _failed(error: BaseException) -> tuple[bool, str]:
    return False, type(error).__name__


def check(cfg: Config, *, embedder: Any = None, load_embedder: Callable[[Config], Any] | None = None) -> dict[str, tuple[bool, str]]:
    results: dict[str, tuple[bool, str]] = {}
    results["memory_store"] = _store(cfg)
    results.update(_personal(cfg))
    results["embedding"] = _embedding(cfg, embedder, load_embedder)
    results["memory_processing"] = _processing(cfg)
    return results


def _store(cfg: Config) -> tuple[bool, str]:
    from .store import MemoryStore
    try:
        cfg.ensure_dirs()
        store = MemoryStore(cfg.db_path)
        try:
            store.db.execute("SELECT 1").fetchone()
        finally:
            store.db.close()
        return True, "ok"
    except Exception as error:
        return _failed(error)


def _personal(cfg: Config) -> dict[str, tuple[bool, str]]:
    if not cfg.personal.enabled:
        return {"personal_memory_write": (False, "personal_disabled"), "personal_memory_read": (False, "personal_disabled")}
    from .autobiography import AutobiographyStore
    try:
        principal = cfg.personal.principal(role="author", session_id="setup-readiness")  # authoring needs a bound session
        store = AutobiographyStore(cfg.db_path, principal, context_budget=cfg.personal.context_budget)
    except Exception as error:
        failed = _failed(error)
        return {"personal_memory_write": failed, "personal_memory_read": failed}
    write = read = (False, "not_reached")
    try:
        with store.transaction():
            store.db.execute("INSERT OR REPLACE INTO autobio_meta VALUES(?, ?)", (PROBE_KEY, "1"))
            write = (True, "ok")
            row = store.db.execute("SELECT value FROM autobio_meta WHERE key=?", (PROBE_KEY,)).fetchone()
            read = (True, "ok") if row and row[0] == "1" else (False, "probe_not_read")
            raise _Rollback()
    except _Rollback:
        pass
    except Exception as error:
        failed = _failed(error)
        write = write if write[0] else failed
        read = failed
    finally:
        store.close()
    return {"personal_memory_write": write, "personal_memory_read": read}


def _embedding(cfg: Config, embedder: Any, load_embedder: Callable[[Config], Any] | None) -> tuple[bool, str]:
    try:
        if embedder is None:
            if load_embedder is None:
                from .embed import make_embedder
                load_embedder = make_embedder
            embedder = load_embedder(cfg)
        vector = [float(value) for value in embedder.embed_query("Is memory ready?")]
    except Exception as error:
        return _failed(error)
    if not vector or not all(math.isfinite(value) for value in vector):
        return False, "invalid_vector"
    if cfg.embed_dims and len(vector) != cfg.embed_dims:
        return False, "dimension_mismatch"
    return True, "ok"


def _processing(cfg: Config) -> tuple[bool, str]:
    if not cfg.engine_profile:
        return False, "no_processing_profile"
    try:
        validate_config(cfg, roles={"distillation"})
        return True, "ok"
    except Exception as error:
        return _failed(error)
