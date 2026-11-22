"""Writing the settings file from the choices made in setup.

The file is ordinary: the same one the guides describe, readable and
editable by hand. An existing settings file is never overwritten without a
dated copy beside it.

Setup decides the sources, the model, the search model and the embedding
model. Everything else an existing file says is carried into the new one: the
options on a source that is kept (its project, what it excludes), the keys of
``[ingest]`` and ``[storage]``, the tuning in ``[llm]`` and ``[search]``, and
every section setup has no page for. What names the earlier provider (a
fallback profile, a reasoning effort) and the ``[embedding]`` table are
replaced. Comments are not carried; the dated copy has them.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

from .providers import SetupError
from .vault import Vault, write_private

SETTINGS = "settings.toml"
# The tables setup writes from the choices made on its pages.
MANAGED = ("data_stores", "llm", "search", "embedding")
# What a source may say besides its name, type and path.
SOURCE_OPTIONS = ("project", "exclude", "skip_copies_of", "user", "team")


def _s(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)  # a TOML basic string


def _key(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z0-9_-]+", name) else _s(name)


def _value(value: Any) -> str:
    """Any TOML value on one line."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }" if value else "{}"
    return _s(value)


def _table(header: str, values: dict[str, Any]) -> list[str]:
    """A table and the tables inside it."""
    inner = {k: v for k, v in values.items() if isinstance(v, dict)}
    lines = [header, *[f"{_key(k)} = {_value(v)}" for k, v in values.items() if k not in inner], ""]
    name = header.strip("[]")
    for key, value in inner.items():
        lines += _table(f"[{name}.{_key(key)}]", value)
    return lines


def _tuning(table: Any) -> dict[str, Any]:
    """What an existing ``[llm]`` or ``[search]`` table says besides which
    model to use. A key that names a profile, a model, a credential or a
    reasoning effort belongs to the provider setup replaces, and a fallback
    profile from the earlier provider file would name one that is not there."""
    names = ("profile", "model", "settings", "auth", "effort")
    return {k: v for k, v in table.items() if not any(word in k for word in names)} if isinstance(table, dict) else {}


def existing(settings: Path) -> dict[str, Any]:
    """The settings file already in the store, or nothing if there is none
    or it cannot be read."""
    try:
        return tomllib.loads(Path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def same_folder(path: Any, settings: Path) -> str:
    """A source's folder in one spelling, however the file wrote it."""
    folder = Path(str(path)).expanduser()
    return os.path.realpath(folder if folder.is_absolute() else Path(settings).parent / folder)


def existing_sources(settings: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """The sources an existing settings file lists, by type and folder."""
    found = {}
    for entry in existing(settings).get("data_stores") or []:
        if isinstance(entry, dict) and entry.get("path"):
            found[(str(entry.get("type", "claude-code")), same_folder(entry["path"], settings))] = entry
    return found


def missing(draft: dict[str, Any]) -> list[str]:
    """Which steps still need an answer before a settings file can be written."""
    gaps = []
    if not [s for s in draft.get("sources") or [] if s.get("enabled", True)]:
        gaps.append("sources")
    model = draft.get("model") or {}
    if not (model.get("verified") and model.get("distill_model") and model.get("search_model")):
        gaps.append("model")
    if not draft.get("embedding"):
        gaps.append("embedding")
    return gaps


def embedding_table(values: dict[str, Any]) -> list[str]:
    """The lines of an ``[embedding]`` table that is being kept as it is."""
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            lines.append(f"{key} = {_s(value)}")
        else:
            raise SetupError("embedding_not_kept")
    return lines


def render(home: Path, registry: Path, draft: dict[str, Any], *, keep_embedding: dict[str, Any] | None = None,
           settings: Path | None = None) -> str:
    """The settings file for the choices made. ``keep_embedding`` writes that
    ``[embedding]`` table instead of the chosen one: the store is still being
    moved to the chosen model and keeps working with its own until then.
    ``settings`` is the file being replaced: what it says outside the parts
    setup decides is carried over."""
    gaps = missing(draft)
    if gaps:
        raise SetupError("incomplete", ", ".join(gaps))
    model, embedding = draft["model"], draft["embedding"]
    kept = existing(settings) if settings else {}
    kept_sources = existing_sources(settings) if settings else {}
    lines = ["# Tess Memory settings, written by `tess-memory setup`.",
             "# Edit by hand if you like; running setup again rewrites this file and keeps a dated copy.", ""]
    loose = {k: v for k, v in kept.items() if not isinstance(v, dict) and k not in MANAGED
             and not (isinstance(v, list) and v and all(isinstance(item, dict) for item in v))}
    if loose:
        lines += [*[f"{_key(k)} = {_value(v)}" for k, v in loose.items()], ""]
    for source in draft["sources"]:
        if not source.get("enabled", True):
            continue
        before = kept_sources.get((source["type"], same_folder(source["path"], settings))) if settings else None
        options = {k: before[k] for k in SOURCE_OPTIONS if k in before} if before else {}
        if source.get("project"):
            options["project"] = source["project"]
        if source["type"] == "transcript":
            options.setdefault("project", source["name"])
        lines += ["[[data_stores]]", f"name = {_s(source['name'])}", f"type = {_s(source['type'])}", f"path = {_s(source['path'])}",
                  *[f"{key} = {_value(value)}" for key, value in options.items()], ""]
    storage = {k: v for k, v in (kept.get("storage") or {}).items() if k != "data_dir"} if isinstance(kept.get("storage"), dict) else {}
    lines += [*_table("[storage]", {"data_dir": str(home), **storage}),
              *_table("[llm]", {"settings": str(registry), "profile": model["profile"], "model": model["distill_model"],
                                "workers": 4, **_tuning(kept.get("llm"))}),
              *_table("[search]", {**_tuning(kept.get("search")), "agent": True, "profile": model["profile"],
                                   "model": model["search_model"]}),
              "[embedding]"]
    if keep_embedding is not None:
        lines += embedding_table(keep_embedding)
    elif embedding["mode"] == "local":
        settings = embedding["settings"]
        lines += ['provider = "local"', f"local_model = {_s(settings['local_model'])}", f"dims = {int(settings['dims'])}",
                  f"query_prefix = {_s(settings['query_prefix'])}", f"passage_prefix = {_s(settings['passage_prefix'])}"]
    else:
        lines += [f"provider = {_s(embedding.get('profile', 'embeddings'))}", f"dims = {int(embedding['dims'])}", 'query_prefix = ""',
                  f"send_dimensions = {'true' if embedding['send_dimensions'] else 'false'}", "batch_size = 64"]
    lines += ["", "# quiet_seconds: leave a session alone until it has been still this long.",
              "# incremental: read only the new, finished part of a session."]
    ingest = kept.get("ingest") if isinstance(kept.get("ingest"), dict) else {}
    lines += _table("[ingest]", {"quiet_seconds": 900, "incremental": True, **ingest})
    for name, value in kept.items():
        if name in MANAGED or name in ("storage", "ingest") or name in loose:
            continue
        if isinstance(value, dict):
            lines += _table(f"[{_key(name)}]", value)
        else:  # an array of tables
            for item in value:
                lines += _table(f"[[{_key(name)}]]", item)
    return "\n".join(lines)


def write(vault: Vault, registry: Path, *, keep_embedding: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the settings file; returns its path and the copy of the one it replaced."""
    path = vault.home / SETTINGS
    text = render(vault.home, registry, vault.draft, keep_embedding=keep_embedding, settings=path)
    from ..config import load_config
    trial = path.with_name(SETTINGS + ".new")
    write_private(trial, text)
    try:
        load_config(trial)  # the file must be one the rest of Tess Memory accepts
    except (ValueError, TypeError) as error:
        trial.unlink(missing_ok=True)  # the settings in use stay as they were
        raise SetupError("invalid_settings", str(error)[:240]) from None
    except Exception:
        trial.unlink(missing_ok=True)
        raise
    backup = None
    if path.exists() and path.read_bytes() != text.encode("utf-8"):
        backup = path.with_name(f"{SETTINGS}.before-setup-{time.strftime('%Y%m%d-%H%M%S')}")
        backup.write_bytes(path.read_bytes())
        backup.chmod(0o600)
    trial.replace(path)
    vault.meta["applied"] = {"settings": str(path), "at": time.time()}
    vault.save()
    return {"settings": str(path), "providers": str(registry), "backup": str(backup) if backup else None}
