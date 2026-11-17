"""Writing the settings file from the choices made in setup.

The file is ordinary: the same one the guides describe, readable and
editable by hand. An existing settings file is never overwritten without a
dated copy beside it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .providers import SetupError
from .vault import Vault, write_private

SETTINGS = "settings.toml"


def _s(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)  # a TOML basic string


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


def render(home: Path, registry: Path, draft: dict[str, Any]) -> str:
    gaps = missing(draft)
    if gaps:
        raise SetupError("incomplete", ", ".join(gaps))
    model, embedding = draft["model"], draft["embedding"]
    lines = ["# Tess Memory settings, written by `tess-memory setup`.",
             "# Edit by hand if you like; running setup again rewrites this file and keeps a dated copy.", ""]
    for source in draft["sources"]:
        if not source.get("enabled", True):
            continue
        lines += ["[[data_stores]]", f"name = {_s(source['name'])}", f"type = {_s(source['type'])}", f"path = {_s(source['path'])}"]
        if source["type"] in ("transcript",) or source.get("project"):
            lines.append(f"project = {_s(source.get('project') or source['name'])}")
        lines.append("")
    lines += ["[storage]", f"data_dir = {_s(home)}", "",
              "[llm]", f"settings = {_s(registry)}", f"profile = {_s(model['profile'])}", f"model = {_s(model['distill_model'])}",
              "workers = 4", "",
              "[search]", "agent = true", f"profile = {_s(model['profile'])}", f"model = {_s(model['search_model'])}", "",
              "[embedding]"]
    if embedding["mode"] == "local":
        settings = embedding["settings"]
        lines += ['provider = "local"', f"local_model = {_s(settings['local_model'])}", f"dims = {int(settings['dims'])}",
                  f"query_prefix = {_s(settings['query_prefix'])}", f"passage_prefix = {_s(settings['passage_prefix'])}"]
    else:
        lines += ['provider = "embeddings"', f"dims = {int(embedding['dims'])}", 'query_prefix = ""',
                  f"send_dimensions = {'true' if embedding['send_dimensions'] else 'false'}", "batch_size = 64"]
    lines += ["", "[ingest]", "quiet_seconds = 900        # leave a session alone until it has been still this long",
              "incremental = true         # read only the new, finished part of a session", ""]
    return "\n".join(lines)


def write(vault: Vault, registry: Path) -> dict[str, Any]:
    """Write the settings file; returns its path and the copy of the one it replaced."""
    path = vault.home / SETTINGS
    text = render(vault.home, registry, vault.draft)
    backup = None
    if path.exists() and path.read_text(encoding="utf-8") != text:
        backup = path.with_name(f"{SETTINGS}.before-setup-{time.strftime('%Y%m%d-%H%M%S')}")
        backup.write_bytes(path.read_bytes())
        backup.chmod(0o600)
    write_private(path, text)
    from ..config import load_config
    load_config(path)  # the file must be one the rest of Tess Memory accepts
    vault.meta["applied"] = {"settings": str(path), "at": time.time()}
    vault.save()
    return {"settings": str(path), "providers": str(registry), "backup": str(backup) if backup else None}
