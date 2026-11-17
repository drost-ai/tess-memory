"""Keeping the store current after setup: the background scheduler.

On macOS setup can install the scheduler as a login item (launchd). It never
replaces a scheduler that is already installed; elsewhere it shows the one
command to run under the platform's own service manager.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .providers import SetupError
from .vault import write_private

CONFIG = "scheduler.toml"


def _config_text(home: Path, settings: Path) -> str:
    quoted = lambda value: json.dumps(str(value))  # noqa: E731
    return "\n".join([
        "# Written by `tess-memory setup`: keeps this store current in the background.",
        "[service]", f"data_dir = {quoted(home / 'service')}", "",
        "[[stores]]", 'name = "main"', f"settings = {quoted(settings)}", "enabled = true",
        "poll_seconds = 300", "sources_per_run = 16", "model_calls_per_run = 128", ""])


def status(home: Path) -> dict[str, Any]:
    from ..scheduler_install import DEFAULT_LABEL, agent_path
    config = home / CONFIG
    command = f"{sys.executable} -m tess_memory.cli scheduler --config {config} serve"
    if sys.platform != "darwin":
        return {"supported": False, "installed": False, "command": command}
    plist = agent_path(DEFAULT_LABEL)
    ours = False
    if plist.exists():
        try:
            import plistlib
            ours = str(config) in (plistlib.loads(plist.read_bytes()).get("ProgramArguments") or [])
        except Exception:  # noqa: BLE001 (unreadable: treated as someone else's)
            ours = False
    return {"supported": True, "installed": plist.exists(), "ours": ours, "command": command}


def install(home: Path, settings: Path) -> dict[str, Any]:
    current = status(home)
    if not current["supported"]:
        raise SetupError("scheduler_unsupported")
    if current["installed"] and not current["ours"]:
        raise SetupError("scheduler_already_installed")
    from ..scheduler_install import install as install_agent
    config = home / CONFIG
    write_private(config, _config_text(home, settings))
    try:
        install_agent(config)
    except Exception as error:  # noqa: BLE001
        raise SetupError("scheduler_install_failed", f"{type(error).__name__}: {str(error)[:200]}") from None
    return status(home)
