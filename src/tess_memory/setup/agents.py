"""Registering the memory server with the agents on this machine.

Whether an agent already has the server is read from its own configuration
file. Adding it goes through the agent's own command (``claude mcp``,
``codex mcp``), so its configuration is only ever changed by the agent
itself, and only when the person asks. Connecting also adds the usage
instructions to the agent's instruction file (see ``guidance``): the tools
alone do not make an agent use them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import guidance
from .providers import SetupError

SERVER = "tess-memory"


def server_entry(settings: Path) -> dict[str, Any]:
    """How any MCP client starts the memory server for this store."""
    return {"command": sys.executable, "args": ["-m", "tess_memory.cli", "serve"],
            "env": {"TESS_MEMORY_SETTINGS": str(settings)}}


def _claude_config(home: Path) -> Path:
    # Claude Code keeps the default home's configuration beside it, and any
    # other home's (CLAUDE_CONFIG_DIR) inside it.
    return home.parent / ".claude.json" if home.name == ".claude" else home / ".claude.json"


def _registered(kind: str, home: Path) -> dict[str, Any] | None:
    try:
        if kind == "claude-code":
            servers = json.loads(_claude_config(home).read_text(encoding="utf-8")).get("mcpServers") or {}
        else:
            servers = tomllib.loads((home / "config.toml").read_text(encoding="utf-8")).get("mcp_servers") or {}
        entry = servers.get(SERVER)
        return entry if isinstance(entry, dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def targets(sources: list[dict[str, Any]], settings: Path) -> list[dict[str, Any]]:
    """Each agent home among the chosen sources, and whether it has the server."""
    found, seen = [], set()
    for source in sources:
        kind, home = source.get("type"), source.get("home")
        if kind not in ("claude-code", "codex") or not home or home in seen:
            continue
        seen.add(home)
        entry = _registered(kind, Path(home))
        env = (entry or {}).get("env") or {}
        found.append({"id": f"{kind}:{home}", "type": kind, "home": home, "label": source.get("label"),
                      "display_path": source.get("display_home") or home,
                      "cli": shutil.which("claude" if kind == "claude-code" else "codex") is not None,
                      "registered": entry is not None,
                      "current": bool(entry) and env.get("TESS_MEMORY_SETTINGS") == str(settings),
                      "guidance": guidance.present(kind, Path(home)),
                      "guidance_file": guidance.FILES[kind]})
    return found


def register(target_id: str, sources: list[dict[str, Any]], settings: Path, *, run: Any = subprocess.run) -> dict[str, Any]:
    """Add the server to one agent, replacing an earlier registration of it,
    and add the usage instructions to the agent's instruction file."""
    target = next((item for item in targets(sources, settings) if item["id"] == target_id), None)
    if target is None:
        raise SetupError("unknown_agent")
    if target["current"]:  # already registered for this store: only the instructions can be missing
        return _instruct(target, target_id, sources, settings)
    if not target["cli"]:
        raise SetupError("agent_command_not_found")
    entry = server_entry(settings)
    home = Path(target["home"])
    environment = dict(os.environ)
    if target["type"] == "claude-code":
        if home.name != ".claude":
            environment["CLAUDE_CONFIG_DIR"] = str(home)
        remove = ["claude", "mcp", "remove", "--scope", "user", SERVER]
        add = ["claude", "mcp", "add-json", "--scope", "user", SERVER, json.dumps({"type": "stdio", **entry})]
    else:
        environment["CODEX_HOME"] = str(home)
        remove = ["codex", "mcp", "remove", SERVER]
        add = ["codex", "mcp", "add", SERVER, *[f"--env={key}={value}" for key, value in entry["env"].items()],
               "--", entry["command"], *entry["args"]]
    try:
        if target["registered"]:
            run(remove, env=environment, capture_output=True, text=True, timeout=60)
        result = run(add, env=environment, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise SetupError("agent_command_failed") from None
    if result.returncode != 0:
        raise SetupError("agent_command_failed", (result.stderr or result.stdout or "").strip()[:300])
    return _instruct(target, target_id, sources, settings)


def _instruct(target: dict[str, Any], target_id: str, sources: list[dict[str, Any]], settings: Path) -> dict[str, Any]:
    try:
        outcome = guidance.install(target["type"], Path(target["home"]))
    except OSError:
        raise SetupError("guidance_not_written", target["guidance_file"]) from None
    return {**next(item for item in targets(sources, settings) if item["id"] == target_id), "guidance_outcome": outcome}
