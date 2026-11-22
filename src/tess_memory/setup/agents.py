"""Registering the memory server with the agents on this machine.

Whether an agent already has the server is read from its own configuration
file. Adding it goes through the agent's own command (``claude mcp``,
``codex mcp``, ``agy mcp``), so its configuration is only ever changed by the agent
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
COMMANDS = {"claude-code": "claude", "codex": "codex", "antigravity": "agy"}


def server_entry(settings: Path) -> dict[str, Any]:
    """How any MCP client starts the memory server for this store."""
    return {"command": sys.executable, "args": ["-m", "tess_memory.cli", "serve"],
            "env": {"TESS_MEMORY_SETTINGS": str(settings)}}


def _claude_config(home: Path) -> Path:
    # Claude Code keeps the default home's configuration beside it, and any
    # other home's (CLAUDE_CONFIG_DIR) inside it.
    return home.parent / ".claude.json" if home.name == ".claude" else home / ".claude.json"


def _is_memory_server(entry: Any) -> bool:
    """Whether a registered server is Tess Memory, whatever it was named."""
    if not isinstance(entry, dict):
        return False
    words = [str(entry.get("command") or ""), *[str(item) for item in entry.get("args") or []]]
    ours = any(word in ("tess-memory", "tess_memory.cli") or word.endswith("/tess-memory") for word in words)
    return ours and any(word in ("serve", "codex-serve") for word in words)


def _settings_of(entry: dict[str, Any]) -> str | None:
    """The settings file a registered memory server was given."""
    named = (entry.get("env") or {}).get("TESS_MEMORY_SETTINGS")
    args = [str(item) for item in entry.get("args") or []]
    if not named and "--settings" in args[:-1]:
        named = args[args.index("--settings") + 1]
    return str(Path(named).expanduser()) if named else None


def _registered(kind: str, home: Path) -> tuple[str, dict[str, Any]] | None:
    """The agent's Tess Memory server and the name it is registered under.
    The usual name wins; otherwise any server that runs Tess Memory counts,
    so an install made by hand under another name is not given a second one."""
    try:
        if kind == "claude-code":
            servers = json.loads(_claude_config(home).read_text(encoding="utf-8")).get("mcpServers") or {}
        elif kind == "antigravity":
            servers = json.loads((home / "config" / "mcp_config.json").read_text(encoding="utf-8")).get("mcpServers") or {}
        else:
            servers = tomllib.loads((home / "config.toml").read_text(encoding="utf-8")).get("mcp_servers") or {}
        if isinstance(servers.get(SERVER), dict):
            return SERVER, servers[SERVER]
        return next(((name, entry) for name, entry in servers.items() if _is_memory_server(entry)), None)
    except (OSError, ValueError, AttributeError):
        return None


def targets(sources: list[dict[str, Any]], settings: Path) -> list[dict[str, Any]]:
    """Each agent home given, and whether it has the server. ``sources`` is
    any list of records with a ``type`` and a ``home``."""
    found, seen = [], set()
    for source in sources:
        kind, home = source.get("type"), source.get("home")
        if kind not in COMMANDS or not home or home in seen:
            continue
        seen.add(home)
        name, entry = _registered(kind, Path(home)) or (None, None)
        found.append({"id": f"{kind}:{home}", "type": kind, "home": home, "label": source.get("label"),
                      "display_path": source.get("display_home") or home,
                      "cli": shutil.which(COMMANDS[kind]) is not None,
                      "registered": entry is not None, "registered_as": name,
                      "current": name == SERVER and _settings_of(entry) == str(settings),
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
        remove = ["claude", "mcp", "remove", "--scope", "user", target["registered_as"] or SERVER]
        add = ["claude", "mcp", "add-json", "--scope", "user", SERVER, json.dumps({"type": "stdio", **entry})]
    elif target["type"] == "antigravity":
        # agy has one home, ~/.gemini, and takes its flags before the server's name.
        remove = ["agy", "mcp", "remove", target["registered_as"] or SERVER]
        add = ["agy", "mcp", "add", *[item for key, value in entry["env"].items() for item in ("--env", f"{key}={value}")],
               SERVER, "--", entry["command"], *entry["args"]]
    else:
        environment["CODEX_HOME"] = str(home)
        remove = ["codex", "mcp", "remove", target["registered_as"] or SERVER]
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
