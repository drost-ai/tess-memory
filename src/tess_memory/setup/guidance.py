"""Telling a connected agent how to use its memory.

Registering the server gives an agent the tools; this gives it the habit. The
text in ``guidance.md`` goes into the agent's own instruction file for every
project: ``CLAUDE.md`` in a Claude Code home, ``AGENTS.md`` in a Codex home,
``GEMINI.md`` in Antigravity's.
It is added once, at the end, and a file that already carries it is left
exactly as it is. A file that is changed is first copied beside itself.
"""
from __future__ import annotations

import time
from pathlib import Path

HEADING = "Memory and planning (tess-memory MCP)"
FILES = {"claude-code": "CLAUDE.md", "codex": "AGENTS.md", "antigravity": "GEMINI.md"}
AGENT_NAMES = {"claude-code": "Claude", "codex": "Codex", "antigravity": "Antigravity"}


def text(kind: str) -> str:
    """The guidance for one kind of agent. The wording is the same for all;
    only the agent's own name in the opening sentence differs."""
    body = (Path(__file__).parent / "guidance.md").read_text(encoding="utf-8")
    return body.replace("shared Claude memory", f"shared {AGENT_NAMES[kind]} memory")


def path(kind: str, home: Path) -> Path:
    return Path(home) / FILES[kind]


def present(kind: str, home: Path) -> bool:
    try:
        return HEADING in path(kind, home).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False


def install(kind: str, home: Path) -> str:
    """Add the guidance to the agent's instruction file.
    Returns ``present`` (already there, untouched), ``added`` or ``created``."""
    target = path(kind, home)
    if present(kind, home):
        return "present"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text(kind), encoding="utf-8")
        return "created"
    existing = target.read_text(encoding="utf-8")
    backup = target.with_name(f"{target.name}.before-tess-memory-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_bytes(target.read_bytes())
    backup.chmod(target.stat().st_mode & 0o777)
    separator = "" if not existing.strip() else ("\n" if existing.endswith("\n") else "\n\n")
    with target.open("a", encoding="utf-8") as handle:  # appending keeps the file's own mode and owner
        handle.write(separator + text(kind))
    return "added"
