"""Find the agents on this machine whose sessions can be read.

Only directory names and file counts are looked at; no session is opened.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

SOURCE_TYPES = ("claude-code", "codex", "tess-core", "transcript")
LABELS = {"claude-code": "Claude Code", "codex": "Codex", "tess-core": "Tess Core agent", "transcript": "Transcript folder"}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "source"


def _count(directory: Path, match, *, depth: int) -> tuple[int, float]:
    """Files under ``directory`` (to ``depth`` levels) that ``match``; newest mtime."""
    total, newest = 0, 0.0
    stack = [(directory, 0)]
    while stack:
        current, level = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    if level < depth:
                        stack.append((Path(entry.path), level + 1))
                elif match(entry.name):
                    total += 1
                    newest = max(newest, entry.stat().st_mtime)
            except OSError:
                continue
    return total, newest


def claude_code(home: Path) -> list[dict[str, Any]]:
    found = []
    for directory in sorted(home.glob(".claude*")):
        projects = directory / "projects"
        if not projects.is_dir():
            continue
        sessions, newest = _count(projects, lambda name: name.endswith(".jsonl"), depth=1)
        if sessions:
            suffix = directory.name[len(".claude"):].strip("-_.")
            found.append({"type": "claude-code", "name": _slug("claude-code-" + suffix) if suffix else "claude-code",
                          "path": str(projects), "home": str(directory), "sessions": sessions, "newest": newest})
    return found


def codex(home: Path) -> list[dict[str, Any]]:
    found = []
    for directory in sorted(home.glob(".codex*")):
        if not directory.is_dir():
            continue
        sessions, newest = 0, 0.0
        for folder in ("sessions", "archived_sessions"):
            count, latest = _count(directory / folder, lambda name: name.startswith("rollout-"), depth=4)
            sessions, newest = sessions + count, max(newest, latest)
        if sessions:
            suffix = directory.name[len(".codex"):].strip("-_.")
            found.append({"type": "codex", "name": _slug("codex-" + suffix) if suffix else "codex",
                          "path": str(directory), "home": str(directory), "sessions": sessions, "newest": newest})
    return found


def detect_sources(home: Path | None = None) -> list[dict[str, Any]]:
    """Every agent home found under the user's home directory, largest first
    within each kind. The default home of each agent comes first."""
    home = Path(home or Path.home())
    found = claude_code(home) + codex(home)
    for item in found:
        item["label"] = LABELS[item["type"]]
        item["display_path"] = _tilde(item["path"], home)
        item["display_home"] = _tilde(item["home"], home)
    return found


def agent_homes(home: Path | None = None) -> list[dict[str, Any]]:
    """Every Claude Code and Codex home on this machine, whether or not it has
    sessions yet and whether or not it was chosen as a source: any of them
    may be given the memory."""
    home = Path(home or Path.home())
    found = []
    for kind, pattern, marks in (("claude-code", ".claude*", ("projects", "settings.json", ".claude.json", "CLAUDE.md")),
                                 ("codex", ".codex*", ("config.toml", "auth.json", "sessions", "AGENTS.md"))):
        for directory in sorted(home.glob(pattern)):
            if directory.is_dir() and any((directory / mark).exists() for mark in marks):
                found.append({"type": kind, "home": str(directory), "label": LABELS[kind], "display_home": _tilde(str(directory), home)})
    return found


def _tilde(path: str, home: Path) -> str:
    try:
        return "~/" + str(Path(path).relative_to(home))
    except ValueError:
        return path


def describe_folder(kind: str, path: str) -> dict[str, Any]:
    """A folder the user named by hand: whether it can be used, and its size."""
    if kind not in SOURCE_TYPES:
        raise ValueError("unknown_source_type")
    directory = Path(path).expanduser()
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("folder_not_found")
    if kind == "claude-code":
        sessions, newest = _count(directory, lambda name: name.endswith(".jsonl"), depth=1)
    elif kind == "codex":
        sessions, newest = 0, 0.0
        for folder in ("sessions", "archived_sessions"):
            count, latest = _count(directory / folder, lambda name: name.startswith("rollout-"), depth=4)
            sessions, newest = sessions + count, max(newest, latest)
    else:
        sessions, newest = _count(directory, lambda name: name.endswith((".json", ".jsonl")), depth=2)
    return {"type": kind, "name": _slug(directory.name.lstrip(".")), "path": str(directory), "home": str(directory),
            "sessions": sessions, "newest": newest, "label": LABELS[kind], "display_path": _tilde(str(directory), Path.home()),
            "display_home": _tilde(str(directory), Path.home())}
