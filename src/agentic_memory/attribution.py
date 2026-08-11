"""Logical project attribution and metadata-only reconciliation.

Discovery always happens in the source's native workspace namespace first so
cwd-based exclusion rules retain their original meaning. A data store may then
map that source project to a stable logical project. Reconciliation changes
only project metadata; it never parses, distills, or embeds a transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import agent_core
from .config import Config, DataStore
from .store import MemoryStore
from .transcript import discover_transcripts, parse_transcript


Parser = Callable[..., object]
Discovered = tuple[Path, str, str | None]


@dataclass
class AttributionStats:
    scanned: int = 0
    tracked: int = 0
    unchanged: int = 0
    changed: int = 0
    reattributed: int = 0
    memories_reattributed: int = 0
    retired_profiles: int = 0
    untracked: int = 0
    changes: list[dict] = field(default_factory=list)


def source_adapter(ds: DataStore) -> tuple[list[Discovered], Parser]:
    """Return native discovery results and parser for one configured store.

    Native discovery intentionally sees the source project before a fixed
    logical project override is applied. This preserves existing exclusion
    semantics for engineering stores and temporary runtime workspaces.
    """
    if ds.type == "claude-code":
        return discover_transcripts(ds.path, ds.exclude), parse_transcript
    if ds.type == "agent-core":
        return agent_core.discover_sessions(ds.path, ds.exclude), agent_core.parse_session
    if ds.type == "transcript":
        return (
            agent_core.discover_transcript_files(ds.path, ds.exclude, ds.project),
            agent_core.parse_transcript_file,
        )
    raise ValueError(
        f"data_store '{ds.name}': unknown type '{ds.type}' "
        "(implemented: 'claude-code', 'agent-core', 'transcript')"
    )


def reconcile_project_attribution(
    cfg: Config, store: MemoryStore, *, dry_run: bool = False
) -> AttributionStats:
    """Repair config-driven project drift without model or embedding calls.

    Only already-tracked sessions participate. New or changed transcript files
    remain the ingestion pipeline's responsibility. The function's signature
    deliberately accepts neither an LLM nor an embedder, making it suitable for
    safe migrations of already-paid-for distilled and embedded artifacts.
    """
    stats = AttributionStats()
    assignments: list[tuple[str, str]] = []
    for ds in cfg.data_stores:
        found, _ = source_adapter(ds)
        for path, source_project, _parent in found:
            stats.scanned += 1
            session = store.session_for_file(str(path))
            if session is None:
                stats.untracked += 1
                continue
            stats.tracked += 1
            logical_project = ds.resolve_project(source_project)
            old_project = session["project"]
            if old_project == logical_project:
                stats.unchanged += 1
                continue
            stats.changed += 1
            assignments.append((session["id"], logical_project))
            stats.changes.append({
                "path": str(path),
                "session_id": session["id"],
                "source_project": source_project,
                "old_project": old_project,
                "new_project": logical_project,
            })

    if dry_run or not assignments:
        return stats

    results = store.reattribute_session_projects(assignments)
    for change, result in zip(stats.changes, results, strict=True):
        stats.reattributed += 1
        stats.memories_reattributed += result["memories"]
        stats.retired_profiles += result["retired_profiles"]
        change.update(
            memories=result["memories"],
            retired_profiles=result["retired_profiles"],
        )
    return stats
