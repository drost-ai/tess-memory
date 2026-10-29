"""Read bounded pages from the source transcript behind an ingested session.

The memory database stores a session's source path; it does not duplicate the
source transcript.  This module resolves that path within configured data-store
roots and reads it without modifying the transcript or the memory store.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .source_io import open_source, source_size


DEFAULT_PAGE_BYTES = 12_000
MAX_PAGE_BYTES = 24_000
MIN_PAGE_BYTES = 256


@dataclass(frozen=True)
class TranscriptPage:
    content: str
    cursor: int
    next_cursor: int | None
    page_bytes: int
    source_size_bytes: int
    complete: bool


def get_session_source_path(db_path: str | Path, session_id: str) -> str | None:
    """Look up a session source using a SQLite-enforced read-only connection."""
    db = Path(db_path).expanduser().resolve()
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT source_path FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
    finally:
        connection.close()
    return row[0] if row and row[0] else None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_transcript_path(
    source_path: str | Path,
    allowed_roots: Iterable[str | Path],
) -> Path:
    """Resolve a session source to its readable transcript inside allowed roots.

    Bare transcript stores point directly at a JSONL file.  Agent Core stores
    point at a session metadata JSON file; their preferred transcript is the
    full transcript-with-tools file, followed by the Python runtime's messages
    sidecar, then the metadata document itself for runtimes with inline messages.
    """
    source = Path(source_path).expanduser()
    roots = [Path(root).expanduser().resolve() for root in allowed_roots]
    if not roots:
        raise PermissionError("No configured transcript roots are available.")

    if source.suffix == ".json":
        candidates = [
            source.parent / source.stem / "full-transcript-with-tools.jsonl",
            source.with_suffix(".messages.jsonl"),
            source,
        ]
    else:
        candidates = [source]
        if source.suffix == '.jsonl':
            candidates.append(source.with_name(source.name + '.zst'))

    found_outside_root = False
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not any(_is_within(resolved, root) for root in roots):
            found_outside_root = True
            continue
        return resolved

    if found_outside_root:
        raise PermissionError("The session transcript is outside configured data-store roots.")
    raise FileNotFoundError(f"No readable transcript exists for source '{source}'.")


def read_transcript_page(
    path: str | Path,
    *,
    cursor: int = 0,
    max_bytes: int = DEFAULT_PAGE_BYTES,
) -> TranscriptPage:
    """Read one UTF-8-safe page using a byte-offset cursor.

    The returned ``content`` is the exact decoded text slice.  Concatenating
    pages in cursor order reconstructs the source text byte-for-byte when it is
    encoded as UTF-8.  A page may stop just before ``max_bytes`` to avoid
    splitting a multi-byte character.
    """
    if cursor < 0:
        raise ValueError("cursor must be non-negative")
    if not MIN_PAGE_BYTES <= max_bytes <= MAX_PAGE_BYTES:
        raise ValueError(
            f"max_bytes must be between {MIN_PAGE_BYTES} and {MAX_PAGE_BYTES}"
        )

    transcript = Path(path)
    size = source_size(transcript)
    if cursor > size:
        raise ValueError(f"cursor {cursor} is beyond transcript size {size}")

    with open_source(transcript) as fh:
        fh.seek(cursor)
        raw = fh.read(min(max_bytes, size - cursor))

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A bounded read may end inside a valid multi-byte character. Retreat to
        # the previous boundary; all cursors returned by this function therefore
        # remain safe for the next page.
        if exc.reason == "unexpected end of data" and exc.end == len(raw):
            raw = raw[: exc.start]
            content = raw.decode("utf-8")
        else:
            raise ValueError(
                "transcript is not valid UTF-8 at the requested cursor"
            ) from exc

    end = cursor + len(raw)
    complete = end >= size
    return TranscriptPage(
        content=content,
        cursor=cursor,
        next_cursor=None if complete else end,
        page_bytes=len(raw),
        source_size_bytes=size,
        complete=complete,
    )
