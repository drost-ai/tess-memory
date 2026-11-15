"""Bounded recent-source priority plus a durable rotating background scan."""
from bisect import bisect_right

SCAN_CURSOR = 'incremental:scan_cursor'
RECENT_SECONDS = 15 * 60


def source_scan_order(found, *, limit, cursor, known_paths, now):
    """Return (source tuple, advances_background_cursor) in visit order.

    Source tuples keep the adapter's (mtime_ns, store, path, project, parent)
    shape. At most half the read budget, capped at eight, can be consumed by
    recent sources before background work gets a turn. Skips consume no reads.
    """
    ranked = sorted(found, key=lambda row: row[0], reverse=True)
    if limit is None:
        return [(row, False) for row in ranked]
    recent_cap = min(8, max(0, limit // 2))
    recent_ids = {i for i, row in enumerate(ranked)
                  if row[0] >= (now - RECENT_SECONDS) * 1e9}
    selected = set(sorted(recent_ids)[:recent_cap])
    recent = [row for i, row in enumerate(ranked) if i in selected]
    background = sorted((row for i, row in enumerate(ranked) if i not in selected), key=lambda row: str(row[2]))
    paths = [str(row[2]) for row in background]
    if cursor is None:
        # First deployment should reach work the old head-first scan never saw.
        start = next((i for i, path in enumerate(paths) if path not in known_paths), 0)
    else:
        start = bisect_right(paths, cursor)
    background = background[start:] + background[:start]
    return [(row, False) for row in recent] + [(row, True) for row in background]
