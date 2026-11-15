"""Time keys shared by ingestion and metadata reconciliation."""

from datetime import datetime, timedelta


def week_start(ts: str | None) -> str | None:
    """Monday of the timestamp's ISO week, matching weekly digest ownership."""
    try:
        day = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None
    return (day - timedelta(days=day.weekday())).isoformat()
