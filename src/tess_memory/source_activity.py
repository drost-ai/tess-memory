"""Fresh physical-source activity checks shared by detection and admission."""
from dataclasses import dataclass
from pathlib import Path
import time

from .audit import source_members


@dataclass(frozen=True)
class SourceActivity:
    modified_at: float
    members: tuple


class SourceNotQuiet(RuntimeError):
    def __init__(self, activity: SourceActivity, quiet_until: float):
        super().__init__('source changed recently; waiting for its quiet window')
        self.activity = activity
        self.quiet_until = quiet_until


def source_activity(path: Path, source_type: str) -> SourceActivity:
    members = []
    for member in source_members(path, source_type):
        info = member.stat()
        members.append((str(member.resolve()), info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns))
    return SourceActivity(max(row[4] for row in members) / 1_000_000_000, tuple(members))


def require_quiet(path: Path, source_type: str, seconds: float):
    activity = source_activity(path, source_type)
    quiet_until = activity.modified_at + seconds
    if seconds and time.time() < quiet_until:
        raise SourceNotQuiet(activity, quiet_until)
    return activity
