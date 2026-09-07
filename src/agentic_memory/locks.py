"""Process exclusion shared by manual and scheduled ingestion of every source type."""
from contextlib import contextmanager
from pathlib import Path
import fcntl


class IngestionBusy(ValueError):
    pass


@contextmanager
def ingestion_lock(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "ingest.lock"
    path.touch(mode=0o600, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IngestionBusy("another ingester owns this memory store") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
