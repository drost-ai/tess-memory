"""Read original or Zstandard-compressed transcripts without materializing them."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


def is_zstd(path: Path) -> bool:
    return path.name.endswith('.jsonl.zst')


@contextmanager
def open_source(path: Path):
    if not is_zstd(path):
        with path.open('rb') as stream:
            yield stream
        return
    from backports import zstd
    try:
        with zstd.open(path, 'rb') as stream:
            yield stream
    except (EOFError, zstd.ZstdError) as exc:
        raise OSError(f'{path}: invalid or incomplete Zstandard archive') from exc


@lru_cache(maxsize=128)
def _decoded_size(path: Path, revision: tuple) -> int:
    # ZstdFile seeks by streaming, with bounded memory. Cache by physical
    # revision so successive transcript pages do not repeat the size scan.
    with open_source(path) as stream:
        stream.seek(0, 2)
        return stream.tell()


def source_size(path: Path) -> int:
    info = path.stat()
    if not is_zstd(path):
        return info.st_size
    return _decoded_size(path, (info.st_dev, info.st_ino, info.st_size,
                               info.st_mtime_ns, info.st_ctime_ns))
