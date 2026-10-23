"""Model downloads: a fake fetch in a temporary home; nothing touches the network."""
import hashlib
import threading
from collections import namedtuple

import pytest

from agentic_memory import model_download
from agentic_memory.model_download import DownloadError

WEIGHTS = b"synthetic weights"


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    entry = dict(model_download.CATALOG["multilingual-e5-small"], bytes=len(WEIGHTS) + 2,
                 weights={"model.safetensors": hashlib.sha256(WEIGHTS).hexdigest()})
    monkeypatch.setitem(model_download.CATALOG, "multilingual-e5-small", entry)


def fetcher(*, weights=WEIGHTS, fail=None, calls=None, steps=2):
    def fetch(repo, *, revision, local_dir, allow_patterns, tqdm_class):
        if calls is not None:
            calls.append((repo, revision, tuple(allow_patterns)))
        with tqdm_class(total=steps) as bar:
            for _ in range(steps):
                bar.update(1)
        if fail:
            raise fail
        from pathlib import Path
        (Path(local_dir) / "model.safetensors").write_bytes(weights)
        (Path(local_dir) / "config.json").write_text("{}")
    return fetch


def test_a_verified_download_is_installed_once_and_described(tmp_path):
    calls, seen = [], []
    path = model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher(calls=calls), progress=lambda done, total: seen.append(done))
    assert path == tmp_path / "models" / "multilingual-e5-small" and (path / "model.safetensors").read_bytes() == WEIGHTS
    assert calls[0][:2] == ("intfloat/multilingual-e5-small", model_download.CATALOG["multilingual-e5-small"]["revision"])
    assert "*.safetensors" in calls[0][2] and not any(".py" in pattern for pattern in calls[0][2])
    assert seen[-1] == len(WEIGHTS) + 2 and model_download.installed(tmp_path, "multilingual-e5-small")
    assert model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher(fail=AssertionError("not fetched again"))) == path
    described = model_download.describe(tmp_path, "multilingual-e5-small")
    assert (described["installed"], described["license"], described["dims"]) == (True, "mit", 384)
    assert model_download.embedding_settings(tmp_path, "multilingual-e5-small")["query_prefix"] == "query: "


def test_bytes_that_fail_verification_are_never_used_or_kept(tmp_path):
    with pytest.raises(DownloadError) as caught:
        model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher(weights=b"tampered"))
    assert caught.value.code == "checksum_mismatch"
    assert not (tmp_path / "models" / "multilingual-e5-small").exists()
    assert not (tmp_path / "models" / ".multilingual-e5-small.partial").exists()


def test_cancel_keeps_the_partial_for_a_resume_and_installs_nothing(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadError) as caught:
        model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher(), cancel=cancel)
    assert caught.value.code == "cancelled" and not model_download.installed(tmp_path, "multilingual-e5-small")
    assert (tmp_path / "models" / ".multilingual-e5-small.partial").is_dir()
    model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher())
    assert model_download.installed(tmp_path, "multilingual-e5-small")


@pytest.mark.parametrize("error,code", [(ConnectionError("offline"), "download_failed"),
                                        (OSError(28, "No space left on device"), "insufficient_space"),
                                        (RuntimeError("proxy said no"), "download_failed")])
def test_failures_have_codes_not_library_text(tmp_path, error, code):
    with pytest.raises(DownloadError) as caught:
        model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher(fail=error))
    assert caught.value.code == code and str(caught.value) == code


def test_too_little_disk_space_refuses_before_fetching(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(model_download.shutil, "disk_usage", lambda path: usage(10, 10, 0))
    with pytest.raises(DownloadError) as caught:
        model_download.download(tmp_path, "multilingual-e5-small", fetch=fetcher(fail=AssertionError("never")))
    assert caught.value.code == "insufficient_space"
    with pytest.raises(DownloadError):
        model_download.describe(tmp_path, "not-in-the-catalogue")
