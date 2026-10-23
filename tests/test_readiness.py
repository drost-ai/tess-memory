"""Memory readiness: temporary homes, synthetic profiles and embedders; no
model is loaded or downloaded and no provider is contacted."""
import sqlite3
from pathlib import Path

import pytest
import yaml

from agentic_memory import readiness
from agentic_memory.config import parse_manifest


class Embedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, query):
        return self.vector


def config(tmp_path, *, personal=True, dims=4, llm="distill", **embedding):
    registry = tmp_path / "providers.yaml"
    registry.write_text(yaml.safe_dump({"profiles": {
        "distill": {"kind": "openai", "baseUrl": "http://127.0.0.1:9/v1", "defaultModel": "m", "auth": {"kind": "none"}},
        "keyed": {"kind": "openai", "defaultModel": "m", "auth": {"kind": "openai_api_key", "envVar": "SYNTHETIC_UNSET_MEMORY_KEY"}},
    }}))
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    memory = {"storage": {"data_dir": "memory"}, "llm": {"profile": llm}, "search": {"agent": False},
              "embedding": {"provider": "local", "local_model": "synthetic-local", "dims": dims, **embedding}}
    if personal:
        memory["personal"] = {"enabled": True, "owner": "owner", "agent": "tess", "context_subjects": ["owner"],
                              "source_mode": "native", "source_paths": ["sessions"], "timezone": "America/Los_Angeles"}
    return parse_manifest({"schema_version": 1, "mode": "embedded", "runtime_home": str(home), "provider_registry": str(registry),
                           "provenance_source_root": str(home / "sessions"), "config": memory})


def test_a_new_home_is_ready_and_the_probe_leaves_nothing_behind(tmp_path):
    cfg = config(tmp_path)
    results = readiness.check(cfg, embedder=Embedder([0.5, 0.5, 0.5, 0.5]))
    assert results == {capability: (True, "ok") for capability in readiness.CAPABILITIES}
    with sqlite3.connect(cfg.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM autobio_meta WHERE key=?", (readiness.PROBE_KEY,)).fetchone() == (0,)
    assert readiness.check(cfg, embedder=Embedder([0.1] * 4))["personal_memory_write"] == (True, "ok")  # repeatable


@pytest.mark.parametrize("vector,code", [([], "invalid_vector"), ([float("nan")] * 4, "invalid_vector"), ([0.1] * 3, "dimension_mismatch")])
def test_an_embedding_must_be_finite_and_the_configured_size(tmp_path, vector, code):
    assert readiness.check(config(tmp_path), embedder=Embedder(vector))["embedding"] == (False, code)


def test_a_model_missing_from_disk_is_reported_not_downloaded(tmp_path):
    def load(cfg):
        raise OSError("synthetic: local files only")
    assert readiness.check(config(tmp_path), load_embedder=load)["embedding"] == (False, "OSError")


def test_disabled_personal_memory_and_unresolvable_processing_are_not_ready(tmp_path):
    results = readiness.check(config(tmp_path, personal=False, llm="keyed"), embedder=Embedder([0.1] * 4))
    assert results["personal_memory_write"] == results["personal_memory_read"] == (False, "personal_disabled")
    assert results["memory_processing"][0] is False
    assert results["memory_store"] == (True, "ok")


def test_an_unwritable_store_is_not_ready(tmp_path):
    cfg = config(tmp_path)
    data = Path(cfg.db_path).parent
    data.mkdir(parents=True, exist_ok=True)
    data.chmod(0o500)
    try:
        results = readiness.check(cfg, embedder=Embedder([0.1] * 4))
    finally:
        data.chmod(0o700)
    assert results["memory_store"][0] is False and results["personal_memory_write"][0] is False
