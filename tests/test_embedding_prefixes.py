"""Explicit embedding prefixes: a fake loaded model, no download or model load."""
import numpy as np
import pytest
import yaml

from tess_memory.config import parse_manifest
from tess_memory.embed import LocalEmbedder, make_embedder


class Model:
    def __init__(self):
        self.seen = []

    def encode(self, texts, **_options):
        self.seen.append(list(texts))
        return np.ones((len(texts), 2), dtype=np.float32)


def loaded(embedder):
    embedder._model = Model()
    return embedder._model


def test_explicit_prefixes_reach_the_model_and_name_the_vector_space():
    embedder = LocalEmbedder("intfloat/multilingual-e5-small", query_prefix="query: ", passage_prefix="passage: ")
    model = loaded(embedder)
    embedder.embed_passages(["a", "b"])
    embedder.embed_query("q")
    assert model.seen == [["passage: a", "passage: b"], ["query: q"]]
    assert embedder.descriptor == "local:intfloat/multilingual-e5-small|q=query: |p=passage: "


def test_without_prefixes_behaviour_and_descriptor_are_unchanged():
    for name, query in (("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: q"),
                        ("sentence-transformers/all-MiniLM-L6-v2", "q")):
        embedder = LocalEmbedder(name)
        model = loaded(embedder)
        embedder.embed_passages(["a"])
        embedder.embed_query("q")
        assert model.seen == [["a"], [query]] and embedder.descriptor == f"local:{name}"


def test_the_manifest_carries_prefixes_to_the_embedder(tmp_path):
    registry = tmp_path / "providers.yaml"
    registry.write_text(yaml.safe_dump({"profiles": {"d": {"kind": "openai", "defaultModel": "m", "auth": {"kind": "none"}}}}))
    def manifest(**embedding):
        return {"schema_version": 1, "mode": "embedded", "runtime_home": str(tmp_path), "provider_registry": str(registry),
                "config": {"search": {"agent": False}, "embedding": {"provider": "local", "local_model": "m", **embedding}}}
    cfg = parse_manifest(manifest(query_prefix="query: ", passage_prefix="passage: "))
    embedder = make_embedder(cfg)
    assert (embedder.query_prefix, embedder.passage_prefix) == ("query: ", "passage: ")
    with pytest.raises(ValueError):
        parse_manifest(manifest(query_prefix=7))
