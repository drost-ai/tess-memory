"""What must never be lost: published memory, and the plan."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from tess_memory import plan, reembed, retrieval
from tess_memory.config import Config
from tess_memory.store import EmbedderMismatch, MemoryRow, MemoryStore, same_vector_space, vector_space


class Embedder:
    """Deterministic vectors of a chosen size; records what it was asked for."""

    def __init__(self, descriptor: str, dims: int, *, fail_after: int | None = None):
        self.descriptor, self.dims, self.batch_size = descriptor, dims, 2
        self.passages: list[str] = []
        self.queries: list[str] = []
        self.fail_after = fail_after

    def _vector(self, text: str) -> np.ndarray:
        values = np.array([(hash((text, i)) % 1000) / 1000 + 0.01 for i in range(self.dims)], dtype=np.float32)
        return values / np.linalg.norm(values)

    def embed_passages(self, texts):
        if self.fail_after is not None and len(self.passages) >= self.fail_after:
            raise RuntimeError("provider went away")
        self.passages += list(texts)
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text):
        self.queries.append(text)
        return self._vector(text)


@pytest.fixture
def store(tmp_path, monkeypatch):
    cfg = Config(data_dir=tmp_path)
    monkeypatch.setattr(retrieval, "CONFIG", cfg)
    result = MemoryStore(cfg.db_path)
    yield result
    result.close()


def fill(store: MemoryStore, embedder: Embedder, texts: list[str]) -> None:
    store.assert_embedder(embedder.descriptor)
    vectors = embedder.embed_passages(texts)
    store.add_memories([MemoryRow(level="memory", kind="decision", content=text, project="p", vector=vector,
                                  ts="2026-11-01T00:00:00Z") for text, vector in zip(texts, vectors)])
    store.commit()


def vectors(store: MemoryStore) -> dict[str, bytes]:
    return {row["content"]: row["vector"] for row in store.db.execute(
        "SELECT m.content, e.vector FROM memories m JOIN embeddings e ON e.memory_id=m.id")}


def test_a_renamed_provider_profile_is_the_same_vector_space():
    assert vector_space("fireworks-embedding:accounts/x/qwen3:4096d") == "accounts/x/qwen3:4096d"
    assert same_vector_space("old-name:model-a:1024d", "embeddings:model-a:1024d")
    assert not same_vector_space("embeddings:model-a:1024d", "embeddings:model-a:512d")
    assert not same_vector_space("embeddings:model-a:1024d", "embeddings:model-b:1024d")
    assert vector_space("local:BAAI/bge-small-en-v1.5|q=a|p=b") == "local:BAAI/bge-small-en-v1.5|q=a|p=b"
    assert not same_vector_space(None, "embeddings:model-a:8d")


def test_renaming_the_profile_does_not_stop_the_store(store):
    fill(store, Embedder("first-name:model-a:4d", 4), ["a decision about caching"])
    store.assert_embedder("second-name:model-a:4d")  # does not raise
    assert store.embedder_matches("second-name:model-a:4d")


def test_another_embedding_model_stops_new_vectors_but_never_the_memory(store):
    old, new = Embedder("embeddings:model-a:4d", 4), Embedder("embeddings:model-b:6d", 6)
    fill(store, old, ["the checkout fix was a zero guard", "we chose UUIDv7 keys"])
    before = vectors(store)
    with pytest.raises(EmbedderMismatch, match="Nothing is lost. Run `tess-memory reembed`"):
        store.assert_embedder(new.descriptor)
    # Search still answers, by keywords, and does not embed a query it could not compare.
    hits = retrieval.search(store, new, "checkout guard")
    assert [hit["content"] for hit in hits][:1] == ["the checkout fix was a zero guard"]
    assert new.queries == [] and vectors(store) == before
    # The plan can still be written and read.
    item = plan.add_item(store, new, project="p", kind="task", title="Ship the cutover")
    assert store.plan_get(item["id"])["title"] == "Ship the cutover"
    assert new.passages == []


def test_reembed_moves_every_row_and_nothing_else(store):
    old, new = Embedder("embeddings:model-a:4d", 4), Embedder("embeddings:model-b:6d", 6)
    fill(store, old, ["alpha decision", "beta solution", "gamma fact"])
    item = plan.add_item(store, old, project="p", kind="epic", title="An authored plan item")
    memories = [tuple(row) for row in store.db.execute("SELECT * FROM memories ORDER BY id")]
    plans = [tuple(row) for row in store.db.execute("SELECT * FROM plan_items")]
    assert reembed.status(store, new.descriptor)["needed"] is True
    seen = []
    result = reembed.reembed(store, new, progress=lambda done, total: seen.append((done, total)))
    assert (result["state"], result["rows"], result["embedded"]) == ("switched", 4, 4)
    assert seen[-1] == (4, 4)
    assert store.get_meta("embedder") == new.descriptor and store.get_meta(reembed.TARGET) is None
    assert {len(np.frombuffer(blob, dtype=np.float32)) for blob in vectors(store).values()} == {6}
    assert [tuple(row) for row in store.db.execute("SELECT * FROM memories ORDER BY id")] == memories
    assert [tuple(row) for row in store.db.execute("SELECT * FROM plan_items")] == plans
    assert not store.db.execute("SELECT name FROM sqlite_master WHERE name='reembed_vectors'").fetchone()
    # Search by meaning works again, in the new space; a second run has nothing to do.
    assert retrieval.search(store, new, "alpha decision")[0]["content"] == "alpha decision"
    assert new.queries == ["alpha decision"]
    assert reembed.reembed(store, new)["state"] == "current"
    assert store.plan_get(item["id"])["title"] == "An authored plan item"


def test_an_interrupted_reembed_changes_nothing_and_resumes(store):
    old = Embedder("embeddings:model-a:4d", 4)
    fill(store, old, ["one", "two", "three", "four", "five"])
    before = vectors(store)
    failing = Embedder("embeddings:model-b:6d", 6, fail_after=2)
    with pytest.raises(RuntimeError, match="provider went away"):
        reembed.reembed(store, failing)
    # Still the old model, the old vectors, and searchable with it.
    assert store.get_meta("embedder") == old.descriptor and vectors(store) == before
    assert retrieval.search(store, old, "three")
    assert reembed.status(store, failing.descriptor)["staged"] == 2
    # The next run embeds only what was not done.
    new = Embedder("embeddings:model-b:6d", 6)
    result = reembed.reembed(store, new)
    assert (result["state"], result["embedded"], sorted(new.passages)) == ("switched", 3, sorted(set(before) - set(failing.passages)))
    # A stop requested between batches is the same: nothing switched.
    third = Embedder("embeddings:model-c:5d", 5)
    assert reembed.reembed(store, third, should_stop=lambda: len(third.passages) >= 2)["state"] == "stopped"
    assert store.get_meta("embedder") == new.descriptor


def test_rows_published_while_reembedding_are_embedded_before_the_switch(store):
    old = Embedder("embeddings:model-a:4d", 4)
    fill(store, old, ["first", "second"])
    new = Embedder("embeddings:model-b:6d", 6)
    added = []

    def progress(done, total):
        if not added:  # ingestion publishes another row, in the old space, mid-way
            added.append(True)
            fill(store, old, ["published meanwhile"])

    result = reembed.reembed(store, new, progress=progress)
    assert result["state"] == "switched" and result["rows"] == 3
    assert {len(np.frombuffer(blob, dtype=np.float32)) for blob in vectors(store).values()} == {6}
    assert "published meanwhile" in new.passages


def test_the_database_itself_refuses_to_delete_the_plan(store):
    item = plan.add_item(store, Embedder("embeddings:model-a:4d", 4), project="p", kind="task", title="Keep me")
    plan.update_item(store, Embedder("embeddings:model-a:4d", 4), item["id"], status="in_progress")
    for statement in ("DELETE FROM plan_items", "DELETE FROM plan_events",
                      f"DELETE FROM plan_items WHERE id={item['id']}"):
        with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
            store.db.execute(statement)
        store.db.rollback()
    assert store.plan_get(item["id"])["status"] == "in_progress"
    assert store.db.execute("SELECT COUNT(*) FROM plan_events").fetchone()[0] >= 2
