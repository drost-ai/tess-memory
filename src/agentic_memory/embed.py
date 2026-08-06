"""Embedding backends.

Two interchangeable implementations behind one interface:

    OpenAICompatEmbedder — remote embeddings via an OpenAI-compatible endpoint,
        configured from a provider profile in ~/.tess-memory/providers.yaml
        (default: embeddings -> Qwen3-Embedding-8B @ 4096d).
        Asymmetric retrieval format: instruct-prefixed queries, plain passages.
    LocalEmbedder — sentence-transformers, fully offline.

All vectors are L2-normalized so cosine similarity is a plain dot product.
Vector spaces are not interchangeable: the store records the embedder
descriptor and refuses to mix (switch provider => re-ingest).
"""

from __future__ import annotations

import json

import numpy as np

from .config import Config

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QWEN_QUERY_INSTRUCT = (
    "Instruct: Given a question about past coding sessions, retrieve the "
    "passages that answer it\nQuery: "
)


class Embedder:
    """Interface: embed_passages / embed_query / descriptor."""

    descriptor: str = "?"

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.descriptor = f"local:{model_name}"
        self._model = None

    def _load(self):
        if self._model is None:
            # Library import/load chatter must never reach stdout: when running
            # as an MCP stdio server, stdout carries the JSON-RPC protocol.
            import contextlib
            import sys

            with contextlib.redirect_stdout(sys.stderr):
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, 1), dtype=np.float32)
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        prefix = _BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        vec = model.encode([prefix + query], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)


class OpenAICompatEmbedder(Embedder):
    def __init__(self, settings_path, profile_name: str, dims: int):
        settings = json.load(open(settings_path))
        prof = (settings.get("providerProfiles") or {}).get(profile_name)
        if not prof:
            raise RuntimeError(f"profile '{profile_name}' not found in {settings_path}")
        self._url = (prof.get("baseUrl") or "").rstrip("/") + "/embeddings"
        self._model = prof.get("defaultModel")
        env_var = (prof.get("auth") or {}).get("envVar")
        import os

        self._key = os.environ.get(env_var or "") or (settings.get("env") or {}).get(env_var)
        if not self._key:
            raise RuntimeError(f"no credential for embedding profile '{profile_name}' ({env_var})")
        self._dims = dims
        self.descriptor = f"{profile_name}:{self._model}:{dims}d"

    def _post(self, inputs: list[str]) -> np.ndarray:
        import httpx

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = httpx.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"model": self._model, "input": inputs, "dimensions": self._dims},
                    timeout=180,
                )
                r.raise_for_status()
                vecs = np.array(
                    [d["embedding"] for d in r.json()["data"]], dtype=np.float32
                )
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / np.clip(norms, 1e-9, None)
            except Exception as e:  # noqa: BLE001 — retry transient network/5xx
                last_err = e
                import time

                time.sleep(2**attempt)
        raise RuntimeError(f"embedding request failed after retries: {last_err}")

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, 1), dtype=np.float32)
        out = [
            self._post(texts[i : i + batch_size])
            for i in range(0, len(texts), batch_size)
        ]
        return np.vstack(out)

    def embed_query(self, query: str) -> np.ndarray:
        return self._post([_QWEN_QUERY_INSTRUCT + query])[0]


def make_embedder(cfg: Config) -> Embedder:
    provider = cfg.embed_provider
    if provider == "local":
        return LocalEmbedder(cfg.embed_model)
    try:
        return OpenAICompatEmbedder(cfg.agent_core_settings, provider, cfg.embed_dims)
    except RuntimeError as e:
        import sys

        print(f"[embed] {e}; falling back to local model", file=sys.stderr)
        return LocalEmbedder(cfg.embed_model)
