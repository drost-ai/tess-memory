"""Embedding backends.

Two interchangeable implementations behind one interface:

    OpenAICompatEmbedder — remote embeddings via an OpenAI-compatible endpoint,
        configured from an explicitly selected provider registry profile.
        Asymmetric retrieval format: instruct-prefixed queries, plain passages.
    LocalEmbedder — sentence-transformers, cached-only unless download opt-in.

All vectors are L2-normalized so cosine similarity is a plain dot product.
Vector spaces are not interchangeable: the store records the embedder
descriptor and refuses to mix (switch provider => re-ingest).
"""

from __future__ import annotations

from contextlib import nullcontext

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
    batch_size: int = 64

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str, batch_size: int = 64, *, allow_download: bool = False,
                 query_prefix: str | None = None, passage_prefix: str | None = None):
        self.allow_download = allow_download
        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        # Explicit prefixes change the vectors, so they are part of the vector
        # space's name; without them the descriptor is the historical one.
        explicit = query_prefix is not None or passage_prefix is not None
        self.descriptor = f"local:{model_name}" + (f"|q={query_prefix or ''}|p={passage_prefix or ''}" if explicit else "")
        self._model = None

    def _load(self):
        if self._model is None:
            # Library import/load chatter must never reach stdout: when running
            # as an MCP stdio server, stdout carries the JSON-RPC protocol.
            import contextlib
            import sys

            with contextlib.redirect_stdout(sys.stderr):
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, local_files_only=not self.allow_download)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1), dtype=np.float32)
        model = self._load()
        if self.passage_prefix:
            texts = [self.passage_prefix + text for text in texts]
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        prefix = self.query_prefix
        if prefix is None:
            prefix = _BGE_QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        vec = model.encode([prefix + query], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)


def _managed_secret(reference: dict) -> str:
    """The secret behind a managed credential reference (a keychain entry).

    Resolved on a thread of its own, so it works whether or not the caller is
    already inside an event loop."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from tess_core.runtime.secrets import resolve_runtime_secret_string

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, resolve_runtime_secret_string(reference, reference.get("purpose"))).result()


class OpenAICompatEmbedder(Embedder):
    _query_prefix = _QWEN_QUERY_INSTRUCT
    _send_dimensions = True

    def __init__(self, settings_path, profile_name: str, dims: int, batch_size: int = 64, *,
                 query_prefix: str | None = None, send_dimensions: bool = True):
        self.batch_size = batch_size
        # None keeps the instruction earlier stores were queried with.
        self._query_prefix = _QWEN_QUERY_INSTRUCT if query_prefix is None else query_prefix
        self._send_dimensions = send_dimensions
        from tess_core.config import ProviderRegistry
        if settings_path is None:
            raise ValueError("Explicit provider registry is required")
        registry = ProviderRegistry(settings_path)
        prof = registry.resolve_profile(profile_name)
        if not prof.get("baseUrl") or not prof.get("defaultModel"):
            raise ValueError(f"Embedding profile '{profile_name}' requires baseUrl and defaultModel")
        self._url = prof["baseUrl"].rstrip("/") + "/embeddings"
        self._model = prof["defaultModel"]
        auth = registry.resolve_auth(profile_name)
        self._key = auth.get("apiKey") or auth.get("authToken") or auth.get("openaiToken")
        if isinstance(self._key, dict):
            self._key = _managed_secret(self._key)
        self._headers = dict(auth.get("customHeaders") or {})
        if self._key:
            self._headers.setdefault("Authorization", f"Bearer {self._key}")
        if not self._headers and (prof.get("auth") or {}).get("kind") != "none":
            raise RuntimeError(f"no credential for embedding profile '{profile_name}'")
        self._dims = dims
        self.descriptor = f"{profile_name}:{self._model}:{dims}d"

    def _post(self, inputs: list[str]) -> np.ndarray:
        import httpx
        from .source_activity import SourceNotQuiet
        from .model_usage import check_source_admission

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                guard = getattr(self, 'request_guard', nullcontext)
                with guard():
                    check_source_admission()
                    r = httpx.post(
                        self._url,
                        headers=self._headers,
                        json={"model": self._model, "input": inputs,
                              **({"dimensions": self._dims} if self._send_dimensions else {})},
                        timeout=180,
                    )
                    r.raise_for_status()
                data = r.json()["data"]
                if all(isinstance(d, dict) and 'index' in d for d in data):
                    if sorted(d['index'] for d in data) != list(range(len(inputs))):
                        raise ValueError('embedding response indices do not match inputs')
                    data = sorted(data, key=lambda d: d['index'])
                vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
                if vecs.shape != (len(inputs), self._dims) or not np.isfinite(vecs).all():
                    raise ValueError('embedding response has invalid dimensions or values')
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs / np.clip(norms, 1e-9, None)
            except SourceNotQuiet:
                raise  # a scheduling deferral must not become a transport retry/error
            except Exception as e:  # noqa: BLE001 — retry transient network/5xx
                last_err = e
                import time

                time.sleep(2**attempt)
        raise RuntimeError(f"embedding request failed after retries: {last_err}")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1), dtype=np.float32)
        out = [
            self._post(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(out)

    def embed_query(self, query: str) -> np.ndarray:
        return self._post([self._query_prefix + query])[0]


def make_embedder(cfg: Config) -> Embedder:
    """Construct only the explicitly selected backend; never silently change vector space."""
    if cfg.embed_provider == "local":
        return LocalEmbedder(cfg.embed_model, cfg.embed_batch_size,
                             allow_download=cfg.embedding_allow_download,
                             query_prefix=cfg.embed_query_prefix, passage_prefix=cfg.embed_passage_prefix)
    return OpenAICompatEmbedder(cfg.provider_registry or cfg.engine_settings,
                                cfg.embed_provider, cfg.embed_dims, cfg.embed_batch_size,
                                query_prefix=cfg.embed_query_prefix, send_dimensions=cfg.embed_send_dimensions)
