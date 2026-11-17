"""Choosing the embedding model: one on this machine, or a hosted one.

A local model is downloaded only when asked, with its size shown first, and
is verified before it is used (see ``model_download``). A hosted model is
reached once to learn the size of its vectors, and once more to learn whether
it accepts a requested size.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

from .. import model_download
from .providers import GOOGLE_OPENAI_BASE, ProviderFile, SetupError, clean_base_url, clean_key, profile_record
from .vault import Vault

PROFILE = "embeddings"
ENV_NAME = "TESS_MEMORY_EMBEDDING_API_KEY"
PRESETS = {
    "google": {"label": "Google", "base_url": GOOGLE_OPENAI_BASE, "model": "gemini-embedding-001"},
    "compatible": {"label": "OpenAI-compatible endpoint", "base_url": "", "model": ""},
}


class LocalModels:
    """The catalogue, and the one download that may be running."""

    def __init__(self, vault: Vault) -> None:
        self.vault = vault
        self.job: dict[str, Any] = {"state": "idle"}
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def catalogue(self) -> list[dict[str, Any]]:
        return [model_download.describe(self.vault.home, name) for name in model_download.CATALOG]

    def start(self, name: str, *, fetch: Any = None, verify: Any = None) -> dict[str, Any]:
        if name not in model_download.CATALOG:
            raise SetupError("unknown_model")
        if self._thread is not None and self._thread.is_alive():
            raise SetupError("download_running")
        self._cancel = threading.Event()
        total = model_download.CATALOG[name]["bytes"]
        self.job = {"state": "downloading", "name": name, "done": 0, "total": total}

        def run() -> None:
            try:
                model_download.download(self.vault.home, name, cancel=self._cancel, fetch=fetch,
                                        progress=lambda done, size: self.job.update(done=done, total=size))
                self.job.update(state="verifying", done=total)
                (verify or _load_once)(self.vault.home, name)
                self.vault.draft["embedding"] = {"mode": "local", "name": name,
                                                 "settings": model_download.embedding_settings(self.vault.home, name)}
                self.vault.save()
                self.job.update(state="ready")
            except model_download.DownloadError as error:
                self.job.update(state="cancelled" if error.code == "cancelled" else "failed", code=error.code)
            except Exception as error:  # noqa: BLE001 (the model did not load)
                self.job.update(state="failed", code="model_not_usable", detail=type(error).__name__)

        self._thread = threading.Thread(target=run, name="tess-memory-model-download", daemon=True)
        self._thread.start()
        return self.job

    def cancel(self) -> None:
        self._cancel.set()


def _load_once(home: Any, name: str) -> None:
    """Load the downloaded model and embed one sentence, from local files only."""
    from ..embed import LocalEmbedder
    entry = model_download.CATALOG[name]
    vector = LocalEmbedder(str(model_download.target(home, name)), query_prefix=entry["query_prefix"],
                           passage_prefix=entry["passage_prefix"]).embed_query("Is memory ready?")
    if len(vector) != entry["dims"]:
        raise ValueError("unexpected vector size")


async def _embed(base_url: str, model: str, key: str | None, dimensions: int | None) -> tuple[int, Any]:
    import httpx
    body: dict[str, Any] = {"model": model, "input": ["Is memory ready?"]}
    if dimensions:
        body["dimensions"] = dimensions
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(base_url + "/embeddings", json=body,
                                     headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return response.status_code, payload


async def connect_hosted(vault: Vault, providers: ProviderFile, *, preset: Any, base_url: Any = None, model: Any = None,
                         api_key: Any = None, transport: Any = _embed) -> dict[str, Any]:
    """Reach a hosted embedding model, learn its vector size, and record it."""
    import httpx
    if preset not in PRESETS:
        raise SetupError("unknown_preset")
    url = clean_base_url(base_url or PRESETS[preset]["base_url"])
    name = (model or PRESETS[preset]["model"] or "").strip() if isinstance(model or "", str) else ""
    if not name or len(name) > 200:
        raise SetupError("invalid_model")
    key = clean_key(api_key) if (preset == "google" or (isinstance(api_key, str) and api_key.strip())) else None
    try:
        status, payload = await transport(url, name, key, None)
    except (httpx.HTTPError, OSError, asyncio.TimeoutError):
        raise SetupError("connection_unavailable") from None
    if status in (401, 403):
        raise SetupError("authentication_denied")
    if status == 404:
        raise SetupError("model_or_endpoint_not_found")
    if status == 429:
        raise SetupError("rate_limited")
    try:
        dims = len(payload["data"][0]["embedding"])
        if status != 200 or dims < 8:
            raise ValueError
    except (TypeError, KeyError, IndexError, ValueError):
        raise SetupError("invalid_provider_response") from None
    try:
        status, payload = await transport(url, name, key, dims)
        send_dimensions = status == 200 and len(payload["data"][0]["embedding"]) == dims
    except Exception:  # noqa: BLE001 (it simply does not take the field)
        send_dimensions = False
    auth = await vault.keep_api_key(key, env_name=ENV_NAME) if key else {}
    previous = vault.draft.get("embedding") or {}
    if previous.get("auth") and previous["auth"] != auth:
        await vault.forget(previous["auth"])
    providers.put(PROFILE, profile_record("embeddings", auth, model=name, base_url=url))
    await providers.save()
    choice = {"mode": "hosted", "preset": preset, "label": PRESETS[preset]["label"], "base_url": url, "model": name,
              "dims": dims, "send_dimensions": send_dimensions, "auth": auth}
    vault.draft["embedding"] = choice
    vault.save()
    return choice


def choose_local(vault: Vault, name: str) -> dict[str, Any]:
    """Use a catalogue model that is already on disk."""
    if name not in model_download.CATALOG or not model_download.installed(vault.home, name):
        raise SetupError("model_not_installed")
    choice = {"mode": "local", "name": name, "settings": model_download.embedding_settings(vault.home, name)}
    vault.draft["embedding"] = choice
    vault.save()
    return choice
