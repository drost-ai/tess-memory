"""The provider file setup writes, and checks against the real providers.

One profile per connection the user makes. A profile never holds a key: it
names a keychain reference or an environment variable (see ``vault``).
Validation is Tess Core's own registry, so a file written here is one the
engine accepts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .vault import Vault, write_private

GOOGLE_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
PLACEHOLDER_MODEL = "catalog"  # a profile must name a model before the list is known

# What each kind of connection is called in the provider file and on screen.
KINDS = {
    "chatgpt": {"profile": "chatgpt", "label": "ChatGPT account"},
    "anthropic": {"profile": "anthropic", "label": "Anthropic API key", "env": "ANTHROPIC_API_KEY"},
    "openai": {"profile": "openai", "label": "OpenAI API key", "env": "OPENAI_API_KEY"},
    "google": {"profile": "google", "label": "Google Gemini API key", "env": "GEMINI_API_KEY"},
    "compatible": {"profile": "endpoint", "label": "OpenAI-compatible endpoint", "env": "TESS_MEMORY_MODEL_API_KEY"},
}


class SetupError(Exception):
    """Something the person can act on; ``code`` is static and safe to show."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code, self.detail = code, detail


def clean_base_url(value: Any) -> str:
    from urllib.parse import urlsplit
    if not isinstance(value, str) or len(value) > 2000:
        raise SetupError("invalid_url")
    value = value.strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise SetupError("invalid_url")
    return value


def clean_key(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000 or any(ord(c) < 33 for c in value.strip()):
        raise SetupError("invalid_key")
    return value.strip()


def profile_record(kind: str, auth: dict[str, Any], *, model: str = PLACEHOLDER_MODEL, base_url: str | None = None) -> dict[str, Any]:
    """The provider-file record for one connection."""
    managed = "credentialRef" in auth
    if kind == "chatgpt":
        return {"kind": "openai", "apiMode": "responses", "defaultModel": model, "auth": {"kind": "chatgpt", **auth}}
    if kind == "anthropic":
        return {"kind": "anthropic", "defaultModel": model, "auth": {"kind": "anthropic_api_key", **auth}}
    if kind == "openai":
        return {"kind": "openai", "defaultModel": model, "auth": {"kind": "openai_api_key", **auth}}
    if kind == "google":
        return {"kind": "google", "defaultModel": model, "auth": {"kind": "google_api_key", **auth}}
    if kind in ("compatible", "embeddings"):
        if not auth:
            auth_record = {"kind": "none"}
        else:  # a keychain reference is only accepted for API-key auth
            auth_record = {"kind": "openai_api_key" if managed else "custom_bearer", **auth}
        return {"kind": "openai", "apiMode": "chat_completions", "baseUrl": base_url, "defaultModel": model, "auth": auth_record}
    raise SetupError("unknown_kind")


class ProviderFile:
    """``providers.yaml`` beside the settings file (``providers.json`` when
    keys are kept in the file, because that format can carry them)."""

    def __init__(self, vault: Vault) -> None:
        self.vault = vault
        self.profiles: dict[str, dict[str, Any]] = {}
        for name in ("providers.yaml", "providers.json"):
            try:
                from tess_core.config import ProviderRegistry
                self.profiles = ProviderRegistry(vault.home / name).profiles
                break
            except Exception:  # noqa: BLE001 (absent, or not one this setup wrote)
                continue

    async def path(self) -> Path:
        return self.vault.home / ("providers.yaml" if await self.vault.mode() == "keychain" else "providers.json")

    def document(self) -> dict[str, Any]:
        return {"schema_version": 1, "providerProfiles": self.profiles}

    def put(self, name: str, record: dict[str, Any]) -> None:
        from tess_core.config import ProviderRegistry, ProviderRegistryError
        candidate = {**self.profiles, name: record}
        try:
            ProviderRegistry.from_document({"schema_version": 1, "providerProfiles": candidate})
        except ProviderRegistryError as error:
            raise SetupError("invalid_profile", str(error)) from None
        self.profiles = candidate

    def drop(self, name: str) -> None:
        self.profiles.pop(name, None)

    async def save(self) -> Path:
        path = await self.path()
        if not self.profiles:
            return path
        if path.suffix == ".json":
            write_private(path, json.dumps({**self.document(), "env": self.vault.env()}, indent=2) + "\n")
        else:
            from tess_core.config import dump_yaml_subset
            write_private(path, "# Written by `tess-memory setup`. Holds no secret: each profile points at a\n"
                                "# keychain entry or names an environment variable.\n" + dump_yaml_subset(self.document()))
        other = path.with_suffix(".json" if path.suffix == ".yaml" else ".yaml")
        other.unlink(missing_ok=True)  # one provider file, never two that disagree
        return path


async def list_models(record: dict[str, Any]) -> dict[str, Any]:
    """Reach the provider with the profile's own credential and list its
    models. Costs nothing; returns ``{state, code, models}``."""
    if record.get("auth", {}).get("kind") == "chatgpt":
        return await _chatgpt_models(record["auth"]["credentialRef"])
    from tess_core.config import ProviderRegistry
    from tess_core.providers.connection import check_provider_connection
    registry = ProviderRegistry.from_document({"schema_version": 1, "providerProfiles": {"candidate": record}})
    return await check_provider_connection(registry, "candidate")


CHATGPT_MODELS = "https://api.openai.com/v1/models"
CHATGPT_CATALOG_LIMIT = 8 * 1024 * 1024


async def _chatgpt_models(reference: dict[str, Any]) -> dict[str, Any]:
    """The models a ChatGPT account may pick. Its catalogue carries each
    model's full instructions and runs to about a megabyte, more than the
    engine's general connection check reads, so it is read here."""
    import httpx
    from tess_core.runtime.secrets import RuntimeSecretError, resolve_runtime_secret_string

    def result(state: str, code: str, models: list[str] | None = None) -> dict[str, Any]:
        return {"state": state, "code": code, "models": models or []}

    try:
        token = await resolve_runtime_secret_string(reference, "provider_oauth_token")
    except RuntimeSecretError:
        return result("failed", "credential_unavailable")
    except Exception as error:  # noqa: BLE001 (a sign-in that could not be refreshed)
        return result("failed", "reauthorization_required" if getattr(error, "terminal", False) else "credential_unavailable")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("GET", CHATGPT_MODELS, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}) as response:
                status, body = response.status_code, b""
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > CHATGPT_CATALOG_LIMIT:
                        return result("failed", "invalid_provider_response")
    except (httpx.HTTPError, OSError):
        return result("failed", "connection_unavailable")
    if status in (401, 403):
        return result("failed", "authentication_denied")
    if status == 429:
        return result("failed", "rate_limited")
    try:
        listed = json.loads(body)
        if status != 200:
            raise ValueError
        if isinstance(listed.get("models"), list):
            models = [item["slug"] for item in listed["models"]
                      if isinstance(item, dict) and isinstance(item.get("slug"), str) and item.get("visibility") == "list"]
        else:
            models = [item["id"] for item in listed["data"] if isinstance(item, dict) and isinstance(item.get("id"), str)]
    except (ValueError, TypeError, KeyError, AttributeError):
        return result("failed", "invalid_provider_response")
    return result("available", "models_listed", models[:200])
