"""Connecting the model that distils sessions and runs the retrieval agent.

Three ways in: a ChatGPT account (OpenAI's sign-in for open-source apps, run
by Tess Core on a loopback address), an API key (Anthropic, OpenAI or Google
Gemini), or any OpenAI-compatible endpoint. A connection is checked by listing the provider's models, which
costs nothing; the chosen models are then proved with one short real request
each, because a model that lists is not always one the account may call.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .providers import KINDS, ProviderFile, SetupError, clean_base_url, clean_key, list_models, profile_record
from .vault import Vault

APP_NAME = "Tess Memory"
SIGN_IN_SECONDS = 300.0
DISCLOSURE = ("Signing in with ChatGPT lets Tess Memory use your ChatGPT plan to distil your sessions. Those requests "
              "count toward your plan's usage, which you manage in ChatGPT's settings. Tess Memory never sees your "
              "ChatGPT conversations.")


def new_host_id() -> str:
    """This installation's stable id for OpenAI's sign-in, made once and kept.
    OpenAI accepts only a UUID in ``urn:uuid:`` form here."""
    import uuid
    return f"urn:uuid:{uuid.uuid4()}"


async def connect_with_key(vault: Vault, providers: ProviderFile, kind: str, *, api_key: Any = None, base_url: Any = None) -> dict[str, Any]:
    """Keep the key, reach the provider with it, and record the connection.
    A key that does not work is not kept."""
    if kind not in ("anthropic", "openai", "google", "compatible"):
        raise SetupError("unknown_kind")
    url = clean_base_url(base_url) if kind == "compatible" else None
    auth: dict[str, Any] = {}
    if kind != "compatible" or (isinstance(api_key, str) and api_key.strip()):
        auth = await vault.keep_api_key(clean_key(api_key), env_name=KINDS[kind]["env"])
    record = profile_record(kind, auth, base_url=url)
    report = await list_models(record)
    if report["state"] == "failed":
        await vault.forget(auth)
        raise SetupError(report["code"])
    previous = vault.draft.get("model") or {}
    if previous.get("auth") and previous["auth"] != auth and previous.get("kind") != "chatgpt":
        await vault.forget(previous["auth"])
    connection = {"kind": kind, "profile": KINDS[kind]["profile"], "label": KINDS[kind]["label"], "base_url": url,
                  "auth": auth, "models": report["models"], "listed": report["state"] == "available",
                  "distill_model": None, "search_model": None, "verified": False}
    vault.draft["model"] = connection
    vault.save()
    return connection


class ChatGPT:
    """One sign-in attempt at a time; it must be finished in a browser on
    this machine, because the callback is this machine's loopback address."""

    def __init__(self, vault: Vault, providers: ProviderFile) -> None:
        self.vault, self.providers = vault, providers
        self._attempt: Any = None
        self._task: asyncio.Task[None] | None = None
        self.outcome: dict[str, Any] = {"state": "idle"}

    async def available(self) -> tuple[bool, str]:
        if await self.vault.mode() != "keychain":
            return False, "needs_keychain"
        try:
            import cryptography  # noqa: F401 (verifies the sign-in's signature)
        except ImportError:
            return False, "needs_cryptography"
        return True, "ok"

    async def start(self) -> dict[str, Any]:
        ok, reason = await self.available()
        if not ok:
            raise SetupError(reason)
        from tess_core.providers.chatgpt_account import SignIn
        await self.cancel()
        saved = self.vault.meta["chatgpt"]
        if isinstance(saved.get("credential_ref"), dict) and saved.get("plan_use") and await self._use(saved["credential_ref"], saved.get("email") or ""):
            return {"url": None, "expires_in": 0}  # still signed in: no browser needed
        if not str(saved.get("host_id") or "").startswith("urn:uuid:"):
            saved["host_id"] = new_host_id()
            self.vault.save()
        try:
            attempt = await SignIn.begin(host_id=saved["host_id"], app_name=APP_NAME, client_id=saved.get("client_id"),
                                         account_id=saved.get("account_id"), login_hint=saved.get("email") or None)
        except OSError:
            raise SetupError("loopback_unavailable") from None
        self._attempt = attempt
        self.outcome = {"state": "waiting"}
        self._task = asyncio.get_running_loop().create_task(self._complete(attempt))
        return {"url": attempt.url, "expires_in": SIGN_IN_SECONDS}

    async def cancel(self) -> None:
        task, self._task = self._task, None
        attempt, self._attempt = self._attempt, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 (it is being abandoned)
                pass
            self.outcome = {"state": "cancelled"}
        if attempt is not None:
            await attempt.close()

    async def _complete(self, attempt: Any) -> None:
        from tess_core.providers.chatgpt_account import ChatGPTAccountError, store_grant
        try:
            grant = await attempt.finish(timeout=SIGN_IN_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 (a static code for the person)
            self.outcome = {"state": "failed", "code": error.code if isinstance(error, ChatGPTAccountError) else "sign_in_failed"}
            return
        finally:
            if self._attempt is attempt:
                self._attempt = None
        try:
            ref = await store_grant(self.vault.store, grant)
            saved = self.vault.meta["chatgpt"]
            old = saved.get("credential_ref")
            saved.update({"client_id": grant.client_id, "account_id": grant.account_id, "email": grant.email,
                          "plan_use": grant.plan_use, "credential_ref": ref})
            self.vault.save()
            if isinstance(old, dict) and old != ref:
                await self.vault.forget({"credentialRef": old})
            if not grant.plan_use:
                self.outcome = {"state": "failed", "code": "plan_use_not_granted", "email": grant.email}
                return
            await self._use(ref, grant.email)
        except Exception as error:  # noqa: BLE001
            self.outcome = {"state": "failed", "code": getattr(error, "code", None) or "sign_in_failed"}

    async def _use(self, ref: dict[str, Any], email: str) -> bool:
        """Make a signed-in account the connection, if it lists models."""
        auth = {"credentialRef": ref}
        report = await list_models(profile_record("chatgpt", auth))
        if report["state"] != "available" or not report["models"]:
            self.outcome = {"state": "failed", "code": report["code"] if report["state"] == "failed" else "no_models", "email": email}
            return False
        self.vault.draft["model"] = {"kind": "chatgpt", "profile": KINDS["chatgpt"]["profile"], "label": KINDS["chatgpt"]["label"],
                                     "base_url": None, "auth": auth, "models": report["models"], "listed": True,
                                     "email": email, "distill_model": None, "search_model": None, "verified": False}
        self.vault.save()
        self.outcome = {"state": "signed_in", "email": email}
        return True


def _one_request(home: Path, registry: Path, profile: str, model: str) -> tuple[bool, str]:
    """One short completion through the same path ingestion uses."""
    from ..config import Config, _export_settings_environment
    from ..llm import EngineBackend
    cfg = Config()
    cfg.data_dir = home
    cfg.engine_settings = registry
    cfg.engine_profile = profile
    cfg.engine_model = model
    _export_settings_environment(registry)
    try:
        EngineBackend(cfg, probe=True)
        return True, "ok"
    except Exception as error:  # noqa: BLE001 (shown to the person, shortened)
        return False, f"{type(error).__name__}: {str(error)[:300]}"


async def choose_models(vault: Vault, providers: ProviderFile, *, distill_model: Any, search_model: Any) -> dict[str, Any]:
    """Record the two models and prove each with a real request."""
    connection = vault.draft.get("model")
    if not connection:
        raise SetupError("no_connection")
    for value in (distill_model, search_model):
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise SetupError("invalid_model")
    distill_model, search_model = distill_model.strip(), search_model.strip()
    providers.put(connection["profile"], profile_record(connection["kind"], connection["auth"], model=distill_model,
                                                        base_url=connection.get("base_url")))
    registry = await providers.save()
    results: dict[str, tuple[bool, str]] = {}
    for model in dict.fromkeys((distill_model, search_model)):
        results[model] = await asyncio.to_thread(_one_request, vault.home, registry, connection["profile"], model)
    connection.update({"distill_model": distill_model, "search_model": search_model,
                       "verified": all(ok for ok, _ in results.values()),
                       "checks": {"distill": list(results[distill_model]), "search": list(results[search_model])}})
    vault.save()
    return connection
