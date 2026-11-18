"""Where setup keeps what it must remember, and where secrets go.

``setup.json`` in the store's folder holds no secret: an installation id, the
choices made so far, and what OpenAI asks an app to keep between sign-ins.

Secrets go to the operating system's keychain (macOS Keychain, or the Secret
Service on Linux) through Tess Core's credential store, and the provider file
then holds only an opaque reference. Where no keychain can be used, keys are
written to the provider file's own ``env`` map instead, in a file only its
owner can read; a ChatGPT sign-in is not offered there, because its tokens
are rewritten on every refresh and need a real store.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

META = "setup.json"


def write_private(path: Path, text: str) -> None:
    """Replace ``path`` atomically with a file only its owner can read."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class Vault:
    def __init__(self, home: Path, *, keychain: bool = True, backend: Any = None) -> None:
        self.home = Path(home)
        self._keychain_allowed = keychain
        self._backend = backend  # a test seam: a keyring-shaped object
        self._store: Any = None
        self._mode: str | None = None
        # Keys to remove once the settings that used them have been replaced.
        self.pending_forget: list[dict[str, Any]] = []
        self.meta = self._load()

    # -- setup.json -----------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            value = json.loads((self.home / META).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        if not isinstance(value, dict):
            value = {}
        value.setdefault("schema_version", 1)
        for key in ("installation_id", "account_id"):
            try:
                if str(uuid.UUID(value.get(key))) != value.get(key):
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                value[key] = str(uuid.uuid4())
        value.setdefault("draft", {})
        value.setdefault("chatgpt", {})
        return value

    def save(self) -> None:
        write_private(self.home / META, json.dumps(self.meta, indent=2, sort_keys=True) + "\n")

    @property
    def draft(self) -> dict[str, Any]:
        return self.meta["draft"]

    # -- the secret store -----------------------------------------------

    @staticmethod
    def platform_backend() -> str | None:
        if sys.platform == "darwin":
            return "macos_keychain"
        if sys.platform.startswith("linux"):
            return "linux_secret_service"
        return None

    async def mode(self) -> str:
        """``keychain`` when the system store works from this process, else ``file``."""
        if self._mode is None:
            self._mode = "file"
            name = self.platform_backend() if self._keychain_allowed else None
            if name is not None:
                try:
                    from tess_core.runtime.credentials import CredentialStore
                    store = (CredentialStore(self.meta["installation_id"], name, self._backend)
                             if self._backend is not None else CredentialStore.desktop(self.meta["installation_id"], name))
                    if await store.bounded("probe") == "available":
                        self._store, self._mode = store, "keychain"
                except Exception:  # noqa: BLE001 (no usable keychain: fall back to the file)
                    self._store = None
        return self._mode

    @property
    def store(self) -> Any:
        return self._store

    async def keep_api_key(self, key: str, *, env_name: str) -> dict[str, Any]:
        """Keep one API key. Returns the ``auth`` fields that point at it:
        a ``credentialRef`` in the keychain, or the ``envVar`` it is filed under."""
        if await self.mode() == "keychain":
            ref = await self._store.bounded("write", self.meta["account_id"], "provider_api_key", {"api_key": key})
            return {"credentialRef": ref}
        self.meta.setdefault("env", {})[env_name] = key
        self.save()
        os.environ[env_name] = key  # this process checks the connection next
        return {"envVar": env_name}

    async def forget(self, auth: dict[str, Any]) -> None:
        """Remove a key kept by ``keep_api_key`` (a failed connection, a replaced key)."""
        ref = auth.get("credentialRef")
        if isinstance(ref, dict) and self._store is not None:
            try:
                await self._store.bounded("delete", ref, account_id=ref["account_id"], purpose=ref["purpose"],
                                          referenced_ids=frozenset())
            except Exception:  # noqa: BLE001 (an orphaned entry is harmless)
                pass
        name = auth.get("envVar")
        if name and name in self.meta.get("env", {}):
            del self.meta["env"][name]
            self.save()
            os.environ.pop(name, None)

    def env(self) -> dict[str, str]:
        """Keys kept in the file (empty when the keychain is used)."""
        return dict(self.meta.get("env") or {})
