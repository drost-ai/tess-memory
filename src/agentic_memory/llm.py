"""LLM backend for distillation: the agent-core-python runtime, in-process.

agent-core-python is a declared dependency. The installed runtime is imported
without mutating sys.path; each completion runs one non-interactive
turn through the engine's runtime and session manager — start_runtime
(daemon-less) → sessions.open → run_turn → stop — with a real system prompt,
tools disabled, and no session persistence.

Provider transport and credentials come only from an explicitly selected
registry profile, through the engine's public configuration/auth APIs.
Optional model/effort overrides belong to this memory instance;
otherwise the provider profile's defaults apply. Shared profiles are never
modified by this package.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .config import Config


class RefusalError(RuntimeError):
    """The model's safety classifiers declined the request (stopReason=refusal)."""


class AgentCoreBackend:
    name = "agent-core"

    @classmethod
    def for_role(cls, cfg: Config, role: str, *, probe: bool = False):
        """Keep retrieval and fallback overrides independent of distillation."""
        if role == "search":
            return cls(cfg, probe=probe, profile=cfg.search_profile or None,
                       model=cfg.search_model or None,
                       reasoning_effort=cfg.search_reasoning_effort or None)
        if role == "fallback":
            if not cfg.agent_core_fallback_profile:
                raise ValueError("No fallback provider profile is configured")
            return cls(cfg, probe=probe, profile=cfg.agent_core_fallback_profile,
                       model=cfg.agent_core_fallback_model or None,
                       reasoning_effort=cfg.agent_core_fallback_reasoning_effort or None)
        raise ValueError(f"Unknown backend role: {role}")

    def __init__(self, cfg: Config, probe: bool = True, profile: str | None = None,
                 *, model: str | None = None, reasoning_effort: str | None = None) -> None:
        self._model = model or (cfg.agent_core_model if profile is None else "")
        self._reasoning_effort = reasoning_effort or (cfg.agent_core_reasoning_effort if profile is None else "")
        self._codex_auth_path = cfg.agent_core_codex_auth_path
        try:
            # The engine's runtime and session manager (agent-core-python
            # v1-refactor, 2026-09): one daemon-less runtime per call, one
            # session, one turn. The dict payload contract is unchanged.
            from agent_core import SessionOptions, start_runtime
        except ImportError as e:
            raise RuntimeError(f"agent-core import failed: {e}") from e
        self._start_runtime = start_runtime
        self._session_options = SessionOptions
        from agent_core.config import ProviderRegistry
        registry_path = cfg.provider_registry or cfg.agent_core_settings
        if registry_path is None:
            raise ValueError("Explicit provider registry is required")
        self._registry = ProviderRegistry(registry_path)
        self._settings_path = str(registry_path)
        self._profile = profile or cfg.agent_core_profile
        if not self._profile:
            raise ValueError("An explicit provider profile is required")
        selected_profile = self._registry.resolve_profile(self._profile, model=self._model or None)
        self._profile_kind = selected_profile.get("kind")
        scratch_root = ((cfg.runtime_home / "run" / "memory") if cfg.runtime_home is not None
                        else cfg.data_dir / "run")
        scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._workdir = tempfile.mkdtemp(prefix="agentic-memory-ac-", dir=scratch_root)
        self._runtime_home = Path(self._workdir) / "runtime"
        atexit.register(shutil.rmtree, self._workdir, ignore_errors=True)
        settings_document = {"providerProfiles": {self._profile: selected_profile}}
        self.name = f"agent-core:{self._profile}"
        if self._model:
            self.name += f":{self._model}"
        if self._reasoning_effort:
            self.name += f"[{self._reasoning_effort}]"
        # Identity follows this profile's semantics. Credential rotation or a
        # change to another runtime's profile must not redistill every session.
        def semantic(value):
            if isinstance(value, dict):
                excluded = {'auth', 'headers', 'httpheaders', 'authorization', 'apikey', 'api_key',
                            'accesstoken', 'refreshtoken', 'token', 'password', 'credentials'}
                return {k: semantic(v) for k, v in value.items() if k.lower() not in excluded}
            if isinstance(value, list):
                return [semantic(v) for v in value]
            return value
        def identity(document):
            selected = semantic((document.get('providerProfiles') or {}).get(self._profile, {}))
            if self._model:
                selected['defaultModel'] = self._model
            if self._reasoning_effort:
                for key in ('reasoningEffort', 'defaultEffort', 'effort', 'effortLevel'):
                    if key in selected:
                        selected[key] = self._reasoning_effort
            globals_used = {k: document[k] for k in ('maxTokens', 'thinkingBudget') if k in document}
            return hashlib.sha256(json.dumps([2, self.name, selected, globals_used], sort_keys=True).encode()).hexdigest()
        self._cache_identity_for = identity
        self.cache_identity = identity(settings_document)
        if probe:
            self._probe()

    def verify_profile(self):
        from agent_core.config import ProviderRegistry
        profile = ProviderRegistry(self._settings_path).resolve_profile(self._profile, model=self._model or None)
        document = {"providerProfiles": {self._profile: profile}}
        if self._cache_identity_for(document) != self.cache_identity:
            raise RuntimeError('provider settings changed during memory work; retry with fresh configuration')

    async def _run_turn(self, *, system: str, prompt: str) -> dict:
        """One daemon-less runtime, one session, one non-interactive turn.

        The runtime is the engine's (start_runtime): it takes no lease and
        owns no daemon, so any number may run beside each other in the
        distillation workers and beside a host's own runtime. Nothing is
        persisted (sessionPersistence off) and no tool is offered.
        """
        cwd = self._workdir
        from .model_usage import current_observer
        observer = current_observer()

        async def adjust(config: dict) -> dict:
            # The host's last word on the resolved config: this store's Codex
            # account and the explicit reasoning effort, as before.
            config = await self._apply_codex_auth_override(config)
            config = self._apply_reasoning_override(config)
            if observer:
                observer.configured(config)
            return config

        class _Host:
            # No ask_question: the turn is non-interactive. Raw stream events
            # are enabled only for usage extraction and never forwarded.
            def on_event(self, _session_id, envelope):
                if observer:
                    observer.event(envelope["event"], envelope["data"])

        options = self._session_options(
            cwd=cwd,
            # Reuse profile transport/auth; explicit per-instance overrides
            # select the model and effort without rewriting the settings file.
            provider=self._profile,
            model=self._model or None,
            effort=self._reasoning_effort or None,
            provider_registry=self._registry,
            settings="{}",
            strict_mcp_config=True,
            provider_auth_path=str(self._codex_auth_path) if self._codex_auth_path else None,
            system_prompt=system,
            adjust_config=adjust,
            stream_provider_messages=bool(observer and self._profile_kind == "openai"),
            extra={"settingSources": "", "tools": [], "sessionPersistence": False, "inputFormat": "text", "outputFormat": "json"},
        )
        runtime = await self._start_runtime(
            self._runtime_home, cwd=cwd, daemon_enabled=False, provider_registry=self._registry,
            session_options=lambda _sid, _reason: options, listener=_Host()
        )
        result = None
        try:
            session = await runtime.sessions.open()
            from .model_usage import check_source_admission
            check_source_admission()
            result = await session.run_turn(prompt)
            return result
        finally:
            try:
                await runtime.stop()
            except Exception as exc:
                if result is None:
                    raise
                # Cleanup is not a failed model request. Return the paid
                # result for ordinary refusal/schema validation and cache
                # publication, with a content-free diagnostic receipt.
                result['memoryCleanupError'] = type(exc).__name__
                import logging
                logging.getLogger(__name__).warning('Runtime cleanup failed after completion (%s); retaining provider result', type(exc).__name__)

    def complete(self, *, system: str, prompt: str) -> str:
        import asyncio

        result = asyncio.run(self._run_turn(system=system, prompt=prompt))
        from .model_usage import current_observer
        observer = current_observer()
        if observer:
            observer.finish(result)
        text = result.get("text") or ""
        if result.get("stopReason") == "refusal":
            raise RefusalError(
                f"model refused ({self.name}): {text[:200] or 'no output'}"
            )
        # agent-core reports a missing credential as a normal end_turn text
        # completion, so match only when the error IS the output — a substring
        # check would reject digests that merely quote the phrase.
        if text.lstrip().startswith("No credential is configured"):
            raise RuntimeError(f"agent-core: {text[:200]}")
        if result.get("stopReason") in ("error", "failed", "cancelled", "permission_denied"):
            raise RuntimeError(
                f"agent-core turn failed: {result.get('error') or text[:200] or result.get('stopReason')}"
            )
        return text

    async def _apply_codex_auth_override(self, config: dict) -> dict:
        """Pin this store's Codex account without changing process-wide env.

        Read on every call so ordinary credential refreshes take effect. Apply
        after agent-core config resolution to override both CODEX_HOME defaults
        and inherited auth-path settings; retain its native secret-reference
        handling. Non-Codex providers keep their own credentials.
        """
        if not self._codex_auth_path or config.get('provider') != 'openai' or config.get('openaiApiMode') != 'codex':
            return config
        from agent_core.config import resolve_codex_auth, resolve_provider_secret
        path = str(self._codex_auth_path)
        token = await resolve_codex_auth(path)
        if not token:
            raise RuntimeError(f'Configured Codex auth file has no usable credential: {path}')
        reference = await resolve_provider_secret(
            token, purpose='provider_oauth_token',
            idempotency_scope=f'memory:{self._profile}:{path}:codex-token')
        result = {**config, 'openaiToken': reference, 'openaiCodexAuthPath': path}
        for key in ('compactionProviderProfile', 'compactionVerificationProviderProfile'):
            binding = config.get(key)
            profile = (binding or {}).get('profile') or {}
            if profile.get('kind') == 'openai' and profile.get('apiMode') == 'codex':
                result[key] = {**binding, 'openaiToken': reference}
        return result

    def _apply_reasoning_override(self, config: dict) -> dict:
        """Preserve explicit effort on compatible Chat Completions backends.

        agent-core currently omits reasoning_effort on non-OpenAI hosts and
        normalizes non-GPT model effort. Its requestOptions escape hatch lets
        this instance send the explicitly requested native value. Calls with
        no override (including existing integrations) are unchanged.
        """
        if not self._reasoning_effort or config.get("provider") != "openai" or config.get("openaiApiMode") != "chat_completions":
            return config
        profile = copy.deepcopy(config.get("providerProfile") or {})
        request_options = profile.setdefault("requestOptions", {})
        request_options["reasoning_effort"] = self._reasoning_effort
        extra = request_options.get("extra_body")
        if isinstance(extra, dict):
            extra.pop("reasoning_effort", None)
        if str(config.get("model", "")).startswith("deepseek-v4-"):
            request_options["thinking"] = {"type": "enabled"}
            if isinstance(extra, dict):
                extra.pop("thinking", None)
        profiles = dict(config.get("providerProfiles") or {})
        if profile.get("name"):
            profiles[profile["name"]] = profile
        return {**config, "providerProfile": profile, "providerProfiles": profiles}

    def _probe(self) -> None:
        # Fail fast and loud at construction: without this, a broken runtime
        # would surface as every session silently degrading to extractive.
        self.complete(system="You are a health check.", prompt="Reply with exactly: OK")
