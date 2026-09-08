"""LLM backend for distillation: the agent-core-python runtime, in-process.

agent-core-python is the only supported path. The runtime is imported from its
checkout (it is dependency-free) and each completion runs one non-interactive
turn through the same seams its own headless runner uses — resolve_config →
resolve_session → bootstrap_runtime → run_agent_prompt — with a real system prompt,
tools disabled, and no session persistence.

Provider transport and credentials come from the runtime's own configuration:
when the Tess settings file exists, its provider profile is selected.
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
import sys
import tempfile

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
        root = cfg.agent_core_path
        if not (root / "agent_core").is_dir():
            raise RuntimeError(f"agent-core-python not found at {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            # agent-core-python's native-Python conversion (v1-refactor)
            # renamed the public functions to snake_case; the dict payload
            # contract (camelCase keys) is unchanged.
            from agent_core import (
                bootstrap_runtime,
                create_agent_runtime_bindings,
                is_task_supervisor_gateway,
                resolve_config,
                resolve_session,
                run_agent_prompt,
            )
        except ImportError as e:
            raise RuntimeError(f"agent-core import failed: {e}") from e
        self._run_agent_prompt = run_agent_prompt
        self._create_bindings = create_agent_runtime_bindings
        self._is_gateway = is_task_supervisor_gateway
        self._resolve_config = resolve_config
        self._bootstrap_runtime = bootstrap_runtime
        self._resolve_session = resolve_session
        self._workdir = tempfile.mkdtemp(prefix="agentic-memory-ac-")
        atexit.register(shutil.rmtree, self._workdir, ignore_errors=True)
        self._settings_path = (
            str(cfg.agent_core_settings) if cfg.agent_core_settings.is_file() else None
        )
        if not self._settings_path and (self._model or self._reasoning_effort):
            raise RuntimeError("Explicit model/effort overrides require an existing provider settings file")
        self._profile = (profile or cfg.agent_core_profile) if self._settings_path else None
        settings_document = {}
        if self._profile:
            with open(self._settings_path) as fh:
                settings_document = json.load(fh)
                profiles = settings_document.get("providerProfiles") or {}
            if self._profile not in profiles:
                raise RuntimeError(
                    f"profile '{self._profile}' not found in {self._settings_path}"
                )
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
                for key in ('reasoningEffort', 'defaultEffort', 'effort'):
                    if key in selected:
                        selected[key] = self._reasoning_effort
            globals_used = {k: document[k] for k in ('maxTokens', 'thinkingBudget') if k in document}
            return hashlib.sha256(json.dumps([2, self.name, selected, globals_used], sort_keys=True).encode()).hexdigest()
        self._cache_identity_for = identity
        self.cache_identity = identity(settings_document)
        if probe:
            self._probe()

    def verify_profile(self):
        if self._settings_path:
            with open(self._settings_path) as handle:
                if self._cache_identity_for(json.load(handle)) != self.cache_identity:
                    raise RuntimeError('provider settings changed during memory work; retry with fresh configuration')

    async def _run_turn(self, *, system: str, prompt: str) -> dict:
        cwd = self._workdir
        options: dict = {
            "tools": [],
            "sessionPersistence": False,
            "systemPrompt": system,
            "inputFormat": "text",
            "outputFormat": "json",
        }
        if self._profile:
            # Reuse profile transport/auth; explicit per-instance overrides
            # select the model and effort without rewriting the settings file.
            options["settings"] = self._settings_path
            options["provider"] = self._profile
        else:
            options["provider"] = "anthropic"
        if self._model:
            options["model"] = self._model
        if self._reasoning_effort:
            options["effort"] = self._reasoning_effort
        config = await self._resolve_config(options, cwd)
        config = await self._apply_codex_auth_override(config)
        config = self._apply_reasoning_override(config)
        from .model_usage import current_observer
        observer = current_observer()
        if observer:
            observer.configured(config)
        session = await self._resolve_session({}, config, {"cwd": cwd, "originalCwd": cwd})
        runtime = await self._bootstrap_runtime(
            {"session": session, "config": config, "cwd": cwd}
        )
        try:
            daemon = runtime.get("daemon")
            def emit(event, data):
                if observer:
                    observer.event(event, data)
                # Raw stream events are enabled only for usage extraction.
                # Never forward their private content to runtime logs.
                if event != 'provider.stream_event':
                    runtime['emitEvent'](event, data)
            bindings = self._create_bindings(
                {
                    "tools": runtime["tools"],
                    "emitEvent": emit,
                    "streamProviderMessages": bool(observer and config.get('provider') == 'openai'),
                    "daemon": {"state": "connected", "client": daemon}
                    if self._is_gateway(daemon)
                    else {
                        "state": "unavailable",
                        "reason": {
                            "code": "headless.daemon_unavailable",
                            "message": "No task supervisor daemon in embedded runs.",
                        },
                    },
                    "refreshMcpToolRegistry": runtime["refreshMcpToolRegistry"],
                    "sessionBinding": runtime["getActiveBinding"](),
                    "questionPolicy": {
                        "interactive": False,
                        "askQuestion": lambda *_a, **_k: "",
                    },
                }
            )
            runtime["startTurn"]({"prompt": prompt})
            try:
                from .model_usage import check_source_admission
                check_source_admission()
                result = await self._run_agent_prompt(
                    {
                        "session": session,
                        "config": config,
                        "runtime": bindings,
                        "prompt": prompt,
                    }
                )
                runtime["finishTurn"](
                    {"kind": "completed", "stopReason": result.get("stopReason"),
                     "model": result.get("model")}
                )
            except Exception as exc:
                runtime["failTurn"](exc)
                raise
            return result
        finally:
            close = runtime.get("close")
            if close:
                maybe = close()
                if hasattr(maybe, "__await__"):
                    await maybe

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
        from agent_core.storage import load_codex_auth_token
        from agent_core.runtime.secret_provider import normalize_runtime_secret_value
        path = str(self._codex_auth_path)
        token = await load_codex_auth_token(path)
        if not token:
            raise RuntimeError(f'Configured Codex auth file has no usable credential: {path}')
        reference = await normalize_runtime_secret_value(
            value=token, purpose='provider_oauth_token',
            acceptedPurposes=('provider_api_key', 'provider_oauth_token'),
            idempotencyScope=f'memory:{self._profile}:{path}:codex-token')
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
