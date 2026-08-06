"""LLM backend for distillation: the agent-core-python runtime, in-process.

agent-core-python is the only supported path. The runtime is imported from its
checkout (it is dependency-free) and each completion runs one non-interactive
turn through the same seams its own headless runner uses — resolveConfig →
resolveSession → bootstrapRuntime → runAgentPrompt — with a real system prompt,
tools disabled, and no session persistence.

Provider, model, and credentials are governed entirely by the runtime's own
configuration: when the Tess settings file exists, its provider profile is
selected and that profile's `defaultModel` rules; otherwise agent-core's
defaults apply. This package never chooses a model.
"""

from __future__ import annotations

import sys
import tempfile

from .config import Config


class RefusalError(RuntimeError):
    """The model's safety classifiers declined the request (stopReason=refusal)."""


class AgentCoreBackend:
    name = "agent-core"

    def __init__(self, cfg: Config, probe: bool = True, profile: str | None = None) -> None:
        root = cfg.agent_core_path
        if not (root / "agent_core").is_dir():
            raise RuntimeError(f"agent-core-python not found at {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from agent_core import (
                bootstrapRuntime,
                createAgentRuntimeBindings,
                isTaskSupervisorGateway,
                resolveConfig,
                resolveSession,
                runAgentPrompt,
            )
        except ImportError as e:
            raise RuntimeError(f"agent-core import failed: {e}") from e
        self._run_agent_prompt = runAgentPrompt
        self._create_bindings = createAgentRuntimeBindings
        self._is_gateway = isTaskSupervisorGateway
        self._resolve_config = resolveConfig
        self._bootstrap_runtime = bootstrapRuntime
        self._resolve_session = resolveSession
        self._workdir = tempfile.mkdtemp(prefix="agentic-memory-ac-")
        self._settings_path = (
            str(cfg.agent_core_settings) if cfg.agent_core_settings.is_file() else None
        )
        self._profile = (profile or cfg.agent_core_profile) if self._settings_path else None
        if self._profile:
            import json

            with open(self._settings_path) as fh:
                profiles = json.load(fh).get("providerProfiles") or {}
            if self._profile not in profiles:
                raise RuntimeError(
                    f"profile '{self._profile}' not found in {self._settings_path}"
                )
            self.name = f"agent-core:{self._profile}"
        if probe:
            self._probe()

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
            # Profile mode: merge the Tess settings file and select the profile;
            # the profile's defaultModel governs.
            options["settings"] = self._settings_path
            options["provider"] = self._profile
        else:
            options["provider"] = "anthropic"
        config = await self._resolve_config(options, cwd)
        session = await self._resolve_session({}, config, {"cwd": cwd, "originalCwd": cwd})
        runtime = await self._bootstrap_runtime(
            {"session": session, "config": config, "cwd": cwd}
        )
        try:
            daemon = runtime.get("daemon")
            bindings = self._create_bindings(
                {
                    "tools": runtime["tools"],
                    "emitEvent": runtime["emitEvent"],
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
                    "print_mode": True,
                    "printMode": True,
                }
            )
            runtime["startTurn"]({"prompt": prompt})
            try:
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

    def _probe(self) -> None:
        # Fail fast and loud at construction: without this, a broken runtime
        # would surface as every session silently degrading to extractive.
        self.complete(system="You are a health check.", prompt="Reply with exactly: OK")
