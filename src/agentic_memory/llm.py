"""LLM backends for distillation.

Three interchangeable backends behind one interface:

    AgentCoreBackend    — routes through ~/agents/agent-core-python (preferred)
    AnthropicSDKBackend — direct Claude API via the anthropic SDK
    ClaudeCLIBackend    — `claude -p` headless mode (reuses Claude Code auth)

`auto` resolution order: agent-core if its runtime is usable, else the SDK if
API credentials exist, else the claude CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config


class LLMBackend(ABC):
    name: str = "?"

    @abstractmethod
    def complete(self, *, system: str, prompt: str, model: str, max_tokens: int = 3000) -> str:
        """Run one system+user completion and return the response text."""


class AnthropicSDKBackend(LLMBackend):
    name = "sdk"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()

    def complete(self, *, system: str, prompt: str, model: str, max_tokens: int = 3000) -> str:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        return "".join(b.text for b in resp.content if b.type == "text")


class ClaudeCLIBackend(LLMBackend):
    """Headless `claude -p`: no API key needed, uses existing Claude Code auth.

    Runs in an empty scratch directory so no project CLAUDE.md/hooks leak into
    the distillation context.
    """

    name = "claude-cli"

    def __init__(self) -> None:
        if not shutil.which("claude"):
            raise RuntimeError("claude CLI not found on PATH")
        self._workdir = Path(tempfile.mkdtemp(prefix="agentic-memory-llm-"))

    def complete(self, *, system: str, prompt: str, model: str, max_tokens: int = 3000) -> str:
        cmd = [
            "claude", "-p",
            "--model", model,
            "--output-format", "json",
            "--append-system-prompt", system,
        ]
        env = dict(os.environ)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=self._workdir,
            timeout=600,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (exit {proc.returncode}): {proc.stderr[:500]}"
            )
        try:
            envelope = json.loads(proc.stdout)
            result = envelope.get("result")
            if isinstance(result, str):
                return result
        except json.JSONDecodeError:
            pass
        return proc.stdout


class AgentCoreBackend(LLMBackend):
    """In-process integration with the agent-core-python runtime.

    Imports the runtime from its checkout (it is dependency-free) and composes
    the same seams its own headless runner uses — resolveConfig →
    resolveSession → bootstrapRuntime → runAgentPrompt — for one
    non-interactive turn per completion: real system prompt via the config's
    `systemPrompt` option, tools disabled, no session persistence.

    Credentials come from the runtime's own auth (`tess auth login` /
    ANTHROPIC_API_KEY); a startup probe detects the missing-credential case so
    `auto` resolution can fall back cleanly.
    """

    name = "agent-core"

    def __init__(self, cfg: Config, probe: bool = True) -> None:
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
        self._fast_model = cfg.fast_model
        self._settings_path = (
            str(cfg.agent_core_settings) if cfg.agent_core_settings.is_file() else None
        )
        self._profile = cfg.agent_core_profile if self._settings_path else None
        if probe:
            self._probe()

    async def _run_turn(self, *, system: str, prompt: str, model: str) -> dict:
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
            # the profile's defaultModel governs, so the model arg is ignored.
            options["settings"] = self._settings_path
            options["provider"] = self._profile
        else:
            options["provider"] = "anthropic"
            options["model"] = model
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
            runtime["startTurn"]({"prompt": prompt, "model": model})
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

    def _run(self, *, system: str, prompt: str, model: str) -> str:
        import asyncio

        result = asyncio.run(self._run_turn(system=system, prompt=prompt, model=model))
        text = result.get("text") or ""
        if "No credential is configured" in text:
            raise RuntimeError(f"agent-core: {text[:200]}")
        if result.get("stopReason") in ("error", "failed", "cancelled", "permission_denied"):
            raise RuntimeError(
                f"agent-core turn failed: {result.get('error') or text[:200] or result.get('stopReason')}"
            )
        return text

    def _probe(self) -> None:
        self._run(system="You are a health check.", prompt="Reply with exactly: OK",
                  model=self._fast_model)

    def complete(self, *, system: str, prompt: str, model: str, max_tokens: int = 3000) -> str:
        return self._run(system=system, prompt=prompt, model=model)


class NullBackend(LLMBackend):
    """Explicitly-disabled LLM (extractive-only ingestion)."""

    name = "none"

    def complete(self, *, system: str, prompt: str, model: str, max_tokens: int = 3000) -> str:
        raise RuntimeError("LLM backend disabled (AGENTIC_MEMORY_BACKEND=none)")


def _sdk_credentials_present() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    cfg_dir = os.environ.get("ANTHROPIC_CONFIG_DIR", "~/.config/anthropic")
    return Path(cfg_dir).expanduser().is_dir()


def make_backend(cfg: Config) -> LLMBackend:
    choice = cfg.llm_backend
    if choice == "none":
        return NullBackend()
    if choice == "agent-core":
        return AgentCoreBackend(cfg)
    if choice == "sdk":
        return AnthropicSDKBackend()
    if choice == "claude-cli":
        return ClaudeCLIBackend()
    # auto
    try:
        return AgentCoreBackend(cfg)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"[llm] agent-core backend unavailable ({e}); falling back", file=sys.stderr)
    if _sdk_credentials_present():
        try:
            return AnthropicSDKBackend()
        except Exception:
            pass
    return ClaudeCLIBackend()
