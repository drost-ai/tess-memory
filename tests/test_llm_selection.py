from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from agentic_memory.config import Config, load_config
from agentic_memory.llm import AgentCoreBackend


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "agent_core").mkdir()
        self.settings = self.root / "settings.json"
        self.profiles = {
            "codex": {"kind": "openai", "apiMode": "codex", "defaultModel": "profile-default", "reasoningEffort": "high"},
            "deepseek-v4-pro": {"kind": "openai", "apiMode": "chat_completions", "defaultModel": "deepseek-v4-pro", "reasoningEffort": "xhigh"},
            "legacy": {"kind": "anthropic", "defaultModel": "legacy-model", "reasoningEffort": "medium"},
        }
        self.settings.write_text(json.dumps({"providerProfiles": self.profiles}))
        self.original_settings = self.settings.read_bytes()
        self.options = []
        self.configs = []

        async def resolve_config(options, cwd):
            self.options.append(copy.deepcopy(options))
            name = options["provider"]
            profile = {**copy.deepcopy(self.profiles[name]), "name": name, "requestOptions": {}}
            return {"provider": profile["kind"], "providerProfileName": name,
                    "providerProfile": profile, "providerProfiles": {name: profile},
                    "openaiApiMode": profile.get("apiMode"),
                    "model": options.get("model") or profile["defaultModel"],
                    "effort": options.get("effort") or profile["reasoningEffort"]}

        async def resolve_session(*args):
            return {"id": "test-session"}

        async def bootstrap_runtime(args):
            return {"tools": [], "emitEvent": mock.Mock(), "refreshMcpToolRegistry": mock.Mock(),
                    "getActiveBinding": lambda: {}, "startTurn": mock.Mock(),
                    "finishTurn": mock.Mock(), "failTurn": mock.Mock()}

        async def run_agent_prompt(args):
            self.configs.append(copy.deepcopy(args["config"]))
            return {"text": "OK", "stopReason": "end_turn", "model": args["config"]["model"]}

        runtime = types.ModuleType("agent_core")
        runtime.resolve_config = resolve_config
        runtime.resolve_session = resolve_session
        runtime.bootstrap_runtime = bootstrap_runtime
        runtime.run_agent_prompt = run_agent_prompt
        runtime.create_agent_runtime_bindings = lambda args: args
        runtime.is_task_supervisor_gateway = lambda _: False
        patches = [mock.patch.dict(sys.modules, {"agent_core": runtime}),
                   mock.patch.object(sys, "path", list(sys.path)),
                   mock.patch("agentic_memory.llm.tempfile.mkdtemp", return_value=str(self.root)),
                   mock.patch("agentic_memory.llm.atexit.register")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.cfg = Config(agent_core_path=self.root, agent_core_settings=self.settings,
                          agent_core_profile="codex", agent_core_model="gpt-5.6-terra",
                          agent_core_reasoning_effort="xhigh",
                          agent_core_fallback_profile="deepseek-v4-pro",
                          agent_core_fallback_model="deepseek-v4-pro",
                          agent_core_fallback_reasoning_effort="max",
                          search_profile="codex", search_model="gpt-5.6-sol",
                          search_reasoning_effort="xhigh")

    def test_each_role_uses_its_own_model_and_effort_without_editing_profiles(self):
        backends = [AgentCoreBackend(self.cfg, probe=False),
                    AgentCoreBackend.for_role(self.cfg, "fallback"),
                    AgentCoreBackend.for_role(self.cfg, "search")]
        for backend in backends:
            self.assertEqual(backend.complete(system="Distill this", prompt="Test only"), "OK")
        self.assertEqual([(o["provider"], o["model"], o["effort"]) for o in self.options], [
            ("codex", "gpt-5.6-terra", "xhigh"),
            ("deepseek-v4-pro", "deepseek-v4-pro", "max"),
            ("codex", "gpt-5.6-sol", "xhigh"),
        ])
        for options in self.options:
            self.assertEqual(options["tools"], [])
            self.assertFalse(options["sessionPersistence"])
        self.assertEqual(self.settings.read_bytes(), self.original_settings)

    def test_deepseek_max_survives_the_custom_host_effort_filter(self):
        backend = AgentCoreBackend.for_role(self.cfg, "fallback")
        backend.complete(system="Test", prompt="Test")
        config = self.configs[-1]
        self.assertEqual(config["effort"], "max")
        options = config["providerProfile"]["requestOptions"]
        self.assertEqual(options["reasoning_effort"], "max")
        self.assertEqual(options["thinking"], {"type": "enabled"})
        self.assertEqual(config["providerProfiles"]["deepseek-v4-pro"]["requestOptions"], options)

    def test_explicit_effort_wins_over_conflicting_request_extras_without_mutation(self):
        backend = AgentCoreBackend.for_role(self.cfg, "fallback")
        original = {"provider": "openai", "openaiApiMode": "chat_completions", "model": "deepseek-v4-pro",
                    "providerProfile": {"name": "deepseek-v4-pro", "requestOptions": {
                        "reasoning_effort": "high", "extra_body": {
                            "reasoning_effort": "high", "thinking": {"type": "disabled"}, "keep": True,
                        }}}}
        before = copy.deepcopy(original)
        changed = backend._apply_reasoning_override(original)
        self.assertEqual(original, before)
        options = changed["providerProfile"]["requestOptions"]
        self.assertEqual(options["reasoning_effort"], "max")
        self.assertEqual(options["thinking"], {"type": "enabled"})
        self.assertEqual(options["extra_body"], {"keep": True})

    def test_legacy_profiles_keep_model_and_effort_defaults(self):
        cfg = Config(agent_core_path=self.root, agent_core_settings=self.settings, agent_core_profile="legacy",
                     agent_core_fallback_profile="deepseek-v4-pro", search_profile="codex")
        for backend in [AgentCoreBackend(cfg, probe=False), AgentCoreBackend.for_role(cfg, "fallback"),
                        AgentCoreBackend.for_role(cfg, "search")]:
            backend.complete(system="Test", prompt="Test")
        for options in self.options:
            self.assertNotIn("model", options)
            self.assertNotIn("effort", options)
        self.assertEqual([(c["model"], c["effort"]) for c in self.configs], [
            ("legacy-model", "medium"), ("deepseek-v4-pro", "xhigh"), ("profile-default", "high"),
        ])
        self.assertEqual(self.configs[1]["providerProfile"]["requestOptions"], {})

    def test_search_profile_defaults_do_not_inherit_distillation_overrides(self):
        self.cfg.search_model = ""
        self.cfg.search_reasoning_effort = ""
        AgentCoreBackend.for_role(self.cfg, "search").complete(system="Test", prompt="Test")
        self.assertNotIn("model", self.options[-1])
        self.assertNotIn("effort", self.options[-1])
        self.assertEqual(self.configs[-1]["model"], "profile-default")

    def test_missing_settings_cannot_silently_change_an_explicit_provider_selection(self):
        self.cfg.agent_core_settings = self.root / "missing.json"
        with self.assertRaisesRegex(RuntimeError, "existing provider settings"):
            AgentCoreBackend(self.cfg, probe=False)

    def test_toml_loads_requested_models_and_preserves_embedding_settings(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "examples/codex-settings.toml")
        self.assertEqual((cfg.agent_core_profile, cfg.agent_core_model, cfg.agent_core_reasoning_effort),
                         ("codex", "gpt-5.6-terra", "xhigh"))
        self.assertEqual((cfg.agent_core_fallback_profile, cfg.agent_core_fallback_model, cfg.agent_core_fallback_reasoning_effort),
                         ("deepseek-v4-pro", "deepseek-v4-pro", "max"))
        self.assertEqual((cfg.search_profile, cfg.search_model, cfg.search_reasoning_effort),
                         ("codex", "gpt-5.6-sol", "xhigh"))
        self.assertEqual((cfg.embed_provider, cfg.embed_dims), ("embeddings", 4096))

    def test_invalid_effort_is_rejected_instead_of_silently_defaulting(self):
        path = self.root / "invalid.toml"
        path.write_text('[search]\nreasoning_effort = "xhig"\n')
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
