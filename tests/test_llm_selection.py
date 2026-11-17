from __future__ import annotations

import copy
import asyncio
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tess_memory.config import Config, load_config
from tess_memory.llm import EngineBackend
from tess_core.config import ProviderRegistry
from tess_core import SessionOptions as EngineSessionOptions
from tess_core.config.parsing import _parse_setting_sources, _load_merged_settings
engine_settings = importlib.import_module("tess_core.storage.settings")


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "tess_core").mkdir()
        self.settings = self.root / "settings.json"
        self.profiles = {
            "codex": {"kind": "openai", "apiMode": "codex", "defaultModel": "profile-default", "reasoningEffort": "high"},
            "deepseek-v4-pro": {"kind": "openai", "apiMode": "chat_completions", "defaultModel": "deepseek-v4-pro", "reasoningEffort": "xhigh"},
            "legacy": {"kind": "anthropic", "defaultModel": "legacy-model", "reasoningEffort": "medium"},
        }
        for profile in self.profiles.values():
            profile["auth"] = {"kind": "none"}
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

        async def run_agent_prompt(args):
            self.configs.append(copy.deepcopy(args["config"]))
            return {"text": "OK", "stopReason": "end_turn", "model": args["config"]["model"]}

        # The engine's runtime surface, faked: start_runtime -> sessions.open -> run_turn,
        # resolving the config from the session options exactly as the manager does.
        # Use the real public constructor: a permissive fake hid API drift.
        SessionOptions = EngineSessionOptions

        test = self

        class FakeSession:
            def __init__(self, runtime):
                self.runtime = runtime

            async def run_turn(self, prompt):
                options = self.runtime.session_options("test-session", "client")
                config = await resolve_config(options.config_overrides("home"), self.runtime.cwd)
                if options.adjust_config is not None:
                    config = await options.adjust_config(config)
                session = await resolve_session({}, config, {})

                def emit(event, data):
                    self.runtime.listener.on_event("test-session", {"event": event, "data": data})

                return await test.run_agent_prompt_hook({"session": session, "config": config, "prompt": prompt,
                                                         "runtime": {"streamProviderMessages": options.stream_provider_messages, "emitEvent": emit}})

        class FakeRuntime:
            def __init__(self, cwd, session_options, listener):
                self.cwd, self.session_options, self.listener = cwd, session_options, listener
                self.sessions = types.SimpleNamespace(open=self.open)
                self.stopped = False

            async def open(self, session_id=None):
                return FakeSession(self)

            async def stop(self, grace=0.0):
                self.stopped = True
                return {}

        async def start_runtime(runtime_home=None, *, cwd=None, daemon_enabled=True, session_options=None, listener=None, **_kw):
            test.assertFalse(daemon_enabled)
            test.assertEqual(Path(runtime_home), test.root / "runtime")
            test.assertIsInstance(_kw["provider_registry"], ProviderRegistry)
            options = session_options("test-session", "client")
            test.assertEqual(options.settings, "{}")
            test.assertEqual(options.config_overrides("home")["settingSources"], "")
            test.assertTrue(options.strict_mcp_config)
            return FakeRuntime(cwd, session_options, listener)

        self.run_agent_prompt_hook = run_agent_prompt
        self.fake_runtime_factory = FakeRuntime
        runtime = types.ModuleType("tess_core")
        runtime.resolve_config = resolve_config
        runtime.resolve_session = resolve_session
        runtime.run_agent_prompt = run_agent_prompt
        runtime.start_runtime = start_runtime
        runtime.SessionOptions = SessionOptions
        patches = [mock.patch.dict(sys.modules, {"tess_core": runtime}),
                   mock.patch.object(sys, "path", list(sys.path)),
                   mock.patch("tess_memory.llm.tempfile.mkdtemp", return_value=str(self.root)),
                   mock.patch("tess_memory.llm.atexit.register")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.cfg = Config(data_dir=self.root / "memory", engine_settings=self.settings,
                          engine_profile="codex", engine_model="gpt-5.6-terra",
                          engine_reasoning_effort="xhigh",
                          engine_fallback_profile="deepseek-v4-pro",
                          engine_fallback_model="deepseek-v4-pro",
                          engine_fallback_reasoning_effort="max",
                          search_profile="codex", search_model="gpt-5.6-sol",
                          search_reasoning_effort="xhigh")

    def test_each_role_uses_its_own_model_and_effort_without_editing_profiles(self):
        backends = [EngineBackend(self.cfg, probe=False),
                    EngineBackend.for_role(self.cfg, "fallback"),
                    EngineBackend.for_role(self.cfg, "search")]
        for backend in backends:
            self.assertEqual(backend.complete(system="Distill this", prompt="Test only"), "OK")
        self.assertEqual([(o["provider"], o["model"], o["effort"]) for o in self.options], [
            ("codex", "gpt-5.6-terra", "xhigh"),
            ("deepseek-v4-pro", "deepseek-v4-pro", "max"),
            ("codex", "gpt-5.6-sol", "xhigh"),
        ])
        for options in self.options:
            self.assertEqual(options["settingSources"], "")
            self.assertEqual(options["settings"], "{}")
            self.assertTrue(options["strictMcpConfig"])
            self.assertEqual(options["tools"], [])
            self.assertFalse(options["sessionPersistence"])
        self.assertEqual(self.settings.read_bytes(), self.original_settings)

    def test_real_source_selection_does_not_read_ambient_settings(self):
        backend = EngineBackend(self.cfg, probe=False)
        self.assertEqual(backend.complete(system="Synthetic", prompt="Test only"), "OK")
        options = self.options[-1]  # actual public SessionOptions.config_overrides
        sources = _parse_setting_sources(options["settingSources"])
        self.assertEqual(sources, [])
        with mock.patch.object(engine_settings, "load_settings", new_callable=mock.AsyncMock,
                               side_effect=AssertionError("ambient source read")) as ambient:
            merged = asyncio.run(_load_merged_settings(sources, str(self.root), options["settings"]))
            self.assertEqual(merged, {"providerProfiles": {}})
            ambient.assert_not_awaited()

    def test_cleanup_failure_preserves_paid_response_without_masking_provider_failure(self):
        from tess_memory.model_usage import UsageObserver, observe_usage
        backend=EngineBackend(self.cfg,probe=False)
        start=backend._start_runtime
        async def failing_stop(grace=0.0):
            raise RuntimeError('synthetic cleanup timeout')
        async def runtime(*args, **kwargs):
            result=await start(*args, **kwargs)
            result.stop=failing_stop
            return result
        backend._start_runtime=runtime
        records=[]
        with observe_usage(UsageObserver(records.append)):
            self.assertEqual(backend.complete(system='Test',prompt='Test'),'OK')
        self.assertEqual(len(self.configs),1)
        self.assertEqual(records[-1]['cleanup_error'],'RuntimeError')
        self.run_agent_prompt_hook=mock.AsyncMock(side_effect=ValueError('provider failed'))
        with self.assertRaises(Exception):backend.complete(system='Test',prompt='Test')

    def test_deepseek_max_survives_the_custom_host_effort_filter(self):
        backend = EngineBackend.for_role(self.cfg, "fallback")
        backend.complete(system="Test", prompt="Test")
        config = self.configs[-1]
        self.assertEqual(config["effort"], "max")
        options = config["providerProfile"]["requestOptions"]
        self.assertEqual(options["reasoning_effort"], "max")
        self.assertEqual(options["thinking"], {"type": "enabled"})
        self.assertEqual(config["providerProfiles"]["deepseek-v4-pro"]["requestOptions"], options)

    def test_explicit_effort_wins_over_conflicting_request_extras_without_mutation(self):
        backend = EngineBackend.for_role(self.cfg, "fallback")
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
        cfg = Config(engine_settings=self.settings, engine_profile="legacy",
                     engine_fallback_profile="deepseek-v4-pro", search_profile="codex")
        for backend in [EngineBackend(cfg, probe=False), EngineBackend.for_role(cfg, "fallback"),
                        EngineBackend.for_role(cfg, "search")]:
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
        EngineBackend.for_role(self.cfg, "search").complete(system="Test", prompt="Test")
        self.assertNotIn("model", self.options[-1])
        self.assertNotIn("effort", self.options[-1])
        self.assertEqual(self.configs[-1]["model"], "profile-default")

    def test_missing_settings_cannot_silently_change_an_explicit_provider_selection(self):
        self.cfg.engine_settings = self.root / "missing.json"
        with self.assertRaisesRegex(ValueError, "explicit provider registry"):
            EngineBackend(self.cfg, probe=False)

    def test_toml_loads_requested_models_and_preserves_embedding_settings(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "examples/codex-settings.toml")
        self.assertEqual((cfg.engine_profile, cfg.engine_model, cfg.engine_reasoning_effort),
                         ("strong", "example-large", "high"))
        self.assertEqual((cfg.engine_fallback_profile, cfg.engine_fallback_model, cfg.engine_fallback_reasoning_effort),
                         ("cheap", "example-small", "medium"))
        self.assertEqual((cfg.search_profile, cfg.search_model, cfg.search_reasoning_effort),
                         ("cheap", "example-small", "medium"))
        self.assertEqual((cfg.embed_provider, cfg.embed_dims), ("embeddings", 1024))

    def test_invalid_effort_is_rejected_instead_of_silently_defaulting(self):
        path = self.root / "invalid.toml"
        path.write_text('[search]\nreasoning_effort = "xhig"\n')
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            load_config(path)

    def test_cache_identity_ignores_credential_rotation_and_unrelated_profiles(self):
        original = EngineBackend(self.cfg, probe=False).cache_identity
        updated = copy.deepcopy(self.profiles)
        updated['legacy']['defaultModel'] = 'a-new-unrelated-model'
        updated['codex']['auth'] = {'kind': 'none', 'envVar': 'ROTATED_CREDENTIAL'}
        updated['codex']['defaultModel'] = 'new-default-overridden-by-explicit-pin'
        updated['codex']['reasoningEffort'] = 'low'
        self.settings.write_text(json.dumps({'providerProfiles': updated, 'env': {'ROTATED_CREDENTIAL': 'synthetic-key'}}))
        self.assertEqual(EngineBackend(self.cfg, probe=False).cache_identity, original)
        self.cfg.engine_model = 'different-explicit-model'
        self.assertNotEqual(EngineBackend(self.cfg, probe=False).cache_identity, original)

    def test_usage_observer_extracts_wire_counters_without_forwarding_raw_events(self):
        from tess_memory.model_usage import UsageObserver, observe_usage
        backend = EngineBackend(self.cfg, probe=False)
        records = []
        observer = UsageObserver(records.append)
        async def turn(args):
            runtime = args['runtime']
            self.assertTrue(runtime['streamProviderMessages'])
            runtime['emitEvent']('provider.request_started', {})
            runtime['emitEvent']('provider.stream_event', {'stream_event': 'response.completed', 'payload': {'response': {
                'model': 'gpt-5.6-terra', 'output': 'private reasoning',
                'usage': {'input_tokens': 10, 'output_tokens': 20, 'output_tokens_details': {'reasoning_tokens': 15}}}}})
            return {'text': 'OK', 'model': 'gpt-5.6-terra', 'usage': {'input_tokens': 10, 'output_tokens': 20}}
        self.run_agent_prompt_hook = turn
        with observe_usage(observer):
            self.assertEqual(backend.complete(system='s', prompt='p'), 'OK')
        self.assertEqual(observer.data()['reasoning_tokens'], 15)
        self.assertEqual(observer.data()['output_tokens'], 20)
        self.assertNotIn('private reasoning', json.dumps(records))

    def test_codex_auth_pin_overrides_inherited_config_and_does_not_invalidate_cache(self):
        import asyncio
        original_identity = EngineBackend(self.cfg, probe=False).cache_identity
        self.cfg.engine_codex_auth_path = self.root / 'alt-auth.json'
        loader = mock.AsyncMock(side_effect=['first-token', 'refreshed-token'])
        normalizer = mock.AsyncMock(side_effect=lambda value, **kw: {'secretRef': value})
        source = {'provider': 'openai', 'openaiApiMode': 'codex', 'openaiToken': 'inherited-token',
                  'openaiCodexAuthPath': '/different/auth.json',
                  'compactionProviderProfile': {'profile': {'kind': 'openai', 'apiMode': 'codex'}, 'openaiToken': 'old'}}
        before = copy.deepcopy(source)
        with mock.patch.object(sys.modules["tess_core.config"], "resolve_codex_auth", loader), \
             mock.patch.object(sys.modules["tess_core.config"], "resolve_provider_secret", normalizer):
            for role, expected in [('primary', 'first-token'), ('search', 'refreshed-token')]:
                backend = EngineBackend(self.cfg, probe=False) if role == 'primary' else EngineBackend.for_role(self.cfg, role)
                if role == 'primary':
                    self.assertEqual(backend.cache_identity, original_identity)
                result = asyncio.run(backend._apply_codex_auth_override(source))
                self.assertEqual(result['openaiCodexAuthPath'], str(self.cfg.engine_codex_auth_path))
                self.assertEqual(result['openaiToken'], {'secretRef': expected})
                self.assertEqual(result['compactionProviderProfile']['openaiToken'], result['openaiToken'])
            fallback = EngineBackend.for_role(self.cfg, 'fallback')
            other = {'provider': 'openai', 'openaiApiMode': 'chat_completions', 'openaiToken': 'deepseek'}
            self.assertIs(asyncio.run(fallback._apply_codex_auth_override(other)), other)
        self.assertEqual(loader.await_count, 2)
        loader.assert_awaited_with(str(self.cfg.engine_codex_auth_path))
        self.assertEqual(source, before)
        self.assertEqual(self.settings.read_bytes(), self.original_settings)

    def test_missing_pinned_codex_credential_never_uses_inherited_account(self):
        import asyncio
        self.cfg.engine_codex_auth_path = self.root / 'missing-auth.json'
        loader = mock.AsyncMock(return_value=None)
        normalizer = mock.AsyncMock()
        with mock.patch.object(sys.modules["tess_core.config"], "resolve_codex_auth", loader), \
             mock.patch.object(sys.modules["tess_core.config"], "resolve_provider_secret", normalizer):
            backend = EngineBackend(self.cfg, probe=False)
            with self.assertRaisesRegex(RuntimeError, 'no usable credential'):
                asyncio.run(backend._apply_codex_auth_override({'provider': 'openai', 'openaiApiMode': 'codex', 'openaiToken': 'old'}))
            normalizer.assert_not_awaited()

    def test_codex_auth_path_loads_and_requires_an_explicit_file_path(self):
        path = self.root / 'memory.toml'
        path.write_text('[llm]\ncodex_auth_path="~/.codex-second/auth.json"\n')
        self.assertEqual(load_config(path).engine_codex_auth_path, Path('~/.codex-second/auth.json').expanduser())
        path.write_text('[llm]\ncodex_auth_path="relative.json"\n')
        self.assertEqual(load_config(path).engine_codex_auth_path, self.root / 'relative.json')
        for value in ['""', 'true']:
            path.write_text('[llm]\ncodex_auth_path='+value+'\n')
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()


def test_an_openai_kind_profile_gets_its_keychain_held_key():
    from tess_memory.llm import EngineBackend

    reference = {"kind": "runtime_secret", "version": 1, "id": "k", "purpose": "provider_api_key"}
    profile = {"auth": {"kind": "openai_api_key", "credentialRef": reference}}
    config = {"provider": "openai", "providerProfile": profile}
    assert EngineBackend._apply_managed_key(config)["openaiToken"] == reference
    # A credential the engine already resolved, another kind of auth, or another provider is left alone.
    assert EngineBackend._apply_managed_key({**config, "openaiToken": "from-env"})["openaiToken"] == "from-env"
    for untouched in ({"provider": "anthropic", "providerProfile": profile},
                      {"provider": "openai", "providerProfile": {"auth": {"kind": "chatgpt", "credentialRef": reference}}},
                      {"provider": "openai", "providerProfile": {"auth": {"kind": "custom_bearer", "envVar": "KEY"}}}):
        assert "openaiToken" not in EngineBackend._apply_managed_key(untouched)
