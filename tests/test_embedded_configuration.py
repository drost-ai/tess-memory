"""Embedded boundary tests: temporary homes, metadata only, no real providers."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
import yaml
from unittest.mock import patch, Mock

from tess_memory.config import DataStore, Config, load_config, load_manifest, parse_manifest, validate_config
from tess_memory.embed import make_embedder
from tess_memory.llm import AgentCoreBackend


class EmbeddedConfigurationTests(unittest.TestCase):
    def test_invalid_provider_configuration_does_not_allocate_scratch_state(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "uninitialized-store"
            with self.assertRaisesRegex(ValueError, "registry"):
                AgentCoreBackend(Config(data_dir=directory), probe=False)
            self.assertFalse(directory.exists())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / 'runtime'
        self.home.mkdir()
        self.registry = self.root / 'providers.yaml'
        self.profiles = {
            'distill': {'kind': 'openai', 'baseUrl': 'https://scripted.invalid/v1',
                        'defaultModel': 'scripted', 'auth': {'kind': 'openai_api_key', 'envVar': 'SCRIPTED_MEMORY_KEY'}},
            'search': {'kind': 'openai', 'baseUrl': 'https://scripted.invalid/v1', 'defaultModel': 'search-model'},
            'fallback': {'kind': 'openai', 'baseUrl': 'https://scripted.invalid/v1', 'defaultModel': 'fallback-model'},
            'embed': {'kind': 'openai', 'baseUrl': 'https://scripted.invalid/v1', 'defaultModel': 'embedding-model'},
        }
        for profile in self.profiles.values():
            profile.setdefault('auth', {'kind': 'none'})
        self.registry.write_text(yaml.safe_dump({'profiles': self.profiles}))
        self.manifest = self.root / 'manifest.json'
        self.envelope = {
            'schema_version': 1, 'mode': 'embedded', 'runtime_home': str(self.home),
            'provider_registry': str(self.registry),
            'config': {'llm': {'profile': 'distill'}, 'search': {'profile': 'search'},
                       'embedding': {'provider': 'local', 'local_model': 'scripted-local'}},
        }
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('network forbidden')).start()

    def load(self, envelope=None):
        self.manifest.write_text(json.dumps(self.envelope if envelope is None else envelope))
        return load_manifest(self.manifest)

    def test_in_memory_manifest_matches_file_loader_without_writes_or_mutation(self):
        from types import MappingProxyType
        before = copy.deepcopy(self.envelope)
        expected = self.load()
        self.manifest.unlink()
        with patch('tess_memory.config.load_manifest', side_effect=AssertionError('no manifest file')):
            actual = parse_manifest(MappingProxyType(self.envelope))
            validate_config(actual, validate_credentials=False)
        self.assertEqual(actual, expected)
        self.assertEqual(self.envelope, before)
        self.assertFalse(self.manifest.exists())
        self.assertFalse(list(self.home.iterdir()))

    def test_file_loader_delegates_to_in_memory_validator(self):
        self.manifest.write_text(json.dumps(self.envelope))
        sentinel = object()
        with patch('tess_memory.config.parse_manifest', return_value=sentinel) as parse:
            self.assertIs(load_manifest(self.manifest, store_settings='store.toml'), sentinel)
        parse.assert_called_once_with(self.envelope, store_settings='store.toml')

    def test_in_memory_manifest_preserves_validation_and_store_overlay_contract(self):
        for updates in ({'schema_version': 2}, {'mode': 'standalone'},
                        {'runtime_home': 'relative'}, {'config': []}):
            document = {**self.envelope, **updates}
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError) as memory_error:
                    parse_manifest(document)
                with self.assertRaises(ValueError) as file_error:
                    self.load(document)
                self.assertEqual(str(memory_error.exception), str(file_error.exception))
        with self.assertRaises(ValueError):
            parse_manifest([])
        overlay = self.home / 'store.toml'
        overlay.write_text('[storage]\ndata_dir="selected-store"\n')
        before = copy.deepcopy(self.envelope)
        actual = parse_manifest(self.envelope, store_settings='store.toml')
        self.assertEqual(actual.data_dir, self.home / 'selected-store')
        self.manifest.write_text(json.dumps(self.envelope))
        self.assertEqual(actual, load_manifest(self.manifest, store_settings='store.toml'))
        self.assertEqual(self.envelope, before)
        overlay.write_text('[llm]\nprofile="forbidden"\n')
        with self.assertRaisesRegex(ValueError, 'override only'):
            parse_manifest(self.envelope, store_settings='store.toml')

    def test_empty_home_no_writes_or_ambient_overrides(self):
        with patch.dict(os.environ, {'HOME': str(self.root / 'empty-user-home'),
                                    'TESS_MEMORY_DATA': '/ambient/data',
                                    'TESS_MEMORY_SOURCE': '/ambient/source',
                                    'TESS_MEMORY_AC_PROFILE': 'ambient',
                                    'TESS_MEMORY_USER': 'ambient',
                                    'TESS_MEMORY_AC_SETTINGS': '/ambient/settings'}):
            cfg = self.load()
        self.assertEqual(cfg.data_dir, self.home / 'memory')
        self.assertEqual(cfg.data_stores, [])
        self.assertEqual(cfg.agent_core_profile, 'distill')
        self.assertIsNone(cfg.identity_user)
        self.assertFalse(list(self.home.iterdir()))
        self.assertIsNone(cfg.provenance_ledger)

    def test_relative_paths_and_absolute_source_identity(self):
        absolute = '/explicit/archive/sessions'
        self.envelope['config'].update({
            'storage': {'data_dir': 'state/memory'},
            'data_stores': [{'path': 'sessions', 'skip_copies_of': ['copies']},
                            {'path': absolute, 'type': 'agent-core'}],
            'personal': {'source_mode': 'native', 'source_paths': ['sessions', absolute]},
        })
        self.envelope['provenance_ledger'] = str(self.root / 'elsewhere' / 'ledger.db')
        self.envelope['provenance_source_root'] = str(self.home / 'sessions')
        cfg = self.load()
        self.assertEqual(cfg.data_dir, self.home / 'state/memory')
        self.assertEqual(cfg.data_stores[0].path, self.home / 'sessions')
        self.assertEqual(cfg.data_stores[0].skip_copies_of, [self.home / 'copies'])
        self.assertEqual(str(cfg.data_stores[1].path), absolute)
        self.assertEqual(cfg.personal.source_paths, (str(self.home / 'sessions'), absolute))
        self.assertEqual(cfg.personal.native_creation_ledger, self.envelope['provenance_ledger'])
        self.assertFalse(cfg.provenance_ledger.exists())

    def test_malformed_envelopes(self):
        cases = [[], {}, {**self.envelope, 'schema_version': True},
                 {**self.envelope, 'schema_version': 2}, {**self.envelope, 'mode': 'standalone'},
                 {**self.envelope, 'runtime_home': 'relative'},
                 {**self.envelope, 'provider_registry': '~/providers.yaml'},
                 {**self.envelope, 'provenance_ledger': 'relative'},
                 {**self.envelope, 'provenance_source_root': 'relative'},
                 {**self.envelope, 'config': []}, {**self.envelope, 'unknown': True}]
        for envelope in cases:
            with self.subTest(envelope=envelope), self.assertRaises(ValueError):
                self.load(envelope)
        self.manifest.write_text('{')
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)
        with self.assertRaises(ValueError):
            load_manifest('relative.json')
        self.manifest.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_manifest(self.manifest)

    def test_malformed_mapping(self):
        cases = [{'llm': []}, {'unknown': {}}, {'data_stores': {}},
                 {'data_stores': [{}]}, {'data_stores': [{'path': 'sessions', 'exclude': 'bad'}]},
                 {'personal': {'source_paths': 'bad'}}, {'data_stores': [{'path': '~/sessions'}]},
                 {'llm': {'profile': 'distill', 'settings': '/legacy'}},
                 {'personal': {'native_creation_ledger': '/legacy'}},
                 {'embedding': {'provider': 'local', 'dims': True}},
                 {'embedding': {'provider': 'local', 'allow_download': 'yes'}}]
        for updates in cases:
            envelope = copy.deepcopy(self.envelope)
            envelope['config'].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.load(envelope)

    def test_all_roles_explicit_and_missing_profiles_fail(self):
        self.envelope["config"]["data_stores"] = [{"path": "sessions"}]
        for section, key in [('llm', 'profile'), ('search', 'profile'), ('embedding', 'provider')]:
            for value in (None, '', 'missing'):
                envelope = copy.deepcopy(self.envelope)
                if value is None:
                    envelope['config'][section].pop(key)
                else:
                    envelope['config'][section][key] = value
                with self.subTest(section=section, value=value), self.assertRaises(ValueError):
                    self.load(envelope)
        self.envelope['config']['llm']['fallback_profile'] = 'missing'
        with self.assertRaises(ValueError):
            self.load()

    def test_metadata_validation_never_constructs_clients_models_or_stores(self):
        self.envelope['config']['llm']['fallback_profile'] = 'fallback'
        with (patch('tess_memory.llm.AgentCoreBackend', side_effect=AssertionError('client')),
             patch('tess_memory.embed.make_embedder', side_effect=AssertionError('embedder')),
             patch('tess_memory.store.MemoryStore', side_effect=AssertionError('store'))):
            cfg = self.load()
        self.assertEqual(cfg.agent_core_fallback_profile, 'fallback')
        self.assertFalse(list(self.home.iterdir()))

    def test_explicit_manifest_environment_selection(self):
        self.load()
        with patch.dict(os.environ, {'TESS_MEMORY_CONFIG': str(self.manifest),
                                    'TESS_MEMORY_SETTINGS': '/must-not-read'}):
            self.assertTrue(load_config().embedded)

    def test_standalone_requires_explicit_selection_and_has_no_default_profile(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            load_config()
        settings = self.root / 'standalone.toml'
        settings.write_text('[llm]\nprofile = "chosen"\n')
        self.assertEqual(load_config(settings).agent_core_profile, 'chosen')
        self.assertEqual(Config().agent_core_profile, '')
        self.assertEqual(Config().data_stores, [])

    def test_local_model_downloads_explicit_only(self):
        constructor = Mock()
        fake = types.ModuleType('sentence_transformers')
        fake.SentenceTransformer = constructor
        with patch.dict(sys.modules, {'sentence_transformers': fake}):
            cfg = self.load()
            embedder = make_embedder(cfg)
            constructor.assert_not_called()
            embedder._load()
            constructor.assert_called_once_with('scripted-local', local_files_only=True)
            constructor.reset_mock()
            self.envelope['config']['embedding']['allow_download'] = True
            make_embedder(self.load())._load()
            constructor.assert_called_once_with('scripted-local', local_files_only=False)

    def test_backend_metadata_and_registry_change_guard(self):
        cfg = self.load()
        backend = AgentCoreBackend(cfg, probe=False)
        backend.verify_profile()
        self.profiles['distill']['defaultModel'] = 'changed-model'
        self.registry.write_text(yaml.safe_dump({'profiles': self.profiles}))
        with self.assertRaises(RuntimeError):
            backend.verify_profile()

    def test_disabled_roles_need_no_profile_or_unrelated_credentials(self):
        self.envelope['config']['llm'] = {}
        self.envelope['config']['search'] = {'agent': False}
        self.envelope['provenance_ledger'] = str(self.root / 'ledger.db')
        cfg = self.load()
        validate_config(cfg)
        self.assertFalse(cfg.personal.enabled)
        self.assertIsNone(cfg.personal.native_creation_ledger)
        self.assertEqual(cfg.agent_core_profile, '')
        self.assertEqual(cfg.search_profile, '')
        self.assertFalse(list(self.home.iterdir()))

    def test_credentials_validate_only_enabled_roles_without_client_or_network(self):
        cfg = self.load()
        with patch.dict(os.environ, {}, clear=True):
            validate_config(cfg)  # unused distill credential is not needed
            with self.assertRaisesRegex(ValueError, 'Missing credential'):
                validate_config(cfg, roles={'distillation'})
            validate_config(cfg, validate_credentials=False, roles={'distillation'})
        with patch.dict(os.environ, {'SCRIPTED_MEMORY_KEY': 'synthetic-not-real'}):
            validate_config(cfg, roles={'distillation'})
        self.assertFalse(list(self.home.iterdir()))

    def test_explicit_local_model_path_resolves_under_runtime_home(self):
        self.envelope['config']['embedding']['local_model'] = './models/cached'
        with self.assertRaisesRegex(ValueError, 'missing or inaccessible'):
            self.load()
        model = self.home / 'models/cached'
        model.mkdir(parents=True)
        cfg = self.load()
        self.assertEqual(cfg.embed_model, str(model))
        metadata = model / 'config_sentence_transformers.json'
        metadata.write_text(json.dumps({'sentence_embedding_dimension': 384}))
        self.envelope['config']['embedding']['dims'] = 128
        with self.assertRaisesRegex(ValueError, 'do not match'):
            self.load()
        self.envelope['config']['embedding']['dims'] = 384
        self.assertEqual(self.load().embed_dims, 384)

    def test_explicit_codex_account_pin_does_not_require_unrelated_env_credential(self):
        self.profiles['distill']['apiMode'] = 'codex'
        self.profiles['distill']['auth'] = {'kind': 'openai_api_key_or_codex', 'envVar': 'UNRELATED_ACCOUNT'}
        self.registry.write_text(yaml.safe_dump({'profiles': self.profiles}))
        auth = self.home / 'auth.json'
        auth.write_text(json.dumps({'tokens': {'access_token': 'synthetic-account-token'}}))
        self.envelope['config']['llm']['codex_auth_path'] = 'auth.json'
        cfg = self.load()
        with patch.dict(os.environ, {}, clear=True):
            validate_config(cfg, roles={'distillation'})
        self.assertEqual(cfg.agent_core_codex_auth_path, auth)
        auth.write_text('{}')
        with patch.dict(os.environ, {'UNRELATED_ACCOUNT': 'must-not-fallback'}):
            with self.assertRaises(ValueError):
                validate_config(cfg, roles={'distillation'})

    def test_embedding_space_mismatch_preserves_vectors_and_source_cursor(self):
        from tess_memory.ingest import ingest
        from tess_memory.store import MemoryStore
        store = MemoryStore(self.root / 'memory.db')
        self.addCleanup(store.close)
        store.assert_embedder('existing-model:2d')
        store.record_source(path='/explicit/session.jsonl', session_id='session', project='project',
                            source_type='agent-core', signature='stable', status='indexed',
                            state={'cursor': 42, 'offset': 1234}, now=1, indexed=True)
        store.db.commit()
        before = list(store.db.iterdump())
        embedder = Mock(descriptor='incompatible-model:3d')
        embedder.embed_passages.side_effect = AssertionError('no reembedding')
        embedder.embed_query.side_effect = AssertionError('no query embedding')
        for incremental in (False, True):
            cfg = Config(data_dir=self.root, incremental_ingestion=incremental)
            with self.subTest(incremental=incremental), self.assertRaisesRegex(RuntimeError, 'store was embedded with'):
                ingest(cfg, store, embedder, None)
            self.assertEqual(list(store.db.iterdump()), before)
        embedder.embed_passages.assert_not_called()
        embedder.embed_query.assert_not_called()

    def test_standalone_paths_resolve_from_selected_file_not_neutral_cwd(self):
        from contextlib import chdir
        directory = self.root / 'selected-config'
        directory.mkdir()
        settings = directory / 'memory.toml'
        absolute_source = '/explicit/archive/sessions'
        settings.write_text("""[storage]
data_dir = "state"
[llm]
settings = "providers.yaml"
codex_auth_path = "credentials/auth.json"
[embedding]
provider = "local"
local_model = "./models/cached"
[personal]
source_mode = "native"
source_paths = ["sessions", "/explicit/archive/sessions"]
native_creation_ledger = "state/ledger.db"
[[data_stores]]
path = "sessions"
skip_copies_of = ["copies", "/explicit/copies"]
[[data_stores]]
path = "/explicit/archive/sessions"
""")
        neutral = self.root / 'neutral-cwd'
        neutral.mkdir()
        with patch.dict(os.environ, {}, clear=True), chdir(neutral):
            cfg = load_config(settings)
        self.assertEqual(cfg.data_dir, directory / 'state')
        self.assertEqual(cfg.agent_core_settings, directory / 'providers.yaml')
        self.assertEqual(cfg.agent_core_codex_auth_path, directory / 'credentials/auth.json')
        self.assertEqual(cfg.embed_model, str(directory / 'models/cached'))
        self.assertEqual(cfg.data_stores[0].path, directory / 'sessions')
        self.assertEqual(cfg.data_stores[0].skip_copies_of, [directory / 'copies', Path('/explicit/copies')])
        self.assertEqual(str(cfg.data_stores[1].path), absolute_source)
        self.assertEqual(cfg.personal.source_paths, (str(directory / 'sessions'), absolute_source))
        self.assertEqual(cfg.personal.native_creation_ledger, str(directory / 'state/ledger.db'))
        overrides = {'TESS_MEMORY_SETTINGS': str(settings), 'TESS_MEMORY_DATA': 'overridden-state',
                     'TESS_MEMORY_SOURCE': 'overridden-sessions', 'TESS_MEMORY_AC_SETTINGS': 'override-providers.yaml'}
        with patch.dict(os.environ, overrides, clear=True), chdir(neutral):
            overridden = load_config()
        self.assertEqual(overridden.data_dir, directory / 'overridden-state')
        self.assertEqual(overridden.data_stores[0].path, directory / 'overridden-sessions')
        self.assertEqual(overridden.agent_core_settings, directory / 'override-providers.yaml')
        self.assertFalse(list(neutral.iterdir()))

    def test_standalone_checkout_overrides_warn_without_echoing_paths(self):
        settings = self.root / 'legacy.toml'
        settings.write_text('[llm]\nagent_core_path="/private-legacy-checkout"\n')
        with patch.dict(os.environ, {'AGENT_CORE_PATH': '/private-environment-checkout'}), \
             self.assertWarns(FutureWarning) as caught:
            load_config(settings)
        self.assertNotIn('/private', str(caught.warning))

    def test_remote_embedding_failure_never_falls_back_to_local(self):
        self.envelope['config']['embedding']['provider'] = 'distill'
        cfg = self.load()
        with patch.dict(os.environ, {}, clear=True), patch('tess_memory.embed.LocalEmbedder') as local:
            with self.assertRaises(ValueError):
                make_embedder(cfg)
            local.assert_not_called()

    def test_native_source_mode_requires_explicit_provenance_root(self):
        self.envelope['config']['personal'] = {'source_mode': 'native', 'source_paths': ['sessions']}
        self.envelope['provenance_ledger'] = str(self.root / 'ledger.db')
        with self.assertRaisesRegex(ValueError, 'provenance_source_root'):
            self.load()
        self.envelope['provenance_source_root'] = str(self.home / 'sessions')
        self.assertEqual(self.load().provenance_source_root, self.home / 'sessions')

    def test_store_overlay_is_explicit_and_cannot_replace_runtime_authority(self):
        self.load()
        store = self.home / 'store.toml'
        store.write_text('[storage]\ndata_dir = "stores/one"\n[[data_stores]]\npath = "sessions"\ntype = "agent-core"\n')
        cfg = load_manifest(self.manifest, store_settings='store.toml')
        self.assertEqual(cfg.data_dir, self.home / 'stores/one')
        self.assertEqual(cfg.data_stores[0].path, self.home / 'sessions')
        self.assertEqual(cfg.agent_core_profile, 'distill')
        self.assertEqual(cfg.provider_registry, self.registry)
        self.assertFalse(cfg.data_dir.exists())
        for invalid in ('[llm]\nprofile="other"', '[personal]\nenabled=true',
                        '[storage]\nother="invalid"', '[[data_stores]]\nname="missing-path"'):
            store.write_text(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_manifest(self.manifest, store_settings=store)

    def test_native_ledger_is_explicit_readonly_and_not_source_sibling(self):
        from dataclasses import replace
        from tess_memory.autobiography_config import PersonalConfig
        from tess_memory.native_creation import native_creation_proof
        ledger = self.root / 'independent-ledger.db'
        with sqlite3.connect(ledger) as db:
            db.execute('CREATE TABLE sessions(id TEXT, engine_session_id TEXT, created_at REAL, deleted_at TEXT)')
            db.execute('INSERT INTO sessions VALUES (?, ?, ?, ?)', ('host', 'session', 11, None))
        source = self.home / 'archive'
        source.mkdir()
        current = {'path': str(source / 'session.json'),
                   'creation': {'id': 'session', 'created_at': 12, 'derived': False},
                   'source_origin': {'kind': 'conversation'}, 'pristine_stream': True, 'first_event_at': 12}
        cfg = Config(provenance_source_root=source, personal=PersonalConfig(source_mode='native', native_creation_ledger=str(ledger)))
        before = ledger.read_bytes()
        proof = native_creation_proof(cfg, DataStore('source', source), current, 10, 15, float)
        self.assertEqual(proof['host_id'], 'host')
        copied = self.home / 'copied'
        copied.mkdir()
        copied_current = {**current, 'path': str(copied / 'session.json')}
        self.assertIsNone(native_creation_proof(cfg, DataStore('copy', copied), copied_current, 10, 15, float))
        legacy_cfg = replace(cfg, provenance_source_root=None)
        self.assertIsNone(native_creation_proof(legacy_cfg, DataStore('source', source), current, 10, 15, float))
        embedded_cfg = replace(cfg, embedded=True, provenance_source_root=None)
        self.assertIsNone(native_creation_proof(embedded_cfg, DataStore('source', source), current, 10, 15, float))
        self.assertEqual(before, ledger.read_bytes())
        absent = self.root / 'absent.db'
        cfg = replace(cfg, personal=replace(cfg.personal, native_creation_ledger=str(absent)))
        self.assertIsNone(native_creation_proof(cfg, DataStore('source', source), current, 10, 15, float))
        self.assertFalse(absent.exists())
        cfg = replace(cfg, personal=replace(cfg.personal, native_creation_ledger=None))
        self.assertIsNone(native_creation_proof(cfg, DataStore('source', source), current, 10, 15, float))

    def test_imports_are_inert_in_fresh_empty_home(self):
        script = '''
import os
from pathlib import Path
import tess_memory.config as config
import tess_memory.server as server
import tess_memory.cli
assert config.CONFIG._value is None
assert server._store is None and server._embedder is None
assert not list(Path(os.environ['HOME']).iterdir())
'''
        empty = self.root / 'empty'
        empty.mkdir()
        env = dict(os.environ, HOME=str(empty), TESS_MEMORY_CONFIG='/must-not-open.json')
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
