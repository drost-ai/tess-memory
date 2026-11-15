"""Synthetic scheduler manifest handoff; no credentials, providers or live sources."""
import json
import os
from unittest.mock import Mock

import pytest

from tess_memory import scheduler
from tess_memory.scheduler_config import ScheduledStore, ServiceConfig


@pytest.fixture
def embedded(tmp_path, monkeypatch):
    registry = tmp_path / 'providers.json'
    registry.write_text(json.dumps({'providerProfiles': {'synthetic': {
        'kind': 'openai', 'baseUrl': 'https://synthetic.invalid/v1',
        'defaultModel': 'synthetic', 'auth': {'kind': 'none'}}}}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'mode': 'embedded',
        'runtime_home': str(tmp_path), 'provider_registry': str(registry),
        'config': {'data_stores': [{'name': 'canonical', 'path': 'canonical-source'}],
            'storage': {'data_dir': 'canonical-destination'},
            'llm': {'profile': 'synthetic', 'fallback_profile': ''},
            'search': {'profile': 'synthetic'}, 'embedding': {'provider': 'local'}}}))
    settings = tmp_path / 'store.toml'
    settings.write_text('[[data_stores]]\nname="selected"\npath="selected-source"\n'
                        '[storage]\ndata_dir="selected-destination"\n')
    monkeypatch.setenv('TESS_MEMORY_CONFIG', str(manifest))
    return manifest, registry, settings


def test_embedded_store_overlay_preserves_canonical_providers_and_is_metadata_only(embedded, tmp_path, monkeypatch):
    manifest, registry, settings = embedded
    before = manifest.read_bytes()
    monkeypatch.setenv('TESS_MEMORY_DATA', '/poison/destination')
    monkeypatch.setenv('TESS_MEMORY_ENGINE_PROFILE', 'poison')
    monkeypatch.setenv('TESS_MEMORY_EMBED_PROVIDER', 'poison')
    monkeypatch.setattr(scheduler, 'MemoryStore', Mock(side_effect=AssertionError('no store during parse')))
    cfg = scheduler._scheduled_config(settings)
    assert cfg.embedded and cfg.runtime_home == tmp_path
    assert cfg.provider_registry == registry
    assert cfg.engine_profile == cfg.search_profile == 'synthetic'
    assert cfg.embed_provider == 'local'
    assert cfg.data_dir == tmp_path / 'selected-destination'
    assert [(store.name, store.path) for store in cfg.data_stores] == [('selected', tmp_path / 'selected-source')]
    assert not cfg.data_dir.exists()
    assert not (tmp_path / 'canonical-destination').exists()
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('extra', [
    '[llm]\nprofile="poison"\n', '[embedding]\nprovider="poison"\n',
    '[personal]\nenabled=true\n',
])
def test_embedded_store_cannot_override_provider_or_personal_policy(embedded, extra):
    _, _, settings = embedded
    settings.write_text(settings.read_text() + extra)
    with pytest.raises(ValueError, match='may override only data_stores and storage.data_dir'):
        scheduler._scheduled_config(settings)


def test_worker_environment_preserves_only_explicit_memory_manifest(embedded, monkeypatch):
    manifest, _, _ = embedded
    monkeypatch.setenv('PYTHONPATH', '/poison/source')
    monkeypatch.setenv('TESS_MEMORY_DATA', '/poison/store')
    env = scheduler._worker_environment()
    assert env['TESS_MEMORY_CONFIG'] == str(manifest)
    assert {key for key in env if key.startswith('TESS_MEMORY_')} == {'TESS_MEMORY_CONFIG'}
    assert 'PYTHONPATH' not in env
    assert os.environ['PYTHONPATH'] == '/poison/source'


def test_worker_environment_rejects_relative_manifest(monkeypatch):
    monkeypatch.setenv('TESS_MEMORY_CONFIG', 'relative.json')
    with pytest.raises(ValueError, match='absolute generated manifest'):
        scheduler._worker_environment()


def test_embedded_spawn_uses_installed_module_and_neutral_runtime_home(embedded, tmp_path, monkeypatch):
    manifest, _, settings = embedded
    config = ServiceConfig(path=tmp_path / 'service.toml', data_dir=tmp_path / 'coordinator',
                           stores=(ScheduledStore('coding', settings, enabled=True),))
    config.data_dir.mkdir()
    child = Mock()
    spawn = Mock(return_value=child)
    monkeypatch.setattr(scheduler.subprocess, 'Popen', spawn)
    monkeypatch.setenv('PYTHONPATH', '/poison/source')
    assert scheduler.spawn_worker(config, {'id': 'synthetic-run', 'instance': 'coding'}) is child
    args, kwargs = spawn.call_args
    assert args[0] == [scheduler.sys.executable, '-u', '-m', 'tess_memory.cli',
                       'scheduler-worker', '--config', str(config.path), '--run-id', 'synthetic-run']
    assert kwargs['cwd'] == tmp_path
    assert kwargs['env']['TESS_MEMORY_CONFIG'] == str(manifest)
    assert 'PYTHONPATH' not in kwargs['env']
    assert kwargs['start_new_session'] is True


def test_standalone_store_loader_still_uses_explicit_toml(tmp_path, monkeypatch):
    monkeypatch.delenv('TESS_MEMORY_CONFIG', raising=False)
    cfg = object()
    load = Mock(return_value=cfg)
    monkeypatch.setattr(scheduler, 'load_config', load)
    settings = tmp_path / 'standalone.toml'
    assert scheduler._scheduled_config(settings) is cfg
    load.assert_called_once_with(settings)
