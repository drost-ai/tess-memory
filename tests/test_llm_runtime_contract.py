"""Exercise the real public runtime through a synthetic provider send only."""
import json
from dataclasses import replace
from tess_core import SessionOptions
from tess_memory.config import Config
from tess_memory.llm import AgentCoreBackend


def test_real_runtime_completion(tmp_path):
    # The guarded runner owns disposal of its entire temporary test root.
    root = tmp_path
    registry = root / 'providers.json'
    registry.write_text(json.dumps({'providerProfiles': {'synthetic': {
        'kind': 'openai', 'apiMode': 'chat_completions', 'defaultModel': 'synthetic',
        'auth': {'kind': 'none'}}}}))
    backend = AgentCoreBackend(Config(data_dir=root / 'data', provider_registry=registry,
                                     agent_core_profile='synthetic'), probe=False)
    calls = []
    async def send(request):
        calls.append(request)
        return {'model': 'synthetic', 'stopReason': 'end_turn', 'usage': {},
                'content': [{'type': 'text', 'text': 'OK'}]}
    backend._session_options = lambda **kw: replace(SessionOptions(**kw), provider_override={'send': send})
    assert backend.complete(system='Synthetic only', prompt='Test') == 'OK'
    assert len(calls) == 1
