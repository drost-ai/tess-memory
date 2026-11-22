"""The guided setup: what it refuses, what it writes, and where keys go."""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from tess_memory.config import load_config
from tess_memory.setup import agents, app as setup_app, apply, detect, embeddings, models
from tess_memory.setup.providers import SetupError
from tess_memory.setup.vault import Vault

PORT = 4517
KEY = "sk-test-0123456789abcdef"


class FakeKeychain:
    def __init__(self):
        self.items = {}

    def set_password(self, service, name, value):
        self.items[(service, name)] = value

    def get_password(self, service, name):
        return self.items.get((service, name))

    def delete_password(self, service, name):
        self.items.pop((service, name), None)


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A home folder with one Claude Code and one Codex session, and
    providers that answer without a network."""
    user = tmp_path / "user"
    (user / ".claude/projects/-work-shop").mkdir(parents=True)
    (user / ".claude/projects/-work-shop/a.jsonl").write_text("{}\n")
    (user / ".codex/sessions/2026/11/01").mkdir(parents=True)
    (user / ".codex/sessions/2026/11/01/rollout-1.jsonl").write_text("{}\n")

    async def listed(_record):
        return {"state": "available", "code": "models_listed", "models": ["large", "small"]}

    monkeypatch.setattr(models, "list_models", listed)
    monkeypatch.setattr(models, "_one_request", lambda *_: (True, "ok"))
    monkeypatch.setattr(Vault, "platform_backend", staticmethod(lambda: "macos_keychain"))
    for name in ("ANTHROPIC_API_KEY", "TESS_MEMORY_EMBEDDING_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path, user


def session(machine, *, keychain: FakeKeychain | None):
    root, user = machine
    home = root / "store"
    vault = Vault(home, keychain=keychain is not None, backend=keychain)
    setup = setup_app.Setup(home, user_home=user, vault=vault)
    client = TestClient(setup_app.create_app(setup, port=PORT), base_url=f"http://127.0.0.1:{PORT}")
    client.headers.update({"X-Setup-Token": setup.token, "Origin": f"http://127.0.0.1:{PORT}"})
    return setup, client


def choose_sources(client):
    found = client.get("/api/state").json()["sources"]
    reply = client.post("/api/sources", json={"sources": [{"type": s["type"], "path": s["path"], "enabled": True} for s in found]})
    assert reply.status_code == 200
    return reply.json()


async def fake_embedding(_url, _model, _key, dimensions):
    return 200, {"data": [{"embedding": [0.1] * 768}]}


def test_only_this_page_on_this_machine_may_drive_setup(machine):
    setup, client = session(machine, keychain=None)
    assert client.get("/api/state", headers={"X-Setup-Token": "wrong"}).status_code == 401
    assert client.get("/api/state", headers={"Host": "attacker.example"}).json() == {"error": "wrong_host"}
    assert client.get("/", headers={"Host": "attacker.example"}).status_code == 421
    assert client.post("/api/sources", json={}, headers={"Origin": "http://attacker.example"}).status_code == 403
    client.headers.pop("Origin")
    assert client.post("/api/quit", json={}).status_code == 403  # a change with no origin at all
    page = client.get("/")
    assert page.status_code == 200 and "default-src 'self'" in page.headers["content-security-policy"]
    assert setup.token not in page.text


def test_sources_are_found_by_counting_files_only(machine):
    _, user = machine
    found = {s["name"]: s for s in detect.detect_sources(user)}
    assert {name: s["sessions"] for name, s in found.items()} == {"claude-code": 1, "codex": 1}
    assert found["claude-code"]["display_path"] == "~/.claude/projects"
    with pytest.raises(ValueError, match="folder_not_found"):
        detect.describe_folder("codex", str(user / "missing"))


def test_a_model_is_required_before_settings_can_be_written(machine):
    setup, client = session(machine, keychain=None)
    assert choose_sources(client)["missing"] == ["model", "embedding"]
    refused = client.post("/api/apply", json={})
    assert refused.status_code == 422 and refused.json()["error"] == "incomplete"
    assert not setup.settings.exists()


def test_keys_go_to_the_keychain_and_no_file_holds_them(machine):
    keychain = FakeKeychain()
    setup, client = session(machine, keychain=keychain)
    choose_sources(client)
    connected = client.post("/api/model/key", json={"kind": "anthropic", "api_key": KEY}).json()
    assert connected["secrets"] == "keychain" and "auth" not in connected["model"]["connection"]
    assert client.post("/api/model/choose", json={"distill_model": "large", "search_model": "small"}).json()["model"]["connection"]["verified"]
    monkey = embeddings.connect_hosted

    async def hosted(vault, providers, **kwargs):
        return await monkey(vault, providers, transport=fake_embedding, **kwargs)

    embeddings.connect_hosted = hosted
    try:
        choice = client.post("/api/embedding/hosted", json={"preset": "google", "api_key": "g-" + KEY}).json()["embedding"]["choice"]
    finally:
        embeddings.connect_hosted = monkey
    assert choice["dims"] == 768 and choice["send_dimensions"] is True
    written = client.post("/api/apply", json={}).json()["result"]
    assert Path(written["providers"]).name == "providers.yaml"
    for path in setup.home.iterdir():
        if path.is_file():
            assert KEY not in path.read_text(), path.name
            assert path.stat().st_mode & 0o077 == 0, path.name
    assert sorted(json.loads(value)["api_key"] for value in keychain.items.values()) == sorted([KEY, "g-" + KEY])
    cfg = load_config(setup.settings)
    assert (cfg.engine_profile, cfg.engine_model, cfg.search_model) == ("anthropic", "large", "small")
    assert (cfg.embed_provider, cfg.embed_dims, cfg.embed_query_prefix, cfg.embed_send_dimensions) == ("embeddings-gemini-embedding-001", 768, "", True)
    assert [(d.name, d.type) for d in cfg.data_stores] == [("claude-code", "claude-code"), ("codex", "codex")]


def test_a_google_gemini_key_becomes_a_google_profile_tess_core_accepts(machine):
    keychain = FakeKeychain()
    setup, client = session(machine, keychain=keychain)
    choose_sources(client)
    connection = client.post("/api/model/key", json={"kind": "google", "api_key": KEY}).json()["model"]["connection"]
    assert (connection["kind"], connection["profile"]) == ("google", "google")
    assert client.post("/api/model/choose", json={"distill_model": "large", "search_model": "small"}).json()["model"]["connection"]["verified"]
    from tess_core.config import ProviderRegistry
    registry = ProviderRegistry(setup.home / "providers.yaml")
    record = registry.profiles["google"]
    assert (record["kind"], record["defaultModel"], record["auth"]["kind"]) == ("google", "large", "google_api_key")
    # The key reaches the engine in Google's own field, still as a keychain reference.
    resolved = registry.resolve_auth("google")
    assert set(resolved) == {"googleApiKey", "customHeaders"} and KEY not in json.dumps(resolved)
    assert KEY not in (setup.home / "providers.yaml").read_text()
    assert client.post("/api/model/key", json={"kind": "mistral", "api_key": KEY}).json()["error"] == "unknown_kind"


def test_without_a_keychain_keys_live_in_one_private_file(machine):
    setup, client = session(machine, keychain=None)
    choose_sources(client)
    state = client.post("/api/model/key", json={"kind": "anthropic", "api_key": KEY}).json()
    assert state["secrets"] == "file" and state["model"]["chatgpt"] == {**state["model"]["chatgpt"], "available": False, "reason": "needs_keychain"}
    client.post("/api/model/choose", json={"distill_model": "large", "search_model": "large"})
    registry = json.loads((setup.home / "providers.json").read_text())
    assert registry["env"] == {"ANTHROPIC_API_KEY": KEY}
    assert registry["providerProfiles"]["anthropic"]["auth"] == {"kind": "anthropic_api_key", "envVar": "ANTHROPIC_API_KEY"}
    assert (setup.home / "providers.json").stat().st_mode & 0o077 == 0


def test_a_key_the_provider_refuses_is_not_kept(machine, monkeypatch):
    async def refused(_record):
        return {"state": "failed", "code": "authentication_denied", "models": []}

    monkeypatch.setattr(models, "list_models", refused)
    keychain = FakeKeychain()
    _, client = session(machine, keychain=keychain)
    reply = client.post("/api/model/key", json={"kind": "openai", "api_key": KEY})
    assert reply.status_code == 422 and reply.json()["error"] == "authentication_denied"
    assert keychain.items == {} and KEY not in reply.text


def test_models_that_fail_a_real_request_do_not_pass(machine, monkeypatch):
    monkeypatch.setattr(models, "_one_request", lambda *_: (False, "RuntimeError: model not available"))
    setup, client = session(machine, keychain=None)
    choose_sources(client)
    client.post("/api/model/key", json={"kind": "compatible", "base_url": "http://localhost:11434/v1"})
    state = client.post("/api/model/choose", json={"distill_model": "local-model", "search_model": "local-model"}).json()
    assert state["model"]["connection"]["verified"] is False and "model" in state["missing"]
    assert json.loads((setup.home / "providers.json").read_text())["providerProfiles"]["endpoint"]["auth"] == {"kind": "none"}


def test_an_existing_settings_file_is_kept_as_a_dated_copy(machine):
    setup, client = session(machine, keychain=None)
    setup.home.mkdir(parents=True)
    setup.settings.write_text("# mine\n")
    assert client.get("/api/state").json()["existing_settings"] is True
    choose_sources(client)
    client.post("/api/model/key", json={"kind": "anthropic", "api_key": KEY})
    client.post("/api/model/choose", json={"distill_model": "large", "search_model": "small"})
    setup.vault.draft["embedding"] = {"mode": "local", "name": "bge-small-en-v1.5",
                                      "settings": {"local_model": "./models/bge-small-en-v1.5", "dims": 384, "query_prefix": "q: ", "passage_prefix": ""}}
    written = client.post("/api/apply", json={}).json()["result"]
    assert Path(written["backup"]).read_text() == "# mine\n"
    embedding = tomllib.loads(setup.settings.read_text())["embedding"]
    assert embedding == {"provider": "local", "local_model": "./models/bge-small-en-v1.5", "dims": 384, "query_prefix": "q: ", "passage_prefix": ""}


def test_a_hosted_embedding_model_that_rejects_a_size_is_remembered(machine):
    async def strict(_url, _model, _key, dimensions):
        return (400, {"error": "unknown field"}) if dimensions else (200, {"data": [{"embedding": [0.0] * 1024}]})

    setup, _ = session(machine, keychain=None)
    import asyncio
    choice = asyncio.run(embeddings.connect_hosted(setup.vault, setup.providers, preset="compatible",
                                                   base_url="https://api.example.com/v1/", model="embed-1", transport=strict))
    assert (choice["dims"], choice["send_dimensions"], choice["base_url"]) == (1024, False, "https://api.example.com/v1")
    with pytest.raises(SetupError, match="authentication_denied"):
        async def denied(*_):
            return 401, None
        asyncio.run(embeddings.connect_hosted(setup.vault, setup.providers, preset="google", api_key=KEY, transport=denied))


def test_agents_are_registered_through_their_own_command(machine, monkeypatch):
    _, user = machine
    settings = user / "store/settings.toml"
    sources = [{**item, "enabled": True} for item in detect.detect_sources(user)]
    (user / ".claude.json").write_text(json.dumps({"mcpServers": {"tess-memory": {"env": {"TESS_MEMORY_SETTINGS": "/elsewhere.toml"}}}}))
    monkeypatch.setattr(agents.shutil, "which", lambda name: "/usr/bin/" + name)
    found = {item["type"]: item for item in agents.targets(sources, settings)}
    assert (found["claude-code"]["registered"], found["claude-code"]["current"]) == (True, False)
    assert (found["codex"]["registered"], found["codex"]["display_path"]) == (False, "~/.codex")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"].get("CODEX_HOME"), kwargs["env"].get("CLAUDE_CONFIG_DIR")))
        return subprocess.CompletedProcess(command, 0, "", "")

    agents.register(found["claude-code"]["id"], sources, settings, run=run)
    agents.register(found["codex"]["id"], sources, settings, run=run)
    assert [call[0][:3] for call in calls] == [["claude", "mcp", "remove"], ["claude", "mcp", "add-json"], ["codex", "mcp", "add"]]
    assert json.loads(calls[1][0][-1])["env"] == {"TESS_MEMORY_SETTINGS": str(settings)}
    assert calls[1][2] is None and calls[2][1] == str(user / ".codex")
    assert f"--env=TESS_MEMORY_SETTINGS={settings}" in calls[2][0]
    with pytest.raises(SetupError, match="unknown_agent"):
        agents.register("codex:/nowhere", sources, settings, run=run)


def test_settings_cannot_be_rendered_from_half_a_setup():
    assert apply.missing({}) == ["sources", "model", "embedding"]
    with pytest.raises(SetupError, match="incomplete"):
        apply.render(Path("/store"), Path("/store/providers.yaml"), {"sources": [{"name": "a", "type": "codex", "path": "/a"}]})


def test_the_chatgpt_sign_in_link_carries_a_host_id_openai_accepts(machine):
    import asyncio
    import uuid
    from urllib.parse import parse_qs, urlsplit

    setup, _ = session(machine, keychain=FakeKeychain())
    setup.vault.meta["chatgpt"]["host_id"] = "an-id-from-before"  # not in the form OpenAI takes: replaced

    async def link():
        started = await setup.chatgpt.start()
        await setup.chatgpt.cancel()
        return started["url"]

    query = parse_qs(urlsplit(asyncio.run(link())).query)
    host_id = query["ext_agent_host_id"][0]
    assert host_id.startswith("urn:uuid:") and str(uuid.UUID(host_id[9:])) == host_id[9:]
    assert host_id == setup.vault.meta["chatgpt"]["host_id"]
    assert (query["client_id"], query["agent_name_hint"]) == (["dynamic_agent_client"], ["Tess Memory"])
    assert query["redirect_uri"][0].startswith("http://127.0.0.1:")


def test_an_account_that_is_still_signed_in_is_used_without_the_browser(machine):
    import asyncio

    setup, _ = session(machine, keychain=FakeKeychain())
    ref = {"kind": "runtime_secret", "version": 1, "id": "0b0e4c1e-6d1c-4b55-9d0e-3f0f8f5b9a11", "purpose": "provider_oauth_token",
           "backend": "macos_keychain", "installation_id": setup.vault.meta["installation_id"], "account_id": setup.vault.meta["account_id"]}
    setup.vault.meta["chatgpt"].update({"host_id": models.new_host_id(), "credential_ref": ref, "plan_use": True, "email": "sam@example.com"})
    assert asyncio.run(setup.chatgpt.start()) == {"url": None, "expires_in": 0}
    connection = setup.vault.draft["model"]
    assert (connection["kind"], connection["email"], connection["models"]) == ("chatgpt", "sam@example.com", ["large", "small"])
    assert setup.chatgpt.outcome == {"state": "signed_in", "email": "sam@example.com"}


def test_connecting_also_gives_the_agent_its_instructions(machine, monkeypatch):
    from tess_memory.setup import guidance

    _, user = machine
    settings = user / "store/settings.toml"
    sources = [{**item, "enabled": True} for item in detect.detect_sources(user)]
    monkeypatch.setattr(agents.shutil, "which", lambda name: "/usr/bin/" + name)
    run = lambda command, **_: subprocess.CompletedProcess(command, 0, "", "")  # noqa: E731
    found = {item["type"]: item for item in agents.targets(sources, settings)}
    assert found["claude-code"]["guidance"] is False and found["codex"]["guidance_file"] == "AGENTS.md"

    (user / ".claude/CLAUDE.md").write_text("# Mine\n\nKeep answers short.")
    (user / ".claude/CLAUDE.md").chmod(0o600)
    claude = agents.register(found["claude-code"]["id"], sources, settings, run=run)
    written = (user / ".claude/CLAUDE.md").read_text()
    assert claude["guidance_outcome"] == "added" and claude["guidance"] is True
    assert written == "# Mine\n\nKeep answers short.\n\n" + guidance.text("claude-code")
    assert (user / ".claude/CLAUDE.md").stat().st_mode & 0o777 == 0o600
    copies = list((user / ".claude").glob("CLAUDE.md.before-tess-memory-*"))
    assert [copy.read_text() for copy in copies] == ["# Mine\n\nKeep answers short."]

    codex = agents.register(found["codex"]["id"], sources, settings, run=run)
    assert codex["guidance_outcome"] == "created"
    assert (user / ".codex/AGENTS.md").read_text() == guidance.text("codex")
    assert "for shared Codex memory and durable" in guidance.text("codex")
    assert guidance.text("claude-code").startswith("## Memory and planning (tess-memory MCP)\n\nUse the `tess-memory` MCP server for shared Claude memory")

    # A second connect, or a file that already carries the section, changes nothing.
    assert guidance.install("claude-code", user / ".claude") == "present"
    assert (user / ".claude/CLAUDE.md").read_text() == written and len(list((user / ".claude").glob("CLAUDE.md.before-*"))) == 1


def test_an_agent_already_connected_only_gets_the_missing_instructions(machine, monkeypatch):
    _, user = machine
    settings = user / "store/settings.toml"
    sources = [{**item, "enabled": True} for item in detect.detect_sources(user)]
    (user / ".claude.json").write_text(json.dumps({"mcpServers": {"tess-memory": {"env": {"TESS_MEMORY_SETTINGS": str(settings)}}}}))
    monkeypatch.setattr(agents.shutil, "which", lambda name: None)  # no command needed for this
    target = next(item for item in agents.targets(sources, settings) if item["type"] == "claude-code")
    assert (target["current"], target["guidance"]) == (True, False)

    def run(*_args, **_kwargs):
        raise AssertionError("the agent's command must not run")

    assert agents.register(target["id"], sources, settings, run=run)["guidance_outcome"] == "created"


def _store_settings(home: Path, user: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    settings = home / "settings.toml"
    settings.write_text(f'[[data_stores]]\nname = "codex"\ntype = "codex"\npath = "{user / ".codex"}"\n\n[storage]\ndata_dir = "{home}"\n')
    return settings


def test_the_scheduler_setup_writes_runs_by_the_documented_rules(machine, monkeypatch):
    from tess_memory.scheduler_config import load_service
    from tess_memory.setup import keepcurrent

    root, user = machine
    home = root / "store"
    settings = _store_settings(home, user)
    (home / "scheduler.toml").write_text(keepcurrent.config_text(home, settings, timezone="America/Los_Angeles"))
    service = load_service(home / "scheduler.toml")
    store = service.stores[0]
    assert (store.poll_seconds, store.profile_seconds, store.digest_seconds) == (300, 3600, 86400)
    assert (store.recovery_hour, store.recovery_timezone, store.model_calls_per_run, store.enabled) == (22, "America/Los_Angeles", 128, True)
    assert service.backlog_seconds == 15 and store.settings == settings.resolve()
    monkeypatch.setenv("TZ", "Europe/Lisbon")
    assert keepcurrent.local_timezone() == "Europe/Lisbon"


def test_on_linux_the_scheduler_becomes_a_user_service_and_can_be_removed(machine, monkeypatch):
    from tess_memory.setup import keepcurrent

    root, user = machine
    home = root / "store"
    settings = _store_settings(home, user)
    monkeypatch.setattr(keepcurrent, "_platform", lambda: "systemd")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    commands = []

    def run(command, **_):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    assert keepcurrent.status(home)["installed"] is False
    installed = keepcurrent.install(home, settings, run=run)
    unit = root / "config/systemd/user/tess-memory-scheduler.service"
    assert (installed["installed"], installed["ours"]) == (True, True)
    assert f"--config {home / 'scheduler.toml'} serve" in unit.read_text() and "Restart=always" in unit.read_text()
    assert commands == [["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", "tess-memory-scheduler.service"]]

    removed = keepcurrent.remove(home, run=run)
    assert removed["installed"] is False and not unit.exists()
    assert (home / "scheduler.toml").exists() and settings.exists()  # nothing of the store is removed


def test_a_scheduler_that_setup_did_not_install_is_never_replaced_or_removed(machine, monkeypatch):
    from tess_memory.setup import keepcurrent

    root, user = machine
    home = root / "store"
    settings = _store_settings(home, user)
    monkeypatch.setattr(keepcurrent, "_platform", lambda: "systemd")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    unit = root / "config/systemd/user/tess-memory-scheduler.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nExecStart=python -m tess_memory.cli scheduler --config /someone/else.toml serve\n")
    assert (keepcurrent.status(home)["installed"], keepcurrent.status(home)["ours"]) == (True, False)
    for action in (lambda: keepcurrent.install(home, settings), lambda: keepcurrent.remove(home)):
        with pytest.raises(SetupError, match="scheduler_already_installed"):
            action()
    assert "/someone/else.toml" in unit.read_text()


def test_the_first_batch_hands_the_rest_to_the_background_service(machine, monkeypatch):
    from tess_memory.setup import keepcurrent

    setup, client = session(machine, keychain=None)
    installs = []
    monkeypatch.setattr(keepcurrent, "status", lambda home: {"supported": True, "kind": "launchd", "installed": bool(installs), "ours": bool(installs),
                                                             "running": False, "command": "", "timezone": "UTC"})
    monkeypatch.setattr(keepcurrent, "install", lambda home, settings: installs.append((home, settings)))
    assert client.get("/api/state").json()["scheduler"]["wanted"] is True  # on unless the person says otherwise

    setup.vault.draft["keep_current"] = False
    setup.ingest.after()
    assert installs == []
    setup.vault.draft["keep_current"] = True
    setup.ingest.after()
    setup.ingest.after()  # already installed: not installed twice
    assert installs == [(setup.home, setup.settings)]

    def refuse(home, settings):
        raise SetupError("scheduler_install_failed", "launchctl said no")

    installs.clear()
    monkeypatch.setattr(keepcurrent, "install", refuse)
    setup.ingest.after()
    assert client.get("/api/state").json()["scheduler"]["error"] == {"code": "scheduler_install_failed", "detail": "launchctl said no"}


def test_a_memory_server_registered_under_another_name_is_recognised(machine, monkeypatch):
    _, user = machine
    settings = user / "store/settings.toml"
    sources = [{**item, "enabled": True} for item in detect.detect_sources(user)]
    (user / ".codex/config.toml").write_text(
        '[mcp_servers.docs]\ncommand = "npx"\nargs = ["docs-server", "serve"]\n\n'
        '[mcp_servers.tess-memory-codex]\ncommand = "/usr/local/bin/uv"\n'
        'args = ["run", "--directory", "/src/tess-memory", "tess-memory", "codex-serve", "--settings", "/elsewhere/settings.toml"]\n')
    monkeypatch.setattr(agents.shutil, "which", lambda name: "/usr/bin/" + name)
    codex = next(item for item in agents.targets(sources, settings) if item["type"] == "codex")
    assert (codex["registered"], codex["registered_as"], codex["current"]) == (True, "tess-memory-codex", False)
    calls = []

    def run(command, **_):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    agents.register(codex["id"], sources, settings, run=run)
    assert calls[0] == ["codex", "mcp", "remove", "tess-memory-codex"] and calls[1][:4] == ["codex", "mcp", "add", "tess-memory"]
    assert agents._settings_of({"args": ["serve", "--settings", "~/x.toml"]}) == str(Path("~/x.toml").expanduser())
    assert agents._is_memory_server({"command": "python", "args": ["-m", "tess_memory.cli", "serve"]})
    assert not agents._is_memory_server({"command": "npx", "args": ["docs-server", "serve"]})


def test_every_agent_can_be_connected_whatever_sources_were_chosen(machine):
    setup, client = session(machine, keychain=None)
    _, user = machine
    (user / ".claude-second").mkdir()
    (user / ".claude-second/settings.json").write_text("{}")  # an agent home with no sessions yet
    (user / ".codex-notes.txt").write_text("not a home")
    found = client.get("/api/state").json()["sources"]
    only_codex = [{"type": s["type"], "path": s["path"], "enabled": s["type"] == "codex"} for s in found]
    state = client.post("/api/sources", json={"sources": only_codex}).json()
    assert [s["name"] for s in state["sources"] if s["enabled"]] == ["codex"]
    assert [(t["type"], t["display_path"]) for t in state["agents"]["targets"]] == [
        ("claude-code", "~/.claude"), ("claude-code", "~/.claude-second"), ("codex", "~/.codex")]


def _built_store(setup, descriptor="embeddings:model-a:4d"):
    """A store that already holds memories and a plan, embedded with ``descriptor``."""
    import numpy as np
    from tess_memory import plan
    from tess_memory.store import MemoryRow, MemoryStore

    class Old:
        batch_size = 8

        def embed_passages(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    Old.descriptor = descriptor
    setup.home.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(setup.home / "memory.db")
    store.assert_embedder(descriptor)
    store.add_memories([MemoryRow(level="memory", kind="decision", content=f"decision {i}", project="p",
                                  vector=np.ones(4, dtype=np.float32)) for i in range(5)])
    store.commit()
    item = plan.add_item(store, Old(), project="p", kind="epic", title="An authored plan")
    store.close()
    return item["id"]


def _configure(setup, client, *, model="model-b", dims=6):
    choose_sources(client)
    client.post("/api/model/key", json={"kind": "anthropic", "api_key": KEY})
    client.post("/api/model/choose", json={"distill_model": "large", "search_model": "small"})
    setup.providers.put(embeddings.profile_name(model), {"kind": "openai", "apiMode": "chat_completions", "baseUrl": "https://embed.invalid/v1",
                                                         "defaultModel": model, "auth": {"kind": "none"}})
    setup.vault.draft["embedding"] = {"mode": "hosted", "preset": "compatible", "label": "OpenAI-compatible endpoint",
                                      "base_url": "https://embed.invalid/v1", "model": model, "dims": dims,
                                      "send_dimensions": True, "auth": {}, "profile": embeddings.profile_name(model)}
    setup.vault.save()


def test_changing_the_embedding_model_of_a_built_store_reembeds_before_any_setting_changes(machine, monkeypatch):
    import sqlite3
    import time
    import numpy as np
    from tess_memory import embed

    setup, client = session(machine, keychain=None)
    plan_id = _built_store(setup)
    setup.settings.write_text("# the settings the store is running on\n")
    _configure(setup, client)

    class New:
        descriptor, batch_size, seen = "embeddings-model-b:model-b:6d", 2, []

        def embed_passages(self, texts):
            New.seen += list(texts)
            return np.full((len(texts), 6), 0.5, dtype=np.float32)

    monkeypatch.setattr(embed, "make_embedder", lambda cfg: New())
    offered = client.post("/api/apply", json={}).json()
    assert offered["result"]["reembed"]["rows"] == 6 and offered["reembed"]["pending"]["current"] == "embeddings:model-a:4d"
    assert offered["reembed"]["pending"]["label"] == "model-b" and offered["applied"] is None
    assert setup.settings.read_text() == "# the settings the store is running on\n"  # nothing changed yet

    def finished():
        assert client.post("/api/reembed/start", json={}).status_code == 200
        for _ in range(200):
            state = client.get("/api/state").json()
            if state["reembed"]["job"]["state"] != "running":
                break
            time.sleep(0.05)
        return state

    # The settings cannot be written once the store has switched: that is said, not "nothing was switched".
    written = apply.write
    monkeypatch.setattr(apply, "write", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    state = finished()
    assert state["reembed"]["job"] == {"state": "failed", "switched": True, "detail": "OSError: disk full", "done": 6, "total": 6}
    assert state["applied"] is None and setup.settings.read_text() == "# the settings the store is running on\n"
    # Starting again embeds nothing more and writes the settings.
    monkeypatch.setattr(apply, "write", written)
    state = finished()
    assert state["reembed"]["job"]["state"] == "switched", state["reembed"]["job"]
    assert state["reembed"]["pending"] is None and state["applied"] is not None

    database = sqlite3.connect(setup.home / "memory.db")
    assert database.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()[0] == New.descriptor
    assert {row[0] for row in database.execute("SELECT length(vector)/4 FROM embeddings")} == {6}
    assert database.execute("SELECT title FROM plan_items WHERE id=?", (plan_id,)).fetchone()[0] == "An authored plan"
    assert database.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 6
    database.close()
    cfg = load_config(setup.settings)
    assert (cfg.embed_provider, cfg.embed_dims) == ("embeddings-model-b", 6)
    assert not (setup.home / "settings.next.toml").exists()
    assert len(list(setup.home.glob("settings.toml.before-setup-*"))) == 1
    assert len(New.seen) == 6


def test_a_reembed_can_be_declined_and_the_store_and_settings_stay_as_they_were(machine, monkeypatch):
    import sqlite3

    setup, client = session(machine, keychain=None)
    _built_store(setup)
    setup.settings.write_text("# running settings\n")
    _configure(setup, client)
    assert client.post("/api/apply", json={}).json()["reembed"]["pending"] is not None
    assert client.post("/api/agents/register", json={"id": "x"}).json()["error"] == "settings_not_written"
    declined = client.post("/api/reembed/cancel", json={}).json()
    assert declined["reembed"] == {"pending": None, "job": {"state": "idle"}}
    assert setup.settings.read_text() == "# running settings\n" and not (setup.home / "settings.next.toml").exists()
    database = sqlite3.connect(setup.home / "memory.db")
    assert database.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()[0] == "embeddings:model-a:4d"
    database.close()
    assert client.post("/api/reembed/start", json={}).json()["error"] == "nothing_to_reembed"


def test_the_same_model_under_another_profile_name_needs_no_reembed(machine):
    setup, client = session(machine, keychain=None)
    _built_store(setup, descriptor="a-name-from-before:model-b:6d")
    _configure(setup, client)
    written = client.post("/api/apply", json={}).json()
    assert written["reembed"]["pending"] is None and written["result"]["settings"] == str(setup.settings)


def test_two_hosted_embedding_models_keep_separate_profiles_until_the_settings_switch(machine):
    import asyncio

    setup, _ = session(machine, keychain=None)
    first = asyncio.run(embeddings.connect_hosted(setup.vault, setup.providers, preset="compatible", base_url="https://one.invalid/v1",
                                                  model="model-a", api_key=KEY + "-a", transport=fake_embedding))
    second = asyncio.run(embeddings.connect_hosted(setup.vault, setup.providers, preset="compatible", base_url="https://two.invalid/v1",
                                                   model="model-b", api_key=KEY + "-b", transport=fake_embedding))
    assert (first["profile"], second["profile"]) == ("embeddings-model-a", "embeddings-model-b")
    # The first model's profile and key are still there: a store on it keeps working meanwhile.
    assert set(setup.providers.profiles) == {"embeddings-model-a", "embeddings-model-b"}
    assert sorted(setup.vault.env().values()) == [KEY + "-a", KEY + "-b"]
    embeddings.prune(setup.vault, setup.providers)
    assert set(setup.providers.profiles) == {"embeddings-model-b"} and len(setup.vault.pending_forget) == 1


def test_only_the_embedding_change_waits_for_the_reembed(machine, monkeypatch):
    import time
    import numpy as np
    from tess_memory import embed

    setup, client = session(machine, keychain=None)
    _built_store(setup, descriptor="embeddings-model-a:model-a:4d")
    # The store runs on settings setup wrote earlier: one source, hosted model-a.
    setup.providers.put("embeddings-model-a", {"kind": "openai", "apiMode": "chat_completions", "baseUrl": "https://old.invalid/v1",
                                               "defaultModel": "model-a", "auth": {"kind": "none"}})
    setup.settings.write_text('[[data_stores]]\nname = "codex"\ntype = "codex"\npath = "/somewhere"\n\n'
                              '[embedding]\nprovider = "embeddings-model-a"\ndims = 4\nquery_prefix = ""\nsend_dimensions = true\n')
    _configure(setup, client)  # now: both sources, a provider, and hosted model-b

    class New:
        descriptor, batch_size = "embeddings-model-b:model-b:6d", 8

        def embed_passages(self, texts):
            return np.full((len(texts), 6), 0.5, dtype=np.float32)

    monkeypatch.setattr(embed, "make_embedder", lambda cfg: New())
    saved = client.post("/api/apply", json={}).json()
    assert saved["reembed"]["pending"]["held"] == "embedding" and saved["applied"] is not None
    interim = load_config(setup.settings)
    # Everything else chosen is in effect; the embedding is still the store's own.
    assert [d.name for d in interim.data_stores] == ["claude-code", "codex"] and interim.engine_profile == "anthropic"
    assert (interim.embed_provider, interim.embed_dims) == ("embeddings-model-a", 4)
    assert set(setup.providers.profiles) >= {"embeddings-model-a", "embeddings-model-b"}

    client.post("/api/reembed/start", json={})
    for _ in range(200):
        state = client.get("/api/state").json()
        if state["reembed"]["job"]["state"] != "running":
            break
        time.sleep(0.05)
    assert state["reembed"]["job"]["state"] == "switched", state["reembed"]["job"]
    final = load_config(setup.settings)
    assert (final.embed_provider, final.embed_dims) == ("embeddings-model-b", 6)
    assert "embeddings-model-a" not in setup.providers.profiles  # dropped once nothing uses it


def test_settings_that_cannot_be_rewritten_around_their_embedding_all_wait(machine):
    setup, client = session(machine, keychain=None)
    _built_store(setup)
    # Hand-written: its embedding profile lives in a provider file setup does not manage.
    # A bad carried-over value is caught before offering a re-embed that could not finish.
    setup.settings.write_text('[embedding]\nprovider = "a-profile-from-elsewhere"\ndims = 4\n\n[codex]\nbatch_size = "many"\n')
    _configure(setup, client)
    refused = client.post("/api/apply", json={})
    assert refused.status_code == 422 and refused.json() == {"error": "invalid_settings", "detail": "[codex] batch_size must be a positive integer"}
    assert client.get("/api/state").json()["reembed"]["pending"] is None and not (setup.home / "settings.next.toml").exists()

    setup.settings.write_text('[embedding]\nprovider = "a-profile-from-elsewhere"\ndims = 4\n\n[planning]\ndefault_project = "assistant"\n')
    held = client.post("/api/apply", json={}).json()
    assert held["reembed"]["pending"]["held"] == "everything" and held["applied"] is None
    assert setup.settings.read_text().startswith('[embedding]\nprovider = "a-profile-from-elsewhere"')
    assert tomllib.loads((setup.home / "settings.next.toml").read_text())["planning"] == {"default_project": "assistant"}


def test_the_background_service_is_on_by_default_from_the_moment_ingestion_starts(machine, monkeypatch):
    from tess_memory.setup import keepcurrent
    from tess_memory.setup.jobs import FirstIngest

    setup, client = session(machine, keychain=None)
    installs = []
    monkeypatch.setattr(keepcurrent, "status", lambda home: {"supported": True, "kind": "launchd", "installed": bool(installs), "ours": bool(installs),
                                                             "running": bool(installs), "command": "", "timezone": "UTC"})
    monkeypatch.setattr(keepcurrent, "install", lambda home, settings: installs.append(settings))
    monkeypatch.setattr(FirstIngest, "start", lambda self, limit: self.status.update(state="running"))
    setup.vault.meta["applied"] = {"settings": str(setup.settings)}
    assert client.get("/api/state").json()["scheduler"]["wanted"] is True
    # The page sends what its ticked-by-default box says; even without that field the default is on.
    started = client.post("/api/ingest/start", json={"limit": 25}).json()
    assert installs == [setup.settings] and started["scheduler"]["ours"] is True
    setup.ingest.after()  # the batch ending does not install it a second time
    assert len(installs) == 1


def test_unticking_the_box_is_the_only_thing_that_leaves_the_service_off(machine, monkeypatch):
    from tess_memory.setup import keepcurrent
    from tess_memory.setup.jobs import FirstIngest

    setup, client = session(machine, keychain=None)
    installs = []
    monkeypatch.setattr(keepcurrent, "status", lambda home: {"supported": True, "kind": "launchd", "installed": False, "ours": False,
                                                             "running": False, "command": "", "timezone": "UTC"})
    monkeypatch.setattr(keepcurrent, "install", lambda home, settings: installs.append(settings))
    monkeypatch.setattr(FirstIngest, "start", lambda self, limit: self.status.update(state="running"))
    setup.vault.meta["applied"] = {"settings": str(setup.settings)}
    state = client.post("/api/ingest/start", json={"limit": 25, "keep_current": False}).json()
    assert installs == [] and state["scheduler"]["wanted"] is False


def test_the_ingestion_setup_starts_outlives_setup(tmp_path):
    """Finishing or closing setup must not end the ingestion it started: the
    child writes to files in the store folder, in a session of its own."""
    import os
    import signal
    import sys
    import time
    from tess_memory.setup.jobs import FirstIngest

    script = ("import json,sys,time,os\n"
              "print('[ingest] db=somewhere', file=sys.stderr, flush=True)\n"
              "for i in (1, 2, 3):\n"
              "    time.sleep(0.15)\n"
              "    print(f'[ingest] ({i}/3) distilled: project / a session', file=sys.stderr, flush=True)\n"
              "print(json.dumps({'distilled': 3, 'errors': 0, 'session': os.getsid(0) == os.getpid()}))\n")
    job = FirstIngest(tmp_path / "store" / "settings.toml")
    job.command = [sys.executable, "-c", script]
    done = []
    job.after = lambda: done.append(True)
    job.start(25)
    process = job._process
    assert process.stdout is None and process.stderr is None      # no pipe to break when setup exits
    assert os.getsid(process.pid) != os.getsid(0)                 # not in setup's session
    for _ in range(100):
        if job.status["state"] != "running":
            break
        time.sleep(0.05)
    assert (job.status["state"], job.status["done"], job.status["total"]) == ("finished", 3, 3)
    assert job.status["stats"] == {"distilled": 3, "errors": 0, "session": True} and done == [True]
    assert job.status["lines"][-1] == "[ingest] (3/3) distilled: project / a session"
    assert "(3/3)" in (tmp_path / "store" / "setup-ingest.log").read_text()

    job.command = [sys.executable, "-c", "import time\ntime.sleep(30)"]
    job.start(None)
    job.stop()
    assert job._process.wait(timeout=5) == -signal.SIGTERM and job.status["state"] == "stopped"


def test_antigravity_is_found_and_connected_through_agy(machine, monkeypatch):
    from tess_memory.setup import guidance
    _, user = machine
    settings = user / "store/settings.toml"
    logs = user / ".gemini/antigravity-cli/brain/11111111-0000-4000-8000-000000000001/.system_generated/logs"
    logs.mkdir(parents=True)
    (logs / "transcript_full.jsonl").write_text("{}\n")
    (logs / "transcript.jsonl").write_text("{}\n")  # the same conversation, shortened: not a second one
    (user / ".gemini/antigravity-backup").mkdir()  # an application folder with no conversations
    found = {s["name"]: s for s in detect.detect_sources(user)}
    assert (found["antigravity-cli"]["sessions"], found["antigravity-cli"]["label"]) == (1, "Antigravity")
    assert (found["antigravity-cli"]["display_path"], found["antigravity-cli"]["display_home"]) == ("~/.gemini/antigravity-cli", "~/.gemini")
    assert "antigravity-backup" not in found
    by_hand = detect.describe_folder("antigravity", str(user / ".gemini/antigravity-cli"))
    assert (by_hand["sessions"], by_hand["home"]) == (1, str(user / ".gemini"))
    assert "type = \"antigravity\"" in apply.render(user / "store", user / "store/providers.json", {
        "sources": [{**found["antigravity-cli"], "enabled": True}], "model": {"profile": "p", "distill_model": "m", "search_model": "m", "verified": True},
        "embedding": {"mode": "hosted", "dims": 4, "send_dimensions": False}})

    (user / ".gemini/config").mkdir()
    (user / ".gemini/config/mcp_config.json").write_text(json.dumps({"mcpServers": {"colab": {"command": "uvx"}}}))
    (user / ".gemini/GEMINI.md").write_text("")
    monkeypatch.setattr(agents.shutil, "which", lambda name: "/usr/bin/" + name)
    homes = detect.agent_homes(user)
    assert [(item["type"], item["display_home"]) for item in homes][-1] == ("antigravity", "~/.gemini")
    target = next(item for item in agents.targets(homes, settings) if item["type"] == "antigravity")
    assert (target["registered"], target["guidance"], target["guidance_file"], target["cli"]) == (False, False, "GEMINI.md", True)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    connected = agents.register(target["id"], homes, settings, run=run)
    assert calls == [["agy", "mcp", "add", "--env", f"TESS_MEMORY_SETTINGS={settings}", "tess-memory", "--",
                      sys.executable, "-m", "tess_memory.cli", "serve"]]
    assert connected["guidance_outcome"] == "added"
    assert (user / ".gemini/GEMINI.md").read_text() == guidance.text("antigravity")
    assert "for shared Antigravity memory and durable" in guidance.text("antigravity")
    (user / ".gemini/config/mcp_config.json").write_text(json.dumps({"mcpServers": {"tess-memory": {
        **agents.server_entry(settings), "disabled": False}}}))
    assert next(item for item in agents.targets(homes, settings) if item["type"] == "antigravity")["current"] is True


def test_setup_on_a_store_with_settings_starts_from_them_and_keeps_what_it_does_not_decide(machine):
    setup, client = session(machine, keychain=None)
    _, user = machine
    (user / "exports").mkdir()
    (user / "exports/one.jsonl").write_text("{}\n")
    setup.home.mkdir(parents=True)
    setup.settings.write_text(f'''# written by hand
provenance_ledger = "/ledgers/main.jsonl"

[[data_stores]]
name = "work"
type = "claude-code"
path = "{user}/.claude/projects"
exclude = ["-private-tmp-*", "*scratch*"]
user = "sam"

[[data_stores]]
name = "old-agent"
type = "transcript"
path = "{user}/exports"
project = "assistant"
skip_copies_of = ["/elsewhere/sessions"]

[storage]
data_dir = "/somewhere/else"
shared = true

[llm]
profile = "mine"
fallback_profile = "other"
workers = 9

[search]
agent = false
max_rounds = 5

[ingest]
quiet_seconds = 60
adopt_published_policies = ["abababababababababababababababababababababababababababababababab"]

[planning]
default_project = "assistant"
lock_writes_to_default = true

[visibility.teams]
"core team" = ["sam", "kit"]

[[users]]
name = "sam"
since = 2026-01-05
''')
    before = tomllib.loads(setup.settings.read_text())
    state = client.get("/api/state").json()
    assert state["existing_settings"] is True
    # Its two sources are ticked under their own names; Codex is on this machine but was not a source.
    assert [(s["name"], s["type"], s["enabled"]) for s in state["sources"]] == [
        ("work", "claude-code", True), ("codex", "codex", False), ("old-agent", "transcript", True)]
    reply = client.post("/api/sources", json={"sources": [{"type": s["type"], "path": s["path"], "enabled": True} for s in state["sources"]]})
    assert [s["name"] for s in reply.json()["sources"]] == ["work", "codex", "old-agent"]
    client.post("/api/model/key", json={"kind": "anthropic", "api_key": KEY})
    client.post("/api/model/choose", json={"distill_model": "large", "search_model": "small"})
    setup.vault.draft["embedding"] = {"mode": "local", "name": "bge-small-en-v1.5",
                                      "settings": {"local_model": "./models/bge-small-en-v1.5", "dims": 384, "query_prefix": "q: ", "passage_prefix": ""}}
    written = client.post("/api/apply", json={}).json()["result"]
    assert Path(written["backup"]).read_text().startswith("# written by hand")
    after = tomllib.loads(setup.settings.read_text())
    work, codex, old = after["data_stores"]
    assert work == {**before["data_stores"][0], "path": str(user / ".claude/projects")}
    assert codex == {"name": "codex", "type": "codex", "path": str(user / ".codex")}
    assert old == before["data_stores"][1]
    for kept in ("provenance_ledger", "planning", "visibility", "users"):
        assert after[kept] == before[kept]
    assert after["storage"] == {"data_dir": str(setup.home), "shared": True}
    assert after["ingest"] == {"quiet_seconds": 60, "incremental": True, "adopt_published_policies": ["abababababababababababababababababababababababababababababababab"]}
    # The model is setup's to decide; how hard to work it is not.
    assert "fallback_profile" not in after["llm"] and (after["llm"]["model"], after["llm"]["workers"]) == ("large", 9)
    assert after["search"] == {"max_rounds": 5, "agent": True, "profile": after["llm"]["profile"], "model": "small"}
    # Running setup again over its own file, edited by hand since, keeps the edit too.
    setup.settings.write_text(setup.settings.read_text() + '\n[retrieval]\nhalf_life_days = 45.5\n')
    client.post("/api/apply", json={})
    assert tomllib.loads(setup.settings.read_text()) == {**after, "retrieval": {"half_life_days": 45.5}}
    # A file whose own settings Tess Memory would refuse is not replaced by one that is refused too.
    backups = sorted(setup.home.glob("settings.toml.before-setup-*"))
    broken = setup.settings.read_text().replace("half_life_days = 45.5", 'half_life_days = 45.5\n\n[codex]\nbatch_size = "many"')
    setup.settings.write_text(broken)
    refused = client.post("/api/apply", json={})
    assert refused.status_code == 422 and refused.json() == {"error": "invalid_settings", "detail": "[codex] batch_size must be a positive integer"}
    assert setup.settings.read_text() == broken and not setup.settings.with_name("settings.toml.new").exists()
    assert sorted(setup.home.glob("settings.toml.before-setup-*")) == backups
