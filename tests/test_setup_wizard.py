"""The guided setup: what it refuses, what it writes, and where keys go."""
from __future__ import annotations

import json
import subprocess
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
    assert (cfg.embed_provider, cfg.embed_dims, cfg.embed_query_prefix, cfg.embed_send_dimensions) == ("embeddings", 768, "", True)
    assert [(d.name, d.type) for d in cfg.data_stores] == [("claude-code", "claude-code"), ("codex", "codex")]


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
