"""One store for several agents is the ordinary case, and it must be schedulable."""
from __future__ import annotations

from pathlib import Path

import pytest

from tess_memory.codex import CodexParseError
from tess_memory.scheduler import store_config
from tess_memory.setup import apply
from tess_memory.store import MemoryStore

DRAFT = {
    "model": {"profile": "chatgpt", "distill_model": "large", "search_model": "small", "verified": True},
    "embedding": {"mode": "hosted", "profile": "embeddings-model", "dims": 8, "send_dimensions": True},
}


def settings_for(home: Path, sources: list[tuple[str, str]]) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    draft = {**DRAFT, "sources": [{"name": kind, "type": kind, "path": str(home / folder), "enabled": True} for kind, folder in sources]}
    path = home / "settings.toml"
    path.write_text(apply.render(home, home / "providers.yaml", draft))
    return path


def with_a_claude_session(home: Path) -> None:
    store = MemoryStore(home / "memory.db")
    store.db.execute("INSERT INTO sessions(id, project, title) VALUES('a-claude-code-session', 'p', 't')")
    store.db.commit()
    store.close()


def test_a_store_setup_writes_for_claude_code_and_codex_together_is_scheduled(tmp_path):
    home = tmp_path / "store"
    settings = settings_for(home, [("claude-code", "claude"), ("codex", "codex")])
    with_a_claude_session(home)  # it already holds sessions that are not Codex's
    cfg = store_config(settings)
    assert [(source.name, source.type) for source in cfg.data_stores] == [("claude-code", "claude-code"), ("codex", "codex")]
    assert cfg.data_dir == home.resolve() or cfg.data_dir == home


def test_a_store_declared_for_codex_alone_still_refuses_a_database_holding_something_else(tmp_path):
    home = tmp_path / "store"
    settings = settings_for(home, [("codex", "codex")])
    assert [source.type for source in store_config(settings).data_stores] == ["codex"]  # empty: fine
    with_a_claude_session(home)
    with pytest.raises(CodexParseError, match="choose a separate data_dir"):
        store_config(settings)


def test_the_scheduler_configuration_setup_writes_points_at_a_store_it_can_run(tmp_path):
    from tess_memory.scheduler_config import load_service
    from tess_memory.setup import keepcurrent

    home = tmp_path / "store"
    settings = settings_for(home, [("claude-code", "claude"), ("codex", "codex")])
    (home / "scheduler.toml").write_text(keepcurrent.config_text(home, settings, timezone="UTC"))
    service = load_service(home / "scheduler.toml")
    assert store_config(service.stores[0].settings).db_path == (home / "memory.db").resolve()
