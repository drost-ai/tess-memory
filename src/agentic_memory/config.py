"""Central configuration for agentic-memory.

Everything is overridable via environment variables so the MCP server and the
CLI can be pointed at different transcript sources or databases without code
changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass
class Config:
    # Where Claude Code session transcripts live (one subdir per project,
    # each containing <session-id>.jsonl files).
    source_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENTIC_MEMORY_SOURCE", Path("~/.claude/projects").expanduser()
        )
    )
    # Where the memory database and caches live.
    data_dir: Path = field(
        default_factory=lambda: _env_path(
            "AGENTIC_MEMORY_DATA", Path("~/.agentic-memory").expanduser()
        )
    )

    # LLM backend: "auto" | "agent-core" | "sdk" | "claude-cli" | "none"
    llm_backend: str = field(
        default_factory=lambda: os.environ.get("AGENTIC_MEMORY_BACKEND", "auto")
    )
    # Fast model: per-session distillation and weekly digests (many calls).
    fast_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTIC_MEMORY_FAST_MODEL", "claude-haiku-4-5"
        )
    )
    # Deep model: project profiles (few calls, cross-session synthesis).
    deep_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTIC_MEMORY_DEEP_MODEL", "claude-opus-5"
        )
    )

    # Embedding provider: a profile name from the Tess settings file, or
    # "local" for offline sentence-transformers.
    embed_provider: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTIC_MEMORY_EMBED_PROVIDER", "embeddings"
        )
    )
    embed_dims: int = field(
        default_factory=lambda: int(os.environ.get("AGENTIC_MEMORY_EMBED_DIMS", "4096"))
    )
    # Local fallback embedding model (sentence-transformers).
    embed_model: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTIC_MEMORY_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
        )
    )

    # Path to the agent-core-python checkout (used by the agent-core backend).
    agent_core_path: Path = field(
        default_factory=lambda: _env_path(
            "AGENT_CORE_PATH", Path("~/agents/agent-core-python").expanduser()
        )
    )
    # Tess settings file with provider profiles, merged into agent-core config.
    agent_core_settings: Path = field(
        default_factory=lambda: _env_path(
            "AGENTIC_MEMORY_AC_SETTINGS", Path("~/.tess-memory/providers.yaml").expanduser()
        )
    )
    # Provider profile to select; its defaultModel governs (claude-opus-5).
    agent_core_profile: str = field(
        default_factory=lambda: os.environ.get(
            "AGENTIC_MEMORY_AC_PROFILE", "strong"
        )
    )

    # Ingestion limits.
    max_transcript_chars: int = 150_000  # single-pass ceiling; above -> map-reduce
    segment_chars: int = 150_000         # sliding-window segment size (map phase)
    segment_overlap: int = 7_500         # overlap between consecutive segments
    max_segments: int = 20               # hard cap on map calls per session
    chunk_chars: int = 6_000             # window cap within an exchange-aligned chunk
    tool_result_snippet: int = 400       # tool-result truncation in rendered turns
    llm_workers: int = 4                 # parallel distillation calls

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
