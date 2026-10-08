"""Central configuration for agentic-memory.

Precedence, lowest to highest:

    1. built-in defaults (the dataclass field defaults below)
    2. settings.toml at the project root (or $AGENTIC_MEMORY_SETTINGS)
    3. environment variables (so the MCP server registration and cron
       ingestion can override without touching the file)
"""

from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from .autobiography_config import PersonalConfig
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Project slugs that are not real work: ephemeral cwds of our own LLM calls
# (ingesting those would feed the memory system its own exhaust) and macOS
# temp-dir sessions.
DEFAULT_EXCLUDES = [
    "-private-var-folders*",
    "-private-tmp-claude*",
    "*agentic-memory-llm*",
    "*agentic-memory-ac*",
]


@dataclass(frozen=True)
class ProjectAttribution:
    """Resolve a source workspace into its durable logical project.

    With no fixed project this preserves the original engineering behavior:
    the source's cwd-derived slug is the project. A fixed project decouples
    memory ownership from execution location for persistent agents and other
    runtimes that move between workspaces.
    """

    fixed_project: str | None = None

    @property
    def mode(self) -> str:
        return "fixed" if (self.fixed_project or "").strip() else "cwd"

    def resolve(self, source_project: str) -> str:
        return (self.fixed_project or "").strip() or source_project


@dataclass
class DataStore:
    """One ingestion source: a directory of session transcripts."""

    name: str
    path: Path
    type: str = "claude-code"  # selects the discovery/parser implementation
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    # Optional stable logical project assigned after source-cwd discovery and
    # exclusion. Required for bare "transcript" stores; for claude-code and
    # agent-core it overrides cwd-derived attribution while cwd stays as
    # provenance on the session row.
    project: str | None = None
    # Whose sessions these are and the team they work in (Epic U); unset =
    # the instance's default owner, as before.
    user: str | None = None
    team: str | None = None

    @property
    def attribution(self) -> ProjectAttribution:
        return ProjectAttribution(self.project)

    def resolve_project(self, source_project: str) -> str:
        return self.attribution.resolve(source_project)

    def is_excluded(self, project_slug: str) -> bool:
        return any(fnmatch.fnmatch(project_slug, pat) for pat in self.exclude)


def _default_stores() -> list[DataStore]:
    return [
        DataStore(name="claude-code", path=Path("~/.claude/projects").expanduser())
    ]


@dataclass(frozen=True)
class CodexSettings:
    """Options applied only to type='codex' sources, never existing parsers."""

    tool_output_chars: int = 2_000
    include_subagents: bool = False
    batch_size: int = 4
    max_record_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("tool_output_chars", "batch_size", "max_record_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"[codex] {name} must be a positive integer")
        if not isinstance(self.include_subagents, bool):
            raise ValueError("[codex] include_subagents must be a boolean")


@dataclass
class Config:
    # -- data stores ---------------------------------------------------------
    data_stores: list[DataStore] = field(default_factory=_default_stores)
    codex: CodexSettings = field(default_factory=CodexSettings)

    # -- planning ------------------------------------------------------------
    # A persistent AI can bind all plan writes to one durable logical project.
    # Engineering instances leave these unset and retain explicit/cwd-derived
    # project behavior exactly as before.
    planning_default_project: str = ""
    planning_lock_writes_to_default: bool = False

    # -- identity (Epic U) ---------------------------------------------------
    # Who reads and writes through this process, bound at start; never a
    # model argument. Unset = the instance's default owner, unrestricted.
    identity_user: str | None = None

    # -- storage -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("~/.agentic-memory").expanduser())

    # -- llm (agent-core-python is the only backend; provider, model, and
    #    credentials are governed by its own configuration) ------------------
    agent_core_path: Path = field(
        default_factory=lambda: Path("~/agents/agent-core-python").expanduser()
    )
    agent_core_settings: Path = field(
        default_factory=lambda: Path("~/.tess-memory/providers.yaml").expanduser()
    )
    agent_core_profile: str = "strong"
    # Empty overrides preserve the selected profile's model/effort defaults.
    agent_core_model: str = ""
    agent_core_reasoning_effort: str = ""
    agent_core_codex_auth_path: Path | None = None  # credential routing only; excluded from cache identity
    # Profile to retry with when the primary model refuses (safety classifiers).
    # Empty = no fallback; refused sessions degrade to extractive digests.
    agent_core_fallback_profile: str = ""
    agent_core_fallback_model: str = ""
    agent_core_fallback_reasoning_effort: str = ""
    llm_workers: int = 4

    # -- search agent (bounded agentic retrieval inside search_memory) -------
    search_agent: bool = True            # False = plain static hybrid search
    search_profile: str = ""             # provider profile; empty = llm profile
    search_model: str = ""
    search_reasoning_effort: str = ""
    search_max_rounds: int = 4           # LLM calls per search (last is forced finish)
    search_queries_per_round: int = 3    # max sub-queries the agent may fan out
    search_per_query_limit: int = 8      # static hits pulled per sub-query
    search_evidence_limit: int = 12      # final curated evidence cap

    # -- embedding -----------------------------------------------------------
    embed_provider: str = "embeddings"  # profile name, or "local"
    embed_dims: int = 4096
    embed_model: str = "BAAI/bge-small-en-v1.5"  # local fallback model
    embed_batch_size: int = 64

    # -- ingest --------------------------------------------------------------
    max_transcript_chars: int = 150_000  # single-pass ceiling; above -> map-reduce
    segment_chars: int = 150_000         # sliding-window segment size (map phase)
    segment_overlap: int = 7_500         # overlap between consecutive segments
    max_segments: int = 20               # hard cap on map calls per session
    segment_hard_cap: int = 600_000      # max grown segment size before even sampling
    segment_workers: int = 1            # per-session map concurrency; shares llm_workers cap
    distillation_cache: bool = False    # exact successful requests, private to this data_dir
    incremental_ingestion: bool = False  # verified prefixes + stable completed-message units
    incremental_reduce_fanout: int = 8
    cache_retention_days: int = 30
    max_model_calls_per_run: int = 0    # 0 = unlimited; checkpoints resume a budgeted run
    incremental_batch_chars: int = 0  # opt in per store; 0 keeps single-unit requests
    incremental_batch_items: int = 8
    incremental_batch_unit_chars: int = 12000
    ingestion_hold_texts: list[str] = field(default_factory=list)  # operator-supplied export holds; never logged
    source_retry_backoff: bool = False  # scheduled workers opt in; explicit manual retries remain immediate
    recovery_ingestion: bool = False  # scheduler-only stable incomplete source recovery
    recovery_min_age_seconds: float = 900
    chunk_chars: int = 6_000             # window cap within an exchange-aligned chunk
    tool_result_snippet: int = 400       # tool-result truncation in rendered turns
    min_session_chars: int = 200         # skip sessions with less dialogue than this
    min_importance: float = 0.25         # drop distilled memories rated below this
    profile_session_limit: int = 30      # sessions fed into a project profile
    digest_session_limit: int = 40       # sessions fed into a weekly digest
    quiet_seconds: float = 0.0          # defer recently modified sources; 0 preserves eager ingest
    aggregate_refresh_seconds: float = 0.0  # minimum time since each successful aggregate refresh

    # -- retrieval -----------------------------------------------------------
    half_life_days: float = 30.0
    w_semantic: float = 0.55
    w_bm25: float = 0.20
    w_importance: float = 0.15
    w_recency: float = 0.10
    candidate_pool: int = 100            # top-N per signal before fusion
    per_session_cap: int = 3             # diversity: max results per session
    default_limit: int = 8               # results when the caller doesn't say
    chunk_neighbor_chars: int = 1_000    # context added around an expanded chunk hit
    vector_cache_bytes: int = 64 * 1024 * 1024
    vector_batch_rows: int = 512
    personal: PersonalConfig = field(default_factory=PersonalConfig)
    adopt_published_policies: tuple[str, ...] = ()  # explicit one-time policy transitions; never rewrite request caches

    @property
    def db_path(self) -> Path:
        return self.data_dir / "memory.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def to_slug(project: str) -> str:
    """Normalize a project reference: absolute or ~ paths become slugs.

    Claude Code's convention: the session cwd with '/' replaced by '-'.
    Relative paths are NOT resolved here (the MCP server cannot see the
    caller's cwd); the CLI resolves them against its own cwd before calling.
    """
    p = (project or "").strip()
    if p.startswith("~"):
        p = os.path.expanduser(p)
    if p.startswith("/"):
        return p.rstrip("/").replace("/", "-") or "-"
    return p


def _path(raw: str) -> Path:
    return Path(raw).expanduser()


def settings_path() -> Path:
    raw = os.environ.get("AGENTIC_MEMORY_SETTINGS")
    return _path(raw) if raw else _PROJECT_ROOT / "settings.toml"


def load_config(path: Path | None = None) -> Config:
    path = path or settings_path()
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raw = {}

    cfg = Config()
    cfg.personal = PersonalConfig.parse(raw.get('personal', {}))
    cfg.codex = CodexSettings(**raw.get("codex", {}))
    planning = raw.get("planning", {})
    storage = raw.get("storage", {})
    llm = raw.get("llm", {})
    search = raw.get("search", {})
    emb = raw.get("embedding", {})
    ing = raw.get("ingest", {})
    ret = raw.get("retrieval", {})
    weights = ret.get("weights", {})

    if raw.get("data_stores"):
        cfg.data_stores = [
            DataStore(
                name=str(d.get("name") or d["path"]),
                path=_path(d["path"]),
                type=str(d.get("type", "claude-code")),
                exclude=[str(p) for p in d["exclude"]] if "exclude" in d else list(DEFAULT_EXCLUDES),
                project=str(d["project"]) if d.get("project") else None,
                user=str(d["user"]) if d.get("user") else None,
                team=str(d["team"]) if d.get("team") else None,
            )
            for d in raw["data_stores"]
        ]
    identity = raw.get("identity", {})
    user = os.environ.get("AGENTIC_MEMORY_USER") or identity.get("user")
    if user:
        from .principals import principal_id
        cfg.identity_user = principal_id(str(user).strip())
    cfg.planning_default_project = str(
        planning.get("default_project", cfg.planning_default_project)
    ).strip()
    cfg.planning_lock_writes_to_default = bool(
        planning.get("lock_writes_to_default", cfg.planning_lock_writes_to_default)
    )
    if cfg.planning_lock_writes_to_default and not cfg.planning_default_project:
        raise ValueError(
            "[planning] lock_writes_to_default=true requires default_project"
        )

    if "data_dir" in storage:
        cfg.data_dir = _path(storage["data_dir"])

    if "agent_core_path" in llm:
        cfg.agent_core_path = _path(llm["agent_core_path"])
    if "settings" in llm:
        cfg.agent_core_settings = _path(llm["settings"])
    if 'codex_auth_path' in llm:
        value = llm['codex_auth_path']
        if not isinstance(value, str) or not value.strip():
            raise ValueError('[llm] codex_auth_path must be a nonempty absolute path')
        cfg.agent_core_codex_auth_path = _path(value.strip())
        if not cfg.agent_core_codex_auth_path.is_absolute():
            raise ValueError('[llm] codex_auth_path must be absolute after expanding ~')
    cfg.agent_core_profile = llm.get("profile", cfg.agent_core_profile)
    cfg.agent_core_fallback_profile = llm.get(
        "fallback_profile", cfg.agent_core_fallback_profile
    )
    cfg.llm_workers = int(llm.get("workers", cfg.llm_workers))

    cfg.search_agent = bool(search.get("agent", cfg.search_agent))
    cfg.search_profile = search.get("profile", cfg.search_profile)
    for section_name, section, fields in (
        ("llm", llm, {
            "model": "agent_core_model", "reasoning_effort": "agent_core_reasoning_effort",
            "fallback_model": "agent_core_fallback_model",
            "fallback_reasoning_effort": "agent_core_fallback_reasoning_effort",
        }),
        ("search", search, {"model": "search_model", "reasoning_effort": "search_reasoning_effort"}),
    ):
        for key, attribute in fields.items():
            if key not in section:
                continue
            value = section[key]
            if not isinstance(value, str):
                raise ValueError(f"[{section_name}] {key} must be a string")
            value = value.strip()
            if "reasoning_effort" in key:
                value = value.lower()
                if value and value not in {"low", "medium", "high", "xhigh", "max"}:
                    raise ValueError(f"[{section_name}] {key} must be low, medium, high, xhigh, or max")
            setattr(cfg, attribute, value)
    for key in ("max_rounds", "queries_per_round", "per_query_limit", "evidence_limit"):
        if key in search:
            setattr(cfg, f"search_{key}", int(search[key]))

    cfg.embed_provider = emb.get("provider", cfg.embed_provider)
    cfg.embed_dims = int(emb.get("dims", cfg.embed_dims))
    cfg.embed_model = emb.get("local_model", cfg.embed_model)
    cfg.embed_batch_size = int(emb.get("batch_size", cfg.embed_batch_size))

    for key in (
        "max_transcript_chars", "segment_chars", "segment_overlap", "max_segments",
        "segment_hard_cap", "chunk_chars", "tool_result_snippet", "min_session_chars",
        "profile_session_limit", "digest_session_limit",
    ):
        if key in ing:
            setattr(cfg, key, int(ing[key]))
    if "min_importance" in ing:
        cfg.min_importance = float(ing["min_importance"])
    for key in ("quiet_seconds", "aggregate_refresh_seconds"):
        if key in ing:
            value = ing[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
                raise ValueError(f"[ingest] {key} must be a finite nonnegative number")
            setattr(cfg, key, float(value))
    if "segment_workers" in ing:
        value = ing["segment_workers"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("[ingest] segment_workers must be a positive integer")
        cfg.segment_workers = value
    if "distillation_cache" in ing:
        if not isinstance(ing["distillation_cache"], bool):
            raise ValueError("[ingest] distillation_cache must be a boolean")
        cfg.distillation_cache = ing["distillation_cache"]
    if "incremental" in ing:
        if not isinstance(ing["incremental"], bool):
            raise ValueError("[ingest] incremental must be a boolean")
        cfg.incremental_ingestion = ing["incremental"]
    if "adopt_published_policies" in ing:
        values=ing["adopt_published_policies"]
        if not isinstance(values,list) or any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v) for v in values):
            raise ValueError('[ingest] adopt_published_policies must be a list of SHA-256 policy identities')
        cfg.adopt_published_policies=tuple(dict.fromkeys(values))
    for key, minimum in (("incremental_reduce_fanout", 2), ("cache_retention_days", 1), ("max_model_calls_per_run", 0),
                         ('incremental_batch_chars', 0), ('incremental_batch_items', 2), ('incremental_batch_unit_chars', 1)):
        if key in ing:
            value = ing[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"[ingest] {key} must be an integer >= {minimum}")
            setattr(cfg, key, value)
    if cfg.incremental_ingestion and not (cfg.segment_chars > 0 and 0 <= cfg.segment_overlap < cfg.segment_chars):
        raise ValueError("incremental segments require 0 <= segment_overlap < segment_chars")

    cfg.half_life_days = float(ret.get("half_life_days", cfg.half_life_days))
    cfg.w_semantic = float(weights.get("semantic", cfg.w_semantic))
    cfg.w_bm25 = float(weights.get("bm25", cfg.w_bm25))
    cfg.w_importance = float(weights.get("importance", cfg.w_importance))
    cfg.w_recency = float(weights.get("recency", cfg.w_recency))
    for key in ("candidate_pool", "per_session_cap", "default_limit", "chunk_neighbor_chars"):
        if key in ret:
            setattr(cfg, key, int(ret[key]))
    for key, minimum in (('vector_cache_bytes', 0), ('vector_batch_rows', 1)):
        if key in ret:
            value = ret[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'[retrieval] {key} must be an integer >= {minimum}')
            setattr(cfg, key, value)

    # ---- environment overrides (highest precedence) ----
    if os.environ.get("AGENTIC_MEMORY_SOURCE"):
        cfg.data_stores = [
            DataStore(name="env-override", path=_path(os.environ["AGENTIC_MEMORY_SOURCE"]))
        ]
    if os.environ.get("AGENTIC_MEMORY_DATA"):
        cfg.data_dir = _path(os.environ["AGENTIC_MEMORY_DATA"])
    if os.environ.get("AGENT_CORE_PATH"):
        cfg.agent_core_path = _path(os.environ["AGENT_CORE_PATH"])
    if os.environ.get("AGENTIC_MEMORY_AC_SETTINGS"):
        cfg.agent_core_settings = _path(os.environ["AGENTIC_MEMORY_AC_SETTINGS"])
    if os.environ.get("AGENTIC_MEMORY_AC_PROFILE"):
        cfg.agent_core_profile = os.environ["AGENTIC_MEMORY_AC_PROFILE"]
    if os.environ.get("AGENTIC_MEMORY_AC_FALLBACK_PROFILE"):
        cfg.agent_core_fallback_profile = os.environ["AGENTIC_MEMORY_AC_FALLBACK_PROFILE"]
    if os.environ.get("AGENTIC_MEMORY_SEARCH_AGENT"):
        cfg.search_agent = os.environ["AGENTIC_MEMORY_SEARCH_AGENT"].lower() not in (
            "0", "false", "no", "off",
        )
    if os.environ.get("AGENTIC_MEMORY_SEARCH_PROFILE"):
        cfg.search_profile = os.environ["AGENTIC_MEMORY_SEARCH_PROFILE"]
    if os.environ.get("AGENTIC_MEMORY_EMBED_PROVIDER"):
        cfg.embed_provider = os.environ["AGENTIC_MEMORY_EMBED_PROVIDER"]
    if os.environ.get("AGENTIC_MEMORY_EMBED_DIMS"):
        cfg.embed_dims = int(os.environ["AGENTIC_MEMORY_EMBED_DIMS"])
    if os.environ.get("AGENTIC_MEMORY_EMBED_MODEL"):
        cfg.embed_model = os.environ["AGENTIC_MEMORY_EMBED_MODEL"]

    return cfg


CONFIG = load_config()
