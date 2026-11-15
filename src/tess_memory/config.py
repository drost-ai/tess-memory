"""Explicit standalone TOML and host-owned embedded manifest configuration.

Imports are inert. Embedded mode reads only TESS_MEMORY_CONFIG's absolute
JSON file and its explicit registry; standalone callers select a TOML path or
TESS_MEMORY_SETTINGS. Legacy environment overrides apply only standalone.
"""

from __future__ import annotations

import json
import fnmatch
import os
import re
import tomllib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from .autobiography_config import PersonalConfig
from pathlib import Path

# Project slugs that are not real work: ephemeral cwds of our own LLM calls
# (ingesting those would feed the memory system its own exhaust) and macOS
# temp-dir sessions.
DEFAULT_EXCLUDES = [
    "-private-var-folders*",
    "-private-tmp-claude*",
    "*tess-memory-llm*",
    "*tess-memory-ac*",
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
    # Directories this store's sessions were copied from: a session whose
    # file name also exists in one of them is a copy, already ingested from
    # there, and is skipped (it would otherwise be distilled again under its
    # new path). E.g. a runtime home copied from another agent's.
    skip_copies_of: list[Path] = field(default_factory=list)

    @property
    def attribution(self) -> ProjectAttribution:
        return ProjectAttribution(self.project)

    def resolve_project(self, source_project: str) -> str:
        return self.attribution.resolve(source_project)

    def is_excluded(self, project_slug: str) -> bool:
        return any(fnmatch.fnmatch(project_slug, pat) for pat in self.exclude)


def _default_policy():
    from .principals import VisibilityPolicy
    return VisibilityPolicy()


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
    data_stores: list[DataStore] = field(default_factory=list)
    embedded: bool = False
    runtime_home: Path | None = None
    provider_registry: Path | None = None
    provenance_ledger: Path | None = None
    provenance_source_root: Path | None = None
    embedding_allow_download: bool = False
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
    # Default visibility by level/kind for rows with none of their own.
    visibility: "VisibilityPolicy" = field(default_factory=lambda: _default_policy())

    # -- storage -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("~/.tess-memory").expanduser())

    # -- llm (agent-core-python is the only backend; provider, model, and
    #    credentials are governed by its own configuration) ------------------
    # Legacy standalone mapping retained, but no checkout is discovered/imported.
    agent_core_path: Path | None = None
    agent_core_settings: Path | None = None
    agent_core_profile: str = ""
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
    # Explicit model prefixes (e5: "query: "/"passage: "). None keeps the
    # historical behaviour: the BGE query instruction for BGE names, else none.
    embed_query_prefix: str | None = None
    embed_passage_prefix: str | None = None
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

    def data_store_for(self, source_path) -> "DataStore | None":
        """The configured store a session's source file lives under."""
        if not source_path:
            return None
        path = Path(str(source_path)).expanduser()
        for ds in self.data_stores:
            try:
                path.resolve().relative_to(ds.path.expanduser().resolve())
                return ds
            except (ValueError, OSError):
                continue
        return None

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
    raw = os.environ.get("TESS_MEMORY_SETTINGS")
    if not raw:
        raise ValueError("Explicit TESS_MEMORY_CONFIG manifest or TESS_MEMORY_SETTINGS is required")
    return _path(raw)


def load_config(path: Path | None = None) -> Config:
    """Load an explicitly selected standalone TOML or embedded manifest."""
    if path is None and os.environ.get("TESS_MEMORY_CONFIG"):
        return load_manifest(Path(os.environ["TESS_MEMORY_CONFIG"]))
    path = (Path(path) if path is not None else settings_path()).expanduser().absolute()
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    base = path.parent

    def anchored(value):
        value = _path(str(value))
        return value if value.is_absolute() else base / value

    # The pure parser enforces absolute credential routing. Resolve this one
    # explicit standalone field before parsing, then anchor all final paths
    # after legacy environment overrides have been applied.
    llm = raw.get("llm", {})
    auth_path = llm.get("codex_auth_path")
    if isinstance(auth_path, str) and auth_path.strip():
        llm["codex_auth_path"] = str(anchored(auth_path.strip()))
    cfg = _parse_config(raw)
    for attribute in ("data_dir", "agent_core_settings", "provider_registry",
                      "agent_core_codex_auth_path", "agent_core_path",
                      "provenance_ledger", "provenance_source_root"):
        value = getattr(cfg, attribute)
        if value is not None:
            setattr(cfg, attribute, anchored(value))
    cfg.data_stores = [replace(ds, path=anchored(ds.path),
                              skip_copies_of=[anchored(p) for p in ds.skip_copies_of])
                       for ds in cfg.data_stores]
    cfg.personal = replace(cfg.personal,
                           source_paths=tuple(str(anchored(p)) for p in cfg.personal.source_paths),
                           native_creation_ledger=(str(anchored(cfg.personal.native_creation_ledger))
                                                   if cfg.personal.native_creation_ledger else None))
    if cfg.embed_model.startswith(("./", "../", "~")):
        cfg.embed_model = str(anchored(cfg.embed_model))
    return cfg


def _parse_config(raw: dict, *, embedded: bool = False) -> Config:
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
                skip_copies_of=[_path(p) for p in d.get("skip_copies_of", [])],
            )
            for d in raw["data_stores"]
        ]
    visibility = raw.get("visibility", {})
    if visibility:
        from .principals import VisibilityPolicy
        defaults = VisibilityPolicy()
        cfg.visibility = VisibilityPolicy(
            levels={**defaults.levels, **{str(k): str(v) for k, v in (visibility.get("levels") or {}).items()}},
            memory_kinds={**defaults.memory_kinds, **{str(k): str(v) for k, v in (visibility.get("memory_kinds") or {}).items()}},
            fallback=str(visibility.get("fallback", defaults.fallback)),
        )
    identity = raw.get("identity", {})
    user = (None if embedded else os.environ.get("TESS_MEMORY_USER")) or identity.get("user")
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
        warnings.warn("llm.agent_core_path is deprecated and ignored; install agent-core as a dependency", FutureWarning, stacklevel=2)
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
    for key in ("query_prefix", "passage_prefix"):
        if key in emb:
            if not isinstance(emb[key], str) or len(emb[key]) > 200:
                raise ValueError(f"[embedding] {key} must be a string of at most 200 characters")
            setattr(cfg, f"embed_{key}", emb[key])

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

    if "allow_download" in emb:
        if not isinstance(emb["allow_download"], bool):
            raise ValueError("[embedding] allow_download must be a boolean")
        cfg.embedding_allow_download = emb["allow_download"]
    if embedded:
        return cfg

    # ---- environment overrides (highest precedence) ----
    if os.environ.get("TESS_MEMORY_SOURCE"):
        cfg.data_stores = [
            DataStore(name="env-override", path=_path(os.environ["TESS_MEMORY_SOURCE"]))
        ]
    if os.environ.get("TESS_MEMORY_DATA"):
        cfg.data_dir = _path(os.environ["TESS_MEMORY_DATA"])
    if os.environ.get("AGENT_CORE_PATH"):
        warnings.warn("AGENT_CORE_PATH is deprecated and ignored; install agent-core as a dependency", FutureWarning, stacklevel=2)
        cfg.agent_core_path = _path(os.environ["AGENT_CORE_PATH"])
    if os.environ.get("TESS_MEMORY_AC_SETTINGS"):
        cfg.agent_core_settings = _path(os.environ["TESS_MEMORY_AC_SETTINGS"])
    if os.environ.get("TESS_MEMORY_AC_PROFILE"):
        cfg.agent_core_profile = os.environ["TESS_MEMORY_AC_PROFILE"]
    if os.environ.get("TESS_MEMORY_AC_FALLBACK_PROFILE"):
        cfg.agent_core_fallback_profile = os.environ["TESS_MEMORY_AC_FALLBACK_PROFILE"]
    if os.environ.get("TESS_MEMORY_SEARCH_AGENT"):
        cfg.search_agent = os.environ["TESS_MEMORY_SEARCH_AGENT"].lower() not in (
            "0", "false", "no", "off",
        )
    if os.environ.get("TESS_MEMORY_SEARCH_PROFILE"):
        cfg.search_profile = os.environ["TESS_MEMORY_SEARCH_PROFILE"]
    if os.environ.get("TESS_MEMORY_EMBED_PROVIDER"):
        cfg.embed_provider = os.environ["TESS_MEMORY_EMBED_PROVIDER"]
    if os.environ.get("TESS_MEMORY_EMBED_DIMS"):
        cfg.embed_dims = int(os.environ["TESS_MEMORY_EMBED_DIMS"])
    if os.environ.get("TESS_MEMORY_EMBED_MODEL"):
        cfg.embed_model = os.environ["TESS_MEMORY_EMBED_MODEL"]

    return cfg


def _absolute(value, name: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def load_manifest(path: Path | str, *, store_settings: Path | str | None = None) -> Config:
    """Validate an embedded manifest without clients, stores, credentials or downloads."""
    path = _absolute(str(path), "TESS_MEMORY_CONFIG")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate manifest field: {key}")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as fh:
        envelope = json.load(fh, object_pairs_hook=unique_object)
    return parse_manifest(envelope, store_settings=store_settings)


def parse_manifest(document: Mapping[str, object], *, store_settings: Path | str | None = None) -> Config:
    """Validate the embedded document in memory, without writing a manifest.

    Shares the file loader's metadata-only validation: explicit registry/model
    metadata may be read, but no credentials, clients, stores or downloads are
    initialized. The caller's document is never mutated. Optional per-store
    settings retain the same explicit, constrained TOML overlay contract.
    """
    envelope = document
    required = {"schema_version", "mode", "runtime_home", "provider_registry", "config"}
    if not isinstance(envelope, Mapping) or set(envelope) - required - {"provenance_ledger", "provenance_source_root"} or required - set(envelope):
        raise ValueError("Invalid embedded manifest fields")
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
        raise ValueError("Unsupported embedded manifest schema_version")
    if envelope["mode"] != "embedded":
        raise ValueError("Manifest mode must be embedded")
    home = _absolute(envelope["runtime_home"], "runtime_home")
    registry = _absolute(envelope["provider_registry"], "provider_registry")
    raw = envelope["config"]
    if not isinstance(raw, dict):
        raise ValueError("Manifest config must be an object")
    raw = json.loads(json.dumps(raw))  # do not mutate the caller's independent mapping
    if store_settings is not None:
        store_path = Path(store_settings)
        if str(store_settings).startswith("~"):
            raise ValueError("Store settings must not use ambient ~ expansion")
        if not store_path.is_absolute():
            store_path = home / store_path
        with store_path.open("rb") as fh:
            store_raw = tomllib.load(fh)
        if set(store_raw) - {"data_stores", "storage"}:
            raise ValueError("Embedded store settings may override only data_stores and storage.data_dir")
        store_storage = store_raw.get("storage", {})
        if not isinstance(store_storage, dict) or set(store_storage) - {"data_dir"}:
            raise ValueError("Embedded store settings storage may contain only data_dir")
        if "data_stores" in store_raw:
            raw["data_stores"] = store_raw["data_stores"]
        if "data_dir" in store_storage:
            if not isinstance(raw.get("storage", {}), dict):
                raise ValueError("[storage] must be an object")
            raw.setdefault("storage", {})["data_dir"] = store_storage["data_dir"]
    sections = {"personal", "codex", "planning", "storage", "llm", "search", "embedding", "ingest", "retrieval", "visibility", "identity"}
    if set(raw) - sections - {"data_stores"}:
        raise ValueError("Unknown memory config section")
    for section in sections:
        if section in raw and not isinstance(raw[section], dict):
            raise ValueError(f"[{section}] must be an object")
    def product_path(value):
        if not isinstance(value, str) or not value or value.startswith("~"):
            raise ValueError("Manifest paths must be nonempty, without ambient ~ expansion")
        p = Path(value)
        return str(p if p.is_absolute() else home / p)
    stores = raw.get("data_stores", [])
    if not isinstance(stores, list) or any(not isinstance(d, dict) or "path" not in d for d in stores):
        raise ValueError("data_stores must be a list of objects with path")
    for store in stores:
        store["path"] = product_path(store["path"])
        for key in ("skip_copies_of", "exclude"):
            values = store.get(key, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError(f"data_stores.{key} must be a list of strings")
        store["skip_copies_of"] = [product_path(p) for p in store.get("skip_copies_of", [])]
    storage = raw.setdefault("storage", {})
    storage["data_dir"] = product_path(storage.get("data_dir", "memory"))
    llm = raw.setdefault("llm", {})
    if "settings" in llm or "agent_core_path" in llm:
        raise ValueError("Embedded llm uses the envelope provider_registry, not legacy settings/path")
    if "codex_auth_path" in llm:
        llm["codex_auth_path"] = product_path(llm["codex_auth_path"])
    personal = raw.setdefault("personal", {})
    if "source_paths" in personal:
        if not isinstance(personal["source_paths"], list):
            raise ValueError("personal.source_paths must be a list of paths")
        personal["source_paths"] = [product_path(p) for p in personal["source_paths"]]
    if "native_creation_ledger" in personal:
        raise ValueError("Embedded provenance requires top-level provenance_ledger")
    ledger = None
    provenance_root = None
    if "provenance_source_root" in envelope:
        provenance_root = _absolute(envelope["provenance_source_root"], "provenance_source_root")
    if personal.get("source_mode") == "native" and provenance_root is None:
        raise ValueError("Native source mode requires explicit provenance_source_root")
    if "provenance_ledger" in envelope:
        ledger = _absolute(envelope["provenance_ledger"], "provenance_ledger")
        if personal.get("source_mode") == "native":
            personal["native_creation_ledger"] = str(ledger)
    search = raw.setdefault("search", {})
    if "agent" in search and not isinstance(search["agent"], bool):
        raise ValueError("[search] agent must be a boolean")
    required_roles = [("embedding", "provider")]
    if stores or personal.get("extraction_enabled") or personal.get("dream_enabled"):
        required_roles.append(("llm", "profile"))
    if search.get("agent", True):
        required_roles.append(("search", "profile"))
    for section, key in (("llm", "profile"), ("search", "profile"), ("embedding", "provider")):
        value = raw.get(section, {}).get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Embedded [{section}] {key} must be a nonempty string")
        if (section, key) in required_roles and value is None:
            raise ValueError(f"Embedded [{section}] {key} must be explicit")
    fallback = llm.get("fallback_profile", "")
    if not isinstance(fallback, str):
        raise ValueError("[llm] fallback_profile must be a string")
    from tess_core.config import ProviderRegistry
    providers = ProviderRegistry(registry)
    for section, name, model in (
        ("llm", llm.get("profile"), llm.get("model") or None),
        ("search", search.get("profile"), search.get("model") or None),
        ("fallback", fallback, llm.get("fallback_model") or None),
    ):
        if name:
            providers.resolve_profile(name, model=model)
    embedding = raw["embedding"]
    local_model = embedding.get("local_model")
    if isinstance(local_model, str) and local_model.startswith(("/", "./", "../", "~")):
        embedding["local_model"] = product_path(local_model)
        model_path = Path(embedding["local_model"])
        if not model_path.is_dir():
            raise ValueError("Explicit local embedding model directory is missing or inaccessible")
        for metadata_name in ("config_sentence_transformers.json", "config.json"):
            metadata_file = model_path / metadata_name
            if not metadata_file.exists():
                continue
            with metadata_file.open(encoding="utf-8") as fh:
                metadata = json.load(fh)
            if not isinstance(metadata, dict):
                raise ValueError("Local embedding model metadata must be an object")
            dimension = metadata.get("sentence_embedding_dimension")
            if dimension is not None:
                if type(dimension) is not int or dimension < 1:
                    raise ValueError("Local embedding model metadata has invalid sentence_embedding_dimension")
                if "dims" in embedding and embedding["dims"] != dimension:
                    raise ValueError("Configured embedding dims do not match local model metadata")
    for key in ("dims", "batch_size"):
        if key in embedding and (type(embedding[key]) is not int or embedding[key] < 1):
            raise ValueError(f"[embedding] {key} must be a positive integer")
    if embedding["provider"] != "local":
        profile = providers.resolve_profile(embedding["provider"])
        if not profile.get("baseUrl") or not profile.get("defaultModel"):
            raise ValueError("Embedding profile requires baseUrl and defaultModel")
    elif "local_model" in embedding and (not isinstance(embedding["local_model"], str) or not embedding["local_model"].strip()):
        raise ValueError("[embedding] local_model must be a nonempty string")
    cfg = _parse_config(raw, embedded=True)
    cfg.embedded = True
    cfg.runtime_home = home
    cfg.provider_registry = registry
    cfg.agent_core_settings = registry
    cfg.provenance_ledger = ledger
    cfg.provenance_source_root = provenance_root
    return cfg


def validate_config(cfg: Config, *, validate_credentials: bool = True, roles: set[str] | None = None) -> None:
    """Validate selected metadata/auth without clients, stores, models, or network.

    Default active roles are embedding, enabled agentic search, and distillation
    (plus its fallback) when sources or personal model jobs are configured.
    Hosts may explicitly restrict active roles to jobs they will actually run.
    All explicitly supplied profiles are metadata-validated regardless.
    """
    from tess_core.config import ProviderRegistry
    registry_path = cfg.provider_registry or cfg.agent_core_settings
    if registry_path is None:
        raise ValueError("Explicit provider registry is required")
    providers = ProviderRegistry(registry_path)
    selected = {"distillation": cfg.agent_core_profile, "search": cfg.search_profile,
                "fallback": cfg.agent_core_fallback_profile, "embedding": cfg.embed_provider}
    models = {"distillation": cfg.agent_core_model, "search": cfg.search_model,
              "fallback": cfg.agent_core_fallback_model, "embedding": ""}
    if roles is None:
        roles = {"embedding"}
        if cfg.search_agent:
            roles.add("search")
        if cfg.data_stores or cfg.personal.extraction_enabled or cfg.personal.dream_enabled:
            roles.add("distillation")
            if cfg.agent_core_fallback_profile:
                roles.add("fallback")
    if roles - set(selected):
        raise ValueError("Unknown provider validation role")
    for role, profile in selected.items():
        if role == "embedding" and profile == "local":
            continue
        if profile:
            providers.resolve_profile(profile, model=models[role] or None)
        if role not in roles:
            continue
        if not profile:
            raise ValueError(f"Explicit {role} profile is required")
        if validate_credentials:
            profile_metadata = providers.resolve_profile(profile, model=models[role] or None)
            codex_path = (cfg.agent_core_codex_auth_path
                          if role != "embedding" and profile_metadata.get("apiMode") == "codex" else None)
            providers.resolve_auth(profile, codex_auth_path=codex_path)


class _LazyConfig:
    """Compatibility binding: importing modules never reads runtime configuration."""
    def __init__(self):
        object.__setattr__(self, "_value", None)

    def _get(self):
        if self._value is None:
            object.__setattr__(self, "_value", load_config())
        return self._value

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        setattr(self._get(), name, value)


CONFIG = _LazyConfig()
