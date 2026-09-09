"""Explicit four-store service configuration; provider/model settings stay per instance."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import tomllib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ScheduledStore:
    name: str
    settings: Path
    enabled: bool = False
    poll_seconds: float = 300
    profile_seconds: float = 3600
    digest_seconds: float = 86400
    sources_per_run: int = 16
    model_calls_per_run: int = 128
    aggregate_calls_per_run: int = 4
    workers: int | None = None
    hold_text_file: Path | None = None
    recovery_hour: int | None = None  # opt in to one bounded nightly recovery pass
    recovery_timezone: str = 'America/Los_Angeles'
    recovery_sources_per_run: int = 256
    recovery_model_calls_per_run: int = 128
    recovery_min_age_seconds: float = 900

    def next_recovery(self, after: float) -> float:
        local = datetime.fromtimestamp(after, ZoneInfo(self.recovery_timezone))
        candidate = local.replace(hour=self.recovery_hour, minute=0, second=0, microsecond=0)
        if candidate.timestamp() <= after:
            candidate += timedelta(days=1)
        return candidate.timestamp()


@dataclass(frozen=True)
class ServiceConfig:
    path: Path
    data_dir: Path
    stores: tuple[ScheduledStore, ...]
    parallel_stores: int = 4
    model_limit: int = 16
    embedding_limit: int = 4
    provider_limits: dict[str, int] = field(default_factory=lambda: {'codex': 16, 'deepseek': 8, 'anthropic': 4, 'fireworks': 4})
    backlog_seconds: float = 15
    retry_seconds: float = 30
    max_retry_seconds: float = 1800
    job_timeout_seconds: float = 3600
    job_drain_seconds: float = 900

    @property
    def db_path(self):
        return self.data_dir / 'coordinator.db'

    def store(self, name):
        for store in self.stores:
            if store.name == name:
                return store
        raise ValueError(f'unknown scheduled store: {name}')


def load_service(path: Path) -> ServiceConfig:
    path = path.expanduser().resolve()
    raw = tomllib.loads(path.read_text())
    service = dict(raw.get('service', {}))
    if not service.get('data_dir'):
        raise ValueError('[service] data_dir is required')
    service['data_dir'] = Path(service['data_dir']).expanduser().resolve()
    stores, names, destinations = [], set(), set()
    for item in raw.get('stores', []):
        values = dict(item)
        name = values.get('name', '')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name) or name in names:
            raise ValueError('store names must be unique lowercase identifiers')
        names.add(name)
        settings = Path(values['settings']).expanduser()
        values['settings'] = settings.resolve() if settings.is_absolute() else (path.parent / settings).resolve()
        contents = tomllib.loads(values['settings'].read_text())
        destination = (contents.get('storage') or {}).get('data_dir')
        if not destination or not contents.get('data_stores'):
            raise ValueError(f'{name}: explicit source and storage settings are required')
        destination = Path(destination).expanduser()
        if not destination.is_absolute():
            raise ValueError('scheduled memory data_dir must be an absolute path')
        destination = destination.resolve()
        if destination in destinations or destination == service['data_dir']:
            raise ValueError('scheduled stores and coordinator require distinct data directories')
        destinations.add(destination)
        if not isinstance(values.get('enabled', False), bool):
            raise ValueError('enabled must be a boolean')
        if values.get('hold_text_file'):
            p = Path(values['hold_text_file']).expanduser()
            values['hold_text_file'] = p.resolve() if p.is_absolute() else (path.parent / p).resolve()
        store = ScheduledStore(**values)
        if store.recovery_hour is not None and (isinstance(store.recovery_hour, bool)
                or not isinstance(store.recovery_hour, int) or not 0 <= store.recovery_hour <= 23):
            raise ValueError('recovery_hour must be an hour from 0 to 23')
        ZoneInfo(store.recovery_timezone)
        for key in ('poll_seconds', 'profile_seconds', 'digest_seconds', 'sources_per_run', 'model_calls_per_run', 'aggregate_calls_per_run', 'recovery_min_age_seconds'):
            value = getattr(store, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
                raise ValueError(f'{key} must be positive and finite')
        for key in ('sources_per_run', 'model_calls_per_run', 'aggregate_calls_per_run', 'workers', 'recovery_sources_per_run', 'recovery_model_calls_per_run'):
            value = getattr(store, key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f'{key} must be a positive integer')
        stores.append(store)
    if not stores:
        raise ValueError('at least one [[stores]] entry is required')
    result = ServiceConfig(path=path, stores=tuple(stores), provider_limits=raw.get('provider_limits', ServiceConfig.__dataclass_fields__['provider_limits'].default_factory()), **service)
    for key in ('parallel_stores', 'model_limit', 'embedding_limit'):
        value = getattr(result, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('backlog_seconds', 'retry_seconds', 'max_retry_seconds', 'job_timeout_seconds', 'job_drain_seconds'):
        value = getattr(result, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise ValueError(f'{key} must be positive and finite')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in result.provider_limits.values()):
        raise ValueError('provider limits must be positive integers')
    return result
