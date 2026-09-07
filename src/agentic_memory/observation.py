"""Persist bounded rollout observations without issuing model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time

from .freshness import safe_error
from .scheduler import service_status
from .scheduler_config import load_service


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def operation_peaks(config, since):
    db = sqlite3.connect(config.db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        rows = db.execute('SELECT provider,kind,started,ended FROM call_receipts WHERE ended IS NULL OR ended>=?', (since,)).fetchall()
    finally:
        db.close()
    events = []
    now = time.time()
    for provider, kind, started, ended in rows:
        events += [(max(started, since), 1, provider, kind), (ended or now, -1, provider, kind)]
    active, peaks, providers, provider_peaks = {}, {}, {}, {}
    for _, delta, provider, kind in sorted(events):
        active[kind] = active.get(kind, 0) + delta
        providers[provider] = providers.get(provider, 0) + delta
        peaks[kind] = max(peaks.get(kind, 0), active[kind])
        provider_peaks[provider] = max(provider_peaks.get(provider, 0), providers[provider])
    within = peaks.get('model', 0) <= config.model_limit and peaks.get('embedding', 0) <= config.embedding_limit
    within = within and all(value <= config.provider_limits.get(key, max(config.model_limit, config.embedding_limit)) for key, value in provider_peaks.items())
    return {'global': peaks, 'providers': provider_peaks, 'within_limits': within}


def observe(config_path: Path, output: Path, *, seconds=86400, interval=60):
    if seconds <= 0 or interval <= 0:
        raise ValueError('observation duration and interval must be positive')
    config = load_service(config_path)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.time()
    history = output.with_suffix('.jsonl')
    observations = sampling_errors = 0
    latest_success = {}
    expected_enabled = None
    disabled_during_observation = set()
    limit_violation_observed = False
    max_gap = 0
    unhealthy = max_unhealthy = 0
    previous = started
    while True:
        now = time.time()
        max_gap = max(max_gap, now - previous)
        previous = now
        observations += 1
        error = None
        try:
            status = service_status(config)
            peaks = operation_peaks(config, started)
        except (sqlite3.Error, OSError) as exc:
            # Monitoring must report failed samples and keep observing. A
            # transient lock must not leave a stale, apparently healthy file.
            status, peaks = None, None
            error = safe_error(exc)
            sampling_errors += 1
        healthy = False
        if status is not None:
            enabled = {j['instance'] for j in status['jobs'] if j['enabled']}
            if expected_enabled is None:
                expected_enabled = enabled
            disabled_during_observation.update(expected_enabled - enabled)
            supervisor = status.get('supervisor') or {}
            stores_available = all(status['stores'].get(name, {}).get('sessions') is not None for name in expected_enabled)
            healthy = bool(stores_available and supervisor.get('process_alive') and
                           supervisor.get('heartbeat_age_seconds', float('inf')) <= max(15, interval * 2))
            for job in status['jobs']:
                if job['kind'] == 'ingest' and job['last_success']:
                    latest_success[job['instance']] = job['last_success']
            summary = {'at': now, 'supervisor': status['supervisor'],
                       'stores': {name: {'sessions': data.get('sessions'), 'ingestion': data.get('ingestion'),
                                         'status': data.get('status'), 'error': data.get('error')}
                                  for name, data in status['stores'].items()},
                       'active_calls': status['providers']['active'], 'enabled_stores': sorted(enabled),
                       'sampling_error': None}
            limit_violation_observed |= not peaks['within_limits']
        else:
            summary = {'at': now, 'supervisor': None, 'stores': {}, 'active_calls': None,
                       'enabled_stores': None, 'sampling_error': error}
        unhealthy = 0 if healthy else unhealthy + 1
        max_unhealthy = max(max_unhealthy, unhealthy)
        with history.open('a') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(summary) + '\n')
        finished = now - started >= seconds
        completed_passes = {name: latest_success.get(name, 0) > started for name in (expected_enabled or ())}
        report = {'status': 'complete' if finished else 'observing', 'started_at': started,
                  'updated_at': now, 'requested_seconds': seconds, 'observed_seconds': now - started,
                  'observations': observations, 'largest_observation_gap_seconds': max_gap,
                  'sampling_errors': sampling_errors, 'last_sampling_error': error,
                  'disabled_during_observation': sorted(disabled_during_observation),
                  'operation_peaks': peaks, 'limit_violation_observed': limit_violation_observed,
                  'ingest_pass_observed': completed_passes,
                  'supervisor_healthy_at_finish': healthy, 'max_consecutive_unhealthy_samples': max_unhealthy,
                  'latest': summary,
                  'passed': bool(finished and completed_passes and all(completed_passes.values()) and
                                 not disabled_during_observation and not limit_violation_observed and
                                 peaks and peaks['within_limits'] and healthy and max_unhealthy <= 1),
                  'scope': 'Operational observation only; source problems, observation gaps and model-quality results require their separate acceptance review.'}
        private_json(output, report)
        if finished:
            return report
        time.sleep(min(interval, max(0.01, seconds - (time.time() - started))))
