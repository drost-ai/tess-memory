"""Persist bounded rollout observations without issuing model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time

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
    db = sqlite3.connect(config.db_path)
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
    observations = 0
    first_success, latest_success = {}, {}
    max_gap = 0
    unhealthy = max_unhealthy = 0
    previous = started
    while True:
        now = time.time()
        status = service_status(config)
        max_gap = max(max_gap, now - previous)
        previous = now
        observations += 1
        enabled = {j['instance'] for j in status['jobs'] if j['enabled']}
        supervisor = status.get('supervisor') or {}
        healthy = bool(supervisor.get('process_alive') and supervisor.get('heartbeat_age_seconds', float('inf')) <= max(15, interval * 2))
        unhealthy = 0 if healthy else unhealthy + 1
        max_unhealthy = max(max_unhealthy, unhealthy)
        for job in status['jobs']:
            if job['kind'] == 'ingest' and job['last_success']:
                first_success.setdefault(job['instance'], job['last_success'])
                latest_success[job['instance']] = job['last_success']
        summary = {'at': now, 'supervisor': status['supervisor'],
                   'stores': {name: {'sessions': data.get('sessions'), 'ingestion': data.get('ingestion')}
                              for name, data in status['stores'].items()},
                   'active_calls': status['providers']['active'], 'enabled_stores': sorted(enabled)}
        with history.open('a') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(summary) + '\n')
        finished = now - started >= seconds
        peaks = operation_peaks(config, started)
        completed_passes = {name: latest_success.get(name, 0) > started for name in enabled}
        report = {'status': 'complete' if finished else 'observing', 'started_at': started,
                  'updated_at': now, 'requested_seconds': seconds, 'observed_seconds': now - started,
                  'observations': observations, 'largest_observation_gap_seconds': max_gap,
                  'operation_peaks': peaks, 'ingest_pass_observed': completed_passes,
                  'supervisor_healthy_at_finish': healthy, 'max_consecutive_unhealthy_samples': max_unhealthy,
                  'latest': summary,
                  'passed': bool(finished and enabled and all(completed_passes.values()) and peaks['within_limits'] and healthy and max_unhealthy <= 1),
                  'scope': 'Operational observation only; source problems and model-quality results require their separate acceptance review.'}
        private_json(output, report)
        if finished:
            return report
        time.sleep(min(interval, max(0.01, seconds - (time.time() - started))))
