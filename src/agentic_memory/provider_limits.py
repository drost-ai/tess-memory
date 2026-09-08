"""Cross-process provider admission, shared cooldowns and call receipts."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlparse
import uuid
import sys

from .distill import CallBudgetExceeded


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def provider_key(cfg, profile_name: str) -> str:
    with cfg.agent_core_settings.open() as handle:
        profile = (json.load(handle).get('providerProfiles') or {}).get(profile_name, {})
    mode = str(profile.get('apiMode', '')).lower()
    host = (urlparse(profile.get('baseUrl') or '').hostname or '').lower()
    kind = str(profile.get('kind', '')).lower()
    if mode == 'codex' or profile_name == 'codex':
        return 'codex'
    if 'fireworks' in host or profile_name.startswith('fireworks'):
        return 'fireworks'
    if 'deepseek' in host or profile_name.startswith('deepseek'):
        return 'deepseek'
    if kind.startswith('anthropic') or profile_name.startswith('anthropic'):
        return 'anthropic'
    return host or profile_name


class ProviderGate:
    def __init__(self, path: Path, *, model_limit: int = 16, embedding_limit: int = 4,
                 provider_limits: dict[str, int] | None = None, poll_seconds: float = 0.2,
                 readonly: bool = False):
        if model_limit < 1 or embedding_limit < 1 or any(v < 1 for v in (provider_limits or {}).values()):
            raise ValueError('provider concurrency limits must be positive')
        self.path = path
        self.readonly = readonly
        self.limits = {'model': model_limit, 'embedding': embedding_limit}
        self.provider_limits = provider_limits or {}
        self.poll_seconds = poll_seconds
        if readonly:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        db = self.connect()
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS call_waiters (
                    ticket INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    owner TEXT NOT NULL, pid INTEGER NOT NULL, instance TEXT NOT NULL,
                    provider TEXT NOT NULL, kind TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS call_leases (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, pid INTEGER NOT NULL,
                    instance TEXT NOT NULL, provider TEXT NOT NULL, kind TEXT NOT NULL,
                    started REAL NOT NULL, heartbeat REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS provider_cooldowns (
                    provider TEXT PRIMARY KEY, until REAL NOT NULL, failures INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS call_receipts (
                    id TEXT PRIMARY KEY, instance TEXT NOT NULL, provider TEXT NOT NULL,
                    kind TEXT NOT NULL, started REAL NOT NULL, ended REAL, outcome TEXT);
                CREATE TABLE IF NOT EXISTS model_requests (
                    call_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    namespace TEXT NOT NULL, phase TEXT NOT NULL, request_hash TEXT NOT NULL,
                    input_chars INTEGER NOT NULL, json_retry INTEGER NOT NULL,
                    details TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS model_requests_run ON model_requests(run_id);
                CREATE INDEX IF NOT EXISTS model_requests_hash ON model_requests(request_hash);
            ''')
            db.commit()
        finally:
            db.close()

    def connect(self):
        if self.readonly:
            db = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
        else:
            db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def release_owner(self, owner: str):
        db = self.connect()
        try:
            with db:
                db.execute("UPDATE call_receipts SET ended=?,outcome='worker_lost' WHERE id IN (SELECT id FROM call_leases WHERE owner=?)", (time.time(), owner))
                db.execute('DELETE FROM call_leases WHERE owner=?', (owner,))
                db.execute('DELETE FROM call_waiters WHERE owner=?', (owner,))
        finally:
            db.close()

    def reap_dead(self):
        db = self.connect()
        try:
            owners = db.execute('SELECT owner,pid FROM call_leases UNION SELECT owner,pid FROM call_waiters').fetchall()
        finally:
            db.close()
        for row in owners:
            if not process_alive(row['pid']):
                self.release_owner(row['owner'])

    def cooldown(self, provider: str, seconds: float | None = None):
        db = self.connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                old = db.execute('SELECT failures FROM provider_cooldowns WHERE provider=?', (provider,)).fetchone()
                failures = min(6, old['failures'] + 1 if old else 1)
                delay = seconds if seconds is not None else min(300, 10 * 2 ** (failures - 1))
                db.execute('INSERT INTO provider_cooldowns VALUES(?,?,?) ON CONFLICT(provider) DO UPDATE SET until=MAX(until,excluded.until),failures=excluded.failures',
                           (provider, time.time() + delay, failures))
        finally:
            db.close()

    @contextmanager
    def acquire(self, provider: str, kind: str, *, owner: str, instance: str, check_admission=None):
        if kind not in self.limits:
            raise ValueError('unknown provider operation kind')
        self.reap_dead()
        identifier = uuid.uuid4().hex
        db = self.connect()
        try:
            with db:
                db.execute('INSERT INTO call_waiters(id,owner,pid,instance,provider,kind,created) VALUES(?,?,?,?,?,?,?)',
                           (identifier, owner, os.getpid(), instance, provider, kind, time.time()))
            last_reap = time.monotonic()
            while True:
                if check_admission:
                    check_admission()
                if time.monotonic() - last_reap >= 1:
                    self.reap_dead()
                    last_reap = time.monotonic()
                admitted = False
                with db:
                    db.execute('BEGIN IMMEDIATE')
                    counts = dict(db.execute('SELECT kind,COUNT(*) FROM call_leases GROUP BY kind'))
                    providers = dict(db.execute('SELECT provider,COUNT(*) FROM call_leases GROUP BY provider'))
                    cooldowns = dict(db.execute('SELECT provider,until FROM provider_cooldowns'))
                    # FIFO among eligible requests. A cooling or saturated
                    # provider cannot prevent other providers from progressing.
                    for row in db.execute('SELECT * FROM call_waiters ORDER BY ticket'):
                        eligible = (counts.get(row['kind'], 0) < self.limits[row['kind']]
                                    and providers.get(row['provider'], 0) < self.provider_limits.get(row['provider'], self.limits[row['kind']])
                                    and cooldowns.get(row['provider'], 0) <= time.time())
                        if not eligible:
                            continue
                        if row['id'] == identifier:
                            now = time.time()
                            db.execute('INSERT INTO call_leases VALUES(?,?,?,?,?,?,?,?)',
                                       (identifier, owner, os.getpid(), instance, provider, kind, now, now))
                            db.execute('INSERT INTO call_receipts VALUES(?,?,?,?,?,?,?)',
                                       (identifier, instance, provider, kind, now, None, 'running'))
                            db.execute('DELETE FROM call_waiters WHERE id=?', (identifier,))
                            admitted = True
                        break
                if admitted:
                    break
                time.sleep(self.poll_seconds)
            stop = threading.Event()
            def heartbeat():
                while not stop.wait(5):
                    connection = self.connect()
                    try:
                        with connection:
                            connection.execute('UPDATE call_leases SET heartbeat=? WHERE id=?', (time.time(), identifier))
                    except sqlite3.Error:
                        pass
                    finally:
                        connection.close()
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            outcome = 'success'
            try:
                yield identifier
            except BaseException as exc:
                outcome = type(exc).__name__
                if re.search(r'rate[ _-]?limit|\b429\b', str(exc), re.I):
                    self.cooldown(provider)
                raise
            finally:
                stop.set()
                thread.join(timeout=1)
                with db:
                    db.execute('DELETE FROM call_leases WHERE id=?', (identifier,))
                    db.execute('UPDATE call_receipts SET ended=?,outcome=? WHERE id=?',
                               (time.time(), outcome, identifier))
                    if outcome == 'success':
                        db.execute('DELETE FROM provider_cooldowns WHERE provider=? AND until<=?', (provider, time.time()))
        finally:
            with db:
                db.execute('DELETE FROM call_waiters WHERE id=?', (identifier,))
            db.close()

    def status(self):
        db = self.connect()
        try:
            return {'active': [dict(r) for r in db.execute('SELECT instance,provider,kind,COUNT(*) AS calls,MIN(started) AS oldest_started FROM call_leases GROUP BY instance,provider,kind')],
                    'waiting': [dict(r) for r in db.execute('SELECT instance,provider,kind,COUNT(*) AS calls FROM call_waiters GROUP BY instance,provider,kind')],
                    'cooldowns': [dict(r) for r in db.execute('SELECT * FROM provider_cooldowns WHERE until>?', (time.time(),))],
                    'completed': [dict(r) for r in db.execute('SELECT instance,provider,kind,outcome,COUNT(*) AS calls FROM call_receipts WHERE ended IS NOT NULL GROUP BY instance,provider,kind,outcome')]}
        finally:
            db.close()

    def usage(self, *, since=0, until=None, instance=None, run_id=None):
        """Observed usage only; include explicit coverage gaps and retry counts."""
        until = time.time() if until is None else until
        db = self.connect()
        try:
            params = [since, until]
            where = "r.kind='model' AND r.started>=? AND r.started<?"
            if instance:
                where += ' AND r.instance=?'
                params.append(instance)
            if run_id:
                where += ' AND m.run_id=?'
                params.append(run_id)
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='model_requests'").fetchone():
                return {'available': False, 'reason': 'usage receipts have not been installed'}
            rows = db.execute('SELECT r.instance,r.provider,r.outcome,m.* FROM call_receipts r '
                              'LEFT JOIN model_requests m ON m.call_id=r.id WHERE ' + where, params).fetchall()
            groups, hashes = {}, {}
            for row in rows:
                details = json.loads(row['details'] or '{}')
                key = row['instance'], row['provider'], row['phase'] or 'unattributed', details.get('requested_model')
                group = groups.setdefault(key, {'instance': key[0], 'provider': key[1], 'phase': key[2], 'model': key[3],
                                                 'calls': 0, 'attributed_calls': 0, 'calls_with_usage': 0,
                                                 'outcomes': {},
                                                 'calls_with_complete_usage': 0, 'wire_attempts_observed': 0,
                                                 'transport_retries_observed': 0, 'json_retries': 0,
                                                 'input_tokens': None, 'output_tokens': None, 'reasoning_tokens': None,
                                                 'cache_read_tokens': None, 'cache_write_tokens': None})
                group['calls'] += 1
                group['outcomes'][row['outcome']] = group['outcomes'].get(row['outcome'], 0) + 1
                group['attributed_calls'] += int(row['call_id'] is not None)
                group['calls_with_usage'] += int(details.get('input_tokens') is not None or details.get('output_tokens') is not None)
                group['calls_with_complete_usage'] += int(bool(details.get('usage_complete')))
                group['wire_attempts_observed'] += details.get('wire_attempts') or 0
                group['transport_retries_observed'] += details.get('transport_retries') or 0
                group['json_retries'] += row['json_retry'] or 0
                for field in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cache_read_tokens', 'cache_write_tokens'):
                    if details.get(field) is not None:
                        group[field] = (group[field] or 0) + details[field]
                if row['outcome'] == 'success' and row['request_hash']:
                    h = row['instance'], row['request_hash']
                    hashes[h] = hashes.get(h, 0) + 1
            return {'available': True, 'since': since, 'until': until, 'groups': list(groups.values()),
                    'repeated_successful_request_hashes': sum(n - 1 for n in hashes.values()),
                    'note': 'Observed provider usage, not Pro allowance or billed cost. Missing/failed-attempt usage remains unknown; reasoning and cached input are subsets, not additional totals. Repeated successes can include invalid output or publication recovery.'}
        finally:
            db.close()


class ModelBudget:
    def __init__(self, maximum: int, *, deadline: float | None = None):
        self.maximum, self.calls = maximum, 0
        self.deadline = deadline
        self.draining = threading.Event()
        self.telemetry_errors = 0
        self.lock = threading.Lock()

    def check(self):
        if self.draining.is_set() or (self.deadline is not None and time.monotonic() >= self.deadline):
            self.draining.set()
            raise CallBudgetExceeded('worker deadline reached; draining active requests into checkpoints')
        if self.maximum and self.calls >= self.maximum:
            raise CallBudgetExceeded('worker model-call budget reached')

    def take(self):
        with self.lock:
            self.check()
            self.calls += 1


class LimitedBackend:
    def __init__(self, backend, gate, budget, *, provider, owner, instance):
        self.backend, self.gate, self.budget = backend, gate, budget
        self.provider, self.owner, self.instance = provider, owner, instance
        self.name = backend.name
        self.cache_identity = getattr(backend, 'cache_identity', backend.name)
    def complete(self, **kwargs):
        verify = getattr(self.backend, 'verify_profile', lambda: None)
        verify()
        self.budget.check()
        with self.gate.acquire(self.provider, 'model', owner=self.owner, instance=self.instance,
                               check_admission=self.budget.check) as call_id:
            verify()
            self.budget.take()
            from .checkpoints import DistillationCheckpoints
            from .model_usage import UsageObserver, current_metadata, observe_usage
            metadata = current_metadata()
            request_hash = DistillationCheckpoints.shared_key(self.cache_identity, kwargs['system'], kwargs['prompt'])
            def persist(details):
                db = None
                try:
                    db = self.gate.connect()
                    with db:
                        db.execute('INSERT OR REPLACE INTO model_requests VALUES(?,?,?,?,?,?,?,?)',
                                   (call_id, self.owner, metadata.get('namespace', ''), metadata.get('phase', 'unknown'),
                                    request_hash, len(kwargs['system']) + len(kwargs['prompt']), metadata.get('json_retry', 0),
                                    json.dumps(details)))
                except sqlite3.Error as exc:
                    # Accounting failure must never discard a paid completion
                    # or trigger another provider call to recover its answer.
                    with self.budget.lock:
                        self.budget.telemetry_errors += 1
                    print(f'[usage] receipt update failed: {type(exc).__name__}', file=sys.stderr, flush=True)
                finally:
                    if db:
                        db.close()
            observer = UsageObserver(persist)
            persist(observer.data())
            with observe_usage(observer):
                result = self.backend.complete(**kwargs)
            verify()
            return result
