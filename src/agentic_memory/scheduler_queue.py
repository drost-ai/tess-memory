"""Durable per-store ingestion/profile/digest jobs with explicit ownership."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

from .provider_limits import process_alive

KINDS = ('ingest', 'profiles', 'digests', 'detect', 'recovery')


class WorkQueue:
    def __init__(self, config, *, readonly=False):
        self.config = config
        self.readonly = readonly
        if readonly:
            return
        config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        config.db_path.touch(mode=0o600, exist_ok=True)
        db = self.connect()
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS scheduled_stores (name TEXT PRIMARY KEY, enabled INTEGER NOT NULL, settings TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    instance TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                    due REAL NOT NULL, run_id TEXT, pid INTEGER, started REAL, heartbeat REAL,
                    attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT, last_stats TEXT,
                    last_success REAL, requested INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(instance,kind));
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, instance TEXT NOT NULL, kind TEXT NOT NULL, pid INTEGER,
                    started REAL NOT NULL, heartbeat REAL, finished REAL, outcome TEXT NOT NULL,
                    stats TEXT, error TEXT, force_retry INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS supervisor (
                    id INTEGER PRIMARY KEY CHECK(id=1), pid INTEGER NOT NULL,
                    started REAL NOT NULL, heartbeat REAL NOT NULL);
            ''')
            with db:
                for store in config.stores:
                    db.execute('INSERT INTO scheduled_stores VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET settings=excluded.settings',
                               (store.name, store.enabled, str(store.settings)))
                    for kind in KINDS:
                        if kind == 'recovery' and store.recovery_hour is None:
                            continue
                        due = store.next_recovery(time.time()) if kind == 'recovery' else time.time()
                        # Existing overdue work survives restarts and sleep for
                        # one catch-up pass; initialization never moves its due time.
                        db.execute('INSERT OR IGNORE INTO jobs(instance,kind,due) VALUES(?,?,?)', (store.name, kind, due))
        finally:
            db.close()

    def connect(self):
        if self.readonly:
            db = sqlite3.connect(self.config.db_path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
        else:
            db = sqlite3.connect(self.config.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def set_enabled(self, instance, enabled):
        self.config.store(instance)
        db = self.connect()
        try:
            with db:
                db.execute('UPDATE scheduled_stores SET enabled=? WHERE name=?', (bool(enabled), instance))
                if enabled:
                    db.execute("UPDATE jobs SET due=? WHERE instance=? AND state!='running' AND kind!='recovery'", (time.time(), instance))
        finally:
            db.close()

    def request(self, instance, kind='ingest'):
        self.config.store(instance)
        if kind not in KINDS:
            raise ValueError('invalid job kind')
        db = self.connect()
        try:
            with db:
                db.execute('UPDATE jobs SET due=?,requested=1 WHERE instance=? AND kind=?', (time.time(), instance, kind))
        finally:
            db.close()

    def claim(self, *, now=None, instance=None, kind=None, explicit=False):
        now = time.time() if now is None else now
        db = self.connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                detection = kind == 'detect'
                # Metadata-only workers have independent capacity. They can
                # observe a store while its one paid ingestion/aggregate owner
                # runs, without authorizing a second paid owner.
                active = {r['instance'] for r in db.execute("SELECT instance,kind FROM jobs WHERE state='running'")
                          if (r['kind'] == 'detect') == detection}
                if len(active) >= self.config.parallel_stores:
                    return None
                rows = db.execute("SELECT j.*,s.enabled FROM jobs j JOIN scheduled_stores s ON s.name=j.instance WHERE j.state!='running' ORDER BY j.due,CASE j.kind WHEN 'ingest' THEN 0 WHEN 'profiles' THEN 1 ELSE 2 END,j.instance").fetchall()
                valid_names = {s.name for s in self.config.stores}
                for row in rows:
                    if (row['kind'] == 'detect') != detection:
                        continue
                    if row['instance'] not in valid_names or row['instance'] in active:
                        continue
                    if instance and row['instance'] != instance:
                        continue
                    if kind and row['kind'] != kind:
                        continue
                    if not explicit and (not row['enabled'] or row['due'] > now):
                        continue
                    if row['kind'] == 'recovery' and self.config.store(row['instance']).recovery_hour is None:
                        continue
                    identifier = uuid.uuid4().hex
                    db.execute("UPDATE jobs SET state='running',run_id=?,pid=NULL,started=?,heartbeat=?,requested=0 WHERE instance=? AND kind=?",
                               (identifier, now, now, row['instance'], row['kind']))
                    db.execute("INSERT INTO runs(id,instance,kind,started,heartbeat,outcome,force_retry) VALUES(?,?,?,?,?,'starting',?)",
                               (identifier, row['instance'], row['kind'], now, now, bool(explicit or row['requested'])))
                    return {'id': identifier, 'instance': row['instance'], 'kind': row['kind']}
                return None
        finally:
            db.close()

    def get_run(self, identifier):
        db = self.connect()
        try:
            row = db.execute('SELECT * FROM runs WHERE id=?', (identifier,)).fetchone()
            return dict(row) if row else None
        finally:
            db.close()

    def start(self, identifier, pid, *, allow_finished=False):
        db = self.connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute("SELECT * FROM jobs WHERE run_id=? AND state='running'", (identifier,)).fetchone()
                if row is None:
                    old = db.execute('SELECT pid,finished FROM runs WHERE id=?', (identifier,)).fetchone()
                    if allow_finished and old and old['pid'] == pid and old['finished'] is not None:
                        return
                    raise ValueError('job ownership expired before worker startup')
                if row['pid'] is not None and row['pid'] != pid:
                    raise ValueError('job ownership belongs to a different worker process')
                db.execute('UPDATE jobs SET pid=?,heartbeat=? WHERE run_id=?', (pid, time.time(), identifier))
                db.execute("UPDATE runs SET pid=?,heartbeat=?,outcome='running' WHERE id=?", (pid, time.time(), identifier))
        finally:
            db.close()

    def heartbeat(self, identifier):
        db = self.connect()
        try:
            with db:
                db.execute("UPDATE jobs SET heartbeat=? WHERE run_id=? AND state='running'", (time.time(), identifier))
                db.execute("UPDATE runs SET heartbeat=? WHERE id=? AND outcome='running'", (time.time(), identifier))
        finally:
            db.close()

    def supervisor_heartbeat(self, pid, started):
        db = self.connect()
        try:
            with db:
                db.execute('INSERT INTO supervisor VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET pid=excluded.pid,started=excluded.started,heartbeat=excluded.heartbeat',
                           (pid, started, time.time()))
        finally:
            db.close()

    def finish(self, identifier, *, outcome='success', stats=None, error=None, remaining=0, now=None):
        now = time.time() if now is None else now
        db = self.connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute("SELECT * FROM jobs WHERE run_id=? AND state='running'", (identifier,)).fetchone()
                if row is None:
                    return False
                store = self.config.store(row['instance'])
                interval = {'ingest': store.poll_seconds, 'detect': store.poll_seconds,
                            'profiles': store.profile_seconds, 'digests': store.digest_seconds,
                            'recovery': 86400}[row['kind']]
                completed = outcome in ('success', 'partial')
                attempts = 0 if completed else row['attempts'] + 1
                delay = (self.config.backlog_seconds if remaining else interval) if completed else min(self.config.max_retry_seconds, self.config.retry_seconds * 2 ** min(attempts - 1, 10))
                due = now if row['requested'] else now + delay
                if row['kind'] == 'recovery' and store.recovery_hour is not None:
                    # A nightly budget is not a license for a 15-second paid
                    # loop. Both failures and unfinished work resume next night.
                    due = store.next_recovery(now)
                if row['kind'] == 'detect' and completed and not row['requested']:
                    # Start-to-start cadence, coalescing a missed tick. This
                    # clock is independent of paid job completion/backlog.
                    due = max(now, row['started'] + interval)
                db.execute("UPDATE jobs SET state='pending',due=?,pid=NULL,heartbeat=?,attempts=?,last_error=?,last_stats=?,last_success=CASE WHEN ? THEN ? ELSE last_success END WHERE run_id=?",
                           (due, now, attempts, error, json.dumps(stats) if stats is not None else None, completed, now, identifier))
                db.execute('UPDATE runs SET finished=?,outcome=?,stats=?,error=? WHERE id=?',
                           (now, outcome, json.dumps(stats) if stats is not None else None, error, identifier))
                return True
        finally:
            db.close()

    def recover(self, *, now=None):
        now = time.time() if now is None else now
        db = self.connect()
        try:
            rows = db.execute("SELECT * FROM jobs WHERE state='running'").fetchall()
        finally:
            db.close()
        lost = []
        for row in rows:
            if row['pid'] is None and now - row['started'] < 30:
                continue
            if not process_alive(row['pid']):
                if self.finish(row['run_id'], outcome='worker_lost', error='worker exited before committing its result', now=now):
                    lost.append(row['run_id'])
        return lost

    def status(self):
        db = self.connect()
        try:
            jobs = []
            for row in db.execute('SELECT j.*,s.enabled FROM jobs j JOIN scheduled_stores s ON s.name=j.instance ORDER BY j.instance,j.kind'):
                data = dict(row)
                data['last_stats'] = json.loads(data['last_stats']) if data['last_stats'] else None
                jobs.append(data)
            supervisor = db.execute('SELECT pid,started,heartbeat FROM supervisor WHERE id=1').fetchone()
            health = dict(supervisor) if supervisor else None
            if health:
                health['process_alive'] = process_alive(health['pid'])
                health['heartbeat_age_seconds'] = max(0, time.time() - health['heartbeat'])
            return {'supervisor': health, 'jobs': jobs, 'recent_runs': [dict(r) for r in db.execute('SELECT id,instance,kind,started,finished,outcome,error FROM runs ORDER BY started DESC LIMIT 30')]}
        finally:
            db.close()
