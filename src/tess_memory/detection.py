"""Independent metadata observations; never modifies ingestion progress or memory."""
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import uuid

from .audit import readonly_database
from .attribution import source_adapter
from .codex import CodexCatalog
from .source_activity import source_activity


def detection_path(cfg):
    return cfg.data_dir / 'source-detection.db'


def detect_sources(cfg, *, stop_requested=lambda: False):
    started = time.time()
    catalog = (CodexCatalog(cfg.data_stores, cfg.codex)
               if any(ds.type == 'codex' for ds in cfg.data_stores) else None)
    errors = len(catalog.errors) if catalog else 0
    rows = {}
    for ds in cfg.data_stores:
        if not ds.path.is_dir():
            # CodexCatalog already reports a missing root.
            if ds.type != 'codex':
                errors += 1
            continue
        for path, project, _ in source_adapter(ds, codex_catalog=catalog)[0]:
            if stop_requested():
                raise InterruptedError('metadata detection stopped; previous complete observation retained')
            try:
                activity = source_activity(path, ds.type)
            except OSError:
                errors += 1
                continue
            path = str(path.resolve())
            sid = ('codex:' + catalog.sources[Path(path)].session_id) if ds.type == 'codex' else None
            signature = hashlib.sha256(json.dumps(activity.members).encode()).hexdigest()
            rows[path] = (path, ds.type, ds.resolve_project(project), sid, signature,
                          activity.modified_at, time.time(), activity.modified_at + cfg.quiet_seconds)
    path = detection_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    generation = uuid.uuid4().hex
    db = sqlite3.connect(path, timeout=10)
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS sources (
                path TEXT PRIMARY KEY, source_type TEXT NOT NULL, project TEXT NOT NULL,
                session_id TEXT, signature TEXT NOT NULL, modified_at REAL NOT NULL,
                observed_at REAL NOT NULL, quiet_until REAL NOT NULL, generation TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY CHECK(id=1), generation TEXT NOT NULL,
                started REAL NOT NULL, finished REAL NOT NULL, sources INTEGER NOT NULL,
                new_sources INTEGER NOT NULL, changed_sources INTEGER NOT NULL, errors INTEGER NOT NULL);
        ''')
        with db:
            old = dict(db.execute('SELECT path,signature FROM sources'))
            new = sum(key not in old for key in rows)
            changed = sum(key in old and old[key] != row[4] for key, row in rows.items())
            db.executemany('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?,?)',
                           [(*row, generation) for row in rows.values()])
            finished = time.time()
            db.execute('INSERT OR REPLACE INTO scans VALUES(1,?,?,?,?,?,?,?)',
                       (generation, started, finished, len(rows), new, changed, errors))
    finally:
        db.close()
    return {'detected_sources': len(rows), 'new_sources': new, 'changed_sources': changed,
            'hot_sources': sum(row[7] > finished for row in rows.values()), 'errors': errors,
            'model_calls': 0, 'embedded_passages': 0, 'started': started, 'finished': finished,
            'duration_seconds': round(finished - started, 3)}


def detection_status(cfg):
    with readonly_database(detection_path(cfg)) as db:
        if db is None:
            return {'status': 'not_initialized'}
        scan = db.execute('SELECT * FROM scans WHERE id=1').fetchone()
        if scan is None:
            return {'status': 'not_initialized'}
        now = time.time()
        hot = db.execute('SELECT COUNT(*) FROM sources WHERE generation=? AND quiet_until>?',
                         (scan['generation'], now)).fetchone()[0]
        return {'status': 'partial' if scan['errors'] else 'observed', 'started_at': scan['started'], 'finished_at': scan['finished'],
                'age_seconds': max(0, now - scan['finished']), 'sources': scan['sources'],
                'hot_sources': hot, 'new_sources': scan['new_sources'], 'changed_sources': scan['changed_sources'],
                'errors': scan['errors'],
                'note': 'Metadata observation only; ingestion independently rechecks files before provider work.'}
