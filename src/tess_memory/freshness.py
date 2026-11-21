"""Cheap, read-only freshness evidence for status, context and retrieved hits."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .source_journal import SourceRevision
from .time_utils import week_start


def safe_error(value):
    text = str(value or '')
    text = re.sub(r'(?i)Bearer\s+[^\s,;]+', 'Bearer [redacted]', text)
    text = re.sub(r'\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,})\b', '[redacted]', text)
    text = re.sub(r'(?i)(api[_-]?key|access[_-]?token|password|authorization)(\s*[:=]\s*)[^\s,;]+', r'\1\2[redacted]', text)
    return text[:400] or None


def _table(store, name):
    return store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _epoch(value):
    try:
        d = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return d.replace(tzinfo=d.tzinfo or timezone.utc).timestamp()
    except (AttributeError, ValueError, TypeError):
        return None


def session_freshness(store, session_id, cfg=None):
    session = store.get_session(session_id)
    if session is None:
        return {'status': 'unknown_session'}
    progress = store.source_progress(session['source_path']) if _table(store, 'ingest_sources') else None
    if progress is None or progress['indexed_at'] is None:
        file = store.db.execute('SELECT * FROM files WHERE path=?', (session['source_path'],)).fetchone()
        base = {'status': 'legacy_unverified', 'indexed_through': session['ended_at'],
                'indexed_at': _epoch(file['ingested_at']) if file else None,
                'source_changed': None, 'pending_tail': None, 'coverage': 'legacy snapshot; source prefix not certified'}
        if progress is not None:
            base.update(status=progress['status'], last_error=safe_error(progress['last_error']), retry_at=progress['retry_at'] if 'retry_at' in progress.keys() else 0)
        return base
    try:
        state = json.loads(progress['state'])
    except (ValueError, TypeError):
        return {'status': 'progress_unreadable', 'indexed_at': progress['indexed_at']}
    if not isinstance(state, dict):
        return {'status': 'progress_unreadable', 'indexed_at': progress['indexed_at']}
    revisions = state.get('revisions') or []
    changed = None if not revisions else False
    available = True
    for raw in revisions:
        try:
            revision = SourceRevision(**raw)
            path = Path(revision.path)
            if cfg and not any(path.resolve().is_relative_to(ds.path.resolve()) for ds in cfg.data_stores):
                available, changed = False, True
                continue
            available = available and path.is_file()
            changed = changed or not revision.unchanged()
        except (ValueError, TypeError, OSError):
            changed = True
            available = False
    status = progress['status']
    if not available:
        status = 'source_unavailable'
    elif changed and status == 'indexed':
        status = 'source_changed'
    kinds = [kind for kind in (state.get('boundary_kinds') or []) if isinstance(kind, str)]
    return {'status': status, 'indexed_at': progress['indexed_at'],
            'indexed_through': state.get('indexed_through_at') or session['ended_at'],
            'indexed_units': len(state.get('indexed_units') or []),
            'pending_units': state.get('pending_units', 0), 'pending_tail': state.get('pending_tail', False),
            'quiet_until': state.get('quiet_until'),
            'source_changed': changed, 'source_available': available,
            'boundary_kinds': dict(Counter(kinds)),
            'recovery_notes': state.get('recovery_notes', []),
            'recovery_only': state.get('recovery_only', False),
            'incomplete_snapshot_recovered': bool(state.get('recovery_signature') and state.get('pending_tail')),
            'coverage': 'eligible persisted prefix; task completion is not inferred',
            'last_error': safe_error(progress['last_error']), 'retry_at': progress['retry_at'] if 'retry_at' in progress.keys() else 0}


def rollup_freshness(store, level, scope):
    if not _table(store, 'aggregate_refresh'):
        return {'status': 'legacy_unverified'}
    row = store.db.execute('SELECT * FROM aggregate_refresh WHERE level=? AND scope=?', (level, scope)).fetchone()
    if row is None:
        return {'status': 'legacy_unverified'}
    return {'status': 'pending_refresh' if row['revision'] > row['applied_revision'] else 'refreshed_snapshot',
            'refreshed_at': row['refreshed_at'], 'revision': row['revision'],
            'applied_revision': row['applied_revision'],
            'coverage': 'indexed sessions only; unprocessed source activity may remain'}


def hit_freshness(store, row, cfg=None):
    if row['session_id']:
        return session_freshness(store, row['session_id'], cfg)
    if row['level'] == 'project':
        return rollup_freshness(store, 'project', row['project'])
    if row['level'] == 'digest':
        return rollup_freshness(store, 'digest', (row['ts'] or '')[:10])
    if row['level'] == 'plan':
        return {'status': 'authored_plan', 'updated_at': row['ts']}
    if row['level'] == 'image':
        return {'status': 'recorded_image', 'updated_at': row['ts']}
    return {'status': 'unknown'}


def describe(freshness):
    status = freshness.get('status', 'unknown').replace('_', ' ')
    text = 'Memory freshness: ' + status
    if freshness.get('incomplete_snapshot_recovered'):
        text += '; available incomplete history recovered; task completion is not inferred'
    if freshness.get('indexed_through'):
        text += '; source events through ' + str(freshness['indexed_through'])
    if freshness.get('pending_units'):
        text += f"; {freshness['pending_units']} eligible units still pending"
    if freshness.get('pending_tail'):
        text += '; unfinished source tail remains'
    if freshness.get('status') == 'quiet_pending' and freshness.get('quiet_until'):
        text += '; model work deferred until source is quiet'
    if freshness.get('source_changed') and freshness.get('status') != 'source_changed':
        text += '; source changed since observation'
    if freshness.get('status') == 'legacy_unverified':
        text += '; indexed source boundaries are not recorded'
    return text


def project_coverage(store, project):
    if not _table(store, 'ingest_sources'):
        return ''
    if project:
        rows = store.db.execute('SELECT status,COUNT(*) FROM ingest_sources WHERE project=? GROUP BY status', (project,)).fetchall()
    else:
        rows = store.db.execute('SELECT status,COUNT(*) FROM ingest_sources GROUP BY status').fetchall()
    if not rows:
        return ''
    return 'Recorded source states: ' + ', '.join(f'{status}={count}' for status, count in rows) + '. Unprocessed or changed sources can contain newer work.'


def aggregate_progress(store):
    """Include legacy scopes that have not entered the refresh queue yet."""
    expected = {
        'project': {r[0] for r in store.db.execute('SELECT DISTINCT project FROM sessions WHERE project IS NOT NULL')},
        'digest': {week for r in store.db.execute('SELECT DISTINCT ended_at FROM sessions') if (week := week_start(r[0]))},
    }
    tracked = {'project': {}, 'digest': {}}
    if _table(store, 'aggregate_refresh'):
        for row in store.db.execute('SELECT level,scope,revision,applied_revision FROM aggregate_refresh'):
            if row['level'] in tracked:
                tracked[row['level']][row['scope']] = row
    result = {}
    for level, required in expected.items():
        rows = tracked[level]
        scopes = required | set(rows)
        current = sum(r['applied_revision'] >= r['revision'] for r in rows.values())
        result[level] = {'total': len(scopes), 'current': current,
                         'pending': sum(r['applied_revision'] < r['revision'] for r in rows.values()),
                         'untracked': len(required - set(rows)),
                         'percent_current': round(100 * current / len(scopes), 1) if scopes else None,
                         'note': 'Current covers indexed sessions and recorded refresh revisions; untracked legacy scopes are unverified, and unprocessed transcripts can contain newer work.'}
    return result


def store_status(store, cfg=None):
    result = store.counts()
    tracked = _table(store, 'ingest_sources')
    statuses = dict(store.db.execute('SELECT status,COUNT(*) FROM ingest_sources GROUP BY status')) if tracked else {}
    problems = []
    if tracked:
        columns = {r['name'] for r in store.db.execute('PRAGMA table_info(ingest_sources)')}
        retry = 'retry_at' if 'retry_at' in columns else '0 AS retry_at'
        for row in store.db.execute("SELECT path,session_id,project,status,state,observed_at,indexed_at,last_error," + retry + " FROM ingest_sources WHERE status NOT IN ('indexed','trivial','shadowed','superseded') ORDER BY CASE status WHEN 'error' THEN 0 WHEN 'held' THEN 1 ELSE 2 END, observed_at DESC LIMIT 20"):
            problem = dict(row)
            try:
                state = json.loads(problem.pop('state') or '{}')
            except (ValueError, TypeError):
                state = {}
            if problem['status'] == 'quiet_pending':
                problem['quiet_until'] = state.get('quiet_until')
            problem['last_error'] = safe_error(problem['last_error'])
            problems.append(problem)
        times = store.db.execute('SELECT MAX(observed_at),MAX(indexed_at) FROM ingest_sources').fetchone()
        legacy = store.db.execute('SELECT COUNT(*) FROM sessions s LEFT JOIN ingest_sources i ON i.path=s.source_path WHERE i.indexed_at IS NULL').fetchone()[0]
    else:
        times, legacy = (None, None), result['sessions']
    pending = dict(store.db.execute('SELECT level,COUNT(*) FROM aggregate_refresh WHERE revision>applied_revision GROUP BY level')) if _table(store, 'aggregate_refresh') else {}
    result['ingestion'] = {'observed_at': time.time(), 'source_statuses': statuses,
                           'last_source_observation': times[0], 'last_publication': times[1],
                           'legacy_sessions_without_prefix_verification': legacy,
                           'pending_aggregates': pending, 'aggregate_progress': aggregate_progress(store), 'problems': problems,
                           'note': 'Counts describe recorded source states. Worker/supervisor heartbeats do not certify source freshness; individual session checks compare current file metadata.'}
    if getattr(store, '_scoped_vectors', None) is not None:
        result['retrieval_index'] = store._scoped_vectors.status()
    if cfg is not None:
        from .detection import detection_status
        try:
            result['ingestion']['detection'] = detection_status(cfg)
        except Exception as exc:
            result['ingestion']['detection'] = {'status': 'unavailable', 'error': safe_error(exc)}
    return result
