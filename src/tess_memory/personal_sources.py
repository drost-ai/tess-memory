"""Personal source policy and durable, contiguous finalized-unit cursors.

Hosts built on one engine share a session format. Format/name/cwd is NOT proof
of host identity: only explicitly configured personal session roots confer it.
No coding discovery, exclusion, ingestion cursor or archive is modified here.
"""
from datetime import datetime
import json
from pathlib import Path

from .autobiography import PersonalError, canonical, digest
from .autobiography_config import NATIVE_SOURCE_MODE


def parse_boundary(value):
    if not isinstance(value, str):
        raise ValueError('personal.source_since must be an ISO-8601 timestamp with timezone')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None:
            raise ValueError('timezone required')
        return result.timestamp()
    except ValueError as exc:
        raise ValueError('personal.source_since must be an ISO-8601 timestamp with timezone') from exc


def after_boundary(turn, boundary):
    if boundary is None:
        return True
    # Missing timestamps are unknown, not evidence of a post-cutoff message.
    try:
        value = turn.ts
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            stamp = value / 1000 if value > 100_000_000_000 else value
        else:
            stamp = parse_boundary(value)
        return stamp >= boundary
    except (ValueError, TypeError, OverflowError):
        return False


def selected_sources(cfg):
    from .config import DataStore
    p = cfg.personal
    if p.source_mode == NATIVE_SOURCE_MODE:
        if not p.source_paths:
            raise PersonalError('invalid', 'explicit native source_paths required')
        roots = sorted({Path(x).expanduser().resolve() for x in p.source_paths})
        return [DataStore('personal-tess-' + digest(str(root))[:16], root,
                          type='tess-core', project=p.agent, exclude=[]) for root in roots]
    selected = [d for d in cfg.data_stores if d.name in p.source_names]
    if not selected or set(p.source_names) - {d.name for d in selected}:
        raise PersonalError('invalid', 'personal extraction requires an explicit valid source allowlist')
    return selected


def source_origin(metadata):
    md = metadata.get('metadata') or {}
    internal = md.get('personalMemoryOrigin')
    parent = md.get('parentSessionId')
    return {'kind': 'internal' if internal in ('dream', 'extractor', 'reconciler', 'follow_up_reconciler', 'topic_reconciler', 'care', 'research') else 'child' if parent else 'conversation',
            'personal_memory_origin': internal, 'parent_session_id': parent,
            # The canonical engine metadata does not establish channel provenance.
            'channel': None, 'channel_evidence': 'not_recorded_in_canonical_session'}


class SourceCursors:
    def __init__(self, store):
        self.store = store
        with store.transaction():
            store.db.execute('''CREATE TABLE IF NOT EXISTS autobio_source_cursors (
                source_id TEXT PRIMARY KEY, generation TEXT NOT NULL,
                boundaries TEXT NOT NULL, units TEXT NOT NULL,
                completed_units INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)''')

    def boundaries(self, logical):
        row = self.store.db.execute('SELECT boundaries FROM autobio_source_cursors WHERE source_id=?', (logical,)).fetchone()
        return json.loads(row[0]) if row else []

    def prepared(self, logical, generation, boundaries, units):
        self.store.db.execute('''INSERT OR REPLACE INTO autobio_source_cursors
            VALUES(?,?,?,?,0,?)''', (logical, generation, canonical(boundaries), canonical(units), self.store.clock()))
        return self.advance(logical)

    def advance(self, logical):
        s = self.store
        row = s.db.execute('SELECT * FROM autobio_source_cursors WHERE source_id=?', (logical,)).fetchone()
        if not row:
            return None
        units = json.loads(row['units']); completed = 0
        for unit in units:
            if unit['disposition'] == 'excluded':
                completed += 1
                continue
            source_id = logical + '/unit/' + str(unit['index'])
            current = s.db.execute('SELECT generation FROM autobio_sources WHERE id=?', (source_id,)).fetchone()
            if not current or current[0] != unit['generation']:
                break
            parts = s.db.execute('''SELECT start,end,state FROM autobio_extraction_parts
                WHERE source_id=? AND generation=? ORDER BY start,end,id''', (source_id, unit['generation'])).fetchall()
            end = 0
            for part in parts:
                if part['start'] > end or part['state'] != 'published':
                    break
                end = max(end, part['end'])
            if end < unit['length']:
                break
            completed += 1
        s.db.execute('UPDATE autobio_source_cursors SET completed_units=?,updated=? WHERE source_id=?',
                     (completed, s.clock(), logical))
        return {'source_id': logical, 'generation': row['generation'], 'completed_units': completed,
                'prepared_units': len(units), 'complete': completed == len(units)}

    def manifest(self, logical):
        row = self.store.db.execute('SELECT * FROM autobio_source_cursors WHERE source_id=?', (logical,)).fetchone()
        if not row:
            return None
        return {'source_id': logical, 'generation': row['generation'],
                'completed_units': row['completed_units'], 'prepared_units': len(json.loads(row['units']))}


MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024


def write_source_snapshot(cfg, store, ranges, cursors):
    """Freeze only this claim's bounded ranges, never expose an unbounded log.

    The path is an immutable private JSON artifact, not permission to read the
    original sessions. Append after claim is considered by the next claim.
    Cursor values are observed finalized progress, not completion receipts.
    """
    import hashlib
    import os
    import tempfile
    entries = []
    for row in ranges:
        md = row['metadata']
        entries.append({'range_id': row['range_id'], 'source_id': row['source_id'],
                        'generation': row['generation'], 'start': row['start'], 'end': row['end'],
                        'text': row['text'], 'source_path': md.get('path'),
                        'session_id': md.get('session_id'), 'source_origin': md.get('source_origin'),
                        'pending_tail': md.get('pending_tail'), 'unit_kind': md.get('unit_kind'),
                        'cursor': cursors.manifest(md.get('logical_source'))})
    payload = {'version': 1, 'coordinate_system': 'cleaned unit character offsets; half-open',
               'source_mode': cfg.personal.source_mode, 'source_since': cfg.personal.source_since,
               'ranges': entries}
    raw = canonical(payload).encode('utf-8')
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise PersonalError('invalid', 'source snapshot exceeds maximum bytes; request fewer ranges')
    sha = hashlib.sha256(raw).hexdigest()
    root = cfg.data_dir / 'personal-extraction-snapshots' / digest([store.principal.owner,store.principal.agent,store.principal.audience])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = root / (sha + '.json')
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as file:
            temporary = file.name
            file.write(raw); file.flush(); os.fsync(file.fileno())
        os.replace(temporary, target)
    return {'version': 1, 'snapshot_path': str(target), 'sha256': sha, 'bytes': len(raw),
            'max_bytes': MAX_SNAPSHOT_BYTES, 'range_count': len(entries),
            'ranges': [{k:v for k,v in e.items() if k != 'text'} for e in entries]}


def metadata_eligible(cfg, metadata):
    if cfg.personal.source_mode != NATIVE_SOURCE_MODE:
        return True
    return (metadata.get('personal_source_mode') == NATIVE_SOURCE_MODE
            and metadata.get('source_root') in {str(Path(x).expanduser().resolve()) for x in cfg.personal.source_paths}
            and metadata.get('source_policy') == digest([cfg.personal.source_mode,cfg.personal.source_since]))


def reconciliation_source_manifest(cfg, store, candidates, cursors):
    ranges=[]
    for candidate in candidates:
        for index, evidence in enumerate(candidate['evidence']):
            source=store.db.execute('SELECT metadata FROM autobio_sources WHERE id=? AND owner=? AND agent=? AND audience=?',
                (evidence['source_id'],store.principal.owner,store.principal.agent,store.principal.audience)).fetchone()
            if not source:
                continue
            ranges.append({'range_id':candidate['id']+':evidence:'+str(index),
                'source_id':evidence['source_id'],'generation':evidence['generation'],
                'start':evidence['start'],'end':evidence['end'],
                'text':evidence.get('quote',candidate['proposal'].get('quote','')),
                'metadata':json.loads(source[0])})
    manifest=write_source_snapshot(cfg,store,ranges,cursors)
    manifest['purpose']='reconciliation_evidence'
    return manifest


def verify_range_source(cfg, store, source_id, generation):
    """Verify a frozen unit, allowing benign append/metadata refresh in Tess.

    Never bless a changed unit just because its containing session still exists.
    A failed physical revision check falls back to exact canonical-unit proof,
    using the saved boundary layout, source policy and origin. Call inside the
    finalization transaction so updated proof and publication are atomic.
    """
    from dataclasses import replace
    from .readers import CompletedSourceReader
    from .source_journal import SourceRevision
    source=store.db.execute('SELECT * FROM autobio_sources WHERE id=?',(source_id,)).fetchone()
    if not source or source['generation'] != generation:
        raise PersonalError('stale_source','source generation changed')
    md=json.loads(source['metadata'])
    if not metadata_eligible(cfg,md):
        raise PersonalError('stale_source','source is no longer eligible')
    revisions=md.get('source_revisions',[])
    if all(SourceRevision(**r).verify() for r in revisions):
        return
    if cfg.personal.source_mode != NATIVE_SOURCE_MODE:
        raise PersonalError('stale_source','source prefix changed')
    try:
        ds=next(d for d in selected_sources(cfg) if d.name==md.get('source_store'))
        path=Path(md['path'])
        if not path.resolve().is_relative_to(ds.path.resolve()):
            raise ValueError('source escaped root')
        metadata=json.loads(path.read_text())
        origin=source_origin(metadata)
        if origin != md.get('source_origin') or origin['kind']=='internal':
            raise ValueError('source origin changed')
        snapshot=CompletedSourceReader(replace(cfg,data_stores=[ds],recovery_ingestion=True)).read(ds,path,ds.project,None)
        if snapshot.transcript is None or snapshot.transcript.session_id != md['session_id']:
            raise ValueError('source identity changed')
        if snapshot.message_indices != list(range(len(snapshot.message_indices))):
            raise ValueError('source index gap')
        saved=store.db.execute('SELECT boundaries FROM autobio_source_cursors WHERE source_id=?',(md['logical_source'],)).fetchone()
        if not saved:
            raise ValueError('no saved source boundaries')
        snapshot.retain_boundaries(json.loads(saved[0]))
        key=snapshot.unit_keys()[md['unit']]
        if digest([key,md['source_policy'],origin]) != md.get('raw_unit_key'):
            raise ValueError('claimed source unit changed')
        latest_metadata=json.loads(path.read_text())
        if source_origin(latest_metadata)!=origin:
            raise ValueError('source origin changed during validation')
        if store.principal.reads:
            from .autobiography_config import session_audience
            if session_audience(latest_metadata,cfg.personal)!=store.principal.audience:
                raise ValueError('source audience changed')
        if not all(r.verify() for r in snapshot.revisions):
            raise ValueError('source changed during validation')
        md['source_revisions']=[vars(r) for r in snapshot.revisions]
        md['pending_tail']=snapshot.pending_tail
        store.db.execute('UPDATE autobio_sources SET metadata=?,updated=? WHERE id=?',
                         (canonical(md),store.clock(),source_id))
    except Exception as exc:
        if isinstance(exc,PersonalError):raise
        raise PersonalError('stale_source','claimed source unit changed or is unavailable') from exc
