"""Read-only, explicitly trusted the host native-creation provenance.

A canonical createdAt (or filesystem birthtime) is NOT creation authority.
The operator opts into a host ledger whose session bindings are written only by
native create/open, never by an import process. The ledger path and trusted source root are explicit and independently bound.
It is not a hostile-writer defense.
"""
from pathlib import Path
import sqlite3


def creation_metadata(meta):
    md = meta.get('metadata') or {}
    # Fail closed for derived/imported histories, including unfamiliar markers.
    derived = any(('import' in k.lower() or 'fork' in k.lower()) and bool(v)
                  for obj in (meta, md) for k, v in obj.items())
    derived = derived or bool(md.get('parentSessionId')) or bool(md.get('compactions'))
    return {'id': meta.get('id'), 'created_at': meta.get('createdAt'), 'derived': derived}


def native_creation_proof(config, ds, current, activated_at, observed_at, stamp):
    """Return independently corroborated creation, or None; never create a DB."""
    ledger = config.personal.native_creation_ledger
    if not ledger or config.personal.source_mode != 'native' or observed_at <= activated_at:
        return None
    try:
        ledger = Path(ledger).expanduser().resolve(strict=True)
        root = ds.path.resolve()
        trusted_root = config.provenance_source_root
        if trusted_root is None:
            if config.embedded:
                return None
            # Preserve the original standalone trust boundary, never any root.
            trusted_root = ledger.parent / 'sessions'
        if root != Path(trusted_root).resolve():
            return None
        path = Path(current['path'])
        creation = current['creation']
        # Compare resolved locations: the configured root may sit behind a symlink.
        if (path.parent.resolve() != root
            or path.stem != creation['id'] or creation['derived']
            or current['source_origin']['kind'] != 'conversation'
            or not current['pristine_stream']):
            return None
        engine_at = stamp(creation['created_at'])
        first_event = current.get('first_event_at')
        if (engine_at is None or not activated_at < engine_at <= observed_at
            or (first_event is not None and first_event < engine_at)):
            return None
        # mode=ro prevents accidental creation/writes. Do not use immutable=1:
        # a live WAL must participate in the read snapshot.
        db = sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True, timeout=1)
        try:
            rows = db.execute('SELECT id, created_at, deleted_at FROM sessions WHERE engine_session_id=?',
                              (creation['id'],)).fetchall()
        finally:
            db.close()
        if len(rows) != 1 or rows[0][2] is not None:
            return None
        host_id, host_created, _ = rows[0]
        host_at = stamp(host_created)
        if not isinstance(host_id, str) or not host_id or host_at is None:
            return None
        if not activated_at < host_at <= engine_at:
            return None
        return {'version': 1, 'ledger': str(ledger), 'host_id': host_id,
                'engine_id': creation['id'], 'host_created_at': host_at,
                'engine_created_at': engine_at}
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        return None
