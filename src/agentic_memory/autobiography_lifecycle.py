"""Append-only lifecycle metadata over immutable authored revision snapshots.

Age never changes validity. Only an explicit authored transition changes the head.
Operator privacy erasure remains the deliberate exception to retention.
"""
import uuid

from .autobiography import PersonalError, canonical, digest

STATES = ('active', 'retired', 'outdated', 'superseded', 'retracted')


def install(db):
    db.execute('''CREATE TABLE IF NOT EXISTS autobio_lifecycle(
        record_id TEXT NOT NULL REFERENCES autobio_records(id) ON DELETE CASCADE,
        revision INTEGER NOT NULL, previous_revision INTEGER,
        previous_state TEXT, state TEXT NOT NULL,
        superseded_by TEXT, superseded_by_revision INTEGER,
        PRIMARY KEY(record_id, revision),
        FOREIGN KEY(record_id, revision) REFERENCES autobio_revisions(record_id, revision) ON DELETE CASCADE)''')
    # Parent deletion remains operator.erase's explicit privacy escape hatch.
    # Block direct deletes and INSERT OR REPLACE as well as ordinary updates.
    for table in ('autobio_revisions', 'autobio_lifecycle'):
        db.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_immutable
            BEFORE UPDATE ON {table} BEGIN
            SELECT RAISE(ABORT, 'committed autobiography history is immutable'); END''')
        db.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_replace
            BEFORE INSERT ON {table} WHEN EXISTS(
                SELECT 1 FROM {table} WHERE record_id=NEW.record_id AND revision=NEW.revision)
            BEGIN SELECT RAISE(ABORT, 'committed autobiography history is immutable'); END''')
        db.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_delete
            BEFORE DELETE ON {table} WHEN EXISTS(
                SELECT 1 FROM autobio_records WHERE id=OLD.record_id)
            BEGIN SELECT RAISE(ABORT, 'use explicit record privacy erasure'); END''')


def append_event(store, record_id, revision, previous, state, replacement=None):
    store.db.execute('INSERT INTO autobio_lifecycle VALUES(?,?,?,?,?,?,?)',
        (record_id, revision, previous['revision'] if previous else None,
         previous['state'] if previous else None, state,
         replacement['id'] if replacement else None,
         replacement['revision'] if replacement else None))


def write_transition(store, edit, commit_id, origin):
    """Internal atomic writer, also used by fenced reconciliation batches."""
    allowed = {'operation', 'record_id', 'expected_revision', 'state', 'reason',
               'superseded_by', 'superseded_by_revision'}
    if set(edit) - allowed:
        raise PersonalError('invalid', 'lifecycle edits cannot rewrite content or evidence')
    record_id, expected_revision = edit.get('record_id'), edit.get('expected_revision')
    state, reason = edit.get('state'), edit.get('reason')
    superseded_by, superseded_by_revision = edit.get('superseded_by'), edit.get('superseded_by_revision')
    if state not in STATES or not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise PersonalError('invalid', 'a known lifecycle state and bounded reason are required')
    if type(expected_revision) is not int or expected_revision < 1:
        raise PersonalError('invalid', 'expected_revision must be a positive integer')
    from .personal_privacy import require_no_credentials
    require_no_credentials(reason)
    if state == 'superseded':
        if not superseded_by or type(superseded_by_revision) is not int:
            raise PersonalError('invalid', 'supersession requires a replacement record and revision')
    elif superseded_by is not None or superseded_by_revision is not None:
        raise PersonalError('invalid', 'replacement is only valid for supersession')
    head = store._head(record_id, write=True)
    store.require_foundation_author(head['segment'])
    if head['revision'] != expected_revision:
        raise PersonalError('conflict', 'record revision changed before lifecycle transition')
    replacement = None
    if superseded_by is not None:
        replacement = store._head(superseded_by, write=True)
        if replacement['id'] == record_id:
            raise PersonalError('invalid', 'a record cannot supersede itself')
        if replacement['revision'] != superseded_by_revision or replacement['state'] != 'active':
            raise PersonalError('conflict', 'replacement must be the expected active revision')
    revision, now = expected_revision + 1, store.clock()
    # SQL copies the exact committed evidence bytes, not a recomputed source view.
    store.db.execute('''INSERT INTO autobio_revisions
        SELECT record_id,?,body,title,?,?,?, ?,attribution,evidence,?,?
        FROM autobio_revisions WHERE record_id=? AND revision=?''',
        (revision, state, reason, store.principal.session_id, origin, now, commit_id, record_id, expected_revision))
    append_event(store, record_id, revision, head, state, replacement)
    store.db.execute('UPDATE autobio_records SET revision=?,state=?,updated=? WHERE id=?',
                     (revision, state, now, record_id))
    return {'record_id': record_id, 'revision': revision}


def transition(store, record_id, *, expected_revision, state, reason,
               idempotency_key, superseded_by=None, superseded_by_revision=None):
    """Append a status revision without rewriting content/evidence; age is not validity."""
    store.principal.require('author', 'operator', 'dream')
    edit = dict(operation='lifecycle', record_id=record_id, expected_revision=expected_revision,
                state=state, reason=reason, superseded_by=superseded_by,
                superseded_by_revision=superseded_by_revision)
    payload_hash = digest({'lifecycle': edit, 'actor': vars(store.principal)})
    with store.transaction():
        receipt = store._receipt(idempotency_key, payload_hash)
        if receipt:
            return receipt
        commit_id, now, p = uuid.uuid4().hex, store.clock(), store.principal
        revision = write_transition(store, edit, commit_id, 'lifecycle')
        store.validate_core_budget()
        receipt = {'commit_id': commit_id, 'edits': [revision], 'committed_at': now}
        store.db.execute('INSERT INTO autobio_commits VALUES(?,?,?,?,?,?,?)',
            (commit_id, p.owner, p.agent, store._request_key(idempotency_key), payload_hash, canonical(receipt), now))
    store.index_record(record_id)
    return receipt
