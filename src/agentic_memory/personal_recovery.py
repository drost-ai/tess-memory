"""Explicit, scoped recovery of retained dead letters; never starts model work."""
from pathlib import Path
import uuid

from .autobiography import PersonalError, canonical
from .autobiography_inbox import PersonalInbox


def retry_failed(config, store, *, kind, target, reason):
    p=store.principal;p.require('operator')
    if kind not in ('candidate','part','source') or not isinstance(target,str) or not target:
        raise PersonalError('invalid','choose one candidate, part or source target')
    if not isinstance(reason,str) or not 0<len(reason.strip())<=1000:
        raise PersonalError('invalid','a bounded recovery reason is required')
    from .personal_privacy import require_no_credentials
    require_no_credentials(reason)
    PersonalInbox(store).initialize()
    with store.transaction():
        store.db.execute('''CREATE TABLE IF NOT EXISTS autobio_recovery_receipts(
            id TEXT PRIMARY KEY,owner TEXT,agent TEXT,audience TEXT,kind TEXT,target TEXT,reason TEXT,previous TEXT,created REAL)''')
        if kind=='source':
            path=Path(target).expanduser().resolve()
            allowed=[d.path.resolve() for d in config.data_stores if d.name in config.personal.source_names]
            if not any(path.is_relative_to(root) for root in allowed):
                raise PersonalError('not_found','recovery target unavailable')
            target=str(path)
            row=store.db.execute("SELECT * FROM autobio_source_errors WHERE path=? AND state='failed'",(target,)).fetchone()
            if not row:raise PersonalError('not_found','failed source unavailable')
            store.db.execute("UPDATE autobio_source_errors SET state='retry',attempts=0,retry_after=0 WHERE path=?",(target,))
        else:
            table='autobio_candidates' if kind=='candidate' else 'autobio_extraction_parts'
            row=store.db.execute(f'''SELECT item.* FROM {table} item JOIN autobio_sources src ON src.id=item.source_id
                WHERE item.id=? AND src.owner=? AND src.agent=? AND src.audience=? AND item.state='failed'
                AND item.generation=src.generation''',(target,p.owner,p.agent,p.audience)).fetchone()
            if not row or store.is_suppressed(row['source_id']):
                raise PersonalError('not_found','failed current work unavailable')
            if kind=='candidate':
                store.db.execute("UPDATE autobio_candidates SET state='pending',attempts=0,review_after=0,cycle_id=NULL,fence=NULL,lease_until=NULL,updated=? WHERE id=?",(store.clock(),target))
            else:
                store.db.execute("UPDATE autobio_extraction_parts SET state='pending',attempts=0,retry_after=0,updated=? WHERE id=?",(store.clock(),target))
        receipt=uuid.uuid4().hex
        previous={k:row[k] for k in ('state','attempts')}
        store.db.execute('INSERT INTO autobio_recovery_receipts VALUES(?,?,?,?,?,?,?,?,?)',
            (receipt,p.owner,p.agent,p.audience,kind,target,reason,canonical(previous),store.clock()))
    return {'receipt_id':receipt,'queued':True,'kind':kind,'target':target,'model_calls':0}
