"""Explicit operator-selected Markdown seeds; no automatic historical imports."""
from pathlib import Path
import hashlib
import uuid
from .autobiography import PersonalError, SEGMENTS, MAX_BODY, canonical, digest


def preview_seeds(files):
    if not isinstance(files,dict) or not files or len(files)>6 or set(files)-set(SEGMENTS):
        raise PersonalError('invalid','select at most one Markdown seed for each known segment')
    records=[]
    for segment,name in sorted(files.items()):
        path=Path(name).expanduser().resolve()
        if not path.is_file() or path.stat().st_size>MAX_BODY:
            raise PersonalError('invalid','seed must be a bounded regular Markdown file')
        data=path.read_bytes();body=data.decode('utf-8')
        from .personal_privacy import require_no_credentials
        require_no_credentials(body)
        if not body.strip():raise PersonalError('invalid','seed body is empty')
        records.append({'segment':segment,'path':str(path),'body':body,
                        'source_id':'import:'+hashlib.sha256(str(path).encode()).hexdigest(),
                        'generation':hashlib.sha256(data).hexdigest()})
    return {'records':records,'preview_hash':digest(records),'default_state':'retired',
            'notice':'Inherited source material is not verified current embodiment. Review before activating the core.'}


def import_seeds(store,files,*,preview_hash,idempotency_key,activate=False):
    store.principal.require('operator')
    preview=preview_seeds(files)
    if preview['preview_hash']!=preview_hash:
        raise PersonalError('conflict','seed files changed after preview')
    payload_hash=digest({'preview':preview_hash,'activate':activate,'actor':vars(store.principal)})
    p=store.principal
    with store.transaction():
        receipt=store._receipt(idempotency_key,payload_hash)
        if receipt:return receipt
        commit_id=uuid.uuid4().hex;edits=[]
        for seed in preview['records']:
            if store.is_suppressed(seed['source_id']):raise PersonalError('suppressed','seed source was erased and suppressed')
            store.db.execute('INSERT OR REPLACE INTO autobio_sources VALUES(?,?,?,?,?,?,?,?)',
                (seed['source_id'],p.owner,p.agent,p.audience,seed['generation'],1,
                 canonical({'kind':'inherited_document','path':seed['path']}),store.clock()))
            evidence={'source_id':seed['source_id'],'generation':seed['generation'],'start':0,'end':1,'attribution':'inherited'}
            edits.append(store._write({'expected_revision':0,'segment':seed['segment'],'title':'Inherited '+seed['segment'],
                'body':seed['body'],'reason':'Explicit previewed seed import; current claims remain subject to review.',
                'attribution':'inherited','evidence':[evidence],'state':'active' if activate else 'retired'},commit_id,'import'))
        store.validate_core_budget()
        receipt={'commit_id':commit_id,'edits':edits,'committed_at':store.clock(),'origin':'import'}
        store.db.execute('INSERT INTO autobio_commits VALUES(?,?,?,?,?,?,?)',
            (commit_id,p.owner,p.agent,store._request_key(idempotency_key),payload_hash,canonical(receipt),store.clock()))
    for edit in edits:store.index_record(edit['record_id'])
    return receipt
