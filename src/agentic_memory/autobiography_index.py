"""Recoverable lexical projection; authored heads are the durable repair queue."""
import sqlite3
import time

from .autobiography import PersonalError


def index_record(store, record_id):
    revision=None
    try:
        with store.transaction():
            record=store._get(record_id);revision=record['revision']
            store.db.execute('DELETE FROM autobio_search WHERE record_id=?',(record_id,))
            if record['state']=='active':
                store.db.execute('INSERT INTO autobio_search(record_id,body) VALUES(?,?)',(record_id,record['title']+'\n'+record['body']))
            store.db.execute('INSERT OR REPLACE INTO autobio_index_state VALUES(?,?)',(record_id,revision))
            store.db.execute('DELETE FROM autobio_index_errors WHERE record_id=?',(record_id,))
        return True
    except (sqlite3.Error,PersonalError) as exc:
        # A missing/current-revision mismatch remains pending even if an I/O
        # failure prevents writing this diagnostic. Never undo an authored commit.
        if revision is not None:
            try:
                with store.transaction():
                    head=store._head(record_id)
                    if head['revision']==revision:
                        old=store.db.execute('SELECT attempts FROM autobio_index_errors WHERE record_id=? AND revision=?',(record_id,revision)).fetchone()
                        attempts=(old[0] if old else 0)+1
                        store.db.execute('INSERT OR REPLACE INTO autobio_index_errors VALUES(?,?,?,?,?,?)',
                            (record_id,revision,attempts,type(exc).__name__,store.clock()+min(86400,300*2**min(attempts,8)),store.clock()))
            except (sqlite3.Error,PersonalError):pass
        return False


def index_status(store, *, limit=20):
    p=store.principal
    with store.snapshot():
        rows=store.db.execute('''SELECT r.id AS record_id,r.revision,i.revision AS indexed_revision,
            e.error,e.attempts,e.retry_after FROM autobio_records r
            LEFT JOIN autobio_index_state i ON i.record_id=r.id
            LEFT JOIN autobio_index_errors e ON e.record_id=r.id AND e.revision=r.revision
            WHERE r.owner=? AND r.agent=? AND r.audience=? AND (i.revision IS NULL OR i.revision!=r.revision OR (r.state='active' AND NOT EXISTS(SELECT 1 FROM autobio_search f WHERE f.record_id=r.id)))
            ORDER BY r.updated,r.id LIMIT ?''',(p.owner,p.agent,p.audience,max(1,min(int(limit),100)))).fetchall()
        count=store.db.execute('''SELECT COUNT(*) FROM autobio_records r LEFT JOIN autobio_index_state i ON i.record_id=r.id
            WHERE r.owner=? AND r.agent=? AND r.audience=? AND (i.revision IS NULL OR i.revision!=r.revision OR (r.state='active' AND NOT EXISTS(SELECT 1 FROM autobio_search f WHERE f.record_id=r.id)))''',
            (p.owner,p.agent,p.audience)).fetchone()[0]
        return {'state':'pending' if count else 'current','pending_count':count,'pending':[dict(r) for r in rows],
                'repair':'automatic bounded maintenance or operator repair-index; exact reads remain available'}


def repair_index(store, *, limit=16, retry_now=False, max_seconds=2):
    store.principal.require('operator','extractor','dream')
    if retry_now:store.principal.require('operator')
    deadline=time.monotonic()+min(10,max(.01,float(max_seconds)));p=store.principal
    with store.snapshot():
        rows=store.db.execute('''SELECT r.id FROM autobio_records r LEFT JOIN autobio_index_state i ON i.record_id=r.id
            LEFT JOIN autobio_index_errors e ON e.record_id=r.id AND e.revision=r.revision
            WHERE r.owner=? AND r.agent=? AND r.audience=? AND (i.revision IS NULL OR i.revision!=r.revision OR (r.state='active' AND NOT EXISTS(SELECT 1 FROM autobio_search f WHERE f.record_id=r.id)))
            AND (? OR e.retry_after IS NULL OR e.retry_after<=?) ORDER BY COALESCE(e.updated,0),r.updated,r.id LIMIT ?''',
            (p.owner,p.agent,p.audience,bool(retry_now),store.clock(),max(1,min(int(limit),100)))).fetchall()
    attempted=repaired=0
    for row in rows:
        if time.monotonic()>=deadline:break
        attempted+=1;repaired+=bool(index_record(store,row[0]))
    return {'attempted':attempted,'repaired':repaired,'index':index_status(store),'model_calls':0}
