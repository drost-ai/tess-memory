"""Personal candidate read surface; publication is independent of coding ingestion."""
import json
import uuid
from .autobiography import PersonalError, SEGMENTS, ATTRIBUTIONS, canonical, digest

INBOX_SCHEMA='''
CREATE TABLE IF NOT EXISTS autobio_candidates(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, agent TEXT NOT NULL, audience TEXT NOT NULL,
 source_id TEXT NOT NULL, generation TEXT NOT NULL, proposal TEXT NOT NULL, evidence TEXT NOT NULL,
 dedupe TEXT NOT NULL UNIQUE, state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
 review_after REAL NOT NULL DEFAULT 0, cycle_id TEXT, fence INTEGER, lease_until REAL,
 attempts INTEGER NOT NULL DEFAULT 0, disposition TEXT, resulting_revisions TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS autobio_extractions(
 source_id TEXT PRIMARY KEY, generation TEXT NOT NULL, keys_json TEXT NOT NULL,
 state TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_extraction_parts(
 id TEXT PRIMARY KEY, source_id TEXT NOT NULL, generation TEXT NOT NULL,
 start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL, state TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0, retry_after REAL NOT NULL DEFAULT 0, error TEXT,
 result TEXT, receipt TEXT, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_model_cache(
 id TEXT PRIMARY KEY, result TEXT NOT NULL, source_id TEXT, record_ids TEXT NOT NULL DEFAULT '[]', created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_baseline_paths(path TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS autobio_source_errors(
 path TEXT PRIMARY KEY, signature TEXT NOT NULL, attempts INTEGER NOT NULL, state TEXT NOT NULL,
 retry_after REAL NOT NULL, error TEXT NOT NULL);
'''


class PersonalInbox:
    def __init__(self,store):
        self.store=store

    def initialize(self):
        with self.store.transaction():
            for statement in INBOX_SCHEMA.split(';'):
                if statement.strip():self.store.db.execute(statement)

    def publish(self,part_id,proposals):
        self.store.principal.require('extractor','operator')
        if not isinstance(proposals,list) or len(proposals)>8:
            raise PersonalError('invalid','return at most eight candidates for a source part')
        s=self.store;p=s.principal
        with s.transaction():
            part=s.db.execute('SELECT * FROM autobio_extraction_parts WHERE id=?',(part_id,)).fetchone()
            if not part:raise PersonalError('not_found','source part unavailable')
            source=s.db.execute('SELECT * FROM autobio_sources WHERE id=? AND owner=? AND agent=? AND audience=?',
                (part['source_id'],p.owner,p.agent,p.audience)).fetchone()
            if not source:raise PersonalError('not_found','source part unavailable')
            if part['state']=='published':
                if json.loads(part['result'])!=proposals:raise PersonalError('idempotency_mismatch','source part already has different published candidates')
                return json.loads(part['receipt'])
            if source['generation']!=part['generation'] or s.is_suppressed(part['source_id']):
                raise PersonalError('stale_source','source evidence changed or was suppressed')
            metadata=json.loads(source['metadata']);made=[]
            for proposal in proposals:
                if not isinstance(proposal,dict) or set(proposal)-{'segment','title','body','attribution','quote','target_record_id'}:
                    raise PersonalError('invalid','invalid candidate fields')
                if proposal.get('segment') not in SEGMENTS or proposal.get('attribution') not in ATTRIBUTIONS:
                    raise PersonalError('invalid','candidate segment or attribution is invalid')
                for key,maximum in [('title',200),('body',4000),('quote',2000)]:
                    if not isinstance(proposal.get(key),str) or not proposal[key].strip() or len(proposal[key])>maximum:
                        raise PersonalError('invalid',f'candidate {key} is missing or oversized')
                quote=proposal['quote'];offset=part['text'].find(quote)
                if offset<0:raise PersonalError('invalid_evidence','candidate quote is absent from the supplied source')
                start=part['start']+offset;end=start+len(quote)
                if proposal['attribution']=='user_statement' and not any(
                    r['role']=='user' and r['start']<=start and r['end']>=end for r in metadata.get('roles',[])):
                    raise PersonalError('invalid_evidence','user-statement quote is not within a user message')
                if proposal.get('target_record_id'):s._head(proposal['target_record_id'])
                evidence=[{'source_id':source['id'],'generation':source['generation'],'start':start,'end':end,'attribution':proposal['attribution']}]
                s._evidence(evidence)
                key=digest([source['id'],source['generation'],start,end,proposal,1])
                cid=uuid.uuid4().hex
                changed=s.db.execute('''INSERT OR IGNORE INTO autobio_candidates
                    (id,owner,agent,audience,source_id,generation,proposal,evidence,dedupe,state,created,updated)
                    VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?)''',
                    (cid,p.owner,p.agent,p.audience,source['id'],source['generation'],canonical(proposal),canonical(evidence),key,s.clock(),s.clock())).rowcount
                if changed:made.append(cid)
            receipt={'published':True,'candidate_ids':made}
            s.db.execute("UPDATE autobio_extraction_parts SET state='published',result=?,receipt=?,updated=? WHERE id=?",
                (canonical(proposals),canonical(receipt),s.clock(),part_id))
            return receipt

    def fail(self,part_id,code):
        self.store.principal.require('extractor','operator')
        with self.store.transaction():
            p=self.store.principal
            row=self.store.db.execute('''SELECT part.attempts,part.state FROM autobio_extraction_parts part JOIN autobio_sources src ON src.id=part.source_id
                WHERE part.id=? AND src.owner=? AND src.agent=? AND src.audience=?''',(part_id,p.owner,p.agent,p.audience)).fetchone()
            if not row:raise PersonalError('not_found','source part unavailable')
            if row['state'] in ('published','stale'):return
            attempts=row[0]+1
            self.store.db.execute('UPDATE autobio_extraction_parts SET attempts=?,state=?,retry_after=?,error=?,updated=? WHERE id=?',
                (attempts,'failed' if attempts>=3 else 'pending',self.store.clock()+min(86400,300*2**attempts),str(code)[:120],self.store.clock(),part_id))

    def list(self,*,state='pending',limit=20,after=''):
        if state not in ('pending','claimed','accepted','merged','dismissed','deferred','stale','failed'):
            raise PersonalError('invalid','unknown candidate state')
        with self.store.transaction(write=False):
            exists=self.store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='autobio_candidates'").fetchone()
            if not exists:
                return {'candidates':[],'next_cursor':None,'extraction_initialized':False}
            p=self.store.principal
            rows=self.store.db.execute('''SELECT * FROM autobio_candidates WHERE owner=? AND agent=? AND audience=?
                AND state=? AND id>? ORDER BY id LIMIT ?''',(p.owner,p.agent,p.audience,state,after,max(1,min(int(limit),100))+1)).fetchall()
            limit=max(1,min(int(limit),100))
            candidates=[]
            for row in rows[:limit]:
                value=dict(row)
                for key in ['proposal','evidence','resulting_revisions']:value[key]=json.loads(value[key])
                candidates.append(value)
            return {'candidates':candidates,'next_cursor':rows[limit-1]['id'] if len(rows)>limit else None,'extraction_initialized':True}
