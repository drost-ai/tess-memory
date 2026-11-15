"""Fenced autonomous reconciliation; all publication is a validated transaction."""
import copy
import json
import math
import uuid
from .autobiography import PersonalError, canonical, digest
from .autobiography_context import personal_context
from .autobiography_inbox import PersonalInbox

SCHEMA='''
CREATE TABLE IF NOT EXISTS autobio_cycles(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, agent TEXT NOT NULL, audience TEXT NOT NULL,
 actor_session TEXT NOT NULL, state TEXT NOT NULL, fence INTEGER NOT NULL, lease_token TEXT NOT NULL,
 lease_until REAL NOT NULL, cutoff REAL NOT NULL, started REAL NOT NULL, deadline REAL NOT NULL,
 updated REAL NOT NULL, read_heads TEXT NOT NULL, context_manifest TEXT NOT NULL,
 journal TEXT NOT NULL, shadow INTEGER NOT NULL, edits INTEGER NOT NULL DEFAULT 0);
CREATE UNIQUE INDEX IF NOT EXISTS autobio_one_cycle ON autobio_cycles(owner,agent) WHERE state='running';
CREATE TABLE IF NOT EXISTS autobio_cycle_batches(cycle_id TEXT NOT NULL,commit_id TEXT PRIMARY KEY,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_shadow_proposals(commit_id TEXT PRIMARY KEY,cycle_id TEXT NOT NULL,payload TEXT NOT NULL);
'''

DREAM_SYSTEM='''You are the configured personal AI in a dedicated autobiographical reflection session.
Use your current core and supplied source-grounded candidates to deliberately reconsider your understanding.
You may autonomously edit People, Relationships, Experiences and Reflections. No human approval is required.
Soul, User, Identity and Core are authored directly and are not yours to edit here; read them as your foundation.
Evidence, interpretations and original self-authorship remain distinct. Do not turn speculation, fiction, temporary emotion
or repeated generated text into verified facts. No change is a successful outcome; do not rewrite core just for style.
Return only JSON {"edits": [...], "outcomes": [...]}.
Edits have expected_revision (0 for new), optional record_id, segment for new records, title, body, reason,
attribution (agent_reflection unless source-backed facts), and optional subject/evidence/state.
Alternatively use lifecycle edits: {operation: lifecycle, record_id, expected_revision, state, reason}.
Lifecycle states: active, retired, outdated, superseded, retracted. Superseded requires superseded_by
and superseded_by_revision naming an existing active replacement. Lifecycle edits preserve body/evidence.
Old memories are not automatically outdated: invalidate only explicitly contradicted or superseded claims.
Outcomes have candidate_id, action (accepted|merged|dismissed|deferred), reason, and edit_indices for resulting edits.
A merged no-change outcome may instead provide existing_revisions [{"record_id":...,"revision":...}].
Deferred outcomes may provide review_after (UTC epoch seconds). Do not claim edits were saved; storage validates and commits them.
Only use supplied candidate/record/evidence IDs or IDs retrieved through the authorized memory tools.'''

INITIALIZE_SYSTEM='''\nThis is a deliberate first-use core initialization, not historical extraction.
Author only the missing Self and/or Principles segments from your current understanding and inherited identity context.
Use concise original agent_reflection attribution. Distinguish inherited identity, intentions and beliefs from verified
runtime capabilities. Preserve uncertainty; do not invent shared history or turn speculative ambitions into achievements.
Do not copy long inherited documents. Keep each body under 2000 UTF-8 bytes. Existing authored segments must remain intact.
Return edits for the missing core segments and an empty outcomes array. No human approval or acknowledgment is needed.'''


class DreamCycle:
    def __init__(self,config,store):
        self.config,self.store=config,store

    def initialize(self):
        PersonalInbox(self.store).initialize()
        with self.store.transaction():
            for statement in SCHEMA.split(';'):
                if statement.strip():self.store.db.execute(statement)

    def _heads(self):
        p=self.store.principal
        return [[r[0],r[1]] for r in self.store.db.execute('SELECT id,revision FROM autobio_records WHERE owner=? AND agent=? AND audience=? ORDER BY id',
                                                       (p.owner,p.agent,p.audience))]

    def _cycle(self,cycle_id,fence,lease_token,*,allow_expired=False):
        p=self.store.principal
        row=self.store.db.execute('SELECT * FROM autobio_cycles WHERE id=? AND owner=? AND agent=? AND audience=?',
                                  (cycle_id,p.owner,p.agent,p.audience)).fetchone()
        if not row:raise PersonalError('not_found','cycle unavailable')
        if (row['state']!='running' or row['fence']!=fence or row['lease_token']!=lease_token
                or row['actor_session']!=p.session_id or (not allow_expired and (row['lease_until']<=self.store.clock() or row['deadline']<=self.store.clock()))):
            raise PersonalError('fenced','cycle publication lease is no longer current')
        return row

    def prepare(self,*,deadline=None,initialize=False,admission=None,full_lease=False):
        s=self.store;p=s.principal;p.require('dream','operator')
        cfg=self.config.personal
        if not cfg.enabled or not cfg.dream_enabled:return {'status':'disabled'}
        if initialize:
            # The core is Soul, User, Identity and Core, written by the agent in
            # conversation or seeded by an operator: a worker does not author it.
            raise PersonalError('unsupported','the core is authored directly (Soul, User, Identity, Core); there is no worker initialization')
        self.initialize()
        from .personal_sources import SourceCursors, metadata_eligible, reconciliation_source_manifest
        cursors=SourceCursors(s)
        with s.transaction():
            if admission:admission()
            now=s.clock()
            context=personal_context(s,budget=cfg.context_budget,subjects=cfg.context_subjects)
            if initialize and not context['manifest']['missing_core']:
                return {'status':'already_initialized','manifest':context['manifest']}
            old=s.db.execute("SELECT * FROM autobio_cycles WHERE owner=? AND agent=? AND state='running'",(p.owner,p.agent)).fetchone()
            if old and old['lease_until']>now and old['deadline']>now:
                return {'status':'busy'}
            if old:
                journal=json.loads(old['journal']);journal['ended_reason']='lease_expired'
                s.db.execute("UPDATE autobio_cycles SET state='partial',updated=?,journal=? WHERE id=?",(now,canonical(journal),old['id']))
                s.db.execute("UPDATE autobio_candidates SET state='pending',cycle_id=NULL,fence=NULL,lease_until=NULL WHERE cycle_id=? AND state='claimed'",(old['id'],))
            prior=s.db.execute('SELECT MAX(fence) FROM autobio_cycles').fetchone()[0] or 0
            cycle_id=uuid.uuid4().hex;fence=prior+1;token=uuid.uuid4().hex
            deadline=min(now+cfg.max_seconds,deadline if deadline is not None else float('inf'));lease_until=deadline if full_lease else min(deadline,now+cfg.lease_seconds)
            if deadline<=now:raise PersonalError('deadline','reflection deadline has expired')
            candidates=[];chars=len(context['text'])
            rows=s.db.execute('''SELECT * FROM autobio_candidates WHERE owner=? AND agent=? AND audience=? AND created<=?
                AND (state='pending' OR (state='deferred' AND review_after<=?)) ORDER BY created,id LIMIT ?''',
                (p.owner,p.agent,p.audience,now,now,0 if initialize else -1 if cfg.max_candidates is None else cfg.max_candidates*4)).fetchall()
            for row in rows:
                source=s.db.execute('SELECT metadata FROM autobio_sources WHERE id=?',(row['source_id'],)).fetchone()
                if source and not metadata_eligible(self.config,json.loads(source[0])):continue
                evidence=json.loads(row['evidence'])
                if not all(s.evidence_available(e) for e in evidence):
                    attempts=row['attempts']+1
                    s.db.execute("UPDATE autobio_candidates SET state=?,attempts=?,review_after=?,disposition='evidence unavailable',updated=? WHERE id=?",('failed' if attempts>=3 else 'deferred',attempts,now+86400,now,row['id']))
                    continue
                value={'id':row['id'],'proposal':json.loads(row['proposal']),'evidence':evidence}
                part=s.db.execute('''SELECT text,start FROM autobio_extraction_parts WHERE source_id=? AND generation=?
                    AND start<=? AND end>=? ORDER BY start LIMIT 1''',(row['source_id'],row['generation'],evidence[0]['start'],evidence[0]['end'])).fetchone()
                if part:
                    offset=evidence[0]['start']-part['start']
                    value['source_context']=part['text'][max(0,offset-1000):offset+len(value['proposal']['quote'])+1000]
                size=len(canonical(value))
                # The prompt must fit: candidates beyond ~48k characters wait for the next cycle.
                if cfg.max_candidates is not None and len(candidates)>=cfg.max_candidates:break
                if candidates and chars+size>48000:continue
                candidates.append(value);chars+=size
                s.db.execute("UPDATE autobio_candidates SET state='claimed',cycle_id=?,fence=?,lease_until=?,updated=? WHERE id=?",
                    (cycle_id,fence,lease_until,now,row['id']))
            status='running' if candidates or initialize else 'completed_no_change'
            journal={'claimed':len(candidates),'commits':[],'outcomes':{},'shadow':cfg.shadow,'initialization':initialize}
            s.db.execute('''INSERT INTO autobio_cycles
                (id,owner,agent,audience,actor_session,state,fence,lease_token,lease_until,cutoff,started,deadline,updated,read_heads,context_manifest,journal,shadow)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (cycle_id,p.owner,p.agent,p.audience,p.session_id,status,fence,token,lease_until,now,now,deadline,now,
                 canonical(self._heads()),canonical(context['manifest']),canonical(journal),int(cfg.shadow)))
            return {'status':status,'cycle_id':cycle_id,'fence':fence,'lease_token':token,'deadline':deadline,
                    'system':DREAM_SYSTEM+(INITIALIZE_SYSTEM if initialize else ''),'core':context,'candidates':candidates,'shadow':cfg.shadow,
                    'source_manifest':reconciliation_source_manifest(self.config,s,candidates,cursors),
                    'limits':{k:v for k,v in {'edits':cfg.max_edits,'model_calls':cfg.model_calls,'seconds':cfg.max_seconds}.items() if v is not None}}

    def renew(self,cycle_id,fence,lease_token):
        self.store.principal.require('dream','operator')
        with self.store.transaction():
            row=self._cycle(cycle_id,fence,lease_token)
            until=min(row['deadline'],self.store.clock()+self.config.personal.lease_seconds)
            self.store.db.execute('UPDATE autobio_cycles SET lease_until=?,updated=? WHERE id=?',(until,self.store.clock(),cycle_id))
            self.store.db.execute("UPDATE autobio_candidates SET lease_until=? WHERE cycle_id=? AND fence=? AND state='claimed'",(until,cycle_id,fence))
            return {'lease_until':until}

    def commit(self,cycle_id,fence,lease_token,*,idempotency_key,edits,outcomes,admission=None):
        s=self.store;p=s.principal;p.require('dream','operator')
        cfg=self.config.personal
        if not isinstance(edits,list) or not isinstance(outcomes,list):
            raise PersonalError('invalid','edits and outcomes must be lists')
        if (cfg.max_edits is not None and len(edits)>cfg.max_edits) or (cfg.max_candidates is not None and len(outcomes)>cfg.max_candidates):
            raise PersonalError('budget','reconciliation batch exceeds configured limits')
        payload_hash=digest({'cycle':cycle_id,'edits':edits,'outcomes':outcomes,'scope':[p.owner,p.agent,p.audience]})
        with s.transaction():
            if admission:admission()
            receipt=s._receipt(idempotency_key,payload_hash)
            if receipt:return receipt
            cycle=self._cycle(cycle_id,fence,lease_token)
            if json.loads(cycle['journal']).get('initialization'):
                missing=set(json.loads(cycle['context_manifest'])['missing_core'])
                if outcomes or any(not isinstance(e,dict) or e.get('segment') not in missing or e.get('record_id') or e.get('expected_revision')!=0 for e in edits):
                    raise PersonalError('invalid','initialization can only author missing core segments')
            if self._heads()!=json.loads(cycle['read_heads']):
                raise PersonalError('conflict','autobiography changed during reflection; reread and reconsider')
            if cfg.max_edits is not None and cycle['edits']+len(edits)>cfg.max_edits:
                raise PersonalError('budget','cycle edit allowance exhausted')
            pending={r['id']:r for r in s.db.execute("SELECT * FROM autobio_candidates WHERE cycle_id=? AND fence=? AND state='claimed'",(cycle_id,fence))}
            prepared=copy.deepcopy(edits);seen=set()
            for outcome in outcomes:
                if not isinstance(outcome,dict) or set(outcome)-{'candidate_id','action','reason','edit_indices','existing_revisions','review_after'}:
                    raise PersonalError('invalid','invalid candidate disposition')
                cid=outcome.get('candidate_id');action=outcome.get('action')
                if cid not in pending or cid in seen:raise PersonalError('fenced','candidate is not currently claimed by this cycle')
                seen.add(cid)
                if action not in ('accepted','merged','dismissed','deferred') or not isinstance(outcome.get('reason'),str) or not 0<len(outcome['reason'])<=1000:
                    raise PersonalError('invalid','a valid disposition and bounded reason are required')
                if 'review_after' in outcome:
                    when=outcome['review_after']
                    if type(when) not in (int,float) or not math.isfinite(when) or not s.clock()<=when<=s.clock()+90*86400:
                        raise PersonalError('invalid','review_after must be a finite future epoch within 90 days')
                indices=outcome.get('edit_indices',[]);existing=outcome.get('existing_revisions',[])
                if not isinstance(indices,list) or any(type(i) is not int or not 0<=i<len(prepared) for i in indices):
                    raise PersonalError('invalid','invalid resulting edit references')
                if not isinstance(existing,list) or (cfg.max_edits is not None and len(existing)>cfg.max_edits):raise PersonalError('invalid','invalid existing revision references')
                if action in ('accepted','merged') and not indices and not existing:
                    raise PersonalError('invalid','accepted evidence must name resulting or already-integrated revisions')
                for ref in existing:
                    if not isinstance(ref,dict) or set(ref)!={'record_id','revision'} or type(ref['revision']) is not int:
                        raise PersonalError('invalid','invalid existing revision reference')
                    record=s._get(ref.get('record_id'))
                    if record['revision']!=ref.get('revision'):raise PersonalError('conflict','already-integrated revision changed')
                evidence=s._evidence(json.loads(pending[cid]['evidence']))
                for index in indices:
                    if not isinstance(prepared[index],dict):raise PersonalError('invalid','edit must be an object')
                    # A lifecycle decision links this candidate via resulting_revisions;
                    # its authored snapshot evidence must remain byte-for-byte unchanged.
                    if prepared[index].get('operation') == 'lifecycle':continue
                    combined=prepared[index].get('evidence',[])+evidence
                    prepared[index]['evidence']=list({canonical(e):e for e in combined}.values())
            commit_id=uuid.uuid4().hex
            if cycle['shadow']:
                # Shadow output is an explicit proposal, never an authored edit
                # or an acknowledged candidate disposition.
                s.db.execute('SAVEPOINT shadow_validation')
                try:
                    for edit in prepared:s._write(edit,commit_id,'dream_cycle')
                    s.validate_core_budget()
                finally:
                    s.db.execute('ROLLBACK TO shadow_validation');s.db.execute('RELEASE shadow_validation')
                s.db.execute('INSERT INTO autobio_shadow_proposals VALUES(?,?,?)',(commit_id,cycle_id,canonical({'edits':prepared,'outcomes':outcomes})))
                receipt={'commit_id':commit_id,'cycle_id':cycle_id,'shadow':True,'proposed_edits':len(edits),'proposed_outcomes':len(outcomes),'edits':[],'committed_at':s.clock()}
            else:
                revisions=[s._write(edit,commit_id,'dream_cycle') for edit in prepared]
                s.validate_core_budget()
                for outcome in outcomes:
                    cid=outcome['candidate_id'];action=outcome['action'];attempts=pending[cid]['attempts']+1
                    review_after=outcome.get('review_after',s.clock()+86400)
                    if type(review_after) not in (int,float) or not math.isfinite(review_after) or not s.clock()<=review_after<=s.clock()+90*86400:
                        raise PersonalError('invalid','review_after must be a finite future epoch within 90 days')
                    state='failed' if action=='deferred' and attempts>=3 else action
                    refs=[revisions[i] for i in outcome.get('edit_indices',[])]+outcome.get('existing_revisions',[])
                    s.db.execute('''UPDATE autobio_candidates SET state=?,review_after=?,attempts=?,disposition=?,resulting_revisions=?,
                        lease_until=NULL,updated=? WHERE id=?''',
                        (state,review_after,attempts,outcome['reason'],canonical(refs),s.clock(),cid))
                receipt={'commit_id':commit_id,'cycle_id':cycle_id,'shadow':False,'edits':revisions,
                         'outcomes':[{k:o[k] for k in ('candidate_id','action')} for o in outcomes],'committed_at':s.clock()}
            s.db.execute('INSERT INTO autobio_commits VALUES(?,?,?,?,?,?,?)',
                (commit_id,p.owner,p.agent,s._request_key(idempotency_key),payload_hash,canonical(receipt),s.clock()))
            s.db.execute('INSERT INTO autobio_cycle_batches VALUES(?,?,?)',(cycle_id,commit_id,s.clock()))
            journal=json.loads(cycle['journal']);journal['commits'].append(commit_id)
            if not cycle['shadow']:
                for outcome in outcomes:
                    action=outcome['action'];journal['outcomes'][action]=journal['outcomes'].get(action,0)+1
            s.db.execute('UPDATE autobio_cycles SET edits=edits+?,read_heads=?,journal=?,updated=? WHERE id=?',
                (0 if cycle['shadow'] else len(edits),canonical(self._heads()),canonical(journal),s.clock(),cycle_id))
            self._cycle(cycle_id,fence,lease_token)
        for revision in receipt['edits']:s.index_record(revision['record_id'])
        return receipt

    def finish(self,cycle_id,fence,lease_token,*,status='completed',error=None):
        s=self.store;p=s.principal;p.require('dream','operator')
        if status not in ('completed','partial','failed','cancelled'):raise PersonalError('invalid','invalid cycle outcome')
        with s.transaction():
            cycle=self._cycle(cycle_id,fence,lease_token,allow_expired=True)
            remaining=s.db.execute("SELECT COUNT(*) FROM autobio_candidates WHERE cycle_id=? AND state='claimed'",(cycle_id,)).fetchone()[0]
            if status=='completed':status='partial' if remaining and not cycle['shadow'] else 'completed' if cycle['edits'] else 'completed_no_change'
            journal=json.loads(cycle['journal']);journal['remaining']=remaining
            if error:journal['error']=str(error)[:120]
            s.db.execute('UPDATE autobio_cycles SET state=?,updated=?,journal=? WHERE id=?',(status,s.clock(),canonical(journal),cycle_id))
            if error in ('invalid','invalid_evidence','context_overflow','idempotency_mismatch','sensitive_content'):
                # Rejected model proposals cannot consume the same allowance
                # every night forever. Keep the evidence in a visible dead letter.
                s.db.execute("""UPDATE autobio_candidates SET state=CASE WHEN attempts+1>=3 THEN 'failed' ELSE 'deferred' END,
                    attempts=attempts+1,review_after=?,disposition=?,cycle_id=NULL,fence=NULL,lease_until=NULL,updated=?
                    WHERE cycle_id=? AND state='claimed'""",(s.clock()+86400,str(error),s.clock(),cycle_id))
            else:
                s.db.execute("UPDATE autobio_candidates SET state='pending',cycle_id=NULL,fence=NULL,lease_until=NULL WHERE cycle_id=? AND state='claimed'",(cycle_id,))
            return {'cycle_id':cycle_id,'status':status,'journal':journal}

    def status(self,limit=10):
        self.initialize();p=self.store.principal
        return [{**dict(r),'journal':json.loads(r['journal'])} for r in self.store.db.execute(
            'SELECT id,state,started,updated,edits,journal FROM autobio_cycles WHERE owner=? AND agent=? AND audience=? ORDER BY started DESC LIMIT ?',
            (p.owner,p.agent,p.audience,max(1,min(int(limit),50))))]
