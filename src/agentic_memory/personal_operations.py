"""Durable personal schedule, conservative budgets and paid-result checkpoints."""
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os
import queue
import threading
import uuid

from .autobiography import PersonalError,canonical,digest
from .autobiography_inbox import PersonalInbox
from .provider_limits import ProviderGate

SCHEMA='''
CREATE TABLE IF NOT EXISTS autobio_schedule(
 kind TEXT PRIMARY KEY,due REAL NOT NULL,operation_id TEXT,actor_session TEXT,fence INTEGER NOT NULL DEFAULT 0,
 lease_until REAL,outcome TEXT,updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_operations(
 id TEXT PRIMARY KEY,kind TEXT NOT NULL,owner TEXT NOT NULL,agent TEXT NOT NULL,audience TEXT NOT NULL,
 actor_session TEXT NOT NULL,fence INTEGER NOT NULL,state TEXT NOT NULL,started REAL NOT NULL,deadline REAL NOT NULL,
 result TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS autobio_model_work(
 id TEXT PRIMARY KEY,operation_id TEXT NOT NULL,work_id TEXT,cache_key TEXT NOT NULL,day TEXT NOT NULL,
 reserved_calls INTEGER NOT NULL,reserved_tokens INTEGER NOT NULL,state TEXT NOT NULL,
 provider TEXT NOT NULL,model TEXT NOT NULL,identity TEXT NOT NULL,source_ids TEXT NOT NULL,
 record_ids TEXT NOT NULL,usage TEXT NOT NULL DEFAULT '{}',created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_cache_uses(operation_id TEXT NOT NULL,cache_key TEXT NOT NULL,PRIMARY KEY(operation_id,cache_key));
'''


class _ProviderLease:
    """Keep a gate's SQLite connection on one thread across separate RPC calls."""
    def __init__(self,context):
        self.context=context;self.ready=queue.Queue();self.release=threading.Event();self.done=threading.Event();self.error=None

    def __enter__(self):
        def hold():
            announced=False
            try:
                with self.context as call_id:
                    self.ready.put((True,call_id));announced=True
                    self.release.wait()
                    if self.error:raise self.error
            except BaseException as exc:
                if not announced:self.ready.put((False,exc))
            finally:self.done.set()
        threading.Thread(target=hold,daemon=True).start()
        okay,result=self.ready.get()
        if not okay:raise result
        return result

    def __exit__(self,kind,error,traceback):
        self.error=error;self.release.set();self.done.wait(5)


class PersonalOperations:
    def __init__(self,config,store):
        self.config,self.store=config,store;self.leases={}
        PersonalInbox(store).initialize()
        with store.transaction():
            for statement in SCHEMA.split(';'):
                if statement.strip():store.db.execute(statement)
            columns={r[1] for r in store.db.execute('PRAGMA table_info(autobio_model_cache)')}
            if 'source_ids' not in columns:store.db.execute("ALTER TABLE autobio_model_cache ADD COLUMN source_ids TEXT NOT NULL DEFAULT '[]'")
            if 'work_id' not in {r[1] for r in store.db.execute('PRAGMA table_info(autobio_model_work)')}:
                store.db.execute('ALTER TABLE autobio_model_work ADD COLUMN work_id TEXT')
        if config.personal.service_config:
            from .scheduler_config import load_service
            service=load_service(Path(config.personal.service_config).expanduser())
            self.gate=ProviderGate(service.db_path,model_limit=service.model_limit,embedding_limit=service.embedding_limit,provider_limits=service.provider_limits)
        else:
            self.gate=ProviderGate(config.data_dir/'personal-provider-gate.db',model_limit=1,embedding_limit=1)

    def next_night(self,now):
        cfg=self.config.personal
        local=datetime.fromtimestamp(now,ZoneInfo(cfg.timezone))
        due=local.replace(hour=cfg.dream_hour,minute=0,second=0,microsecond=0)
        if due.timestamp()<=now:due+=timedelta(days=1)
        return due.timestamp()

    def _role(self,kind):
        if kind not in ('extractor','dream'):raise PersonalError('invalid','unknown personal operation')
        self.store.principal.require(kind,'operator')

    def claim(self,kind,*,manual=False):
        self._role(kind);s=self.store;p=s.principal;cfg=self.config.personal
        enabled=cfg.extraction_enabled if kind=='extractor' else cfg.dream_enabled
        if not cfg.enabled or not enabled:return {'status':'disabled'}
        now=s.clock()
        with s.transaction():
            due=now if kind=='extractor' else self.next_night(now)
            s.db.execute('INSERT OR IGNORE INTO autobio_schedule(kind,due,updated) VALUES(?,?,?)',(kind,due,now))
            row=s.db.execute('SELECT * FROM autobio_schedule WHERE kind=?',(kind,)).fetchone()
            if row['operation_id'] and row['lease_until']>now:return {'status':'busy','due':row['due']}
            if row['due']>now and not manual:return {'status':'not_due','due':row['due']}
            if row['operation_id']:
                s.db.execute("UPDATE autobio_operations SET state='interrupted',result=? WHERE id=?",(canonical({'reason':'lease_expired'}),row['operation_id']))
            op=uuid.uuid4().hex;fence=row['fence']+1;deadline=now+cfg.max_seconds
            s.db.execute('INSERT INTO autobio_operations(id,kind,owner,agent,audience,actor_session,fence,state,started,deadline) VALUES(?,?,?,?,?,?,?,\'running\',?,?)',
                         (op,kind,p.owner,p.agent,p.audience,p.session_id,fence,now,deadline))
            s.db.execute('UPDATE autobio_schedule SET operation_id=?,actor_session=?,fence=?,lease_until=?,updated=? WHERE kind=?',
                         (op,p.session_id,fence,deadline,now,kind))
            return {'status':'running','operation_id':op,'fence':fence,'deadline':deadline,
                    'limits':{'model_calls':cfg.model_calls,'max_output_tokens':cfg.max_output_tokens,'max_seconds':cfg.max_seconds}}

    def _operation(self,operation_id,fence):
        s=self.store;p=s.principal
        with s.lock:
            row=s.db.execute('SELECT * FROM autobio_operations WHERE id=? AND owner=? AND agent=? AND audience=?',
                             (operation_id,p.owner,p.agent,p.audience)).fetchone()
        if not row:raise PersonalError('not_found','operation unavailable')
        self._role(row['kind'])
        if row['actor_session']!=p.session_id or row['fence']!=fence or row['state']!='running' or row['deadline']<=s.clock():
            raise PersonalError('fenced','personal operation lease expired or was replaced')
        return row

    def reserve(self,operation_id,fence,*,work_id,identity,provider,model,system,prompt,input_units,slots=1):
        s=self.store;p=s.principal;cfg=self.config.personal
        if type(slots) is not int or not 1<=slots<=3 or type(input_units) is not int or not 0<input_units<=256000:
            raise PersonalError('budget','invalid bounded model reservation')
        if any(not isinstance(x,str) or len(x)>maximum for x,maximum in [(identity,128),(provider,200),(model,200),(system,64000),(prompt,128000)]):
            raise PersonalError('invalid','invalid model request metadata')
        key=digest([p.owner,p.agent,p.audience,identity,system,prompt])
        day=datetime.fromtimestamp(s.clock(),ZoneInfo(cfg.timezone)).date().isoformat()
        # Budget a conservative estimate for envelopes, tools and follow-up
        # context as well as the bounded request/output text. This is not billing.
        tokens=slots*(input_units+cfg.max_output_tokens+8192)+max(0,slots-1)*32768
        with s.transaction():
            op=self._operation(operation_id,fence)
            if op['kind']=='extractor':
                source=s.db.execute('''SELECT src.id FROM autobio_sources src JOIN autobio_extraction_parts part ON src.id=part.source_id
                    WHERE part.id=? AND part.state='pending' AND src.owner=? AND src.agent=? AND src.audience=? AND src.generation=part.generation''',
                    (work_id,p.owner,p.agent,p.audience)).fetchone()
                if not source or s.is_suppressed(source['id']):raise PersonalError('stale_source','source part is no longer eligible')
                source_ids=[source['id']];record_ids=[]
            else:
                cycle=s.db.execute("SELECT * FROM autobio_cycles WHERE id=? AND owner=? AND agent=? AND audience=? AND state='running' AND actor_session=?",
                    (work_id,p.owner,p.agent,p.audience,p.session_id)).fetchone()
                if not cycle or cycle['lease_until']<=s.clock():raise PersonalError('fenced','reflection cycle is not live')
                record_ids=[r['record_id'] for r in json.loads(cycle['context_manifest'])['records']]
                source_ids=[r[0] for r in s.db.execute("SELECT DISTINCT source_id FROM autobio_candidates WHERE cycle_id=? AND state='claimed'",(work_id,))]
            cached=s.db.execute('SELECT result FROM autobio_model_cache WHERE id=?',(key,)).fetchone()
            if cached:
                s.db.execute('INSERT OR IGNORE INTO autobio_cache_uses VALUES(?,?)',(operation_id,key))
                return {'status':'cached','response':json.loads(cached[0]),'cache_key':key}
            prior=s.db.execute("SELECT 1 FROM autobio_model_work w JOIN autobio_operations o ON o.id=w.operation_id WHERE w.cache_key=? AND w.state IN ('waiting','running') AND o.deadline>?",(key,s.clock())).fetchone()
            if prior:return {'status':'busy'}
            used=s.db.execute("SELECT COALESCE(SUM(reserved_calls),0),COALESCE(SUM(reserved_tokens),0) FROM autobio_model_work WHERE day=? AND state!='not_dispatched'",(day,)).fetchone()
            run_used=s.db.execute("SELECT COALESCE(SUM(reserved_calls),0) FROM autobio_model_work WHERE operation_id=? AND state!='not_dispatched'",(operation_id,)).fetchone()[0]
            call_ceiling=cfg.daily_model_calls
            token_ceiling=cfg.daily_token_budget
            if op['kind']=='extractor':
                call_ceiling-=min(cfg.model_calls,cfg.daily_model_calls//4)
                token_ceiling=cfg.daily_token_budget*3//4
            if used[0]+slots>call_ceiling or used[1]+tokens>token_ceiling or run_used+slots>cfg.model_calls:
                raise PersonalError('budget','personal daily or operation allowance exhausted; work remains queued')
            request_id=uuid.uuid4().hex
            s.db.execute('''INSERT INTO autobio_model_work(id,operation_id,work_id,cache_key,day,reserved_calls,reserved_tokens,state,provider,model,identity,source_ids,record_ids,created)
                VALUES(?,?,?,?,?,?,?,'waiting',?,?,?,?,?,?)''',
                (request_id,operation_id,work_id,key,day,slots,tokens,provider,model,identity,canonical(source_ids),canonical(record_ids),s.clock()))
        def admission():
            self._operation(operation_id,fence)
            if op['kind']=='extractor':
                from .source_activity import require_quiet,SourceNotQuiet
                with s.lock:metadata=json.loads(s.db.execute('SELECT metadata FROM autobio_sources WHERE id=?',(source_ids[0],)).fetchone()[0])
                try:require_quiet(Path(metadata['path']),metadata['source_type'],self.config.quiet_seconds)
                except SourceNotQuiet as exc:raise PersonalError('quiet','source advanced; wait for the quiet window') from exc
        context=_ProviderLease(self.gate.acquire(provider,'model',owner=request_id,instance='personal:'+p.agent,check_admission=admission))
        try:
            call_id=context.__enter__()
        except BaseException:
            with s.transaction():s.db.execute("UPDATE autobio_model_work SET state='not_dispatched' WHERE id=?",(request_id,))
            raise
        self.leases[request_id]=(context,call_id)
        with s.transaction():s.db.execute("UPDATE autobio_model_work SET state='running' WHERE id=?",(request_id,))
        return {'status':'reserved','request_id':request_id,'max_provider_calls':slots,'max_output_tokens':cfg.max_output_tokens,'cache_key':key}

    def reject_cache(self,operation_id,fence,cache_key,code):
        if code not in ('invalid','invalid_evidence','context_overflow','idempotency_mismatch','sensitive_content'):
            return {'invalidated':False}
        with self.store.transaction():
            self._operation(operation_id,fence)
            known=self.store.db.execute('''SELECT 1 FROM autobio_model_work WHERE operation_id=? AND cache_key=?
                UNION SELECT 1 FROM autobio_cache_uses WHERE operation_id=? AND cache_key=?''',
                (operation_id,cache_key,operation_id,cache_key)).fetchone()
            if not known:raise PersonalError('forbidden','cache entry is not bound to this operation')
            self.store.db.execute('DELETE FROM autobio_model_cache WHERE id=?',(cache_key,))
            return {'invalidated':True,'reason':code}

    def complete_model(self,operation_id,fence,request_id,*,response=None,usage=None,error=None):
        s=self.store
        if request_id not in self.leases:raise PersonalError('fenced','model reservation is not owned by this service instance')
        context,call_id=self.leases.pop(request_id)
        try:
            with s.transaction():
                op=s.db.execute('SELECT * FROM autobio_operations WHERE id=?',(operation_id,)).fetchone()
                row=s.db.execute('SELECT * FROM autobio_model_work WHERE id=? AND operation_id=?',(request_id,operation_id)).fetchone()
                if not row or op['fence']!=fence or op['actor_session']!=s.principal.session_id:
                    raise PersonalError('fenced','model reservation owner changed')
                clean_usage={k:v for k,v in (usage or {}).items() if k in ('input_tokens','output_tokens','reasoning_tokens','cache_read_tokens','wire_attempts','transport_retries') and type(v) is int and v>=0}
                if response is not None:
                    if not isinstance(response,dict) or len(canonical(response))>128000:raise PersonalError('invalid','model response is not bounded JSON')
                    from .personal_privacy import require_no_credentials
                    require_no_credentials(canonical(response))
                    s.db.execute('INSERT OR IGNORE INTO autobio_model_cache(id,result,source_id,record_ids,created,source_ids) VALUES(?,?,?,?,?,?)',
                        (row['cache_key'],canonical(response),json.loads(row['source_ids'])[0] if op['kind']=='extractor' else None,row['record_ids'],s.clock(),row['source_ids']))
                s.db.execute('UPDATE autobio_model_work SET state=?,usage=? WHERE id=?',('complete' if response is not None else 'failed',canonical(clean_usage),request_id))
            db=None;telemetry_error=False
            try:
                db=self.gate.connect()
                details={**clean_usage,'requested_model':row['model'],'runtime':'agent-core-python','usage_complete':all(k in clean_usage for k in ('input_tokens','output_tokens'))}
                with db:db.execute('INSERT OR REPLACE INTO model_requests VALUES(?,?,?,?,?,?,?,?)',
                    (call_id,operation_id,'personal/'+s.principal.agent,op['kind'],row['cache_key'],0,0,canonical(details)))
            except Exception:telemetry_error=True
            finally:
                if db:db.close()
            return {'checkpointed':response is not None,'telemetry_error':telemetry_error}
        finally:
            failed=response is None
            context.__exit__(RuntimeError if failed else None,RuntimeError('personal model failed') if failed else None,None)

    def finish(self,operation_id,fence,*,outcome='completed',result=None):
        s=self.store
        if outcome not in ('completed','partial','failed','cancelled'):raise PersonalError('invalid','invalid operation outcome')
        with s.transaction():
            op=self._operation(operation_id,fence)
            due=self.next_night(s.clock()) if op['kind']=='dream' else s.clock()+self.config.personal.extraction_interval
            s.db.execute('UPDATE autobio_operations SET state=?,result=? WHERE id=?',(outcome,canonical(result or {}),operation_id))
            s.db.execute('UPDATE autobio_schedule SET due=?,operation_id=NULL,lease_until=NULL,outcome=?,updated=? WHERE kind=? AND fence=?',
                         (due,outcome,s.clock(),op['kind'],fence))
            return {'outcome':outcome,'next_due':due}

    def status(self):
        day=datetime.fromtimestamp(self.store.clock(),ZoneInfo(self.config.personal.timezone)).date().isoformat()
        row=self.store.db.execute("SELECT COUNT(*),COALESCE(SUM(reserved_calls),0),COALESCE(SUM(reserved_tokens),0) FROM autobio_model_work WHERE day=? AND state!='not_dispatched'",(day,)).fetchone()
        p=self.store.principal
        candidates=dict(self.store.db.execute('SELECT state,COUNT(*) FROM autobio_candidates WHERE owner=? AND agent=? AND audience=? GROUP BY state',(p.owner,p.agent,p.audience)))
        parts=dict(self.store.db.execute('SELECT part.state,COUNT(*) FROM autobio_extraction_parts part JOIN autobio_sources src ON src.id=part.source_id WHERE src.owner=? AND src.agent=? AND src.audience=? GROUP BY part.state',(p.owner,p.agent,p.audience)))
        oldest=self.store.db.execute("SELECT MIN(created) FROM autobio_candidates WHERE owner=? AND agent=? AND audience=? AND state IN ('pending','deferred','failed')",(p.owner,p.agent,p.audience)).fetchone()[0]
        failures=[dict(r) for r in self.store.db.execute('''SELECT part.id,part.state,part.attempts,part.error FROM autobio_extraction_parts part
            JOIN autobio_sources src ON src.id=part.source_id WHERE src.owner=? AND src.agent=? AND src.audience=? AND part.state='failed' LIMIT 20''',(p.owner,p.agent,p.audience))]
        sources=[dict(r) for r in self.store.db.execute("SELECT path,state,attempts,error FROM autobio_source_errors ORDER BY path LIMIT 20")]
        return {'schedule':[dict(r) for r in self.store.db.execute('SELECT kind,due,operation_id,lease_until,outcome FROM autobio_schedule')],
                'candidate_states':candidates,'extraction_states':parts,
                'oldest_candidate_age_seconds':max(0,self.store.clock()-oldest) if oldest else None,
                'failed_parts':failures,'source_errors':sources,
                'day':day,'model_turns':row[0],'reserved_provider_calls':row[1],'reserved_token_estimate':row[2],
                'limits':{'daily_provider_calls':self.config.personal.daily_model_calls,'daily_token_reservation_budget':self.config.personal.daily_token_budget},
                'note':'Conservative reservations include potential logical provider calls; transport retries and billed cost remain separately observed.'}

    def close(self):
        for context,_ in list(self.leases.values()):
            context.__exit__(RuntimeError,RuntimeError('personal service closing'),None)
        self.leases.clear()
