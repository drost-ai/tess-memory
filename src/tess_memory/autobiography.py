"""Opt-in authored personal state. No coding-memory tables or providers are used."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

# Soul, User, Identity and Core are the foundation documents an agent is given
# or writes about itself and its person, kept in their own words. Core is the
# short list of memories essential to remaining itself, which the agent adds
# to over time. They are authored directly (the agent in conversation, or an operator's seed): the
# background extractor and reconciler neither propose nor edit them.
FOUNDATION_SEGMENTS = ('Soul', 'User', 'Identity', 'Core')
# What background workers extract into and reconcile.
WORKER_SEGMENTS = ('People', 'Relationships', 'Experiences', 'Reflections')
SEGMENTS = FOUNDATION_SEGMENTS + WORKER_SEGMENTS
# One current document each (subject 'core'), loaded whole into the context:
# the agent's core is its foundation documents.
CORE_SEGMENTS = FOUNDATION_SEGMENTS
# Retired: Self and Principles were the core before the foundation documents.
# Records already in a store stay readable (by id, in history and search) and
# can be retired by the author or an operator; nothing new is written to them,
# and they are loaded only until a store has its first foundation document.
RETIRED_SEGMENTS = ('Self', 'Principles')
ATTRIBUTIONS = ('user_statement', 'observed_fact', 'agent_reflection', 'extractor_inference', 'inherited')
ROLES = ('reader', 'author', 'extractor', 'dream', 'reconciler', 'follow_up_reconciler', 'topic_reconciler', 'care', 'research', 'operator')
SCHEMA_VERSION = 1
MAX_BODY = 32000
MAX_EVIDENCE = 32


class PersonalError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value, name='identifier'):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}', value):
        raise PersonalError('invalid', f'invalid {name}')
    return value


# The agent's own identity: shared across audiences. User and Core are about
# the owner and the life shared with them, and stay in the owner's audience.
SHARED_SEGMENTS = ('Soul', 'Identity') + RETIRED_SEGMENTS


@dataclass(frozen=True)
class PersonalPrincipal:
    owner: str
    agent: str
    audience: str = 'owner'
    role: str = 'reader'
    session_id: str = 'unbound'
    # '' = this audience only (single-audience stores, unchanged);
    # '*' = every audience (the owner's, in a multi-audience store);
    # otherwise the owner audience whose Self/Principles this audience shares.
    reads: str = ''

    def __post_init__(self):
        for field in ('owner', 'agent', 'audience', 'session_id'):
            identifier(getattr(self, field), field)
        if self.reads not in ('', '*'):
            identifier(self.reads, 'reads')
        if self.role not in ROLES:
            raise PersonalError('forbidden', 'unavailable personal role')
        if self.role in ('author','dream','reconciler','follow_up_reconciler','topic_reconciler','care','research') and self.session_id == 'unbound':
            raise PersonalError('forbidden','authoring requires a trusted session binding')

    def audience_sql(self, alias=''):
        """Which records this principal may READ (writes stay in its own audience)."""
        column = f'{alias}.' if alias else ''
        if self.reads == '*':
            return '1=1', []
        if self.reads:
            shared = ','.join('?' for _ in SHARED_SEGMENTS)
            return (f'({column}audience=? OR ({column}audience=? AND {column}segment IN ({shared})))',
                    [self.audience, self.reads, *SHARED_SEGMENTS])
        return f'{column}audience=?', [self.audience]

    def require(self, *roles):
        if ('dream' if self.role=='reconciler' else self.role) not in tuple('dream' if role=='reconciler' else role for role in roles):
            raise PersonalError('forbidden', 'operation unavailable to this principal')


SCHEMA = '''
CREATE TABLE IF NOT EXISTS autobio_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_records(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, agent TEXT NOT NULL, audience TEXT NOT NULL,
 segment TEXT NOT NULL, subject TEXT NOT NULL, title TEXT NOT NULL,
 revision INTEGER NOT NULL, state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
 UNIQUE(owner,agent,audience,segment,subject));
CREATE TABLE IF NOT EXISTS autobio_revisions(
 record_id TEXT NOT NULL REFERENCES autobio_records(id) ON DELETE CASCADE,
 revision INTEGER NOT NULL, body TEXT NOT NULL, title TEXT NOT NULL, state TEXT NOT NULL,
 reason TEXT NOT NULL, actor_session TEXT NOT NULL, origin TEXT NOT NULL,
 attribution TEXT NOT NULL, evidence TEXT NOT NULL, created REAL NOT NULL, commit_id TEXT NOT NULL,
 PRIMARY KEY(record_id,revision));
CREATE TABLE IF NOT EXISTS autobio_commits(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, agent TEXT NOT NULL, request_key TEXT NOT NULL,
 payload_hash TEXT NOT NULL, receipt TEXT NOT NULL, created REAL NOT NULL,
 UNIQUE(owner,agent,request_key));
CREATE TABLE IF NOT EXISTS autobio_index_state(record_id TEXT PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_index_errors(record_id TEXT PRIMARY KEY REFERENCES autobio_records(id) ON DELETE CASCADE,
 revision INTEGER NOT NULL,attempts INTEGER NOT NULL,error TEXT NOT NULL,retry_after REAL NOT NULL,updated REAL NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS autobio_search USING fts5(record_id UNINDEXED, body, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS autobio_sources(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, agent TEXT NOT NULL, audience TEXT NOT NULL,
 generation TEXT NOT NULL, span_count INTEGER NOT NULL, metadata TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_suppressions(
 owner TEXT NOT NULL, agent TEXT NOT NULL, source_id TEXT NOT NULL, created REAL NOT NULL,
 PRIMARY KEY(owner,agent,source_id));
CREATE TABLE IF NOT EXISTS autobio_erasure_receipts(
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, agent TEXT NOT NULL, record_ids TEXT NOT NULL,
 source_ids TEXT NOT NULL, created REAL NOT NULL);
'''


class AutobiographyStore:
    """One configured owner/agent scope; callers cannot override it in tool input."""
    def __init__(self, path: Path, principal: PersonalPrincipal, *, clock=time.time, context_budget=6000):
        self.path, self.principal, self.clock = Path(path), principal, clock
        self.context_budget=context_budget
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA secure_delete=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.lock = threading.RLock()
        try:
            with self.transaction():
                # executescript would commit implicitly; run the additive DDL
                # inside the same explicit transaction as its version marker.
                for statement in SCHEMA.split(';'):
                    if statement.strip():
                        self.db.execute(statement)
                row = self.db.execute("SELECT value FROM autobio_meta WHERE key='version'").fetchone()
                if row and int(row[0]) != SCHEMA_VERSION:
                    raise PersonalError('schema_version', 'unsupported personal schema version')
                from .autobiography_lifecycle import install
                install(self.db)
                # Engine sessions were labelled 'agent-core' before the engine
                # was named tess-core. Only the label in the metadata changes.
                old, new = '"source_type":"agent-core"', '"source_type":"tess-core"'
                if self.db.execute("SELECT 1 FROM autobio_sources WHERE instr(metadata,?) LIMIT 1", (old,)).fetchone():
                    self.db.execute("UPDATE autobio_sources SET metadata=replace(metadata,?,?) WHERE instr(metadata,?)", (old, new, old))
                scope = canonical([principal.owner, principal.agent])
                bound = self.db.execute("SELECT value FROM autobio_meta WHERE key='scope'").fetchone()
                if bound and bound[0] != scope:
                    raise PersonalError('forbidden', 'personal store scope unavailable')
                self.db.execute("INSERT OR IGNORE INTO autobio_meta VALUES('scope',?)", (scope,))
                self.db.execute("INSERT OR IGNORE INTO autobio_meta VALUES('version',?)", (str(SCHEMA_VERSION),))
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @contextmanager
    def snapshot(self):
        with self.lock:
            owns=not self.db.in_transaction
            if owns:self.db.execute('BEGIN')
            try:yield
            finally:
                if owns:self.db.rollback()

    @contextmanager
    def transaction(self, *, write=True):
        with self.lock:
            if self.db.in_transaction:
                raise RuntimeError('nested personal transaction')
            self.db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def _head(self, record_id, *, write=False):
        p = self.principal
        audience, args = ('audience=?', [p.audience]) if write else p.audience_sql()
        row = self.db.execute(f'SELECT * FROM autobio_records WHERE id=? AND owner=? AND agent=? AND {audience}',
                              (record_id, p.owner, p.agent, *args)).fetchone()
        if not row:
            raise PersonalError('not_found', 'personal record unavailable')
        return dict(row)

    def _get(self, record_id, revision=None):
        head = self._head(record_id)
        revision = head['revision'] if revision is None else revision
        row = self.db.execute('SELECT * FROM autobio_revisions WHERE record_id=? AND revision=?',
                              (record_id, revision)).fetchone()
        if not row:
            raise PersonalError('not_found', 'personal revision unavailable')
        result = {**head, **dict(row), 'id': record_id}
        result['evidence'] = json.loads(result['evidence'])
        result['current_revision'] = head['revision']
        event = self.db.execute('SELECT * FROM autobio_lifecycle WHERE record_id=? AND revision=?', (record_id, revision)).fetchone()
        result['lifecycle'] = dict(event) if event else None
        result['evidence_availability'] = [self.evidence_available(e) for e in result['evidence']]
        return result

    def get(self, record_id, revision=None, *, include_inactive=False):
        with self.transaction(write=False):
            if revision is None and not include_inactive and self._head(record_id)['state'] != 'active':
                raise PersonalError('not_found', 'personal record is no longer active; request history or an explicit revision')
            return self._get(record_id, revision)

    def list_records(self, *, segment=None, limit=100, after='', include_inactive=False):
        if segment is not None and segment not in SEGMENTS+RETIRED_SEGMENTS:
            raise PersonalError('invalid', 'unknown personal segment')
        limit = max(1, min(int(limit), 100))
        p = self.principal
        with self.transaction(write=False):
            audience, args = p.audience_sql()
            rows = self.db.execute(f'''SELECT id FROM autobio_records WHERE owner=? AND agent=? AND {audience}
                AND (? OR state='active') AND id>? AND (? IS NULL OR segment=?) ORDER BY id LIMIT ?''',
                (p.owner, p.agent, *args, bool(include_inactive), after, segment, segment, limit+1)).fetchall()
            return {'records': [self._get(r['id']) for r in rows[:limit]],
                    'next_cursor': rows[limit-1]['id'] if len(rows)>limit else None}

    def history(self, record_id, *, before=None, limit=20):
        with self.transaction(write=False):
            head = self._head(record_id)
            rows = self.db.execute('SELECT revision FROM autobio_revisions WHERE record_id=? AND revision<? ORDER BY revision DESC LIMIT ?',
                (record_id, before if before is not None else head['revision']+1, max(1,min(int(limit),50)))).fetchall()
            return {'revisions': [self._get(record_id, r[0]) for r in rows],
                    'next_cursor': rows[-1][0] if rows and rows[-1][0]>1 else None}

    def evidence_available(self, evidence):
        p = self.principal
        source = self.db.execute('SELECT generation,metadata FROM autobio_sources WHERE id=? AND owner=? AND agent=? AND audience=?',
            (evidence.get('source_id'),p.owner,p.agent,p.audience)).fetchone()
        if not source or source[0]!=evidence.get('generation') or self.is_suppressed(evidence.get('source_id')):
            return False
        try:
            metadata=json.loads(source['metadata'])
            if metadata.get('kind')=='inherited_document':
                return hashlib.sha256(Path(metadata['path']).read_bytes()).hexdigest()==source[0]
            from .source_journal import SourceRevision
            return all(SourceRevision(**r).verify() for r in metadata.get('source_revisions',[]))
        except (OSError,ValueError,TypeError):
            return False

    def _evidence(self, evidence):
        if not isinstance(evidence,list) or len(evidence)>MAX_EVIDENCE:
            raise PersonalError('invalid_evidence', 'invalid evidence list')
        result=[]
        for e in evidence:
            if not isinstance(e,dict) or set(e)-{'source_id','generation','start','end','attribution'}:
                raise PersonalError('invalid_evidence','invalid evidence reference')
            p=self.principal
            row=self.db.execute('SELECT * FROM autobio_sources WHERE id=? AND owner=? AND agent=? AND audience=?',
                (e.get('source_id'),p.owner,p.agent,p.audience)).fetchone()
            if (not row or not self.evidence_available(e)
                or type(e.get('start')) is not int or type(e.get('end')) is not int
                or not 0<=e['start']<e['end']<=row['span_count'] or e.get('attribution') not in ATTRIBUTIONS):
                raise PersonalError('invalid_evidence','evidence unavailable or changed')
            result.append(dict(e))
        return result

    def is_suppressed(self, source_id):
        p=self.principal
        return bool(self.db.execute('SELECT 1 FROM autobio_suppressions WHERE owner=? AND agent=? AND source_id=?',
                                   (p.owner,p.agent,source_id)).fetchone())

    def require_foundation_author(self, segment):
        """Soul, User, Identity and Core change only through the author or an
        operator, never through a background worker's batch. The same holds
        for retiring what is left in Self and Principles."""
        if segment in FOUNDATION_SEGMENTS+RETIRED_SEGMENTS and self.principal.role not in ('author','operator'):
            raise PersonalError('forbidden',f'{segment} is authored directly; background workers cannot change it')

    def _write(self, edit, commit_id, origin):
        if isinstance(edit, dict) and edit.get('operation') == 'lifecycle':
            from .autobiography_lifecycle import write_transition
            return write_transition(self, edit, commit_id, origin)
        if not isinstance(edit,dict) or set(edit)-{'record_id','expected_revision','segment','subject','title','body','reason','evidence','attribution','state'}:
            raise PersonalError('invalid','invalid authored edit')
        body, reason, title = edit.get('body'),edit.get('reason'),edit.get('title')
        if not isinstance(body,str) or not body.strip() or len(body)>MAX_BODY:
            raise PersonalError('invalid','body must be nonempty and at most 32000 characters')
        if not isinstance(reason,str) or not reason.strip() or len(reason)>2000:
            raise PersonalError('invalid','a bounded revision reason is required')
        if not isinstance(title,str) or not title.strip() or len(title)>200:
            raise PersonalError('invalid','a bounded title is required')
        from .personal_privacy import require_no_credentials
        require_no_credentials(body+'\n'+title+'\n'+reason)
        attribution=edit.get('attribution','agent_reflection')
        if attribution not in ATTRIBUTIONS:
            raise PersonalError('invalid','unknown attribution')
        evidence=self._evidence(edit.get('evidence',[]))
        if attribution in ('user_statement','observed_fact','extractor_inference','inherited') and not evidence:
            raise PersonalError('invalid_evidence','this attribution requires source evidence')
        state=edit.get('state','active')
        if state not in ('active','retired'):
            raise PersonalError('invalid','unknown record state')
        p=self.principal
        now=self.clock()
        record_id=edit.get('record_id')
        head=None
        expected=edit.get('expected_revision')
        if type(expected) is not int or expected<0:
            raise PersonalError('invalid','expected_revision must be a nonnegative integer')
        if record_id:
            head=self._head(record_id, write=True)  # only this audience's own records
            if expected!=head['revision']:
                raise PersonalError('conflict','record revision changed; reread before proposing another edit')
            if edit.get('segment',head['segment'])!=head['segment'] or edit.get('subject',head['subject'])!=head['subject']:
                raise PersonalError('invalid','segment and subject are immutable')
            if head['segment'] in RETIRED_SEGMENTS:
                raise PersonalError('invalid',head['segment']+' is retired: its history stays readable, and its content is no longer edited')
            self.require_foundation_author(head['segment'])
            revision=expected+1
            self.db.execute('UPDATE autobio_records SET title=?,revision=?,state=?,updated=? WHERE id=?',
                            (title,revision,state,now,record_id))
        else:
            if expected!=0 or edit.get('segment') not in SEGMENTS:
                raise PersonalError('invalid','new records require expected_revision=0 and a segment')
            self.require_foundation_author(edit['segment'])
            record_id=uuid.uuid4().hex
            subject=edit.get('subject') or ('core' if edit['segment'] in CORE_SEGMENTS else record_id)
            identifier(subject,'subject')
            if edit['segment'] in CORE_SEGMENTS and subject!='core':
                raise PersonalError('invalid','core segments have one current document')
            revision=1
            try:
                self.db.execute('INSERT INTO autobio_records VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (record_id,p.owner,p.agent,p.audience,edit['segment'],subject,title,revision,state,now,now))
            except sqlite3.IntegrityError as exc:
                raise PersonalError('conflict','a record for this segment and subject already exists') from exc
        self.db.execute('INSERT INTO autobio_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (record_id,revision,body,title,state,reason,p.session_id,origin,attribution,canonical(evidence),now,commit_id))
        from .autobiography_lifecycle import append_event
        append_event(self, record_id, revision, head, state)
        return {'record_id':record_id,'revision':revision}

    def _receipt(self, request_key, payload_hash):
        request_key=self._request_key(request_key)
        p=self.principal
        row=self.db.execute('SELECT * FROM autobio_commits WHERE owner=? AND agent=? AND request_key=?',
                            (p.owner,p.agent,request_key)).fetchone()
        if row and row['payload_hash']!=payload_hash:
            raise PersonalError('idempotency_mismatch','idempotency key belongs to a different request')
        return json.loads(row['receipt']) if row else None

    def _request_key(self, request_key):
        identifier(request_key,'idempotency key')
        return digest([self.principal.audience,request_key])

    def validate_core_budget(self):
        from .autobiography_context import personal_context
        personal_context(self,budget=self.context_budget)

    def update(self, *, idempotency_key, **edit):
        self.principal.require('author','operator')
        payload_hash=digest({'edit':edit,'actor':vars(self.principal),'origin':'interactive'})
        with self.transaction():
            receipt=self._receipt(idempotency_key,payload_hash)
            if receipt:return receipt
            commit_id=uuid.uuid4().hex
            revision=self._write(edit,commit_id,'interactive')
            self.validate_core_budget()
            receipt={'commit_id':commit_id,'edits':[revision],'committed_at':self.clock()}
            p=self.principal
            self.db.execute('INSERT INTO autobio_commits VALUES(?,?,?,?,?,?,?)',
                (commit_id,p.owner,p.agent,self._request_key(idempotency_key),payload_hash,canonical(receipt),self.clock()))
        self.index_record(revision['record_id'])
        return receipt

    def transition(self, record_id, **options):
        from .autobiography_lifecycle import transition
        return transition(self, record_id, **options)

    def restore(self, record_id, revision, *, expected_revision, reason, idempotency_key):
        old=self.get(record_id,revision)
        return self.update(record_id=record_id,expected_revision=expected_revision,title=old['title'],body=old['body'],
                           state=old['state'],reason=reason,attribution=old['attribution'],evidence=old['evidence'],idempotency_key=idempotency_key)

    def index_record(self, record_id):
        from .autobiography_index import index_record
        return index_record(self,record_id)

    def index_status(self, *, limit=20):
        from .autobiography_index import index_status
        return index_status(self,limit=limit)

    def repair_index(self, **options):
        from .autobiography_index import repair_index
        return repair_index(self,**options)

    def search(self, query, *, segment=None, limit=12):
        tokens=re.findall(r'\w+',str(query),flags=re.UNICODE)[:16]
        if not tokens:return {'records':[],'mode':'lexical','index':self.index_status()}
        expression=' OR '.join('"'+t+'"' for t in tokens)
        p=self.principal
        with self.transaction(write=False):
            audience, args = p.audience_sql('r')
            rows=self.db.execute(f'''SELECT r.id FROM autobio_search f JOIN autobio_records r ON r.id=f.record_id
                JOIN autobio_index_state i ON i.record_id=r.id AND i.revision=r.revision
                WHERE autobio_search MATCH ? AND r.owner=? AND r.agent=? AND {audience} AND r.state='active'
                AND (? IS NULL OR r.segment=?) ORDER BY rank,r.id LIMIT ?''',
                (expression,p.owner,p.agent,*args,segment,segment,max(1,min(int(limit),50)))).fetchall()
            return {'records':[self._get(r[0]) for r in rows],'mode':'lexical','index':self.index_status()}

    def export(self):
        self.principal.require('operator','author')
        from .personal_privacy import derivative_registers
        registers=derivative_registers(self)
        with self.transaction(write=False):
            p=self.principal
            records=[self._get(r[0]) for r in self.db.execute('SELECT id FROM autobio_records WHERE owner=? AND agent=? AND audience=? ORDER BY id',(p.owner,p.agent,p.audience))]
            if len(records)>1000:
                raise PersonalError('budget','use paginated record/history reads for exports exceeding 1000 records')
            revisions=[dict(r) for record in records for r in self.db.execute('SELECT * FROM autobio_revisions WHERE record_id=? ORDER BY revision',(record['id'],))]
            return {'registers':registers.export_scoped() if registers and p.role=='operator' else {},'schema_version':SCHEMA_VERSION,'scope':[p.owner,p.agent,p.audience],'records':records,'revisions':revisions,
                    'lifecycle':[dict(e) for record in records for e in self.db.execute('SELECT * FROM autobio_lifecycle WHERE record_id=? ORDER BY revision',(record['id'],))],
                    'documents':[{'filename':r['id']+'.md','markdown':r['body'],'revision':r['revision']} for r in records],
                    'privacy_notice':'Downloaded exports and external backups require operator-managed deletion and retention.'}

    def erase(self, record_id, *, expected_revision):
        """Privileged erasure of a record and related provisional evidence.

        Suppress its source identities across future generations. External raw
        transcripts, downloaded exports and backups remain operator-managed.
        """
        self.principal.require('operator')
        from .personal_privacy import derivative_registers
        registers=derivative_registers(self)
        with self.transaction():
            head=self._head(record_id)
            if head['revision']!=expected_revision:
                raise PersonalError('conflict','record changed before erasure')
            sources=set()
            for row in self.db.execute('SELECT evidence FROM autobio_revisions WHERE record_id=?',(record_id,)):
                sources.update(e['source_id'] for e in json.loads(row[0]))
            p=self.principal
            derivatives=registers.purge_sources(sources) if registers else {}
            for source in sources:
                self.db.execute('INSERT OR IGNORE INTO autobio_suppressions VALUES(?,?,?,?)',(p.owner,p.agent,source,self.clock()))
                if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='autobio_candidates'").fetchone():
                    self.db.execute('DELETE FROM autobio_candidates WHERE source_id=? AND owner=? AND agent=?',(source,p.owner,p.agent))
                    self.db.execute('DELETE FROM autobio_extraction_parts WHERE source_id=?',(source,))
                    self.db.execute('DELETE FROM autobio_model_cache WHERE source_id=?',(source,))
                    if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='autobio_model_errors'").fetchone():
                        self.db.execute('DELETE FROM autobio_model_errors WHERE source_id=?',(source,))
                self.db.execute('DELETE FROM autobio_sources WHERE id=? AND owner=? AND agent=?',(source,p.owner,p.agent))
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='autobio_model_cache'").fetchone():
                for cached in self.db.execute('SELECT id,record_ids FROM autobio_model_cache').fetchall():
                    if record_id in json.loads(cached['record_ids']):self.db.execute('DELETE FROM autobio_model_cache WHERE id=?',(cached['id'],))
                if 'source_ids' in {r[1] for r in self.db.execute('PRAGMA table_info(autobio_model_cache)')}:
                    for cached in self.db.execute('SELECT id,source_ids FROM autobio_model_cache').fetchall():
                        if sources.intersection(json.loads(cached['source_ids'])):self.db.execute('DELETE FROM autobio_model_cache WHERE id=?',(cached['id'],))
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE name='autobio_shadow_proposals'").fetchone():
                for row in self.db.execute('SELECT id,context_manifest FROM autobio_cycles').fetchall():
                    if record_id in {r['record_id'] for r in json.loads(row['context_manifest']).get('records',[])}:
                        self.db.execute('DELETE FROM autobio_shadow_proposals WHERE cycle_id=?',(row['id'],))
            self.db.execute('DELETE FROM autobio_search WHERE record_id=?',(record_id,))
            self.db.execute('DELETE FROM autobio_index_state WHERE record_id=?',(record_id,))
            self.db.execute('DELETE FROM autobio_records WHERE id=?',(record_id,))
            receipt=uuid.uuid4().hex
            self.db.execute('INSERT INTO autobio_erasure_receipts VALUES(?,?,?,?,?,?)',
                (receipt,p.owner,p.agent,canonical([record_id]),canonical(sorted(sources)),self.clock()))
            return {'register_derivatives':derivatives,'erasure_id':receipt,'erased_record_ids':[record_id],'suppressed_source_ids':sorted(sources),
                    'erasure_scope':'logical_application_state','forensic_erasure':False,
                    'external_copies':'SQLite WAL/freelist artifacts, filesystem snapshots, raw transcripts, downloaded exports and backups are not forensically erased by this transaction.'}
