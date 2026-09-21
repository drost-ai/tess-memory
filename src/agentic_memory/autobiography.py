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

SEGMENTS = ('Self', 'Principles', 'People', 'Relationships', 'Experiences', 'Reflections')
ATTRIBUTIONS = ('user_statement', 'observed_fact', 'agent_reflection', 'extractor_inference', 'inherited')
ROLES = ('reader', 'author', 'extractor', 'dream', 'operator')
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


@dataclass(frozen=True)
class PersonalPrincipal:
    owner: str
    agent: str
    audience: str = 'owner'
    role: str = 'reader'
    session_id: str = 'unbound'

    def __post_init__(self):
        for field in ('owner', 'agent', 'audience', 'session_id'):
            identifier(getattr(self, field), field)
        if self.role not in ROLES:
            raise PersonalError('forbidden', 'unavailable personal role')
        if self.role in ('author','dream') and self.session_id == 'unbound':
            raise PersonalError('forbidden','authoring requires a trusted session binding')

    def require(self, *roles):
        if self.role not in roles:
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

    def _head(self, record_id):
        p = self.principal
        row = self.db.execute('SELECT * FROM autobio_records WHERE id=? AND owner=? AND agent=? AND audience=?',
                              (record_id, p.owner, p.agent, p.audience)).fetchone()
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
        result['evidence_availability'] = [self.evidence_available(e) for e in result['evidence']]
        return result

    def get(self, record_id, revision=None):
        with self.transaction(write=False):
            return self._get(record_id, revision)

    def list_records(self, *, segment=None, limit=100, after=''):
        if segment is not None and segment not in SEGMENTS:
            raise PersonalError('invalid', 'unknown personal segment')
        limit = max(1, min(int(limit), 100))
        p = self.principal
        with self.transaction(write=False):
            rows = self.db.execute('''SELECT id FROM autobio_records WHERE owner=? AND agent=? AND audience=?
                AND id>? AND (? IS NULL OR segment=?) ORDER BY id LIMIT ?''',
                (p.owner, p.agent, p.audience, after, segment, segment, limit+1)).fetchall()
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

    def _write(self, edit, commit_id, origin):
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
        expected=edit.get('expected_revision')
        if type(expected) is not int or expected<0:
            raise PersonalError('invalid','expected_revision must be a nonnegative integer')
        if record_id:
            head=self._head(record_id)
            if expected!=head['revision']:
                raise PersonalError('conflict','record revision changed; reread before proposing another edit')
            if edit.get('segment',head['segment'])!=head['segment'] or edit.get('subject',head['subject'])!=head['subject']:
                raise PersonalError('invalid','segment and subject are immutable')
            revision=expected+1
            self.db.execute('UPDATE autobio_records SET title=?,revision=?,state=?,updated=? WHERE id=?',
                            (title,revision,state,now,record_id))
        else:
            if expected!=0 or edit.get('segment') not in SEGMENTS:
                raise PersonalError('invalid','new records require expected_revision=0 and a segment')
            record_id=uuid.uuid4().hex
            subject=edit.get('subject') or ('core' if edit['segment'] in ('Self','Principles') else record_id)
            identifier(subject,'subject')
            if edit['segment'] in ('Self','Principles') and subject!='core':
                raise PersonalError('invalid','core segments have one current document')
            revision=1
            try:
                self.db.execute('INSERT INTO autobio_records VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (record_id,p.owner,p.agent,p.audience,edit['segment'],subject,title,revision,state,now,now))
            except sqlite3.IntegrityError as exc:
                raise PersonalError('conflict','a record for this segment and subject already exists') from exc
        self.db.execute('INSERT INTO autobio_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (record_id,revision,body,title,state,reason,p.session_id,origin,attribution,canonical(evidence),now,commit_id))
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

    def restore(self, record_id, revision, *, expected_revision, reason, idempotency_key):
        old=self.get(record_id,revision)
        return self.update(record_id=record_id,expected_revision=expected_revision,title=old['title'],body=old['body'],
                           state=old['state'],reason=reason,attribution=old['attribution'],evidence=old['evidence'],idempotency_key=idempotency_key)

    def index_record(self, record_id):
        try:
            with self.transaction():
                record=self._get(record_id)
                self.db.execute('DELETE FROM autobio_search WHERE record_id=?',(record_id,))
                if record['state']=='active':
                    self.db.execute('INSERT INTO autobio_search(record_id,body) VALUES(?,?)',(record_id,record['title']+'\n'+record['body']))
                self.db.execute('INSERT OR REPLACE INTO autobio_index_state VALUES(?,?)',(record_id,record['revision']))
            return True
        except (sqlite3.Error,PersonalError):
            return False  # acknowledged authored state survives index failure

    def search(self, query, *, segment=None, limit=12):
        tokens=re.findall(r'\w+',str(query),flags=re.UNICODE)[:16]
        if not tokens:return {'records':[],'mode':'lexical'}
        expression=' OR '.join('"'+t+'"' for t in tokens)
        p=self.principal
        with self.transaction(write=False):
            rows=self.db.execute('''SELECT r.id FROM autobio_search f JOIN autobio_records r ON r.id=f.record_id
                JOIN autobio_index_state i ON i.record_id=r.id AND i.revision=r.revision
                WHERE autobio_search MATCH ? AND r.owner=? AND r.agent=? AND r.audience=? AND r.state='active'
                AND (? IS NULL OR r.segment=?) ORDER BY rank,r.id LIMIT ?''',
                (expression,p.owner,p.agent,p.audience,segment,segment,max(1,min(int(limit),50)))).fetchall()
            return {'records':[self._get(r[0]) for r in rows],'mode':'lexical'}

    def export(self):
        self.principal.require('operator','author')
        with self.transaction(write=False):
            p=self.principal
            records=[self._get(r[0]) for r in self.db.execute('SELECT id FROM autobio_records WHERE owner=? AND agent=? AND audience=? ORDER BY id',(p.owner,p.agent,p.audience))]
            if len(records)>1000:
                raise PersonalError('budget','use paginated record/history reads for exports exceeding 1000 records')
            revisions=[dict(r) for record in records for r in self.db.execute('SELECT * FROM autobio_revisions WHERE record_id=? ORDER BY revision',(record['id'],))]
            return {'schema_version':SCHEMA_VERSION,'scope':[p.owner,p.agent,p.audience],'records':records,'revisions':revisions,
                    'documents':[{'filename':r['id']+'.md','markdown':r['body'],'revision':r['revision']} for r in records],
                    'privacy_notice':'Downloaded exports and external backups require operator-managed deletion and retention.'}

    def erase(self, record_id, *, expected_revision):
        """Privileged erasure of a record and related provisional evidence.

        Suppress its source identities across future generations. External raw
        transcripts, downloaded exports and backups remain operator-managed.
        """
        self.principal.require('operator')
        with self.transaction():
            head=self._head(record_id)
            if head['revision']!=expected_revision:
                raise PersonalError('conflict','record changed before erasure')
            sources=set()
            for row in self.db.execute('SELECT evidence FROM autobio_revisions WHERE record_id=?',(record_id,)):
                sources.update(e['source_id'] for e in json.loads(row[0]))
            p=self.principal
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
            return {'erasure_id':receipt,'erased_record_ids':[record_id],'suppressed_source_ids':sorted(sources),
                    'external_copies':'Raw transcripts, downloaded exports and backups are outside this erase transaction.'}
