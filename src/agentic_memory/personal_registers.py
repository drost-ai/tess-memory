"""Scoped personal registers. Internal extraction/privacy hooks never commit.

Leases authorize database work only: a claim/receipt is not platform delivery.
All mutating public methods serialize CAS and receipts in the store transaction.
"""
import json
import math
import uuid

from .autobiography import PersonalError, canonical, digest

SCHEMA = '''
CREATE TABLE IF NOT EXISTS autobio_register_candidates(
 id TEXT PRIMARY KEY,owner TEXT NOT NULL,agent TEXT NOT NULL,audience TEXT NOT NULL,
 kind TEXT NOT NULL,source_id TEXT NOT NULL,source_generation TEXT NOT NULL,range_id TEXT NOT NULL,
 candidate_id TEXT NOT NULL,proposal TEXT NOT NULL,evidence TEXT NOT NULL,activation_version TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending',review_after REAL,disposition TEXT,reason TEXT,
 operation_id TEXT,fence INTEGER NOT NULL DEFAULT 0,lease_token TEXT,lease_until REAL,
 UNIQUE(owner,agent,audience,kind,source_id,source_generation,range_id,candidate_id));
CREATE TABLE IF NOT EXISTS autobio_register_items(
 id TEXT PRIMARY KEY,owner TEXT NOT NULL,agent TEXT NOT NULL,audience TEXT NOT NULL,kind TEXT NOT NULL,
 key TEXT NOT NULL,title TEXT NOT NULL,body TEXT NOT NULL,question TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'active',revision INTEGER NOT NULL DEFAULT 1,due_at REAL,review_after REAL,
 reason TEXT NOT NULL DEFAULT '',operation_id TEXT,fence INTEGER NOT NULL DEFAULT 0,lease_token TEXT,lease_until REAL,
 UNIQUE(owner,agent,audience,kind,key));
CREATE TABLE IF NOT EXISTS autobio_register_evidence(
 owner TEXT NOT NULL,agent TEXT NOT NULL,audience TEXT NOT NULL,kind TEXT NOT NULL,
 item_id TEXT NOT NULL,candidate_id TEXT NOT NULL,source_id TEXT NOT NULL,evidence TEXT NOT NULL,
 PRIMARY KEY(item_id,candidate_id));
CREATE TABLE IF NOT EXISTS autobio_register_progress(
 id TEXT PRIMARY KEY,owner TEXT NOT NULL,agent TEXT NOT NULL,audience TEXT NOT NULL,kind TEXT NOT NULL,
 item_id TEXT NOT NULL,operation_id TEXT NOT NULL,summary TEXT NOT NULL,findings TEXT NOT NULL,
 next_step TEXT NOT NULL,outcome TEXT NOT NULL,review_after REAL,delivery_receipt TEXT,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS autobio_register_receipts(
 owner TEXT NOT NULL,agent TEXT NOT NULL,audience TEXT NOT NULL,request_id TEXT NOT NULL,
 operation_id TEXT,request_hash TEXT NOT NULL,response TEXT NOT NULL,
 PRIMARY KEY(owner,agent,audience,request_id));
'''


class PersonalRegisters:
    def __init__(self, store, operations):
        self.store, self.operations = store, operations
        with store.transaction():
            for sql in SCHEMA.split(';'):
                if sql.strip():
                    store.db.execute(sql)

    @property
    def scope(self):
        p = self.store.principal
        return (p.owner, p.agent, p.audience)

    def _kind(self, kind):
        if kind not in ('follow_up', 'topic'):
            raise PersonalError('invalid', 'unknown register kind')

    def _read_role(self, kind, candidates=False):
        self._kind(kind)
        roles = [kind + '_reconciler', 'operator']
        if not candidates:
            roles.append({'follow_up': 'care', 'topic': 'research'}[kind])
        self.store.principal.require(*roles)

    def _source_live(self, source_id, generation):
        row = self.store.db.execute('SELECT generation FROM autobio_sources WHERE id=? AND owner=? AND agent=? AND audience=?', (source_id, *self.scope)).fetchone()
        return bool(row and row['generation'] == generation and not self.store.is_suppressed(source_id))

    def _candidate_live(self, row):
        if not self._source_live(row['source_id'], row['source_generation']):
            return False
        for evidence in json.loads(row['evidence']):
            if not isinstance(evidence, dict):
                return False
            if not self._source_live(evidence.get('source_id'), evidence.get('generation')):
                return False
        return True

    def _item_live(self, item_id):
        rows = self.store.db.execute('SELECT c.* FROM autobio_register_evidence e JOIN autobio_register_candidates c ON c.id=e.candidate_id WHERE e.owner=? AND e.agent=? AND e.audience=? AND e.item_id=?', (*self.scope, item_id)).fetchall()
        return bool(rows) and all(self._candidate_live(r) for r in rows)

    def _fresh_item(self, kind, item_id):
        row = self._item(kind, item_id)
        if row['state'] == 'suppressed' or not self._item_live(item_id):
            raise PersonalError('stale_source', 'register item suppressed or evidence unavailable')
        return row

    def _authority(self, operation_id, operation_fence, kind, reconcile=False):
        self._kind(kind)
        if type(operation_fence) is not int:
            raise PersonalError('fenced', 'operation fence must be an integer')
        role = kind + '_reconciler' if reconcile else {'follow_up': 'care', 'topic': 'research'}[kind]
        self.store.principal.require(role, 'operator')
        op = self.operations._operation(operation_id, operation_fence)
        if op['kind'] != role:
            raise PersonalError('forbidden', 'operation has wrong register role')
        return op

    @staticmethod
    def _text(value, name, maximum=32000, empty=True):
        if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
            raise PersonalError('invalid', 'invalid ' + name)
        return value

    @staticmethod
    def _time(value):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise PersonalError('invalid', 'timestamp must be finite epoch seconds')
        return float(value)

    def _rows(self, table, kind, where='1', args=()):
        return self.store.db.execute(
            f'SELECT * FROM autobio_register_{table} WHERE owner=? AND agent=? AND audience=? AND kind=? AND {where}',
            (*self.scope, kind, *args)).fetchall()

    def _decode(self, row):
        result = dict(row)
        for key in ('proposal', 'evidence', 'findings', 'delivery_receipt'):
            if result.get(key) is not None:
                result[key] = json.loads(result[key])
        if 'key' in result and 'revision' in result:
            history = self.store.db.execute('SELECT * FROM autobio_register_progress WHERE owner=? AND agent=? AND audience=? AND kind=? AND item_id=? ORDER BY created DESC,rowid DESC LIMIT 20', (*self.scope, result['kind'], result['id'])).fetchall()
            result['progress'] = [self._decode(entry) for entry in reversed(history)]
        if 'proposal' in result:
            result['source_candidate_id'] = result['candidate_id']
            result['candidate_id'] = result['id']
        return result

    def _item(self, kind, item_id, revision=None):
        rows = self._rows('items', kind, 'id=?', (item_id,))
        if not rows:
            raise PersonalError('not_found', 'register item unavailable')
        row = rows[0]
        if revision is not None and (type(revision) is not int or revision != row['revision']):
            raise PersonalError('conflict', 'register revision changed')
        return row

    def _receipt(self, request_id, payload):
        self._text(request_id, 'request_id', 128, False)
        fingerprint = digest(payload)
        row = self.store.db.execute('SELECT * FROM autobio_register_receipts WHERE owner=? AND agent=? AND audience=? AND request_id=?', (*self.scope, request_id)).fetchone()
        if row and row['request_hash'] != fingerprint:
            raise PersonalError('conflict', 'request_id reused with different input')
        response = json.loads(row['response']) if row else None
        if response is not None and payload[0] != 'suppress':
            items = response.get('items', []) + ([response['item']] if 'item' in response else [])
            for item in items:
                if payload[0] == 'reconcile' and item['state'] == 'suppressed':
                    current = self._item(item['kind'], item['id'])
                    if current['state'] != 'suppressed' or not self._item_live(item['id']):
                        raise PersonalError('stale_source', 'receipt evidence unavailable')
                else:
                    self._fresh_item(item['kind'], item['id'])
            for outcome in response.get('outcomes', []):
                candidates = self._rows('candidates', payload[3], 'id=?', (outcome['candidate_id'],))
                if not candidates or not self._candidate_live(candidates[0]):
                    raise PersonalError('stale_source', 'receipt evidence unavailable')
        return fingerprint, response

    def _save(self, request_id, operation_id, fingerprint, response):
        self.store.db.execute('INSERT INTO autobio_register_receipts VALUES(?,?,?,?,?,?,?)', (*self.scope, request_id, operation_id, fingerprint, canonical(response)))
        return response

    def publish_candidate(self, kind, source_id, source_generation, range_id, candidate_id, proposal, evidence, activation_version):
        """Extraction transaction hook; caller owns activation gate and transaction."""
        self._kind(kind)
        if not self.store.db.in_transaction:
            raise RuntimeError('publish_candidate requires extraction transaction')
        if not isinstance(proposal, dict) or not isinstance(evidence, list) or len(evidence) > 32 or len(canonical(proposal)) > 40000 or len(canonical(evidence)) > 64000:
            raise PersonalError('invalid', 'unbounded register candidate')
        if not evidence or not self._source_live(source_id, source_generation) or any(not isinstance(e, dict) or not self._source_live(e.get('source_id'), e.get('generation')) for e in evidence):
            raise PersonalError('stale_source', 'candidate source evidence unavailable')
        identity = (*self.scope, kind, source_id, source_generation, range_id, candidate_id)
        cursor = self.store.db.execute('''INSERT OR IGNORE INTO autobio_register_candidates
            (id,owner,agent,audience,kind,source_id,source_generation,range_id,candidate_id,proposal,evidence,activation_version)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''', (digest(identity), *identity, canonical(proposal), canonical(evidence), str(activation_version)))
        return {'candidate_id': digest(identity), 'inserted': bool(cursor.rowcount)}

    def _page(self, table, kind, state, limit, after, due_before=None):
        self._kind(kind)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise PersonalError('invalid', 'limit must be 1..100')
        self._text(after, 'after', 128)
        where, args = 'state=? AND id>?', [state, after]
        if due_before is not None:
            where += ' AND COALESCE(review_after,due_at,0)<=?'
            args.append(self._time(due_before))
        rows = self._rows(table, kind, where + ' ORDER BY id LIMIT ?', (*args, limit))
        visible = [r for r in rows if (self._candidate_live(r) if table == 'candidates' else self._item_live(r['id']))]
        return {'kind': kind, table: [self._decode(r) for r in visible], 'after': rows[-1]['id'] if rows else ''}

    def candidates(self, kind, state='pending', limit=20, after=''):
        self._read_role(kind, True)
        with self.store.snapshot():
            return self._page('candidates', kind, state, limit, after)

    def list(self, kind, state='active', due_before=None, limit=20, after=''):
        self._read_role(kind)
        with self.store.snapshot():
            return self._page('items', kind, state, limit, after, due_before)

    def reconcile_claim(self, operation_id, operation_fence, kind, limit=20):
        with self.store.transaction():
            op = self._authority(operation_id, operation_fence, kind, True)
            if type(limit) is not int or not 1 <= limit <= 100:
                raise PersonalError('invalid', 'limit must be 1..100')
            self.expire_leases()
            rows = self._rows('candidates', kind, 'operation_id=?', (operation_id,))
            if not rows:
                rows = self._rows('candidates', kind, "state='pending' AND operation_id IS NULL AND (review_after IS NULL OR review_after<=?) ORDER BY id", (self.store.clock(),))
                rows = [r for r in rows if self._candidate_live(r)][:limit]
                if rows:
                    token, fence = uuid.uuid4().hex, max(r['fence'] for r in rows) + 1
                    until = op['deadline']
                    for row in rows:
                        self.store.db.execute('UPDATE autobio_register_candidates SET operation_id=?,fence=?,lease_token=?,lease_until=? WHERE id=?', (operation_id, fence, token, until, row['id']))
                    rows = self._rows('candidates', kind, 'operation_id=?', (operation_id,))
            if any(not self._candidate_live(r) for r in rows):
                raise PersonalError('stale_source', 'claimed candidate evidence changed')
            return {'status': 'claimed' if rows else 'empty', 'candidates': [self._decode(r) for r in rows],
                    'items': [self._decode(r) for r in self._rows('items', kind, "state='active' ORDER BY id") if self._item_live(r['id'])],
                    'fence': rows[0]['fence'] if rows else 0, 'lease_token': rows[0]['lease_token'] if rows else '',
                    'lease_until': rows[0]['lease_until'] if rows else None}

    def _lease(self, row, operation_id, fence, lease_token):
        if type(fence) is not int or row['operation_id'] != operation_id or row['fence'] != fence or not lease_token or row['lease_token'] != lease_token or row['lease_until'] is None or row['lease_until'] <= self.store.clock():
            raise PersonalError('fenced', 'register lease expired or replaced')

    def reconcile_commit(self, operation_id, operation_fence, kind, fence, lease_token, request_id, edits, outcomes):
        with self.store.transaction():
            self._authority(operation_id, operation_fence, kind, True)
            fingerprint, replay = self._receipt(request_id, ['reconcile', operation_id, operation_fence, kind, fence, lease_token, edits, outcomes])
            if replay is not None:
                return replay
            if not isinstance(edits, list) or len(edits) > 100 or not isinstance(outcomes, list) or len(outcomes) > 100:
                raise PersonalError('invalid', 'invalid reconciliation batch')
            rows = self._rows('candidates', kind, 'operation_id=?', (operation_id,))
            if not rows:
                raise PersonalError('fenced', 'no claimed candidates')
            for row in rows:
                if not self._candidate_live(row):
                    raise PersonalError('stale_source', 'candidate evidence unavailable')
                self._lease(row, operation_id, fence, lease_token)
            # Public candidate identity is the stable scoped row id, not extractor-local candidate_id.
            candidates = {r['id']: r for r in rows}
            if any(not isinstance(o, dict) for o in outcomes) or len(outcomes) != len(candidates) or {o.get('candidate_id') for o in outcomes} != set(candidates):
                raise PersonalError('invalid', 'outcomes must cover the whole claimed batch exactly once')
            changed = []
            for edit in edits:
                if not isinstance(edit, dict):
                    raise PersonalError('invalid', 'invalid edit')
                action = edit.get('action', edit.get('op', edit.get('operation')))
                if action not in ('create', 'update', 'suppress'):
                    raise PersonalError('invalid', 'invalid edit action')
                key = self._text(edit.get('key'), 'key', 256, False)
                title = self._text(edit.get('title'), 'title', 1000, False)
                body = self._text(edit.get('body'), 'body')
                reason = self._text(edit.get('reason', ''), 'reason', 4000)
                question = self._text(edit.get('question', ''), 'question', 4000)
                due = self._time(edit.get('due_at'))
                if action == 'create':
                    if edit.get('item_id') or type(edit.get('expected_revision')) is not int or edit['expected_revision'] != 0:
                        raise PersonalError('conflict', 'create requires revision zero and no item_id')
                    if self._rows('items', kind, 'key=?', (key,)):
                        raise PersonalError('conflict', 'register key already exists')
                    item_id = uuid.uuid4().hex
                    self.store.db.execute('INSERT INTO autobio_register_items(id,owner,agent,audience,kind,key,title,body,question,due_at,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)', (item_id, *self.scope, kind, key, title, body, question, due, reason))
                else:
                    if action != 'suppress':
                        self._fresh_item(kind, edit.get('item_id'))
                    row = self._item(kind, edit.get('item_id'), edit.get('expected_revision'))
                    if edit.get('expected_revision') is None:
                        raise PersonalError('invalid', 'expected_revision required')
                    if row['key'] != key:
                        raise PersonalError('invalid', 'register key is immutable')
                    if row['state'] == 'suppressed' and action != 'suppress':
                        raise PersonalError('conflict', 'suppressed item cannot be reactivated')
                    item_id = row['id']
                    self.store.db.execute('''UPDATE autobio_register_items SET title=?,body=?,question=?,due_at=?,reason=?,state=?,revision=revision+1,
                        operation_id=NULL,lease_token=NULL,lease_until=NULL,fence=fence+1 WHERE id=?''', (title, body, question, due, reason, 'suppressed' if action == 'suppress' else row['state'], item_id))
                changed.append(self._decode(self._item(kind, item_id)))
            normalized = []
            for outcome in outcomes:
                disposition = outcome.get('disposition')
                if disposition not in ('applied', 'duplicate', 'rejected', 'deferred'):
                    raise PersonalError('invalid', 'invalid candidate disposition')
                row = candidates[outcome['candidate_id']]
                item_id = outcome.get('item_id')
                # Creates can be linked by their immutable key, avoiding client-generated IDs.
                if not item_id and outcome.get('key'):
                    matches = self._rows('items', kind, 'key=?', (outcome['key'],))
                    item_id = matches[0]['id'] if matches else None
                if disposition in ('applied', 'duplicate') and not item_id:
                    if len(changed) == 1:
                        item_id = changed[0]['id']
                    else:
                        raise PersonalError('invalid', 'applied/duplicate outcome requires item_id or key')
                if item_id:
                    linked = self._item(kind, item_id)
                    if linked['state'] == 'suppressed' and disposition != 'rejected':
                        raise PersonalError('conflict', 'cannot apply candidate to suppressed item')
                    self.store.db.execute('INSERT OR IGNORE INTO autobio_register_evidence VALUES(?,?,?,?,?,?,?,?)', (*self.scope, kind, item_id, row['id'], row['source_id'], row['evidence']))
                reason = self._text(outcome.get('reason', ''), 'reason', 4000)
                review = self._time(outcome.get('review_after'))
                if disposition == 'deferred' and (review is None or review <= self.store.clock()):
                    raise PersonalError('invalid', 'deferred candidate needs future review_after')
                self.store.db.execute('''UPDATE autobio_register_candidates SET state=?,disposition=?,reason=?,review_after=?,operation_id=NULL,lease_token=NULL,lease_until=NULL WHERE id=?''', ('pending' if disposition == 'deferred' else disposition, disposition, reason, review, row['id']))
                normalized.append({**outcome, 'item_id': item_id})
            for item in changed:
                if not self._item_live(item['id']):
                    raise PersonalError('invalid', 'every edited item must retain live candidate evidence')
            return self._save(request_id, operation_id, fingerprint, {'status': 'committed', 'items': changed, 'outcomes': normalized, 'request_id': request_id})

    def claim(self, operation_id, operation_fence, kind, item_id, expected_revision, request_id):
        with self.store.transaction():
            op = self._authority(operation_id, operation_fence, kind)
            fingerprint, replay = self._receipt(request_id, ['claim', operation_id, operation_fence, kind, item_id, expected_revision])
            if replay is not None:
                return replay
            if type(expected_revision) is not int:
                raise PersonalError('invalid', 'expected_revision must be an integer')
            self._fresh_item(kind, item_id)
            row = self._item(kind, item_id, expected_revision)
            if row['state'] != 'active' or (row['review_after'] is not None and row['review_after'] > self.store.clock()) or (row['lease_until'] and row['lease_until'] > self.store.clock()):
                raise PersonalError('conflict', 'item not available for claim')
            token, fence = uuid.uuid4().hex, row['fence'] + 1
            until = op['deadline']
            self.store.db.execute('UPDATE autobio_register_items SET operation_id=?,fence=?,lease_token=?,lease_until=? WHERE id=?', (operation_id, fence, token, until, item_id))
            return self._save(request_id, operation_id, fingerprint, {'status': 'claimed', 'item': self._decode(self._item(kind, item_id)), 'fence': fence, 'lease_token': token, 'lease_until': until, 'request_id': request_id})

    def progress(self, operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision, request_id, summary, findings=None, next_step='', review_after=None):
        return self._advance(operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision, request_id, summary, [] if findings is None else findings, next_step, review_after, 'progressed', None)

    def complete(self, operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision, request_id, outcome, summary, review_after=None, delivery_receipt=None):
        if outcome not in ('completed', 'deferred', 'exhausted'):
            raise PersonalError('invalid', 'invalid completion outcome')
        return self._advance(operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision, request_id, summary, [], '', review_after, outcome, delivery_receipt)

    def _advance(self, operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision, request_id, summary, findings, next_step, review_after, outcome, delivery_receipt):
        with self.store.transaction():
            self._authority(operation_id, operation_fence, kind)
            fingerprint, replay = self._receipt(request_id, ['advance', operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision, summary, findings, next_step, review_after, outcome, delivery_receipt])
            if replay is not None:
                return replay
            if type(expected_revision) is not int:
                raise PersonalError('invalid', 'expected_revision must be an integer')
            self._fresh_item(kind, item_id)
            row = self._item(kind, item_id, expected_revision)
            self._lease(row, operation_id, fence, lease_token)
            if row['state'] != 'active':
                raise PersonalError('conflict', 'item is not active')
            self._text(summary, 'summary', 8000, False)
            self._text(next_step, 'next_step', 4000)
            if not isinstance(findings, list) or len(findings) > 20:
                raise PersonalError('invalid', 'findings must have at most 20 entries')
            for finding in findings:
                if not isinstance(finding, dict) or set(finding) - {'text', 'url'}:
                    raise PersonalError('invalid', 'invalid finding')
                self._text(finding.get('text'), 'finding text', 4000, False)
                if 'url' in finding:
                    self._text(finding['url'], 'finding URL', 2048, False)
                    if not finding['url'].startswith(('https://', 'http://')):
                        raise PersonalError('invalid', 'finding URL must be HTTP(S)')
            review = self._time(review_after)
            if outcome == 'deferred' and (review is None or review <= self.store.clock()):
                raise PersonalError('invalid', 'deferred work needs future review_after')
            if delivery_receipt is not None and (not isinstance(delivery_receipt, dict) or len(canonical(delivery_receipt)) > 8000):
                raise PersonalError('invalid', 'invalid delivery receipt')
            progress_id = uuid.uuid4().hex
            self.store.db.execute('INSERT INTO autobio_register_progress VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (progress_id, *self.scope, kind, item_id, operation_id, summary, canonical(findings), next_step, outcome, review, canonical(delivery_receipt) if delivery_receipt is not None else None, self.store.clock()))
            state = outcome if outcome in ('completed', 'exhausted') else 'active'
            self.store.db.execute('UPDATE autobio_register_items SET state=?,revision=revision+1,review_after=? WHERE id=?', (state, review, item_id))
            if outcome != 'progressed':
                self.store.db.execute('UPDATE autobio_register_items SET operation_id=NULL,lease_token=NULL,lease_until=NULL,fence=fence+1 WHERE id=?', (item_id,))
            return self._save(request_id, operation_id, fingerprint, {'status': outcome, 'item': self._decode(self._item(kind, item_id)), 'progress_id': progress_id, 'request_id': request_id})

    def suppress(self, kind, item_id, expected_revision, request_id, reason):
        self.store.principal.require('operator')
        self._kind(kind)
        with self.store.transaction():
            fingerprint, replay = self._receipt(request_id, ['suppress', kind, item_id, expected_revision, reason])
            if replay is not None:
                return replay
            self._text(reason, 'reason', 4000, False)
            if type(expected_revision) is not int:
                raise PersonalError('invalid', 'expected_revision must be an integer')
            self._item(kind, item_id, expected_revision)
            self.store.db.execute("UPDATE autobio_register_items SET state='suppressed',reason=?,revision=revision+1,fence=fence+1,operation_id=NULL,lease_token=NULL,lease_until=NULL WHERE id=?", (reason, item_id))
            return self._save(request_id, None, fingerprint, {'status': 'suppressed', 'item': self._decode(self._item(kind, item_id)), 'request_id': request_id})

    def validate_claim(self, operation_id, operation_fence, kind, item_id, fence, lease_token, expected_revision):
        with self.store.snapshot():
            try:
                self._authority(operation_id, operation_fence, kind)
                if type(expected_revision) is not int:
                    return {'valid': False}
                self._fresh_item(kind, item_id)
                row = self._item(kind, item_id, expected_revision)
                self._lease(row, operation_id, fence, lease_token)
                return {'valid': row['state'] == 'active'}
            except PersonalError:
                return {'valid': False}

    def export_scoped(self):
        self.store.principal.require('operator')
        with self.store.snapshot():
            result = {}
            for table in ('candidates', 'items', 'evidence', 'progress', 'receipts'):
                rows = self.store.db.execute(f'SELECT * FROM autobio_register_{table} WHERE owner=? AND agent=? AND audience=?', self.scope).fetchall()
                result[table] = [self._decode(r) for r in rows]
            return result

    def _release(self, clause, args):
        if not self.store.db.in_transaction:
            raise RuntimeError('register maintenance requires caller transaction')
        count = 0
        for table in ('items', 'candidates'):
            count += self.store.db.execute(f'''UPDATE autobio_register_{table} SET operation_id=NULL,lease_token=NULL,lease_until=NULL,fence=fence+1
                WHERE owner=? AND agent=? AND audience=? AND operation_id IS NOT NULL AND {clause}''', (*self.scope, *args)).rowcount
        return count

    def release_operation(self, operation_id):
        return self._release('operation_id=?', (operation_id,))

    def expire_leases(self, now=None):
        return self._release('lease_until<=?', (self.store.clock() if now is None else now,))

    def invalidate_all(self):
        return self._release('1', ())

    def purge_sources(self, source_ids):
        """Conservatively remove entire items derived from erased sources, including receipts.

        Caller owns privacy authority and transaction. No prose survives in progress
        or receipt snapshots, even when the item also had non-erased evidence.
        """
        if not self.store.db.in_transaction:
            raise RuntimeError('register purge requires caller transaction')
        source_ids = tuple(source_ids)
        if not source_ids:
            return {'items': 0, 'candidates': 0, 'receipts': 0}
        scope = 'owner=? AND agent=? AND audience=?'
        candidates = self.store.db.execute(f'SELECT * FROM autobio_register_candidates WHERE {scope}', self.scope).fetchall()
        erased = set(source_ids)
        candidate_ids = [r['id'] for r in candidates if r['source_id'] in erased or any(e.get('source_id') in erased for e in json.loads(r['evidence']))]
        ids = set()
        for candidate_id in candidate_ids:
            ids.update(r[0] for r in self.store.db.execute(f'SELECT item_id FROM autobio_register_evidence WHERE {scope} AND candidate_id=?', (*self.scope, candidate_id)))
        for item_id in ids:
            for table in ('progress', 'evidence'):
                self.store.db.execute(f'DELETE FROM autobio_register_{table} WHERE {scope} AND item_id=?', (*self.scope, item_id))
            self.store.db.execute(f'DELETE FROM autobio_register_items WHERE {scope} AND id=?', (*self.scope, item_id))
        for candidate_id in candidate_ids:
            self.store.db.execute(f'DELETE FROM autobio_register_candidates WHERE {scope} AND id=?', (*self.scope, candidate_id))
        receipts = self.store.db.execute(f'DELETE FROM autobio_register_receipts WHERE {scope}', self.scope).rowcount
        self.invalidate_all()
        return {'items': len(ids), 'candidates': len(candidate_ids), 'receipts': receipts}
