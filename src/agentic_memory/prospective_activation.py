"""Scoped prospective opt-in fences; deliberately independent of extraction progress.

Only canonical agent-core sources currently have trustworthy position adapters.
Other backends fail closed. Source files and their timestamps are trusted local
inputs, not cryptographically authenticated events. A hostile filesystem writer
can forge them. Validation must occur in the caller's publication transaction;
filesystem reads cannot themselves be made atomic with SQLite publication.
"""
from contextlib import contextmanager
import uuid
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .autobiography import PersonalError, canonical, digest
from .personal_sources import selected_sources, parse_boundary, source_origin
from .readers import CompletedSourceReader


def _stamp(value):
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            result = value / 1000 if value > 100_000_000_000 else value
        else:
            result = parse_boundary(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


class ProspectiveActivation:
    def __init__(self, config, store):
        self.config, self.store = config, store
        # Constructors and validators never commit a caller-owned transaction.
        with self._transaction():
            store.db.execute('''CREATE TABLE IF NOT EXISTS autobio_prospective_policy (
                owner TEXT NOT NULL, agent TEXT NOT NULL, audience TEXT NOT NULL,
                activation_version INTEGER NOT NULL, request_id TEXT NOT NULL,
                activated_at REAL NOT NULL, receipt TEXT NOT NULL,
                PRIMARY KEY(owner,agent,audience), UNIQUE(owner,agent,audience,request_id))''')
            store.db.execute('''CREATE TABLE IF NOT EXISTS autobio_prospective_baselines (
                owner TEXT NOT NULL, agent TEXT NOT NULL, audience TEXT NOT NULL,
                activation_version INTEGER NOT NULL, source_path TEXT NOT NULL,
                observed_at REAL NOT NULL, baseline TEXT NOT NULL,
                PRIMARY KEY(owner,agent,audience,activation_version,source_path))''')

    @contextmanager
    def _transaction(self):
        if not self.store.db.in_transaction:
            with self.store.transaction():
                yield
            return
        name = 'prospective_' + uuid.uuid4().hex
        self.store.db.execute('SAVEPOINT ' + name)
        try:
            yield
        except BaseException:
            self.store.db.execute('ROLLBACK TO ' + name)
            self.store.db.execute('RELEASE ' + name)
            raise
        else:
            self.store.db.execute('RELEASE ' + name)

    @property
    def _scope(self):
        p = self.store.principal
        return p.owner, p.agent, p.audience

    def status(self):
        row = self.store.db.execute('''SELECT receipt FROM autobio_prospective_policy
            WHERE owner=? AND agent=? AND audience=?''', self._scope).fetchone()
        return json.loads(row[0]) if row else {'active': False, 'activation_version': 0}

    def _selected(self):
        sources = selected_sources(self.config)
        if any(ds.type != 'agent-core' for ds in sources):
            raise PersonalError('unsupported_source', 'prospective activation requires canonical agent-core positions')
        for ds in sources:
            if not ds.path.is_dir():
                raise PersonalError('stale_source', 'configured source root is unavailable')
        return sources

    def _capture(self, ds, path, *, rewrite_fence=None):
        """Reuse canonical recovery parser, with a disposable cache, not a cursor.

        Byte positions below are a conservative physical envelope, not a second
        extraction parser. At baseline, replay canonical rewrites to establish CURRENT
        positions. During admission, only complete records inside the immutable
        baseline byte prefix may rewrite positions; the appended tail never may.
        None means baseline observation, not evidence admission.
        """
        path = Path(path).resolve()
        if not path.is_relative_to(ds.path.resolve()):
            raise PersonalError('stale_source', 'source outside configured root')
        try:
            meta_raw = path.read_bytes()
            meta = json.loads(meta_raw)
            if not isinstance(meta, dict):
                raise ValueError('invalid metadata')
            full = path.parent / path.stem / 'full-transcript-with-tools.jsonl'
            sidecar = path.with_name(path.stem + '.messages.jsonl')
            body = full if full.is_file() else sidecar if sidecar.is_file() else None
            if body and not body.resolve().is_relative_to(ds.path.resolve()):
                raise ValueError('body outside configured root')
            raw = body.read_bytes() if body else b''
            with tempfile.TemporaryDirectory(prefix='prospective-source-') as tmp:
                cfg = replace(self.config, data_dir=Path(tmp), data_stores=[ds], recovery_ingestion=True)
                snapshot = CompletedSourceReader(cfg).read(ds, path, ds.project or self.config.personal.agent)
            if path.read_bytes() != meta_raw or (body and body.read_bytes() != raw):
                raise ValueError('source changed during observation')
            if not all(r.verify() for r in snapshot.revisions):
                raise ValueError('source revision changed')
            indices = snapshot.message_indices
            if indices is None or indices != list(range(len(indices))):
                raise ValueError('canonical message positions unavailable or noncontiguous')
            messages = meta.get('messages', [])
            if not isinstance(messages, list):
                raise ValueError('invalid inline messages')
            positions = {}
            if body:
                offset, next_index = 0, 0
                for line in raw.splitlines(keepends=True):
                    begin, offset = offset, offset + len(line)
                    if not line.endswith(b'\n'):
                        break  # partial bytes are still included in baseline byte highwater
                    record = json.loads(line)
                    kind = record.get('type')
                    historical = rewrite_fence is None or offset <= rewrite_fence
                    if kind in ('truncate', 'repair'):
                        if not historical:
                            raise ValueError('rewrite after baseline cannot establish prospective continuity')
                        count = record.get('message_count', record.get('truncated_to_message_index'))
                        if type(count) is not int or count < 0:
                            raise ValueError('invalid message truncation boundary')
                        positions = {k: v for k, v in positions.items() if k < count}
                        next_index = count
                    if kind == 'message':
                        index = record.get('message_index', next_index)
                        if (type(index) is not int or index < 0
                            or (index in positions and not historical)):
                            raise ValueError('ambiguous message position')
                        positions[index] = begin
                        next_index = index + 1
            identity = digest([ds.name, str(path), meta.get('id'), source_origin(meta),
                               (meta.get('metadata') or {}).get('audience')])
            baseline = {'path': str(path), 'store': ds.name, 'identity': identity,
                        'message_highwater': len(indices), 'body': str(body.resolve()) if body else None,
                        'byte_highwater': len(raw), 'prefix_hash': hashlib.sha256(raw).hexdigest(),
                        'inline_highwater': len(messages), 'inline_hash': digest(messages),
                        'source_origin': source_origin(meta)}
            return baseline, snapshot, raw, messages, positions
        except PersonalError:
            raise
        except Exception as exc:
            raise PersonalError('stale_source', 'cannot establish authoritative source positions') from exc

    def _insert(self, version, at, baseline):
        self.store.db.execute('''INSERT INTO autobio_prospective_baselines
            VALUES(?,?,?,?,?,?,?)''', (*self._scope, version, baseline['path'], at, canonical(baseline)))

    def activate(self, request_id, expected_policy_version=0):
        self.store.principal.require('operator')
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
            raise PersonalError('invalid', 'activation request_id is required')
        if type(expected_policy_version) is not int:
            raise PersonalError('invalid', 'expected policy version must be an integer')
        with self._transaction():
            prior = self.status()
            if prior['active'] and prior['request_id'] == request_id:
                return prior
            if prior['activation_version'] != expected_policy_version or prior['active']:
                raise PersonalError('conflict', 'prospective activation is immutable')
            # Take the timestamp BEFORE scanning: every observed tail is excluded.
            at = self.store.clock()
            captured = []
            for ds in self._selected():
                for path in sorted(ds.path.glob('*.json')):
                    captured.append(self._capture(ds, path)[0])
            receipt = {'active': True, 'activation_version': 1, 'request_id': request_id,
                       'activated_at': at, 'baseline_count': len(captured)}
            # No policy is written if even one capture failed.
            self.store.db.execute('INSERT INTO autobio_prospective_policy VALUES(?,?,?,?,?,?,?)',
                                  (*self._scope, 1, request_id, at, canonical(receipt)))
            for baseline in captured:
                self._insert(1, at, baseline)
            return receipt

    def _baseline(self, version, path):
        row = self.store.db.execute('''SELECT baseline,observed_at FROM autobio_prospective_baselines
            WHERE owner=? AND agent=? AND audience=? AND activation_version=? AND source_path=?''',
            (*self._scope, version, str(Path(path).resolve()))).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def observe_path(self, ds, path):
        with self._transaction():
            policy = self.status()
            if not policy['active']:
                return None
            allowed = {d.name: d for d in self._selected()}
            if ds.name not in allowed or ds.path.resolve() != allowed[ds.name].path.resolve():
                raise PersonalError('stale_source', 'source store is not selected')
            existing = self._baseline(policy['activation_version'], path)
            if existing:
                return existing[0]
            at = self.store.clock()
            baseline = self._capture(allowed[ds.name], path)[0]
            self._insert(policy['activation_version'], at, baseline)
            return baseline

    def _source(self, source_id, generation=None):
        row = self.store.db.execute('''SELECT * FROM autobio_sources
            WHERE id=? AND owner=? AND agent=? AND audience=?''', (source_id, *self._scope)).fetchone()
        if not row or (generation is not None and row['generation'] != generation) or self.store.is_suppressed(source_id):
            raise PersonalError('invalid_evidence', 'source unavailable or changed')
        md = json.loads(row['metadata'])
        ds = next((d for d in self._selected() if d.name == md.get('source_store')), None)
        if ds is None or md.get('source_type') != 'agent-core':
            raise PersonalError('invalid_evidence', 'source backend is not selected')
        return row, md, ds

    def observe_source(self, source_id):
        _, md, ds = self._source(source_id)
        return self.observe_path(ds, md['path'])

    def _eligible(self, source_id, generation, start, end):
        policy = self.status()
        if not policy['active']:
            raise PersonalError('activation_required', 'prospective activation is required')
        row, md, ds = self._source(source_id, generation)
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= row['span_count']:
            raise PersonalError('invalid_evidence', 'invalid source range')
        saved = self._baseline(policy['activation_version'], md['path'])
        if not saved:
            return policy, []
        baseline, observed_at = saved
        current, snapshot, raw, messages, positions = self._capture(
            ds, md['path'], rewrite_fence=baseline['byte_highwater'])
        if (current['identity'] != baseline['identity'] or current['body'] != baseline['body']
            or current['message_highwater'] < baseline['message_highwater']
            or len(raw) < baseline['byte_highwater']
            or hashlib.sha256(raw[:baseline['byte_highwater']]).hexdigest() != baseline['prefix_hash']
            or len(messages) < baseline['inline_highwater']
            or digest(messages[:baseline['inline_highwater']]) != baseline['inline_hash']):
            raise PersonalError('stale_source', 'activation source prefix changed')
        # Unit generations are immutable too: edits outside the quote must not
        # preserve a stale claim merely because its quoted substring survived.
        logical = md.get('logical_source')
        if logical and self.store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='autobio_source_cursors'").fetchone():
            cursor = self.store.db.execute('SELECT boundaries FROM autobio_source_cursors WHERE source_id=?', (logical,)).fetchone()
            if cursor:
                snapshot.retain_boundaries(json.loads(cursor[0]))
        unit, key = md.get('unit'), md.get('raw_unit_key')
        keys = snapshot.unit_keys()
        expected_policy = digest([self.config.personal.source_mode, self.config.personal.source_since])
        origin = current['source_origin']
        if (md.get('source_policy') != expected_policy or md.get('source_origin') != origin
            or origin['kind'] == 'internal'):
            raise PersonalError('stale_source', 'source policy or origin changed')
        # Match the extraction adapter's strict/source_since key derivation,
        # not an alternate permissive raw-key fallback.
        if self.config.personal.source_mode == 'native' or self.config.personal.source_since is not None:
            keys = [digest([k, expected_policy, origin]) for k in keys]
        if type(unit) is not int or not 0 <= unit < len(keys) or not key or keys[unit] != key:
            raise PersonalError('stale_source', 'canonical source unit changed or proof unavailable')
        from .personal_privacy import redact_credentials
        turns = snapshot.transcript.turns if snapshot.transcript else []
        eligible = []
        for role in md.get('roles', []):
            index, stamp = role.get('message_index'), _stamp(role.get('source_event_time'))
            a, b = role.get('start'), role.get('end')
            if (type(index) is not int or index < baseline['message_highwater'] or stamp is None
                or stamp <= max(policy['activated_at'], observed_at)
                or type(a) is not int or type(b) is not int or not 0 <= a < b <= row['span_count']):
                continue
            if baseline['body'] and (index not in positions or positions[index] < baseline['byte_highwater']):
                continue
            matches = [t for t in turns if t.message_index == index and t.message_id == role.get('message_id')
                       and t.role == role.get('role') and _stamp(t.ts) == stamp]
            for turn in matches:
                value = redact_credentials(turn.text)
                if len(value) != b - a or '<autobiography_context>' in value or 'personalContextManifest' in value:
                    continue
                lo, hi = max(start, a), min(end, b)
                if lo < hi:
                    eligible.append({'start': lo, 'end': hi, '_text': value, '_origin': a})
        return policy, eligible

    def eligible_spans(self, source_id, source_generation, range_start, range_end):
        _, spans = self._eligible(source_id, source_generation, range_start, range_end)
        return [{'start': s['start'], 'end': s['end']} for s in spans]

    def validate_evidence(self, source_id, source_generation, range_start, range_end, quote):
        policy, spans = self._eligible(source_id, source_generation, range_start, range_end)
        if not isinstance(quote, str) or not quote or len(quote) != range_end - range_start:
            raise PersonalError('invalid_evidence', 'quote must exactly cover the evidence range')
        for span in spans:
            if span['start'] == range_start and span['end'] == range_end:
                a = range_start - span['_origin']
                if span['_text'][a:a + len(quote)] == quote:
                    return {'source_id': source_id, 'generation': source_generation,
                            'start': range_start, 'end': range_end, 'quote': quote,
                            'activation_version': policy['activation_version']}
        raise PersonalError('invalid_evidence', 'entire quote must be prospectively eligible')
