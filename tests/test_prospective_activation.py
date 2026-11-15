"""Prospective fences use only temporary databases and temporary source files."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from tess_memory.autobiography import AutobiographyStore, PersonalError, canonical, digest
from tess_memory.autobiography_config import PersonalConfig
from tess_memory.autobiography_extract import clean_unit
from tess_memory.config import Config
from tess_memory.personal_sources import selected_sources
from tess_memory.prospective_activation import ProspectiveActivation


class ProspectiveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sessions = self.root / 'sessions'
        self.sessions.mkdir()
        p = PersonalConfig(enabled=True, extraction_enabled=True, owner='sam', agent='assistant',
                           source_mode='native', source_paths=(str(self.sessions),))
        self.cfg = Config(data_dir=self.root / 'memory', personal=p, quiet_seconds=0)
        self.store = AutobiographyStore(self.cfg.db_path, p.principal(role='operator', session_id='test'))
        self.addCleanup(self.store.close)
        self.now = 1800000000
        self.store.clock = lambda: self.now
        self.api = ProspectiveActivation(self.cfg, self.store)
        self.ds = selected_sources(self.cfg)[0]

    def source(self, name='main', inline=False):
        path = self.sessions / (name + '.json')
        path.write_text(json.dumps({'id': name, 'messages': []}))
        if inline:
            return path, None
        folder = self.sessions / name
        folder.mkdir()
        body = folder / 'full-transcript-with-tools.jsonl'
        body.write_bytes(b'')
        return path, body

    def append(self, path, body, text, stamp=None, index=None):
        stamp = self.now + 10 if stamp is None else stamp
        if body:
            records = body.read_text().splitlines()
            index = len(records) if index is None else index
            record = {'type': 'message', 'message_index': index, 'message_id': f'm{index}',
                      'role': 'user', 'content': text, 'persisted_at': stamp}
            with body.open('a') as f:
                f.write(json.dumps(record) + '\n')
        else:
            meta = json.loads(path.read_text())
            meta['messages'].append({'role': 'user', 'content': text, 'timestamp': stamp})
            path.write_text(json.dumps(meta))

    def register(self, path, source_id='unit'):
        baseline, snapshot, *_ = self.api._capture(self.ds, path)
        text, roles = clean_unit(snapshot.transcript.turns)
        policy = digest([self.cfg.personal.source_mode, self.cfg.personal.source_since])
        origin = baseline['source_origin']
        md = {'path': str(path), 'source_type': 'agent-core', 'source_store': self.ds.name, 'roles': roles,
              'source_policy': policy, 'source_origin': origin,
              'unit': 0, 'raw_unit_key': digest([snapshot.unit_keys()[0], policy, origin])}
        with self.store.transaction():
            self.store.db.execute('INSERT OR REPLACE INTO autobio_sources VALUES(?,?,?,?,?,?,?,?)',
                (source_id, *self.api._scope, 'g', len(text), canonical(md), self.now))
        return text, roles

    def valid(self, role, quote, source_id='unit'):
        return self.api.validate_evidence(source_id, 'g', role['start'], role['end'], quote)

    def rewrite(self, body, kind, count, field='message_count'):
        with body.open('a') as f:
            f.write(json.dumps({'type': kind, field: count}) + '\n')

    def test_prebaseline_rewrites_current_only_and_future_append(self):
        for kind, field in [('truncate', 'message_count'), ('repair', 'truncated_to_message_index')]:
            with self.subTest(kind=kind):
                path, body = self.source(kind)
                self.append(path, body, 'retained history', self.now + 10000, index=0)
                self.append(path, body, 'removed history', self.now + 10000, index=1)
                self.rewrite(body, kind, 1, field)
                self.append(path, body, 'replacement history', self.now + 10000, index=1)
        self.api.activate('a')
        for kind in ['truncate', 'repair']:
            path = self.sessions / (kind + '.json')
            body = self.sessions / kind / 'full-transcript-with-tools.jsonl'
            baseline, snapshot, _, _, positions = self.api._capture(self.ds, path)
            self.assertEqual(baseline['message_highwater'], 2)
            self.assertEqual(set(positions), {0, 1})
            self.append(path, body, 'genuine future', index=2)
            text, roles = self.register(path)
            self.assertNotIn('removed history', text)
            for role, quote in zip(roles[:2], ['retained history', 'replacement history']):
                with self.assertRaises(PersonalError):
                    self.valid(role, quote)
            self.assertEqual(self.valid(roles[2], 'genuine future')['activation_version'], 1)
            original = body.read_bytes()
            body.write_bytes(original.replace(b'retained history', b'mutated! history'))
            with self.assertRaises(PersonalError):
                self.valid(roles[2], 'genuine future')
            body.write_bytes(original)
            # A no-op rewrite is still forbidden after baseline.
            self.rewrite(body, kind, 3)
            with self.assertRaises(PersonalError):
                self.valid(roles[2], 'genuine future')

    def test_prebaseline_duplicate_current_only_postbaseline_duplicate_denied(self):
        path, body = self.source()
        self.append(path, body, 'superseded', index=0)
        self.append(path, body, 'current history', index=0)
        self.api.activate('a')
        self.append(path, body, 'future', index=1)
        text, roles = self.register(path)
        self.assertNotIn('superseded', text)
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'current history')
        self.valid(roles[1], 'future')
        self.append(path, body, 'future', index=1)
        with self.assertRaises(PersonalError):
            self.valid(roles[1], 'future')

    def test_malformed_baseline_rewrite_activation_atomic(self):
        good, good_body = self.source('a-good')
        self.append(good, good_body, 'good history')
        path, body = self.source('z-bad')
        for count in [True, -1, '1', None, 1.5]:
            with self.subTest(count=count):
                body.write_bytes(b'')
                self.append(path, body, 'history', index=0)
                self.rewrite(body, 'repair', count)
                with self.assertRaises(PersonalError):
                    self.api.activate('a')
                self.assertFalse(self.api.status()['active'])
                self.assertEqual(self.store.db.execute(
                    'SELECT count(*) FROM autobio_prospective_baselines').fetchone()[0], 0)

    def test_pending_tail_future_timestamp_and_new_append(self):
        path, body = self.source()
        self.append(path, body, 'old pending', self.now + 10000)
        receipt = self.api.activate('a')
        self.assertEqual(receipt, self.api.activate('a'))
        self.append(path, body, 'new eligible')
        _, roles = self.register(path)
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'old pending')
        self.assertEqual(self.valid(roles[1], 'new eligible')['activation_version'], 1)
        with self.assertRaises(PersonalError):
            self.api.activate('different', expected_policy_version=1)
        self.assertFalse((self.cfg.data_dir / 'source-cache.db').exists())

    def test_same_timestamp_missing_and_backdated_deny(self):
        path, body = self.source()
        self.api.activate('a')
        for stamp in [self.now, self.now - 1, 'invalid']:
            self.append(path, body, 'not eligible', stamp)
        # Mixed invalid timestamp types cannot form a canonical snapshot: fail closed.
        with self.assertRaises(PersonalError):
            self.register(path)
        body.write_bytes(b'')
        for stamp in [self.now, self.now - 1]:
            self.append(path, body, 'not eligible', stamp)
        _, roles = self.register(path)
        for role in roles:
            with self.assertRaises(PersonalError):
                self.valid(role, 'not eligible')

    def test_first_observation_excludes_imported_future_events(self):
        self.api.activate('a')
        path, body = self.source('imported')
        self.append(path, body, 'imported future', self.now + 10000)
        _, roles = self.register(path)
        self.assertEqual(self.api.eligible_spans('unit', 'g', roles[0]['start'], roles[0]['end']), [])
        self.api.observe_source('unit')
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'imported future')
        self.append(path, body, 'genuinely new')
        _, roles = self.register(path)
        self.valid(roles[-1], 'genuinely new')

    def test_prefix_rewrite_and_truncate_deny(self):
        path, body = self.source()
        self.append(path, body, 'before')
        self.api.activate('a')
        self.append(path, body, 'after')
        _, roles = self.register(path)
        original = body.read_bytes()
        body.write_bytes(original.replace(b'before', b'BEFORE'))
        with self.assertRaises(PersonalError):
            self.valid(roles[-1], 'after')
        body.write_bytes(b'')
        with self.assertRaises(PersonalError):
            self.valid(roles[-1], 'after')

    def test_partial_tail_position_cannot_cross_baseline(self):
        path, body = self.source()
        record = json.dumps({'type': 'message', 'message_index': 0, 'message_id': 'm0',
                            'role': 'user', 'content': 'partial old', 'persisted_at': self.now + 10}).encode()
        body.write_bytes(record[:30])
        self.api.activate('a')
        with body.open('ab') as f:
            f.write(record[30:] + b'\n')
        self.append(path, body, 'new')
        _, roles = self.register(path)
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'partial old')
        self.valid(roles[-1], 'new')

    def test_atomic_failure_and_scope(self):
        self.source()
        (self.sessions / 'broken.json').write_text('{')
        with self.assertRaises(PersonalError):
            self.api.activate('a')
        self.assertFalse(self.api.status()['active'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM autobio_prospective_baselines').fetchone()[0], 0)
        (self.sessions / 'broken.json').unlink()
        self.api.activate('a')
        self.store.principal = replace(self.store.principal, audience='other')
        self.assertFalse(self.api.status()['active'])

    def test_validation_is_readonly_in_outer_transaction_and_quote_exact(self):
        path, body = self.source()
        self.api.activate('a')
        self.append(path, body, 'new text')
        _, roles = self.register(path)
        count = self.store.db.total_changes
        with self.store.transaction():
            self.valid(roles[0], 'new text')
            self.assertTrue(self.store.db.in_transaction)
            self.assertEqual(self.store.db.total_changes, count)
            with self.assertRaises(PersonalError):
                self.valid(roles[0], 'bad text')
        with self.assertRaises(PersonalError):
            self.api.validate_evidence('unit', 'g', 0, roles[0]['end'], '[user]\nnew text')

    def test_inactive_and_unknown_position_deny(self):
        path, body = self.source()
        self.append(path, body, 'old')
        _, roles = self.register(path)
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'old')
        self.api.activate('a')
        self.append(path, body, 'new')
        _, roles = self.register(path)
        row = self.store.db.execute('SELECT metadata FROM autobio_sources').fetchone()
        md = json.loads(row[0])
        md['roles'][-1]['message_index'] = None
        with self.store.transaction():
            self.store.db.execute('UPDATE autobio_sources SET metadata=?', (canonical(md),))
        with self.assertRaises(PersonalError):
            self.valid(roles[-1], 'new')

    def test_inline_prefix_and_outer_rollback(self):
        path, body = self.source(inline=True)
        self.append(path, body, 'old')
        with self.assertRaises(RuntimeError):
            with self.store.transaction():
                self.api.activate('a')
                self.assertTrue(self.store.db.in_transaction)
                raise RuntimeError('rollback')
        self.assertFalse(self.api.status()['active'])
        self.api.activate('a')
        self.append(path, body, 'new')
        _, roles = self.register(path)
        self.valid(roles[-1], 'new')
        meta = json.loads(path.read_text())
        meta['messages'][0]['content'] = 'rewritten'
        path.write_text(json.dumps(meta))
        with self.assertRaises(PersonalError):
            self.valid(roles[-1], 'new')

    def test_sidecar_uncommitted_tail_is_baselined(self):
        path, _ = self.source(inline=True)
        body = path.with_name(path.stem + '.messages.jsonl')
        body.write_text(json.dumps({'type': 'message', 'message_index': 0,
            'persisted_at': self.now + 10000,
            'message': {'id': 'm0', 'role': 'user', 'content': 'uncommitted old'}}) + '\n')
        self.api.activate('a')
        with body.open('a') as f:
            f.write(json.dumps({'type': 'message', 'message_index': 1,
                'persisted_at': self.now + 10,
                'message': {'id': 'm1', 'role': 'user', 'content': 'new sidecar'}}) + '\n')
        _, roles = self.register(path)
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'uncommitted old')
        self.valid(roles[-1], 'new sidecar')

    def test_stale_unit_even_when_quote_survives(self):
        path, body = self.source()
        self.api.activate('a')
        self.append(path, body, 'kept quote')
        self.append(path, body, 'other text')
        _, roles = self.register(path)
        body.write_bytes(body.read_bytes().replace(b'other text', b'OTHER TEXT'))
        with self.assertRaises(PersonalError):
            self.valid(roles[0], 'kept quote')

    def test_unsupported_backend_and_nonoperator_cannot_activate(self):
        from tess_memory.config import DataStore
        cfg = replace(self.cfg, data_stores=[DataStore('unsupported', self.sessions, type='claude-code')],
            personal=replace(self.cfg.personal, source_mode='configured', source_names=('unsupported',)))
        with self.assertRaises(PersonalError):
            ProspectiveActivation(cfg, self.store).activate('unsupported')
        self.assertFalse(self.api.status()['active'])
        self.store.principal = replace(self.store.principal, role='extractor')
        with self.assertRaises(PersonalError):
            self.api.activate('denied')

    def test_unstable_capture_fails_atomic_activation(self):
        from unittest.mock import patch
        path, body = self.source()
        self.append(path, body, 'old')
        from tess_memory.readers import CompletedSourceReader
        original = CompletedSourceReader.read
        def racing(reader, *args, **kwargs):
            snapshot = original(reader, *args, **kwargs)
            self.append(path, body, 'raced append')
            return snapshot
        with patch.object(CompletedSourceReader, 'read', racing):
            with self.assertRaises(PersonalError):
                self.api.activate('raced')
        self.assertFalse(self.api.status()['active'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM autobio_prospective_baselines').fetchone()[0], 0)

    def test_strict_key_requires_current_policy_and_origin_no_raw_fallback(self):
        path, body = self.source()
        self.api.activate('a')
        self.append(path, body, 'new')
        _, roles = self.register(path)
        self.valid(roles[0], 'new')
        original = json.loads(self.store.db.execute('SELECT metadata FROM autobio_sources').fetchone()[0])
        _, snapshot, *_ = self.api._capture(self.ds, path)
        for key, bad in [('source_policy', 'foreign-policy'), ('source_origin', {}),
                         ('raw_unit_key', snapshot.unit_keys()[0])]:
            md = dict(original)
            md[key] = bad
            with self.store.transaction():
                self.store.db.execute('UPDATE autobio_sources SET metadata=?', (canonical(md),))
            with self.assertRaises(PersonalError):
                self.valid(roles[0], 'new')
