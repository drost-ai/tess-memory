"""Native late discovery: temporary host ledger, sources, and memory only."""
from dataclasses import replace
import json
import sqlite3

import test_prospective_activation as base
from tess_memory.autobiography import PersonalError, canonical
from tess_memory.autobiography_config import PersonalConfig
from tess_memory.prospective_activation import ProspectiveActivation


class NativeCreationTests(base.ProspectiveTests):
    def setUp(self):
        super().setUp()
        self.ledger = self.root / 'gateway.db'
        with sqlite3.connect(self.ledger) as db:
            db.execute('CREATE TABLE sessions(id TEXT, engine_session_id TEXT, created_at REAL, deleted_at TEXT)')
        self.cfg = replace(self.cfg, personal=replace(self.cfg.personal, native_creation_ledger=str(self.ledger)))
        self.api = ProspectiveActivation(self.cfg, self.store)

    def invalid(self, role, quote):
        with self.assertRaises(PersonalError) as ctx:
            self.valid(role, quote)
        self.assertEqual(ctx.exception.code, 'invalid_evidence')

    def native(self, *, ledger=True, created=None):
        created = self.now + 2 if created is None else created
        path, body = self.source()
        path.write_text(json.dumps({'id': 'main', 'createdAt': created, 'messages': []}))
        if ledger:
            with sqlite3.connect(self.ledger) as db:
                db.execute('INSERT INTO sessions VALUES(?,?,?,NULL)', ('host', 'main', created - 1))
        self.append(path, body, 'new native conversation', stamp=created + 1, index=0)
        record = json.loads(body.read_text())
        record['session_id'] = 'main'
        body.write_text(json.dumps(record) + '\n')
        return path, body

    def discover(self, path):
        self.now += 100
        self.api.observe_path(self.ds, path)
        return self.register(path)

    def test_late_completed_native_and_legacy_baseline(self):
        self.api.activate('a')
        path, body = self.native()
        text, roles = self.discover(path)
        before = self.api._baseline(1, path)
        self.valid(roles[0], 'new native conversation')
        # Model an actual activation-v1 baseline, not a reset or replacement fence.
        old = {k: v for k, v in before[0].items() if k not in
               ('creation', 'pristine_stream', 'first_event_at', 'native_creation_proof')}
        self.store.db.execute('UPDATE autobio_prospective_baselines SET baseline=?', (canonical(old),))
        self.valid(roles[0], 'new native conversation')
        self.assertEqual(self.api._baseline(1, path), (old, before[1]))
        self.assertEqual(self.api.status()['activation_version'], 1)

    def test_created_timestamp_without_ledger_is_not_proof(self):
        self.api.activate('a')
        path, _ = self.native(ledger=False)
        _, roles = self.discover(path)
        self.invalid(roles[0], 'new native conversation')

    def test_old_session_and_known_activation_baseline_denied(self):
        path, _ = self.native(created=self.now - 50)
        self.api.activate('a')
        _, roles = self.discover(path)
        self.invalid(roles[0], 'new native conversation')

    def test_late_old_session_denied(self):
        self.api.activate('a')
        path, _ = self.native(created=self.now - 50)
        _, roles = self.discover(path)
        self.invalid(roles[0], 'new native conversation')

    def test_import_and_fork_metadata_denied(self):
        for marker in ('importedFrom', 'forkedFromSessionId', 'parentSessionId'):
            with self.subTest(marker=marker):
                # Independent temporary fixture per variant.
                self.setUp()
                self.api.activate('a')
                path, _ = self.native()
                meta = json.loads(path.read_text()); meta['metadata'] = {marker: 'old-session'}
                path.write_text(json.dumps(meta))
                _, roles = self.discover(path)
                self.invalid(roles[0], 'new native conversation')

    def test_prebaseline_duplicate_and_repair_denied(self):
        for kind in ('duplicate', 'repair'):
            with self.subTest(kind=kind):
                self.setUp()
                self.api.activate('a')
                path, body = self.native()
                raw = body.read_text()
                if kind == 'repair':
                    raw += json.dumps({'type': 'repair', 'truncated_to_message_index': 0}) + '\n'
                body.write_text(raw + body.read_text())
                _, roles = self.discover(path)
                self.invalid(roles[0], 'new native conversation')

    def test_postbaseline_rewrite_remains_stale(self):
        self.api.activate('a')
        path, body = self.native()
        _, roles = self.discover(path)
        with body.open('a') as f:
            f.write(body.read_text())
        with self.assertRaises(PersonalError) as ctx:
            self.valid(roles[0], 'new native conversation')
        self.assertEqual(ctx.exception.code, 'stale_source')

    def test_changed_prefix_remains_stale(self):
        self.api.activate('a')
        path, body = self.native()
        _, roles = self.discover(path)
        body.write_text(body.read_text().replace('new native', 'old native'))
        with self.assertRaises(PersonalError) as ctx:
            self.valid(roles[0], 'new native conversation')
        self.assertEqual(ctx.exception.code, 'stale_source')

    def test_deleted_ambiguous_missing_or_changed_ledger_denied(self):
        for mutation in ('delete', 'duplicate', 'missing', 'old', 'host_changed'):
            with self.subTest(mutation=mutation):
                self.setUp()
                self.api.activate('a')
                path, _ = self.native()
                _, roles = self.discover(path)
                with sqlite3.connect(self.ledger) as db:
                    if mutation == 'delete': db.execute("UPDATE sessions SET deleted_at='now'")
                    if mutation == 'duplicate': db.execute('INSERT INTO sessions SELECT * FROM sessions')
                    if mutation == 'missing': db.execute('DELETE FROM sessions')
                    if mutation == 'old': db.execute('UPDATE sessions SET created_at=1')
                    if mutation == 'host_changed': db.execute("UPDATE sessions SET id='other-host'")
                self.invalid(roles[0], 'new native conversation')

    def test_config_is_explicit_and_validated(self):
        for value in ('', False, 42):
            with self.assertRaises(ValueError):
                PersonalConfig.parse({'source_mode': 'native', 'native_creation_ledger': value})
        with self.assertRaises(ValueError):
            PersonalConfig.parse({'native_creation_ledger': str(self.ledger)})
        self.api.activate('a')
        path, _ = self.native()
        _, roles = self.discover(path)
        self.api.config = replace(self.cfg, personal=replace(self.cfg.personal, native_creation_ledger=None))
        self.invalid(roles[0], 'new native conversation')

    def test_creation_or_event_identity_change_is_not_native_proof(self):
        for mutation in ('precreation_event', 'wrong_session', 'missing_session', 'changed_creation', 'wrong_root'):
            with self.subTest(mutation=mutation):
                self.setUp()
                self.api.activate('a')
                path, body = self.native()
                if mutation in ('precreation_event', 'wrong_session', 'missing_session'):
                    record = json.loads(body.read_text())
                    if mutation == 'precreation_event': record['persisted_at'] = self.now + 1
                    if mutation == 'wrong_session': record['session_id'] = 'another'
                    if mutation == 'missing_session': del record['session_id']
                    body.write_text(json.dumps(record) + '\n')
                _, roles = self.discover(path)
                if mutation == 'changed_creation':
                    meta = json.loads(path.read_text()); meta['createdAt'] += 0.5
                    path.write_text(json.dumps(meta))
                if mutation == 'wrong_root':
                    other = self.root / 'other'; other.mkdir()
                    copy = other / 'gateway.db'; copy.write_bytes(self.ledger.read_bytes())
                    self.api.config = replace(self.cfg, personal=replace(self.cfg.personal, native_creation_ledger=str(copy)))
                self.invalid(roles[0], 'new native conversation')

    def test_known_activation_source_never_relaxed_even_with_postactivation_clock(self):
        path, _ = self.native()  # Deliberately future-dated but already exists at activation.
        self.api.activate('a')
        _, roles = self.discover(path)
        self.invalid(roles[0], 'new native conversation')

    def test_inline_import_not_relaxed_by_host_row(self):
        self.api.activate('a')
        path, body = self.native()
        record = json.loads(body.read_text())
        body.unlink()
        meta = json.loads(path.read_text())
        meta['messages'] = [{'role': 'user', 'content': record['content'], 'timestamp': record['persisted_at']}]
        path.write_text(json.dumps(meta))
        _, roles = self.discover(path)
        self.invalid(roles[0], 'new native conversation')

    def test_native_postdiscovery_append_and_baseline_both_eligible(self):
        self.api.activate('a')
        path, body = self.native()
        self.discover(path)
        record = {'type': 'message', 'message_index': 1, 'session_id': 'main',
                  'message_id': 'm1', 'role': 'user', 'content': 'new tail', 'persisted_at': self.now + 1}
        with body.open('a') as f: f.write(json.dumps(record) + '\n')
        self.now += 2
        _, roles = self.register(path)
        self.valid(roles[0], 'new native conversation')
        self.valid(roles[1], 'new tail')

    def test_malformed_metadata_fails_closed(self):
        self.api.activate('a')
        path, _ = self.native()
        meta = json.loads(path.read_text()); meta['metadata'] = ['not', 'an', 'object']
        path.write_text(json.dumps(meta))
        self.now += 100
        with self.assertRaises(PersonalError) as ctx:
            self.api.observe_path(self.ds, path)
        self.assertEqual(ctx.exception.code, 'stale_source')
        self.assertIsNone(self.api._baseline(1, path))
