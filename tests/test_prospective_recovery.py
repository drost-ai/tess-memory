import unittest

import test_native_creation as fixtures
from agentic_memory.autobiography import PersonalError, canonical, digest
from agentic_memory.autobiography_inbox import PersonalInbox
from agentic_memory.prospective_recovery import plan_native_recovery


class RecoveryPlanTests(unittest.TestCase):
    # Reuse fixtures without inheriting/re-running the entire provenance suite.
    source = fixtures.base.ProspectiveTests.source
    append = fixtures.base.ProspectiveTests.append
    register = fixtures.base.ProspectiveTests.register
    native = fixtures.NativeCreationTests.native
    discover = fixtures.NativeCreationTests.discover

    def setUp(self):
        fixtures.base.ProspectiveTests.setUp(self)
        import sqlite3
        from dataclasses import replace
        from agentic_memory.prospective_activation import ProspectiveActivation
        self.ledger = self.root / 'gateway.db'
        with sqlite3.connect(self.ledger) as db:
            db.execute('CREATE TABLE sessions(id TEXT, engine_session_id TEXT, created_at REAL, deleted_at TEXT)')
        self.cfg = replace(self.cfg, personal=replace(self.cfg.personal, native_creation_ledger=str(self.ledger)))
        self.api = ProspectiveActivation(self.cfg, self.store)
        self.api.activate('a')
        path, _ = self.native()
        text, roles = self.discover(path)
        PersonalInbox(self.store).initialize()
        self.store.db.execute("ALTER TABLE autobio_extraction_parts ADD COLUMN drafts TEXT NOT NULL DEFAULT '{}'")
        self.allowlist = {}
        for i in range(11):
            receipt = {'range_id': str(i), 'published': True, 'candidate_ids': []}
            self.store.db.execute('''INSERT INTO autobio_extraction_parts
                (id, source_id, generation, start, end, text, state, result, receipt, updated)
                VALUES(?, 'unit', 'g', ?, ?, ?, 'published', '[]', ?, ?)''',
                (str(i), roles[0]['start'] + i, roles[0]['start'] + i + 1, text[i:i+1], canonical(receipt), self.now))
            self.allowlist[str(i)] = digest(receipt)
        self.store.db.commit()

    def test_eleven_range_dry_run_is_bounded_deterministic_and_read_only(self):
        before = list(self.store.db.iterdump())
        changes = self.store.db.total_changes
        plan = plan_native_recovery(self.api, self.allowlist)
        self.assertEqual(len(plan['ranges']), 11)
        self.assertFalse(plan['apply_supported'])
        self.assertEqual(plan_native_recovery(self.api, self.allowlist), plan)
        self.assertEqual(self.store.db.total_changes, changes)
        self.assertEqual(list(self.store.db.iterdump()), before)
        self.assertEqual(self.api.status()['activation_version'], 1)

    def test_nonempty_changed_receipt_pending_and_stale_generation_denied(self):
        for column, value in [('state', 'pending'), ('generation', 'changed'),
                              ('result', '[{}]'), ('drafts', '{"x":{"candidate_kind":"topic"}}'),
                              ('receipt', '{"published":true}')]:
            with self.subTest(column=column):
                old = self.store.db.execute(f'SELECT {column} FROM autobio_extraction_parts WHERE id=\'0\'').fetchone()[0]
                self.store.db.execute(f'UPDATE autobio_extraction_parts SET {column}=? WHERE id=\'0\'', (value,))
                with self.assertRaises(PersonalError):
                    plan_native_recovery(self.api, self.allowlist)
                self.store.db.execute(f'UPDATE autobio_extraction_parts SET {column}=? WHERE id=\'0\'', (old,))

    def test_missing_authority_and_overbroad_requests_denied(self):
        for ids in ({}, {**self.allowlist, '12': 'hash'}, {'missing': 'hash'}, {'0': 'wrong-hash'}):
            with self.assertRaises(PersonalError):
                plan_native_recovery(self.api, ids)
        self.ledger.unlink()
        with self.assertRaises(PersonalError):
            plan_native_recovery(self.api, self.allowlist)
        self.assertFalse(self.ledger.exists())

    def test_existing_memory_is_preserved_not_replayed(self):
        receipt = {'range_id': '0', 'published': True, 'candidate_ids': ['existing-memory']}
        self.store.db.execute("UPDATE autobio_extraction_parts SET receipt=?,result='[{}]',drafts=? WHERE id='0'",
                              (canonical(receipt), canonical({'draft': {'candidate_kind': 'memory'}})))
        self.allowlist['0'] = digest(receipt)
        before = list(self.store.db.iterdump())
        plan = plan_native_recovery(self.api, self.allowlist)
        self.assertEqual(plan['ranges'][0]['preserved_memory_candidate_ids'], ['existing-memory'])
        self.assertEqual(list(self.store.db.iterdump()), before)
