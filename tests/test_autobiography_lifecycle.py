"""Disposable lifecycle/history regressions; never open configured/live stores."""
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agentic_memory.autobiography import AutobiographyStore, PersonalError, PersonalPrincipal
from agentic_memory.autobiography_context import personal_context


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'personal.db'
        self.principal = PersonalPrincipal('owner', 'agent', role='operator', session_id='test')
        self.store = AutobiographyStore(self.path, self.principal, clock=lambda: 1)
        self.addCleanup(lambda: self.store.close())

    def create(self, subject='core', segment='Identity', **kwargs):
        return self.store.update(expected_revision=0, segment=segment, subject=subject,
            title='Original title', body='Ancient sapphire memory', reason='Authored deliberately',
            idempotency_key='create-' + subject + segment, **kwargs)['edits'][0]['record_id']

    def raw(self, rid, revision=1):
        return tuple(self.store.db.execute('SELECT * FROM autobio_revisions WHERE record_id=? AND revision=?',
                                          (rid, revision)).fetchone())

    def change(self, rid, state, expected=1, **kwargs):
        return self.store.transition(rid, expected_revision=expected, state=state,
            reason='Explicit validity decision', idempotency_key=f'{rid}-{state}-{expected}', **kwargs)

    def test_old_is_not_outdated_and_restart_does_not_age_out(self):
        rid = self.create()
        self.store.close()
        self.store = AutobiographyStore(self.path, self.principal, clock=lambda: 10**12)
        self.assertEqual(self.store.get(rid)['state'], 'active')
        self.assertEqual(self.store.search('sapphire')['records'][0]['id'], rid)
        self.assertIn('sapphire', personal_context(self.store)['text'])

    def test_all_invalid_states_excluded_even_when_index_write_fails(self):
        for state in ('retired', 'outdated', 'retracted', 'superseded'):
            with self.subTest(state=state):
                rid = self.create(state, 'People')
                original = self.raw(rid)
                replacement = self.create('replacement', 'People') if state == 'superseded' else None
                kwargs = {'superseded_by': replacement, 'superseded_by_revision': 1} if replacement else {}
                with patch.object(self.store, 'index_record', return_value=False):
                    self.change(rid, state, **kwargs)
                with self.assertRaises(PersonalError):
                    self.store.get(rid)
                self.assertEqual(self.store.get(rid, include_inactive=True)['state'], state)
                self.assertNotIn(rid, [r['id'] for r in self.store.list_records()['records']])
                self.assertNotIn(rid, [r['id'] for r in self.store.search('sapphire')['records']])
                self.assertNotIn(rid, [r['record_id'] for r in personal_context(self.store, subjects=[state])['manifest']['records']])
                self.assertEqual(self.raw(rid), original)
                self.assertEqual(self.store.get(rid, 1)['body'], 'Ancient sapphire memory')
                self.assertEqual(len(self.store.history(rid)['revisions']), 2)
                self.assertIn(rid, [r['id'] for r in self.store.list_records(include_inactive=True)['records']])
        self.assertEqual(len(self.store.export()['lifecycle']), 9)

    def test_supersession_links_exact_revision_and_restart_history(self):
        rid = self.create()
        target = self.create('target', 'People')
        before = self.raw(rid)
        receipt = self.change(rid, 'superseded', superseded_by=target, superseded_by_revision=1)
        self.assertEqual(receipt, self.change(rid, 'superseded', superseded_by=target, superseded_by_revision=1))
        self.store.update(record_id=target, expected_revision=1, title='Updated', body='New content',
                          reason='Correction', idempotency_key='target-update')
        self.store.close()
        self.store = AutobiographyStore(self.path, self.principal)
        event = self.store.get(rid, 2)['lifecycle']
        self.assertEqual((event['superseded_by'], event['superseded_by_revision']), (target, 1))
        self.assertEqual(self.raw(rid), before)
        self.assertEqual(self.store.get(rid, 1)['state'], 'active')
        self.assertNotIn('sapphire', personal_context(self.store)['text'])

    def test_snapshot_updates_replacements_and_deletes_blocked_privacy_erase_allowed(self):
        rid = self.create()
        original = self.raw(rid)
        for table in ('autobio_revisions', 'autobio_lifecycle'):
            for sql in (f"UPDATE {table} SET state='retracted' WHERE record_id=?",
                        f'DELETE FROM {table} WHERE record_id=?',
                        f'INSERT OR REPLACE INTO {table} SELECT * FROM {table} WHERE record_id=?'):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    self.store.db.execute(sql, (rid,))
        self.assertEqual(self.raw(rid), original)
        self.store.erase(rid, expected_revision=1)
        for table in ('autobio_revisions', 'autobio_lifecycle'):
            self.assertEqual(self.store.db.execute(f'SELECT count(*) FROM {table}').fetchone()[0], 0)

    def test_missing_evidence_does_not_block_retraction_or_rewrite_history(self):
        # Register a disposable evidence identity using the established source schema.
        self.store.db.execute('INSERT INTO autobio_sources VALUES(?,?,?,?,?,?,?,?)',
                              ('source', 'owner', 'agent', 'owner', 'gen', 1, '{}', 1))
        evidence = [{'source_id': 'source', 'generation': 'gen', 'start': 0, 'end': 1,
                     'attribution': 'user_statement'}]
        rid = self.create(evidence=evidence, attribution='user_statement')
        original = self.raw(rid)
        self.store.db.execute("DELETE FROM autobio_sources WHERE id='source'")
        self.change(rid, 'retracted')
        self.assertEqual(self.raw(rid), original)
        for revision in (1, 2):
            record = self.store.get(rid, revision)
            self.assertEqual(record['evidence'], evidence)
            self.assertEqual(record['evidence_availability'], [False])

    def test_conflicts_permissions_and_atomicity(self):
        rid = self.create()
        target = self.create('target', 'People')
        for kwargs in ({}, {'superseded_by': rid, 'superseded_by_revision': 1},
                       {'superseded_by': target, 'superseded_by_revision': 42}):
            with self.assertRaises(PersonalError):
                self.change(rid, 'superseded', **kwargs)
        self.assertEqual(self.store.get(rid)['revision'], 1)
        self.change(rid, 'outdated')
        with self.assertRaises(PersonalError):
            self.change(rid, 'retracted', expected=1)
        reader = AutobiographyStore(self.path, replace(self.principal, role='reader'))
        self.addCleanup(reader.close)
        with self.assertRaises(PersonalError):
            reader.transition(rid, expected_revision=2, state='active', reason='No', idempotency_key='reader')
        other = AutobiographyStore(self.path, replace(self.principal, audience='other', reads='*'))
        self.addCleanup(other.close)
        with self.assertRaises(PersonalError):
            other.transition(rid, expected_revision=2, state='active', reason='No', idempotency_key='other')
        before = self.raw(rid, 2)
        with patch.object(self.store, 'validate_core_budget', side_effect=PersonalError('budget', 'overflow')):
            with self.assertRaises(PersonalError):
                self.change(rid, 'active', expected=2)
        self.assertEqual(self.raw(rid, 2), before)
        self.assertEqual(len(self.store.history(rid)['revisions']), 2)
        self.change(rid, 'active', expected=2)
        self.assertEqual(self.store.get(rid)['revision'], 3)

    def test_existing_schema_upgrade_is_additive_and_preserves_revision_bytes(self):
        rid = self.create()
        original = self.raw(rid)
        # Emulate a pre-lifecycle database; its committed content must not be backfilled/rewritten.
        for table in ('autobio_lifecycle', 'autobio_revisions'):
            for suffix in ('immutable', 'no_delete', 'no_replace'):
                self.store.db.execute(f'DROP TRIGGER {table}_{suffix}')
        self.store.db.execute('DROP TABLE autobio_lifecycle')
        self.store.close()
        self.store = AutobiographyStore(self.path, self.principal)
        self.assertEqual(self.raw(rid), original)
        self.assertIsNone(self.store.get(rid)['lifecycle'])
        self.change(rid, 'outdated')
        self.assertEqual(self.store.get(rid, 2)['lifecycle']['previous_revision'], 1)
        self.assertEqual(self.raw(rid), original)


if __name__ == '__main__':
    unittest.main()
