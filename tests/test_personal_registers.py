"""Temporary SQLite tests; no configured/live personal data is opened."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from agentic_memory.autobiography import AutobiographyStore, PersonalPrincipal, PersonalError
from agentic_memory.personal_registers import PersonalRegisters
from agentic_memory.personal_operations import PersonalOperations, SCHEMA


class PersonalRegisterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.now = 1000.0
        self.store = AutobiographyStore(Path(tmp.name) / 'personal.db', PersonalPrincipal('owner', 'agent', role='operator', session_id='test'), clock=lambda: self.now)
        self.addCleanup(self.store.close)
        with self.store.transaction():
            for sql in SCHEMA.split(';'):
                if sql.strip():
                    self.store.db.execute(sql)
        # Real operation authority, without provider/config construction.
        self.ops = object.__new__(PersonalOperations)
        self.ops.store = self.store
        self.registers = PersonalRegisters(self.store, self.ops)
        self.operation('reconcile', 'follow_up_reconciler')
        self.operation('care', 'care')
        self.operation('topic', 'topic_reconciler')
        self.operation('research', 'research')
        with self.store.transaction():
            for source in ('s1', 's2', 's'):
                self.store.db.execute("INSERT INTO autobio_sources VALUES(?,?,?,?,?,10,'{}',?)", (source, 'owner', 'agent', 'owner', 'g1', self.now))

    def operation(self, name, role):
        with self.store.transaction():
            self.store.db.execute('INSERT INTO autobio_operations(id,kind,owner,agent,audience,actor_session,fence,state,started,deadline) VALUES(?,?,?,?,?,?,1,\'running\',?,?)', (name, role, 'owner', 'agent', 'owner', 'test', self.now, self.now + 600))

    def publish(self, candidate='c1', kind='follow_up', source='s1'):
        with self.store.transaction():
            return self.registers.publish_candidate(kind, source, 'g1', 'r1', candidate, {'title': 'Remember'}, [{'source_id': source, 'generation': 'g1', 'start': 0, 'end': 5, 'quote': 'hello'}], 1)

    def create(self, kind='follow_up'):
        candidate = self.publish(kind=kind)
        op = 'reconcile' if kind == 'follow_up' else 'topic'
        lease = self.registers.reconcile_claim(op, 1, kind)
        args = (op, 1, kind, lease['fence'], lease['lease_token'], 'create-' + kind,
                [{'action': 'create', 'expected_revision': 0, 'key': 'k1', 'title': 'Title', 'body': 'Body'}],
                [{'candidate_id': candidate['candidate_id'], 'disposition': 'applied', 'key': 'k1'}])
        result = self.registers.reconcile_commit(*args)
        self.assertEqual(result, self.registers.reconcile_commit(*args))
        return result['items'][0]

    def claim(self, kind='follow_up'):
        item = self.create(kind)
        op = 'care' if kind == 'follow_up' else 'research'
        lease = self.registers.claim(op, 1, kind, item['id'], 1, 'claim-' + kind)
        return (op, 1, kind, item['id'], lease['fence'], lease['lease_token'], 1)

    def test_extraction_atomic_dedupe_and_scoped_ids(self):
        first = self.publish()
        self.assertTrue(first['inserted'])
        self.assertEqual(self.publish(), {'candidate_id': first['candidate_id'], 'inserted': False})
        with self.assertRaises(RuntimeError):
            with self.store.transaction():
                self.registers.publish_candidate('topic', 's', 'g1', 'r', 'x', {}, [{'source_id': 's', 'generation': 'g1'}], 1)
                raise RuntimeError('rollback extraction')
        self.assertEqual(self.registers.candidates('topic')['candidates'], [])
        with self.assertRaises(RuntimeError):
            self.registers.publish_candidate('topic', 's', 'g1', 'r', 'x', {}, [{'source_id': 's', 'generation': 'g1'}], 1)

    def test_whole_batch_and_atomic_edit_rollback(self):
        self.publish('c1')
        self.publish('c2')
        lease = self.registers.reconcile_claim('reconcile', 1, 'follow_up')
        args = ('reconcile', 1, 'follow_up', lease['fence'], lease['lease_token'], 'bad')
        with self.assertRaises(PersonalError):
            self.registers.reconcile_commit(*args, [], [{'candidate_id': lease['candidates'][0]['id'], 'disposition': 'rejected'}])
        outcomes = [{'candidate_id': r['id'], 'disposition': 'rejected'} for r in lease['candidates']]
        with self.assertRaises(PersonalError):
            self.registers.reconcile_commit(*args, [{'action': 'create', 'expected_revision': 0, 'key': 'x', 'title': 'X', 'body': 'x'}, {'action': 'bad'}], outcomes)
        self.assertEqual(self.registers.list('follow_up')['items'], [])

    def test_claim_progress_complete_replay_and_live_authority(self):
        args = self.claim()
        self.assertEqual(self.registers.validate_claim(*args), {'valid': True})
        first = self.registers.progress(*args, 'p1', 'Investigated', [{'text': 'Finding', 'url': 'https://example.test'}])
        self.assertEqual(first['item']['revision'], 2)
        self.assertEqual(self.registers.progress(*args, 'p1', 'Investigated', [{'text': 'Finding', 'url': 'https://example.test'}]), first)
        self.assertEqual(self.registers.validate_claim(*args), {'valid': False})
        with self.assertRaises(PersonalError):
            self.registers.progress(*args, 'p1', 'Changed')
        current = (*args[:-1], 2)
        result = self.registers.complete(*current, 'done', 'completed', 'Finished')
        self.assertEqual(result['item']['state'], 'completed')
        self.assertEqual(self.registers.complete(*current, 'done', 'completed', 'Finished'), result)
        self.now += 601
        with self.assertRaises(PersonalError):
            self.registers.complete(*current, 'done', 'completed', 'Finished')

    def test_suppression_fences_and_no_cross_scope_access(self):
        args = self.claim()
        self.registers.suppress('follow_up', args[3], 1, 'suppress', 'Do not pursue')
        self.assertEqual(self.registers.validate_claim(*args), {'valid': False})
        with self.assertRaises(PersonalError):
            self.registers.progress(*args, 'p', 'stale')
        self.store.principal = replace(self.store.principal, audience='other')
        self.assertEqual(self.registers.list('follow_up', state='suppressed')['items'], [])
        self.assertEqual(self.registers.candidates('follow_up', state='applied')['candidates'], [])
        with self.assertRaises(PersonalError):
            self.registers.suppress('follow_up', args[3], 2, 'other', 'No access')

    def test_operation_role_and_session_guards(self):
        item = self.create()
        with self.assertRaises(PersonalError):
            self.registers.claim('research', 1, 'follow_up', item['id'], 1, 'bad')
        self.store.principal = replace(self.store.principal, role='research')
        with self.assertRaises(PersonalError):
            self.registers.claim('care', 1, 'follow_up', item['id'], 1, 'bad')
        self.store.principal = replace(self.store.principal, role='operator', session_id='different')
        with self.assertRaises(PersonalError):
            self.registers.claim('care', 1, 'follow_up', item['id'], 1, 'bad')

    def test_expiry_release_and_revision_cas(self):
        args = self.claim('topic')
        self.now += 601
        self.assertFalse(self.registers.validate_claim(*args)['valid'])
        with self.store.transaction():
            self.assertEqual(self.registers.expire_leases(), 1)
        self.operation('research2', 'research')
        lease = self.registers.claim('research2', 1, 'topic', args[3], 1, 'reclaim')
        self.assertGreater(lease['fence'], args[4])
        with self.store.transaction():
            self.assertEqual(self.registers.release_operation('research2'), 1)
        self.assertFalse(self.registers.validate_claim('research2', 1, 'topic', args[3], lease['fence'], lease['lease_token'], 1)['valid'])

    def test_privacy_purges_all_derivative_prose(self):
        args = self.claim()
        self.registers.progress(*args, 'progress', 'Private derivative', [{'text': 'Private research'}])
        with self.store.transaction():
            counts = self.registers.purge_sources(['s1'])
        self.assertEqual(counts['items'], 1)
        self.assertEqual(counts['candidates'], 1)
        self.assertTrue(all(not rows for rows in self.registers.export_scoped().values()))

    def test_deferred_candidate_and_work_due_time(self):
        c = self.publish()
        lease = self.registers.reconcile_claim('reconcile', 1, 'follow_up')
        self.registers.reconcile_commit('reconcile', 1, 'follow_up', lease['fence'], lease['lease_token'], 'defer', [], [{'candidate_id': c['candidate_id'], 'disposition': 'deferred', 'review_after': 1200}])
        self.assertEqual(self.registers.reconcile_claim('reconcile', 1, 'follow_up')['status'], 'empty')
        self.now = 1201
        self.assertEqual(self.registers.reconcile_claim('reconcile', 1, 'follow_up')['status'], 'claimed')

    def test_finding_bounds_and_delivery_is_optional_metadata(self):
        args = self.claim()
        for findings in ([{'text': 'x'}] * 21, [{'text': 'x', 'url': 'file:///secret'}]):
            with self.assertRaises(PersonalError):
                self.registers.progress(*args, 'invalid', 'summary', findings)
        result = self.registers.complete(*args, 'defer', 'deferred', 'Retry later', 1400)
        self.assertEqual(result['item']['state'], 'active')
        self.assertEqual(self.registers.list('follow_up', due_before=1300)['items'], [])
        self.assertEqual(len(self.registers.list('follow_up', due_before=1500)['items']), 1)

    def test_read_role_isolation_and_required_cas(self):
        item = self.create()
        with self.assertRaises(PersonalError):
            self.registers.claim('care', 1, 'follow_up', item['id'], None, 'none')
        with self.assertRaises(PersonalError):
            self.registers.suppress('follow_up', item['id'], None, 'none', 'reason')
        self.store.principal = replace(self.store.principal, role='care')
        self.assertEqual(len(self.registers.list('follow_up')['items']), 1)
        with self.assertRaises(PersonalError):
            self.registers.list('topic')
        with self.assertRaises(PersonalError):
            self.registers.candidates('follow_up')
        with self.assertRaises(PersonalError):
            self.registers.export_scoped()

    def test_source_change_hides_candidates_items_and_fences_replay(self):
        args = self.claim()
        with self.store.transaction():
            self.store.db.execute("UPDATE autobio_sources SET generation='g2' WHERE id='s1'")
        self.assertEqual(self.registers.list('follow_up')['items'], [])
        self.assertEqual(self.registers.candidates('follow_up', state='applied')['candidates'], [])
        self.assertFalse(self.registers.validate_claim(*args)['valid'])
        with self.assertRaises(PersonalError):
            self.registers.claim('care', 1, 'follow_up', args[3], 1, 'claim-follow_up')
        with self.assertRaises(PersonalError):
            self.registers.progress(*args, 'progress', 'Stale evidence')

    def test_suppression_blocks_old_claim_receipt(self):
        args = self.claim()
        self.registers.suppress('follow_up', args[3], 1, 'suppress', 'Stop')
        with self.assertRaises(PersonalError):
            self.registers.claim('care', 1, 'follow_up', args[3], 1, 'claim-follow_up')

    def test_all_evidence_sources_checked_and_privacy_purged(self):
        with self.store.transaction():
            self.registers.publish_candidate('follow_up', 's1', 'g1', 'r1', 'c1', {},
                [{'source_id': 's1', 'generation': 'g1'}, {'source_id': 's2', 'generation': 'g1'}], 1)
        args = self.claim()
        with self.store.transaction():
            self.store.db.execute("INSERT INTO autobio_suppressions VALUES('owner','agent','s2',?)", (self.now,))
        self.assertFalse(self.registers.validate_claim(*args)['valid'])
        self.assertEqual(self.registers.list('follow_up')['items'], [])
        with self.store.transaction():
            result = self.registers.purge_sources(['s2'])
        self.assertEqual(result['items'], 1)
        self.assertTrue(all(not rows for rows in self.registers.export_scoped().values()))

    def test_multiple_creates_resolve_server_ids_by_key(self):
        candidates = [self.publish('c1'), self.publish('c2')]
        lease = self.registers.reconcile_claim('reconcile', 1, 'follow_up')
        edits = [{'action': 'create', 'expected_revision': 0, 'key': 'k' + str(i), 'title': 'Title', 'body': 'Body'} for i in range(2)]
        outcomes = [{'candidate_id': c['candidate_id'], 'disposition': 'applied', 'key': 'k' + str(i)} for i, c in enumerate(candidates)]
        result = self.registers.reconcile_commit('reconcile', 1, 'follow_up', lease['fence'], lease['lease_token'], 'multi', edits, outcomes)
        self.assertEqual(len(result['items']), 2)
        self.assertEqual({i['id'] for i in result['items']}, {o['item_id'] for o in result['outcomes']})

    def test_deadline_bound_and_epoch_validation(self):
        self.now = 1550
        args = self.claim()
        self.assertEqual(self.registers._item('follow_up', args[3])['lease_until'], 1600)
        for value in ('2026-01-01T00:00:00Z', True, float('nan'), float('inf')):
            with self.assertRaises(PersonalError):
                self.registers.complete(*args, 'badtime', 'deferred', 'Wait', value)


    def test_execution_deadline_and_bounded_resume_history(self):
        args = self.claim('topic')
        self.assertEqual(self.registers._item('topic', args[3])['lease_until'], 1600)
        self.now += 301
        self.assertTrue(self.registers.validate_claim(*args)['valid'])
        for index in range(22):
            current = (*args[:-1], index + 1)
            self.registers.progress(*current, 'step-' + str(index), 'summary-' + str(index), [{'text': 'finding-' + str(index)}], 'Next step')
        listed = self.registers.list('topic')['items'][0]
        self.assertEqual(len(listed['progress']), 20)
        self.assertEqual(listed['progress'][0]['summary'], 'summary-2')
        self.assertEqual(listed['progress'][-1]['findings'], [{'text': 'finding-21'}])
        with self.store.transaction():
            self.registers.release_operation('research')
        self.operation('resume', 'research')
        claimed = self.registers.claim('resume', 1, 'topic', args[3], 23, 'resume')
        self.assertEqual(claimed['item']['progress'], listed['progress'])
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_register_progress').fetchone()[0], 22)

    def test_reconcile_lease_is_operation_bounded_and_cannot_revive(self):
        self.publish()
        lease = self.registers.reconcile_claim('reconcile', 1, 'follow_up')
        self.assertEqual(lease['lease_until'], 1600)
        self.now = 1301
        self.assertEqual(self.registers.reconcile_claim('reconcile', 1, 'follow_up')['lease_token'], lease['lease_token'])
        self.now = 1600
        with self.assertRaises(PersonalError):
            self.registers.reconcile_claim('reconcile', 1, 'follow_up')
        with self.store.transaction():
            self.assertEqual(self.registers.expire_leases(), 1)
        self.operation('new-reconcile', 'follow_up_reconciler')
        new = self.registers.reconcile_claim('new-reconcile', 1, 'follow_up')
        self.assertGreater(new['fence'], lease['fence'])
        self.assertNotEqual(new['lease_token'], lease['lease_token'])
        self.assertEqual(new['lease_until'], 2200)


if __name__ == '__main__':
    unittest.main()
