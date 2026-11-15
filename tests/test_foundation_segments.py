"""Soul, User, Identity and Core: the core of personal memory; Self and Principles are retired.

Isolated temporary stores only.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from agentic_memory.autobiography import (
    AutobiographyStore, FOUNDATION_SEGMENTS, PersonalError, PersonalPrincipal, RETIRED_SEGMENTS, SEGMENTS, WORKER_SEGMENTS,
)
from agentic_memory.autobiography_context import personal_context
from agentic_memory.autobiography_inbox import PersonalInbox
from agentic_memory.autobiography_lifecycle import write_transition


class FoundationSegmentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'memory.db'
        self.principal = PersonalPrincipal('sam', 'assistant', role='author', session_id='root-1')
        self.store = self.open(self.principal)

    def open(self, principal):
        store = AutobiographyStore(self.path, principal, context_budget=64000)
        self.addCleanup(store.close)
        return store

    def write(self, segment, body, store=None, **extra):
        receipt = (store or self.store).update(expected_revision=0, segment=segment, title=segment, body=body,
                                               reason='authored', idempotency_key='create-' + segment, **extra)
        return receipt['edits'][0]['record_id']

    def test_the_four_core_segments_and_the_four_worker_segments(self):
        self.assertEqual(FOUNDATION_SEGMENTS, ('Soul', 'User', 'Identity', 'Core'))
        self.assertEqual(WORKER_SEGMENTS, ('People', 'Relationships', 'Experiences', 'Reflections'))
        self.assertEqual(RETIRED_SEGMENTS, ('Self', 'Principles'))
        self.assertFalse(set(RETIRED_SEGMENTS) & set(SEGMENTS))
        self.assertEqual(set(SEGMENTS), set(FOUNDATION_SEGMENTS) | set(WORKER_SEGMENTS))
        for segment in FOUNDATION_SEGMENTS:
            self.assertEqual(self.store.list_records(segment=segment)['records'], [])

    def test_each_is_one_current_document_the_author_writes_and_revises(self):
        for segment in FOUNDATION_SEGMENTS:
            record_id = self.write(segment, 'In my own words.')
            record = self.store.get(record_id)
            self.assertEqual((record['segment'], record['subject'], record['revision']), (segment, 'core', 1))
            with self.assertRaises(PersonalError) as second:
                self.store.update(expected_revision=0, segment=segment, title='Another', body='A second document.',
                                  reason='authored', idempotency_key='again-' + segment)
            self.assertEqual(second.exception.code, 'conflict')
            with self.assertRaises(PersonalError):
                self.store.update(expected_revision=0, segment=segment, subject='other', title='Another',
                                  body='A second document.', reason='authored', idempotency_key='subject-' + segment)
            self.store.update(record_id=record_id, expected_revision=1, title=segment, body='Revised in my own words.',
                              reason='I changed', idempotency_key='revise-' + segment)
            self.assertEqual(self.store.get(record_id)['body'], 'Revised in my own words.')

    def legacy(self, segment, body):
        """A Self or Principles record as an older store holds it: such
        records can no longer be created through the store."""
        record_id = 'legacy-' + segment.lower()
        with self.store.transaction():
            self.store.db.execute('INSERT INTO autobio_records VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                  (record_id, 'sam', 'assistant', 'owner', segment, 'core', segment, 1, 'active', 1.0, 1.0))
            self.store.db.execute('INSERT INTO autobio_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                                  (record_id, 1, body, segment, 'active', 'Authored before the foundation documents.', 'root-0',
                                   'interactive', 'agent_reflection', '[]', 1.0, 'commit-' + record_id))
            self.store.db.execute('INSERT INTO autobio_lifecycle VALUES(?,?,?,?,?,?,?)', (record_id, 1, None, None, 'active', None, None))
        return record_id

    def test_the_four_are_the_core_and_load_in_order(self):
        empty = personal_context(self.store, budget=64000)
        self.assertEqual((empty['status'], empty['manifest']['missing_core']), ('uninitialized', ['Soul', 'User', 'Identity', 'Core']))
        self.write('Core', 'CORE TEXT')
        self.write('Identity', 'IDENTITY TEXT')
        partial = personal_context(self.store, budget=64000)
        self.assertEqual((partial['status'], partial['manifest']['missing_core']), ('partial_core', ['Soul', 'User']))
        self.write('User', 'USER TEXT')
        self.write('Soul', 'SOUL TEXT')
        context = personal_context(self.store, budget=64000)
        self.assertEqual((context['status'], context['manifest']['missing_core']), ('ready', []))
        self.assertEqual([record['segment'] for record in context['manifest']['records']], ['Soul', 'User', 'Identity', 'Core'])
        positions = [context['text'].index(text) for text in ('SOUL TEXT', 'USER TEXT', 'IDENTITY TEXT', 'CORE TEXT')]
        self.assertEqual(positions, sorted(positions))

    def test_self_and_principles_are_retired_nothing_new_is_written_to_them(self):
        for segment in RETIRED_SEGMENTS:
            with self.assertRaises(PersonalError) as created:
                self.store.update(expected_revision=0, segment=segment, title=segment, body='A new one.',
                                  reason='authored', idempotency_key='new-' + segment)
            self.assertEqual(created.exception.code, 'invalid')
            record_id = self.legacy(segment, segment.upper() + ' BODY')
            with self.assertRaises(PersonalError) as edited:
                self.store.update(record_id=record_id, expected_revision=1, title=segment, body='Rewritten.',
                                  reason='authored', idempotency_key='edit-' + segment)
            self.assertEqual(edited.exception.code, 'invalid')
            # What a store already holds stays readable, with its history.
            self.assertEqual(self.store.get(record_id)['body'], segment.upper() + ' BODY')
            self.assertEqual(len(self.store.history(record_id)['revisions']), 1)
            self.assertEqual([r['id'] for r in self.store.list_records(segment=segment)['records']], [record_id])

    def test_an_existing_self_and_principles_load_only_until_the_first_foundation_document(self):
        self.legacy('Self', 'SELF BODY')
        self.legacy('Principles', 'PRINCIPLES BODY')
        before = personal_context(self.store, budget=64000)
        # An existing identity whose foundation documents are still to be written: not a new one.
        self.assertEqual(before['status'], 'partial_core')
        self.assertIn('SELF BODY', before['text'])
        self.assertIn('PRINCIPLES BODY', before['text'])
        self.write('Soul', 'SOUL TEXT')
        after = personal_context(self.store, budget=64000)
        self.assertEqual(after['status'], 'partial_core')
        self.assertEqual([record['segment'] for record in after['manifest']['records']], ['Soul'])
        self.assertNotIn('SELF BODY', after['text'])
        self.assertNotIn('PRINCIPLES BODY', after['text'])

    def test_the_author_retires_an_existing_self_and_a_worker_cannot(self):
        record_id = self.legacy('Self', 'SELF BODY')
        worker = self.open(replace(self.principal, role='reconciler', session_id='worker-1'))
        with worker.transaction():
            with self.assertRaises(PersonalError) as refused:
                write_transition(worker, {'operation': 'lifecycle', 'record_id': record_id, 'expected_revision': 1,
                                          'state': 'retired', 'reason': 'reconciled'}, 'commit', 'dream_cycle')
            self.assertEqual(refused.exception.code, 'forbidden')
        self.store.transition(record_id, expected_revision=1, state='retired', reason='Replaced by Identity and Soul.',
                              idempotency_key='retire-self')
        self.assertEqual(self.store.get(record_id, include_inactive=True)['state'], 'retired')
        self.assertNotIn('SELF BODY', personal_context(self.store, budget=64000)['text'])
        self.assertEqual(personal_context(self.store, budget=64000)['status'], 'uninitialized')

    def test_seeds_are_the_four_foundation_documents_only(self):
        from agentic_memory.autobiography_import import import_seeds, preview_seeds
        folder = self.path.parent
        files = {}
        for segment in FOUNDATION_SEGMENTS:
            (folder / (segment + '.md')).write_text('The ' + segment + ' document.')
            files[segment] = str(folder / (segment + '.md'))
        for segment in RETIRED_SEGMENTS + WORKER_SEGMENTS:
            (folder / 'other.md').write_text('Not a foundation document.')
            with self.assertRaises(PersonalError):
                preview_seeds({segment: str(folder / 'other.md')})
        operator = self.open(replace(self.principal, role='operator'))
        preview = preview_seeds(files)
        receipt = import_seeds(operator, files, preview_hash=preview['preview_hash'], idempotency_key='seed', activate=True)
        self.assertEqual(len(receipt['edits']), 4)
        context = personal_context(self.store, budget=64000)
        self.assertEqual(context['status'], 'ready')
        self.assertIn('The Soul document.', context['text'])

    def test_a_foundation_document_too_large_for_the_context_is_refused_at_write(self):
        small = AutobiographyStore(self.path, self.principal, context_budget=600)
        self.addCleanup(small.close)
        with self.assertRaises(PersonalError) as refused:
            self.write('Soul', 'x' * 2000, store=small)
        self.assertEqual(refused.exception.code, 'context_overflow')
        self.assertEqual(self.store.list_records(segment='Soul')['records'], [])

    def test_background_workers_cannot_create_edit_or_retire_them(self):
        soul = self.write('Soul', 'In my own words.')
        for role in ('reconciler', 'dream', 'extractor'):
            worker = self.open(replace(self.principal, role=role, session_id='worker-1'))
            with worker.transaction():
                with self.assertRaises(PersonalError) as edit:
                    worker._write({'record_id': soul, 'expected_revision': 1, 'title': 'Soul', 'body': 'Rewritten.',
                                   'reason': 'reconciled'}, 'commit', 'dream_cycle')
                self.assertEqual(edit.exception.code, 'forbidden')
                with self.assertRaises(PersonalError) as create:
                    worker._write({'expected_revision': 0, 'segment': 'Identity', 'title': 'Identity', 'body': 'Invented.',
                                   'reason': 'reconciled'}, 'commit', 'dream_cycle')
                self.assertEqual(create.exception.code, 'forbidden')
                with self.assertRaises(PersonalError) as retire:
                    write_transition(worker, {'operation': 'lifecycle', 'record_id': soul, 'expected_revision': 1,
                                              'state': 'retired', 'reason': 'reconciled'}, 'commit', 'dream_cycle')
                self.assertEqual(retire.exception.code, 'forbidden')
                # The segments workers do reconcile stay writable.
                worker._write({'expected_revision': 0, 'segment': 'Reflections', 'subject': 'r-' + role, 'title': 'A reflection',
                               'body': 'Reconciled.', 'reason': 'reconciled'}, 'commit', 'dream_cycle')
        self.assertEqual(self.store.get(soul)['body'], 'In my own words.')
        self.assertEqual(self.store.get(soul)['revision'], 1)

    def test_extraction_cannot_propose_a_candidate_for_them(self):
        extractor = self.open(replace(self.principal, role='extractor'))
        inbox = PersonalInbox(extractor)
        inbox.initialize()
        with extractor.transaction():
            extractor.db.execute('INSERT INTO autobio_sources VALUES(?,?,?,?,?,?,?,?)',
                                 ('source-1', 'sam', 'assistant', 'owner', 'g1', 40, '{"roles":[]}', 1.0))
            extractor.db.execute("INSERT INTO autobio_extraction_parts(id,source_id,generation,start,end,text,state,updated) "
                                 "VALUES('part-1','source-1','g1',0,40,'Sam said the soul is the part that stays.','pending',1.0)")
        for segment in FOUNDATION_SEGMENTS:
            with self.assertRaises(PersonalError) as refused:
                inbox.publish('part-1', [{'segment': segment, 'title': 'A title', 'body': 'A body.',
                                          'attribution': 'agent_reflection', 'quote': 'the soul is the part that stays'}])
            self.assertEqual(refused.exception.code, 'invalid')

    def test_other_audiences_share_soul_and_identity_but_not_user_or_core(self):
        for segment in FOUNDATION_SEGMENTS:
            self.write(segment, segment.upper() + ' TEXT')
        guest = self.open(PersonalPrincipal('sam', 'assistant', audience='kim', role='reader', reads='owner'))
        visible = {record['segment'] for record in guest.list_records()['records']}
        self.assertEqual(visible, {'Soul', 'Identity'})
        text = personal_context(guest, budget=64000)['text']
        self.assertIn('SOUL TEXT', text)
        self.assertIn('IDENTITY TEXT', text)
        self.assertNotIn('USER TEXT', text)
        self.assertNotIn('CORE TEXT', text)


if __name__ == '__main__':
    unittest.main()
