from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from agentic_memory.autobiography import AutobiographyStore,PersonalPrincipal,PersonalError
from agentic_memory.autobiography_import import preview_seeds,import_seeds
from agentic_memory.autobiography_inbox import PersonalInbox


class PersonalContractTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.principal=PersonalPrincipal('owner','agent',role='operator',session_id='first')
        self.store=AutobiographyStore(self.root/'memory.db',self.principal);self.addCleanup(self.store.close)

    def test_same_actor_retry_and_new_actor_retry_are_explicitly_different(self):
        edit=dict(expected_revision=0,segment='Self',title='Self',body='Deliberate reflection',reason='Initial authorship',idempotency_key='request')
        receipt=self.store.update(**edit)
        reopened=AutobiographyStore(self.store.path,self.principal);self.addCleanup(reopened.close)
        self.assertEqual(reopened.update(**edit),receipt)
        new_actor=AutobiographyStore(self.store.path,replace(self.principal,session_id='second'));self.addCleanup(new_actor.close)
        with self.assertRaises(PersonalError) as error:new_actor.update(**edit)
        self.assertEqual(error.exception.code,'idempotency_mismatch')
        self.assertEqual(new_actor.get(receipt['edits'][0]['record_id'])['revision'],1)

    def test_restore_missing_evidence_preserves_history_and_rejects_atomically(self):
        path=self.root/'inherited.md';path.write_text('An inherited statement.')
        files={'Self':str(path)};preview=preview_seeds(files)
        receipt=import_seeds(self.store,files,preview_hash=preview['preview_hash'],idempotency_key='import')
        rid=receipt['edits'][0]['record_id'];path.unlink()
        self.assertEqual(self.store.get(rid)['evidence_availability'],[False])
        with self.assertRaises(PersonalError) as error:
            self.store.restore(rid,1,expected_revision=1,reason='Restore old state',idempotency_key='restore')
        self.assertEqual(error.exception.code,'invalid_evidence')
        self.assertEqual(len(self.store.history(rid)['revisions']),1)

    def test_logical_erasure_receipt_does_not_claim_forensic_removal(self):
        receipt=self.store.update(expected_revision=0,segment='Self',title='Self',body='Private sentence.',reason='Self-authored',idempotency_key='initial')
        erased=self.store.erase(receipt['edits'][0]['record_id'],expected_revision=1)
        self.assertEqual(erased['erasure_scope'],'logical_application_state');self.assertFalse(erased['forensic_erasure'])
        self.assertEqual(self.store.list_records()['records'],[])
        self.assertEqual(self.store.search('private')['records'],[])

    def test_producer_audience_is_pinned_without_changing_authored_read_scoping(self):
        PersonalInbox(self.store).initialize()
        guest=AutobiographyStore(self.store.path,replace(self.principal,audience='guest'));self.addCleanup(guest.close)
        self.assertEqual(guest.list_records()['records'],[])
        with self.assertRaises(PersonalError) as error:PersonalInbox(guest).initialize()
        self.assertEqual(error.exception.code,'operational_audience')
