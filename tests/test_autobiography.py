from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from agentic_memory.autobiography import AutobiographyStore, PersonalPrincipal, PersonalError, SEGMENTS
from agentic_memory.store import MemoryStore


class AutobiographyTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.path=Path(tmp.name)/'memory.db'
        self.principal=PersonalPrincipal('sam','assistant',role='author',session_id='root-1')
        self.store=AutobiographyStore(self.path,self.principal)
        self.addCleanup(self.store.close)

    def create(self,**overrides):
        edit={'expected_revision':0,'segment':'Identity','title':'Current self','body':'I verify current capabilities.',
              'reason':'Deliberate self-authorship','idempotency_key':'first'}
        edit.update(overrides)
        return self.store.update(**edit)

    def test_autonomous_all_segments_with_full_revision_and_restore(self):
        for segment in SEGMENTS:
            receipt=self.create(segment=segment,idempotency_key=segment)
            record_id=receipt['edits'][0]['record_id']
            self.store.update(record_id=record_id,expected_revision=1,title=segment,body='A revised principle or interpretation.',
                              reason='Autonomous reconsideration',idempotency_key=segment+'-edit')
            self.assertEqual(self.store.get(record_id)['revision'],2)
            self.assertEqual(len(self.store.history(record_id)['revisions']),2)
            self.store.restore(record_id,1,expected_revision=2,reason='Restore deliberately',idempotency_key=segment+'-restore')
            self.assertEqual(self.store.get(record_id)['revision'],3)
            self.assertEqual(self.store.get(record_id)['body'],'I verify current capabilities.')

    def test_idempotency_replays_receipt_and_rejects_changed_payload(self):
        first=self.create();self.assertEqual(first,self.create())
        with self.assertRaisesRegex(PersonalError,'different request'):self.create(body='Different content')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_revisions').fetchone()[0],1)

    def test_credentials_and_oversized_mandatory_core_cannot_strand_next_session(self):
        for text,code in [('sk-'+'A'*24,'sensitive_content'),('x'*7000,'context_overflow')]:
            with self.assertRaises(PersonalError) as error:self.create(body=text)
            self.assertEqual(error.exception.code,code)
            self.assertEqual(self.store.list_records()['records'],[])
        self.create()

    def test_concurrent_stale_writer_cannot_overwrite_or_publish_partial_transaction(self):
        record_id=self.create()['edits'][0]['record_id']
        other=AutobiographyStore(self.path,replace(self.principal,session_id='root-2'));self.addCleanup(other.close)
        other.update(record_id=record_id,expected_revision=1,title='New',body='New body',reason='Live edit',idempotency_key='other')
        with self.assertRaisesRegex(PersonalError,'revision changed'):
            self.store.update(record_id=record_id,expected_revision=1,title='Stale',body='Old overwrite',reason='Old dream',idempotency_key='stale')
        self.assertEqual(self.store.get(record_id)['body'],'New body')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_commits').fetchone()[0],2)

    def test_wrong_owner_audience_and_readonly_role_fail_without_existence_leak(self):
        record_id=self.create()['edits'][0]['record_id']
        with self.assertRaisesRegex(PersonalError,'scope unavailable'):
            AutobiographyStore(self.path,replace(self.principal,owner='other'))
        guest=AutobiographyStore(self.path,replace(self.principal,audience='guest',role='reader'))
        self.addCleanup(guest.close)
        errors=[]
        for rid in [record_id,'does-not-exist']:
            with self.assertRaises(PersonalError) as e:guest.get(rid)
            errors.append((e.exception.code,str(e.exception)))
        self.assertEqual(errors[0],errors[1])
        self.assertEqual(guest.search('capabilities')['records'],[])
        with self.assertRaisesRegex(PersonalError,'unavailable to this principal'):
            guest.update(idempotency_key='forbidden',expected_revision=0,segment='Identity',title='x',body='x',reason='x')

    def test_coding_rows_and_default_search_projection_are_untouched(self):
        coding=MemoryStore(self.path);self.addCleanup(coding.close)
        before={t:[tuple(r) for r in coding.db.execute('SELECT * FROM '+t)] for t in ['memories','plan_items','sessions']}
        self.create(body='Private personal memory sentinel')
        self.assertEqual(before,{t:[tuple(r) for r in coding.db.execute('SELECT * FROM '+t)] for t in before})
        self.assertEqual(coding.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0],0)
        self.assertEqual(len(self.store.search('sentinel')['records']),1)

    def test_acknowledged_write_survives_index_failure_and_reopen(self):
        with mock.patch.object(self.store,'index_record',return_value=False):record_id=self.create()['edits'][0]['record_id']
        reopened=AutobiographyStore(self.path,self.principal);self.addCleanup(reopened.close)
        self.assertEqual(reopened.get(record_id)['revision'],1)
        self.assertEqual(reopened.search('capabilities')['records'],[])
        reopened.index_record(record_id)
        self.assertEqual(len(reopened.search('capabilities')['records']),1)

    def test_mid_write_failure_rolls_back_head_revision_and_receipt(self):
        original=self.store._write
        def fail(*args):original(*args);raise RuntimeError('crash before commit')
        with mock.patch.object(self.store,'_write',side_effect=fail):
            with self.assertRaises(RuntimeError):self.create()
        for table in ['autobio_records','autobio_revisions','autobio_commits']:
            self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0],0)
        self.create()

    def test_evidence_requires_matching_owned_generation_and_valid_span(self):
        with self.store.transaction():
            self.store.db.execute('INSERT INTO autobio_sources VALUES(?,?,?,?,?,?,?,?)',
                ('source1','sam','assistant','owner','g1',2,'{}',1))
        valid={'source_id':'source1','generation':'g1','start':0,'end':1,'attribution':'user_statement'}
        self.create(evidence=[valid],attribution='user_statement')
        for change in [{'generation':'g2'},{'source_id':'foreign'},{'end':3},{'start':True}]:
            with self.assertRaises(PersonalError):self.create(idempotency_key='bad',segment='People',evidence=[{**valid,**change}])
        with self.assertRaises(PersonalError):self.create(segment='People',idempotency_key='fact',attribution='observed_fact')

    def test_retire_and_erase_do_not_reimport_or_keep_search_text(self):
        record_id=self.create(body='Private obsolete secretlike sentence')['edits'][0]['record_id']
        self.store.update(record_id=record_id,expected_revision=1,title='Retired',body='Private obsolete secretlike sentence',
                          reason='Retire',state='retired',idempotency_key='retire')
        self.assertEqual(self.store.search('obsolete')['records'],[])
        self.assertEqual(len(self.store.history(record_id)['revisions']),2)
        with self.assertRaises(PersonalError):self.store.erase(record_id,expected_revision=2)
        operator=AutobiographyStore(self.path,replace(self.principal,role='operator'));self.addCleanup(operator.close)
        operator.erase(record_id,expected_revision=2)
        self.assertEqual(operator.export()['records'],[])
        for table in ['autobio_revisions','autobio_search']:
            self.assertEqual(operator.db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0],0)
        # The original receipt remains idempotent, and cannot recreate erased text.
        self.create(body='Private obsolete secretlike sentence')
        self.assertEqual(operator.export()['records'],[])

    def test_unsupported_schema_is_rejected_and_normal_data_unchanged(self):
        self.create()
        self.store.db.execute("UPDATE autobio_meta SET value='999' WHERE key='version'")
        with self.assertRaises(PersonalError):AutobiographyStore(self.path,self.principal)
        self.assertEqual(self.store.get(self.store.list_records()['records'][0]['id'])['revision'],1)

    def test_seed_import_requires_preview_and_is_inherited_retired_and_idempotent(self):
        from agentic_memory.autobiography_import import preview_seeds,import_seeds
        file=self.path.parent/'self.md';file.write_text('An inherited identity claim.')
        files={'Identity':str(file)};preview=preview_seeds(files)
        operator=AutobiographyStore(self.path,replace(self.principal,role='operator'));self.addCleanup(operator.close)
        receipt=import_seeds(operator,files,preview_hash=preview['preview_hash'],idempotency_key='seed')
        self.assertEqual(receipt,import_seeds(operator,files,preview_hash=preview['preview_hash'],idempotency_key='seed'))
        record=operator.get(receipt['edits'][0]['record_id'],include_inactive=True)
        self.assertEqual((record['state'],record['origin'],record['attribution']),('retired','import','inherited'))
        file.write_text('Changed after preview')
        with self.assertRaises(PersonalError):import_seeds(operator,files,preview_hash=preview['preview_hash'],idempotency_key='changed')
