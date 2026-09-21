from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agentic_memory.autobiography import AutobiographyStore, PersonalPrincipal, PersonalError


class IndexRepairTests(unittest.TestCase):
    def setUp(self):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        self.path=Path(root.name)/'memory.db';self.now=[2000000000.0]
        self.principal=PersonalPrincipal('owner','agent',role='operator',session_id='one')
        self.store=AutobiographyStore(self.path,self.principal,clock=lambda:self.now[0]);self.addCleanup(self.store.close)

    def create(self, **changes):
        edit=dict(expected_revision=0,segment='Principles',title='Principle',body='Durable sentinel',reason='Deliberate authorship',idempotency_key='first')
        return self.store.update(**{**edit,**changes})['edits'][0]['record_id']

    def test_crash_gap_survives_reopen_and_repair_changes_no_authored_history(self):
        with patch.object(self.store,'index_record',return_value=False):rid=self.create()
        reopened=AutobiographyStore(self.path,self.principal);self.addCleanup(reopened.close)
        search=reopened.search('sentinel')
        self.assertEqual(search['records'],[]);self.assertEqual(search['index']['pending_count'],1)
        before=[tuple(r) for r in reopened.db.execute('SELECT * FROM autobio_revisions')]
        self.assertEqual(reopened.repair_index()['repaired'],1)
        self.assertEqual(reopened.search('sentinel')['records'][0]['id'],rid)
        self.assertEqual(before,[tuple(r) for r in reopened.db.execute('SELECT * FROM autobio_revisions')])
        self.assertEqual(reopened.repair_index()['attempted'],0)

    def test_real_index_error_is_visible_backed_off_and_repair_uses_newest_head(self):
        rid=self.create()
        def deny_insert(action, table, *args):
            return sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_INSERT and table=='autobio_search' else sqlite3.SQLITE_OK
        self.store.db.set_authorizer(deny_insert)
        self.create(record_id=rid,expected_revision=1,idempotency_key='second',body='Newest sentinel')
        self.store.db.set_authorizer(None)
        pending=self.store.index_status()['pending'][0]
        self.assertEqual(pending['revision'],2);self.assertEqual(pending['indexed_revision'],1)
        self.assertEqual(pending['error'],'DatabaseError')
        self.assertEqual(self.store.repair_index()['attempted'],0)
        self.now[0]+=601
        self.assertEqual(self.store.repair_index()['repaired'],1)
        self.assertEqual(self.store.search('newest')['records'][0]['revision'],2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_index_errors').fetchone()[0],0)

    def test_missing_projection_repair_is_bounded_and_scope_isolated(self):
        self.create()
        self.create(segment='Reflections',idempotency_key='reflection')
        with self.store.transaction():self.store.db.execute('DELETE FROM autobio_search')
        self.assertEqual(self.store.index_status()['pending_count'],2)
        guest=AutobiographyStore(self.path,replace(self.principal,audience='guest',role='reader'));self.addCleanup(guest.close)
        self.assertEqual(guest.index_status()['pending_count'],0)
        with self.assertRaises(PersonalError):guest.repair_index()
        result=self.store.repair_index(limit=1)
        self.assertEqual(result['repaired'],1);self.assertEqual(result['index']['pending_count'],1)
        self.assertEqual(self.store.repair_index(limit=1)['index']['state'],'current')
