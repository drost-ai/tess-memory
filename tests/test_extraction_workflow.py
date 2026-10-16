import unittest
from test_autobiography_extract import ExtractionTests
from agentic_memory.extraction_workflow import ExtractionWorkflow
from agentic_memory.personal_operations import PersonalOperations
from agentic_memory.autobiography import PersonalError


class WorkflowTests(unittest.TestCase):
    candidate = ExtractionTests.candidate

    def setUp(self):
        ExtractionTests.setUp(self)
        self.ops = PersonalOperations(self.cfg, self.store)
        self.addCleanup(self.ops.close)
        self.flow = ExtractionWorkflow(self.cfg, self.store, self.ops)
        self.op = self.ops.claim('extractor', manual=True)
        self.auth = (self.op['operation_id'], self.op['fence'])
        self.range = self.flow.claim(*self.auth)['ranges'][0]
        self.args = (*self.auth, self.range['range_id'], self.range['lease_token'])

    def test_drafts_hidden_until_atomic_finalize(self):
        first = self.flow.candidate(*self.args,'one','r1',0,candidate=self.candidate())
        self.assertEqual(self.flow.candidate(*self.args,'one','r1',0,candidate=self.candidate()),first)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_candidates').fetchone()[0],0)
        with self.assertRaises(PersonalError):
            self.flow.candidate(*self.args,'one','r1',0,candidate={**self.candidate(),'body':'changed'})
        with self.assertRaises(PersonalError):
            self.extract.publish(self.range['range_id'], [self.candidate()])
        receipt = self.flow.finalize(*self.args,'final1')
        self.assertEqual(receipt['candidate_ids'],[first['candidate_id']])
        self.assertEqual(self.flow.finalize(*self.args,'final1'),receipt)
        self.assertEqual(self.flow.read(*self.args)['receipt'],receipt)
        self.assertEqual(receipt['cursor']['end'],self.range['end'])

    def test_withdraw_and_empty_completion(self):
        first = self.flow.candidate(*self.args,'one','r1',0,candidate=self.candidate())
        withdrawn = self.flow.candidate(*self.args,'one','r2',first['revision'],'withdraw')
        self.assertEqual(withdrawn['state'],'withdrawn')
        receipt = self.flow.finalize(*self.args,'empty')
        self.assertEqual(receipt['candidate_ids'],[])
        self.assertEqual(self.flow.finalize(*self.args,'empty'),receipt)

    def test_reopen_and_fencing(self):
        first = self.flow.candidate(*self.args,'one','r1',0,candidate=self.candidate())
        reopened = ExtractionWorkflow(self.cfg,self.store,self.ops)
        self.assertEqual(reopened.read(*self.args)['drafts']['one']['candidate_id'],first['candidate_id'])
        with self.assertRaises(PersonalError):
            reopened.finalize(*self.auth,self.range['range_id'],'wrong','invalid')
        self.ops.finish(*self.auth,outcome='completed',result={})
        new = self.ops.claim('extractor',manual=True)
        ranges = reopened.claim(new['operation_id'],new['fence'])['ranges']
        reclaimed = next(r for r in ranges if r['range_id']==self.range['range_id'])
        with self.assertRaises(PersonalError):
            reopened.finalize(*self.args,'old')
        receipt = reopened.finalize(new['operation_id'],new['fence'],reclaimed['range_id'],reclaimed['lease_token'],'recover')
        self.assertEqual(receipt['candidate_ids'],[first['candidate_id']])

    def test_changed_source_rejects_finalization(self):
        self.flow.candidate(*self.args,'one','r1',0,candidate=self.candidate())
        with self.store.transaction():
            self.store.db.execute("UPDATE autobio_sources SET generation='changed'")
        with self.assertRaises(PersonalError):
            self.flow.finalize(*self.args,'stale')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_candidates').fetchone()[0],0)

    def test_finalize_callback_failure_rolls_back_all_publication(self):
        from unittest.mock import patch
        first=self.flow.candidate(*self.args,'one','draft',0,candidate=self.candidate())
        with patch.object(self.flow,'_remember',side_effect=RuntimeError('lost transaction')):
            with self.assertRaises(RuntimeError):self.flow.finalize(*self.args,'finish')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_candidates').fetchone()[0],0)
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts WHERE id=?',(self.range['range_id'],)).fetchone()[0],'pending')
        self.assertEqual(self.flow.finalize(*self.args,'finish')['candidate_ids'],[first['candidate_id']])

    def test_synthetic_user_turn_is_not_extraction_evidence(self):
        from agentic_memory.transcript import Turn
        from agentic_memory.autobiography_extract import clean_unit
        text,roles=clean_unit([Turn('user','invented scheduled request',human_evidence=False),Turn('assistant','A reflection')])
        self.assertNotIn('invented scheduled request',text)
        self.assertFalse(any(r['role']=='user' for r in roles))

    def test_out_of_order_empty_finalization_does_not_skip_gap(self):
        import uuid
        from agentic_memory.autobiography import canonical
        rid=self.range['range_id'];second=uuid.uuid4().hex
        with self.store.transaction():
            row=self.store.db.execute('SELECT * FROM autobio_extraction_parts WHERE id=?',(rid,)).fetchone()
            values=dict(row);split=len(row['text'])//2
            self.store.db.execute('UPDATE autobio_extraction_parts SET end=?,text=? WHERE id=?',(split,row['text'][:split],rid))
            values.update(id=second,start=split,text=row['text'][split:],work_token='second')
            columns=list(values)
            self.store.db.execute('INSERT INTO autobio_extraction_parts ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',tuple(values[c] for c in columns))
        last=self.flow.finalize(*self.auth,second,'second','last-first')
        self.assertEqual(last['cursor']['end'],0)
        first=self.flow.finalize(*self.args,'first-last')
        self.assertEqual(first['cursor']['end'],self.range['end'])

    def test_expired_operation_adopts_drafts_and_fences_old_writer(self):
        first=self.flow.candidate(*self.args,'one','draft',0,candidate=self.candidate())
        now=self.store.clock()
        self.store.clock=lambda: now+self.cfg.personal.max_seconds+1
        new=self.ops.claim('extractor',manual=True)
        reclaimed=next(r for r in self.flow.claim(new['operation_id'],new['fence'])['ranges'] if r['range_id']==self.range['range_id'])
        self.assertEqual(reclaimed['drafts']['one']['candidate_id'],first['candidate_id'])
        with self.assertRaises(PersonalError):self.flow.finalize(*self.args,'stale-writer')
        result=self.flow.finalize(new['operation_id'],new['fence'],reclaimed['range_id'],reclaimed['lease_token'],'adopted')
        self.assertEqual(result['candidate_ids'],[first['candidate_id']])

    def test_foreign_audience_cannot_read_or_finalize_range(self):
        from dataclasses import replace
        from agentic_memory.autobiography import AutobiographyStore
        foreign=AutobiographyStore(self.cfg.db_path,replace(self.store.principal,audience='foreign'))
        self.addCleanup(foreign.close)
        # The operational audience gate rejects the foreign producer before
        # it can even create a workflow or obtain a range-reading capability.
        with self.assertRaisesRegex(PersonalError,'one operational audience'):
            ExtractionWorkflow(self.cfg,foreign,self.ops)
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts WHERE id=?',(self.range['range_id'],)).fetchone()[0],'pending')
