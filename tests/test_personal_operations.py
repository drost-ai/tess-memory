from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from agentic_memory.autobiography import AutobiographyStore,PersonalPrincipal,PersonalError
from agentic_memory.autobiography_config import PersonalConfig
from agentic_memory.autobiography_extract import PersonalExtraction
from agentic_memory.config import Config,DataStore
from agentic_memory.personal_operations import PersonalOperations
from test_codex import write_rollout,message


class PersonalOperationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();home=self.root/'source'
        write_rollout(home,items=[message('user','A new meaningful event.'),message('assistant','Understood.')])
        self.now=[datetime(2026,9,20,20,tzinfo=ZoneInfo('UTC')).timestamp()]
        personal=PersonalConfig(enabled=True,owner='owner',agent='agent',extraction_enabled=True,dream_enabled=True,
            historical=True,source_names=('source',),daily_model_calls=2,model_calls=2,daily_token_budget=100000)
        self.cfg=Config(data_dir=self.root/'memory',data_stores=[DataStore('source',home,type='codex')],personal=personal)
        self.store=AutobiographyStore(self.cfg.db_path,PersonalPrincipal('owner','agent',role='extractor',session_id='worker'),clock=lambda:self.now[0]);self.addCleanup(self.store.close)
        self.part=PersonalExtraction(self.cfg,self.store).prepare()['parts'][0]
        self.ops=PersonalOperations(self.cfg,self.store);self.addCleanup(self.ops.close)

    def reserve(self,operation,prompt='source'):
        return self.ops.reserve(operation['operation_id'],operation['fence'],work_id=self.part['id'],identity='fixture',provider='fixture',model='fixture',system='extract',prompt=prompt,input_units=100,slots=1)

    def complete(self,operation,reservation,response=None):
        return self.ops.complete_model(operation['operation_id'],operation['fence'],reservation['request_id'],response=response or {'candidates':[]},usage={'input_tokens':10,'output_tokens':5})

    def test_budget_and_exact_checkpoint_prevent_duplicate_paid_calls(self):
        operation=self.ops.claim('extractor');first=self.reserve(operation)
        self.assertEqual(first['status'],'reserved')
        self.assertEqual(self.reserve(operation)['status'],'busy')
        self.complete(operation,first)
        self.assertEqual(self.reserve(operation)['status'],'cached')
        second=self.reserve(operation,'different');self.complete(operation,second)
        with self.assertRaisesRegex(PersonalError,'allowance exhausted'):self.reserve(operation,'third')
        self.assertEqual(self.ops.status()['reserved_provider_calls'],2)
        self.assertEqual(self.ops.gate.status()['active'],[])

    def test_daily_budget_survives_new_operations_and_resets_by_local_day(self):
        op=self.ops.claim('extractor')
        for prompt in ['one','two']:self.complete(op,self.reserve(op,prompt))
        self.ops.finish(op['operation_id'],op['fence'])
        next_op=self.ops.claim('extractor',manual=True)
        with self.assertRaises(PersonalError):self.reserve(next_op,'three')
        self.now[0]+=86400
        newer=self.ops.claim('extractor')
        third=self.reserve(newer,'three');self.complete(newer,third)
        self.assertEqual(self.ops.status()['reserved_provider_calls'],1)

    def test_schedule_is_calendar_based_with_dst_and_no_early_dream(self):
        operator=AutobiographyStore(self.cfg.db_path,replace(self.store.principal,role='operator'),clock=lambda:self.now[0]);self.addCleanup(operator.close)
        ops=PersonalOperations(self.cfg,operator);self.addCleanup(ops.close)
        result=ops.claim('dream');self.assertEqual(result['status'],'not_due')
        self.assertEqual(datetime.fromtimestamp(result['due'],ZoneInfo('UTC')).hour,23)
        # Test a zone with both transitions in this year. Vancouver's current
        # IANA rules need not have the same fall transition as Los Angeles.
        ops.config=replace(self.cfg,personal=replace(self.cfg.personal,timezone='America/Los_Angeles'))
        for day,hours in [('2026-03-07',23),('2026-10-31',25)]:
            now=datetime.fromisoformat(day+'T23:00:00').replace(tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
            self.assertEqual((ops.next_night(now)-now)/3600,hours)

    def test_expired_operation_cannot_finish_over_replacement(self):
        old=self.ops.claim('extractor');self.assertEqual(self.ops.claim('extractor')['status'],'busy')
        self.now[0]+=1000
        replacement=self.ops.claim('extractor')
        self.assertGreater(replacement['fence'],old['fence'])
        with self.assertRaisesRegex(PersonalError,'expired or was replaced'):self.ops.finish(old['operation_id'],old['fence'])

    def test_disabled_job_makes_no_model_reservation(self):
        ops=PersonalOperations(replace(self.cfg,personal=replace(self.cfg.personal,extraction_enabled=False)),self.store)
        self.addCleanup(ops.close)
        self.assertEqual(ops.claim('extractor')['status'],'disabled')
        self.assertEqual(ops.status()['model_turns'],0)

    def test_model_reservation_and_completion_can_arrive_on_different_rpc_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        operation=self.ops.claim('extractor')
        with ThreadPoolExecutor(max_workers=1) as pool:
            reservation=pool.submit(self.reserve,operation).result(timeout=5)
        self.complete(operation,reservation)
        self.assertEqual(self.ops.gate.status()['active'],[])

    def test_credentials_are_never_checkpointed_and_lease_is_released(self):
        operation=self.ops.claim('extractor');reservation=self.reserve(operation)
        with self.assertRaisesRegex(PersonalError,'credentials belong'):
            self.complete(operation,reservation,{'candidates':[], 'leak':'sk-'+'Z'*24})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_model_cache').fetchone()[0],0)
        self.assertEqual(self.ops.gate.status()['active'],[])

    def test_dead_letter_can_be_explicitly_requeued_without_model_calls(self):
        from agentic_memory.autobiography_inbox import PersonalInbox
        from agentic_memory.personal_recovery import retry_failed
        for _ in range(3):PersonalInbox(self.store).fail(self.part['id'],'invalid')
        self.assertEqual(self.ops.status()['extraction_states'],{'failed':1})
        operator=AutobiographyStore(self.cfg.db_path,replace(self.store.principal,role='operator'))
        self.addCleanup(operator.close)
        result=retry_failed(self.cfg,operator,kind='part',target=self.part['id'],reason='Corrected the provider configuration.')
        self.assertEqual(result['model_calls'],0)
        self.assertEqual(self.ops.status()['extraction_states'],{'pending':1})
        self.assertEqual(operator.db.execute('SELECT COUNT(*) FROM autobio_recovery_receipts').fetchone()[0],1)
        with self.assertRaises(PersonalError):retry_failed(self.cfg,self.store,kind='part',target=self.part['id'],reason='Not an operator')

    def test_completion_receipt_survives_lost_ack_and_new_service_instance(self):
        op=self.ops.claim('extractor');reserved=self.reserve(op)
        receipt=self.complete(op,reserved)
        reopened=PersonalOperations(self.cfg,self.store);self.addCleanup(reopened.close)
        self.assertEqual(reopened.complete_model(op['operation_id'],op['fence'],reserved['request_id'],error='lost_ack'),receipt)
        with self.assertRaisesRegex(PersonalError,'different response'):
            reopened.complete_model(op['operation_id'],op['fence'],reserved['request_id'],response={'different':True})
        self.assertEqual(self.ops.status()['observed_usage']['input_tokens'],10)
        self.assertEqual(self.ops.status()['observed_usage']['complete_usage_turns'],1)
        self.assertEqual(self.ops.status()['remaining_reservations']['provider_calls'],1)

    def test_finish_is_replayable_and_expired_owner_can_only_finalize(self):
        op=self.ops.claim('extractor',request_deadline=self.now[0]+1)
        self.now[0]+=2
        with self.assertRaises(PersonalError):self.reserve(op)
        receipt=self.ops.finish(op['operation_id'],op['fence'],outcome='partial',result={'commits':1})
        self.assertEqual(self.ops.finish(op['operation_id'],op['fence'],outcome='partial',result={'commits':1}),receipt)
        self.assertEqual(self.ops.get_operation(op['operation_id'])['result'],{'commits':1})
        with self.assertRaisesRegex(PersonalError,'different outcome'):
            self.ops.finish(op['operation_id'],op['fence'],outcome='failed')

    def test_host_deadline_only_shortens_and_late_publication_rolls_back(self):
        op=self.ops.claim('extractor',request_deadline=self.now[0]+5)
        self.assertEqual(op['deadline'],self.now[0]+5)
        checks=[]
        def admission():
            checks.append(True)
            if len(checks)==2:self.now[0]+=6
            self.ops._operation(op['operation_id'],op['fence'])
        with self.assertRaises(PersonalError):
            PersonalExtraction(self.cfg,self.store).publish(self.part['id'],[],admission=admission)
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts WHERE id=?',(self.part['id'],)).fetchone()[0],'pending')
