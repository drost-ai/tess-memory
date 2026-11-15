"""Personal memory without work or spending limits (the default): no daily
call or token allowance, no per-job call allowance, no cap on a reflection's
candidates or edits. Configured limits still apply (test_personal_operations)."""

from dataclasses import fields
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from tess_memory.autobiography import AutobiographyStore,PersonalPrincipal
from tess_memory.autobiography_config import OPTIONAL_LIMITS,PersonalConfig
from tess_memory.autobiography_extract import PersonalExtraction,_part_limit
from tess_memory.config import Config,DataStore
from tess_memory.dream_cycle import DreamCycle
from tess_memory.personal_operations import PersonalOperations
from test_codex import write_rollout,message


class LimitsConfigTests(unittest.TestCase):
    def test_limits_are_off_unless_configured(self):
        defaults={f.name:f.default for f in fields(PersonalConfig)}
        self.assertTrue(all(defaults[key] is None for key in OPTIONAL_LIMITS))
        parsed=PersonalConfig.parse({'max_edits':50,'max_candidates':500})
        self.assertEqual((parsed.max_edits,parsed.max_candidates,parsed.daily_token_budget),(50,500,None))
        with self.assertRaisesRegex(ValueError,'positive integer'):
            PersonalConfig.parse({'model_calls':0})

    def test_context_budget_allows_about_128k_tokens(self):
        self.assertEqual(PersonalConfig.parse({'context_budget':512000}).context_budget,512000)
        for bad in (255,512001):
            with self.assertRaisesRegex(ValueError,'context_budget'):
                PersonalConfig.parse({'context_budget':bad})

    def test_part_limit(self):
        self.assertEqual([_part_limit(None,None),_part_limit(5,None),_part_limit(None,3),_part_limit(5,3)],[-1,5,3,3])


class UnlimitedWorkTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();home=self.root/'source'
        text='I prefer concise updates. '*30
        write_rollout(home,items=[message('user',text),message('assistant','Understood.')])
        self.now=[datetime(2026,9,20,20,tzinfo=ZoneInfo('America/Vancouver')).timestamp()]
        personal=PersonalConfig(enabled=True,owner='owner',agent='agent',extraction_enabled=True,dream_enabled=True,
            historical=True,source_names=('source',),shadow=False)
        self.cfg=Config(data_dir=self.root/'memory',data_stores=[DataStore('source',home,type='codex')],personal=personal)
        self.store=AutobiographyStore(self.cfg.db_path,PersonalPrincipal('owner','agent',role='extractor',session_id='worker'),clock=lambda:self.now[0])
        self.addCleanup(self.store.close)
        self.extract=PersonalExtraction(self.cfg,self.store)
        self.part=self.extract.prepare()['parts'][0]

    def test_reservations_never_run_out(self):
        ops=PersonalOperations(self.cfg,self.store);self.addCleanup(ops.close)
        operation=ops.claim('extractor')
        for n in range(40):
            reservation=ops.reserve(operation['operation_id'],operation['fence'],work_id=self.part['id'],identity='fixture',provider='fixture',
                                    model='fixture',system='extract',prompt=f'prompt {n}',input_units=120000,slots=1)
            self.assertEqual(reservation['status'],'reserved')
            ops.complete_model(operation['operation_id'],operation['fence'],reservation['request_id'],response={'candidates':[]},usage={'input_tokens':10,'output_tokens':5})
        status=ops.status()
        self.assertEqual(status['reserved_provider_calls'],40)
        self.assertEqual(status['remaining_reservations'],{'provider_calls':None,'tokens':None})
        self.assertEqual(status['limits'],{'daily_provider_calls':None,'daily_token_reservation_budget':None})

    def test_a_reflection_takes_every_candidate_and_any_number_of_edits(self):
        candidates=[{'segment':'Experiences','title':f'Update {n}','body':f'Observation number {n} about updates.',
                     'attribution':'user_statement','quote':'I prefer concise updates.'} for n in range(20)]
        ids=self.extract.publish(self.part['id'],candidates)['candidate_ids']
        self.assertEqual(len(ids),20)
        dreamer=AutobiographyStore(self.cfg.db_path,PersonalPrincipal('owner','agent',role='dream',session_id='dream'),clock=lambda:self.now[0])
        self.addCleanup(dreamer.close)
        dream=DreamCycle(self.cfg,dreamer);cycle=dream.prepare()
        self.assertEqual(len(cycle['candidates']),20)
        self.assertNotIn('edits',cycle['limits'])
        edits=[{'expected_revision':0,'segment':'Experiences','title':f'Update {n}','body':f'Observation number {n}.',
                'reason':'Integrate the statement.','attribution':'user_statement'} for n in range(20)]
        outcomes=[{'candidate_id':cid,'action':'accepted','reason':'Explicit statement.','edit_indices':[n]} for n,cid in enumerate(ids)]
        receipt=dream.commit(cycle['cycle_id'],cycle['fence'],cycle['lease_token'],idempotency_key='all',edits=edits,outcomes=outcomes)
        self.assertEqual(len(receipt['edits']),20)


if __name__=='__main__':
    unittest.main()
