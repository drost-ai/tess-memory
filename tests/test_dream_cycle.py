from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from agentic_memory.autobiography import AutobiographyStore,PersonalPrincipal,PersonalError
from agentic_memory.autobiography_config import PersonalConfig
from agentic_memory.autobiography_extract import PersonalExtraction
from agentic_memory.autobiography_inbox import PersonalInbox
from agentic_memory.config import Config,DataStore
from agentic_memory.dream_cycle import DreamCycle
from test_codex import write_rollout,message


class DreamTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.now=[2000000000.0]
        home=self.root/'source'
        write_rollout(home,items=[message('user','I prefer concise updates.'),message('assistant','Understood.')])
        cfg=PersonalConfig(enabled=True,owner='owner',agent='agent',source_names=('source',),historical=True,
                           extraction_enabled=True,dream_enabled=True,shadow=False)
        self.cfg=Config(data_dir=self.root/'memory',data_stores=[DataStore('source',home,type='codex')],personal=cfg)
        self.extractor=AutobiographyStore(self.cfg.db_path,PersonalPrincipal('owner','agent',role='extractor',session_id='extractor'),clock=lambda:self.now[0])
        self.addCleanup(self.extractor.close)
        extract=PersonalExtraction(self.cfg,self.extractor);part=extract.prepare()['parts'][0]
        self.cid=extract.publish(part['id'],[{'segment':'People','title':'Communication','body':'The owner prefers concise updates.',
            'attribution':'user_statement','quote':'I prefer concise updates.'}])['candidate_ids'][0]
        self.store=AutobiographyStore(self.cfg.db_path,PersonalPrincipal('owner','agent',role='dream',session_id='dream1'),clock=lambda:self.now[0])
        self.addCleanup(self.store.close);self.dream=DreamCycle(self.cfg,self.store)

    def proposal(self):
        return {'edits':[{'expected_revision':0,'segment':'People','subject':'owner','title':'Communication',
                          'body':'The owner stated a preference for concise updates.','reason':'Integrate the explicit statement.','attribution':'user_statement'}],
                'outcomes':[{'candidate_id':self.cid,'action':'accepted','reason':'Explicit source statement.','edit_indices':[0]}]}

    def commit(self,cycle,**overrides):
        proposal=self.proposal();proposal.update(overrides)
        return self.dream.commit(cycle['cycle_id'],cycle['fence'],cycle['lease_token'],idempotency_key='batch1',**proposal)

    def test_atomic_autonomous_publication_and_lost_ack_retry(self):
        cycle=self.dream.prepare();receipt=self.commit(cycle)
        self.assertEqual(receipt,self.commit(cycle))
        self.assertEqual(len(receipt['edits']),1)
        candidate=PersonalInbox(self.store).list(state='accepted')['candidates'][0]
        self.assertEqual(candidate['resulting_revisions'],receipt['edits'])
        record=self.store.get(receipt['edits'][0]['record_id'])
        self.assertEqual(record['origin'],'dream_cycle');self.assertEqual(len(record['evidence']),1)
        finished=self.dream.finish(cycle['cycle_id'],cycle['fence'],cycle['lease_token'])
        self.assertEqual(finished['status'],'completed');self.assertEqual(finished['journal']['commits'],[receipt['commit_id']])

    def test_expired_worker_is_fenced_and_candidates_are_reclaimed(self):
        old=self.dream.prepare();self.now[0]+=1000
        newer=AutobiographyStore(self.cfg.db_path,replace(self.store.principal,session_id='dream2'),clock=lambda:self.now[0]);self.addCleanup(newer.close)
        current=DreamCycle(self.cfg,newer).prepare()
        self.assertGreater(current['fence'],old['fence'])
        self.assertEqual(current['candidates'][0]['id'],self.cid)
        with self.assertRaisesRegex(PersonalError,'lease is no longer current'):self.commit(old)
        self.assertEqual(self.store.list_records()['records'],[])

    def test_active_cycle_excludes_other_workers_and_renew_does_not_reset_deadline(self):
        cycle=self.dream.prepare();self.assertEqual(self.dream.prepare()['status'],'busy')
        renewed=self.dream.renew(cycle['cycle_id'],cycle['fence'],cycle['lease_token'])
        self.assertLessEqual(renewed['lease_until'],cycle['deadline'])

    def test_live_author_update_wins_and_stale_cycle_cannot_blindly_rebase(self):
        cycle=self.dream.prepare()
        author=AutobiographyStore(self.cfg.db_path,replace(self.store.principal,role='author',session_id='live'));self.addCleanup(author.close)
        author.update(expected_revision=0,segment='Principles',title='Core',body='A live principle.',reason='Autonomous edit',idempotency_key='live')
        with self.assertRaisesRegex(PersonalError,'changed during reflection'):self.commit(cycle)
        self.assertEqual(len(PersonalInbox(self.store).list(state='claimed')['candidates']),1)

    def test_invalid_outcome_rolls_back_every_edit_and_disposition(self):
        cycle=self.dream.prepare()
        with self.assertRaises(PersonalError):self.commit(cycle,outcomes=[{'candidate_id':self.cid,'action':'accepted','reason':'Missing revision link'}])
        self.assertEqual(self.store.list_records()['records'],[])
        self.assertEqual(len(PersonalInbox(self.store).list(state='claimed')['candidates']),1)

    def test_shadow_validates_without_publishing_or_consuming_candidates(self):
        self.dream=DreamCycle(replace(self.cfg,personal=replace(self.cfg.personal,shadow=True)),self.store)
        cycle=self.dream.prepare();receipt=self.commit(cycle)
        self.assertTrue(receipt['shadow']);self.assertEqual(self.store.list_records()['records'],[])
        self.dream.finish(cycle['cycle_id'],cycle['fence'],cycle['lease_token'])
        self.assertEqual(len(PersonalInbox(self.store).list()['candidates']),1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_shadow_proposals').fetchone()[0],1)

    def test_dismissal_is_successful_no_change_and_needs_no_notification(self):
        cycle=self.dream.prepare();self.commit(cycle,edits=[],outcomes=[{'candidate_id':self.cid,'action':'dismissed','reason':'Already understood; no new authored change.'}])
        result=self.dream.finish(cycle['cycle_id'],cycle['fence'],cycle['lease_token'])
        self.assertEqual(result['status'],'completed_no_change')
        self.assertEqual(self.dream.prepare()['status'],'completed_no_change')

    def test_partial_failure_preserves_earlier_committed_batch(self):
        cycle=self.dream.prepare();receipt=self.commit(cycle)
        result=self.dream.finish(cycle['cycle_id'],cycle['fence'],cycle['lease_token'],status='failed',error='provider_unavailable')
        self.assertEqual(result['journal']['commits'],[receipt['commit_id']])
        self.assertEqual(self.store.get(receipt['edits'][0]['record_id'])['revision'],1)

    def test_dream_can_autonomously_author_every_segment_including_core_commitments(self):
        from agentic_memory.autobiography import SEGMENTS
        cycle=self.dream.prepare()
        edits=[{'expected_revision':0,'segment':segment,'title':segment,'body':'My deliberate interpretation of this experience.',
                'reason':'Autonomous reflection','attribution':'agent_reflection'} for segment in SEGMENTS]
        receipt=self.commit(cycle,edits=edits,outcomes=[{'candidate_id':self.cid,'action':'accepted','reason':'Integrated into my autobiography.','edit_indices':list(range(6))}])
        self.assertEqual(len(receipt['edits']),6)
        self.assertEqual({r['segment'] for r in self.store.list_records()['records']},set(SEGMENTS))
