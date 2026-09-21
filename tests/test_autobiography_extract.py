from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from agentic_memory.autobiography import AutobiographyStore,PersonalPrincipal,PersonalError
from agentic_memory.autobiography_config import PersonalConfig
from agentic_memory.autobiography_extract import PersonalExtraction,clean_unit
from agentic_memory.autobiography_inbox import PersonalInbox
from agentic_memory.config import Config,DataStore
from agentic_memory.transcript import Turn
from test_codex import write_rollout,append,wrapped,message


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.home=self.root/'sources'
        self.path=write_rollout(self.home,items=[message('user','I prefer concise updates.'),message('assistant','I will keep that in mind.')])
        personal=PersonalConfig(enabled=True,owner='owner',agent='agent',extraction_enabled=True,source_names=('selected',),historical=True)
        self.cfg=Config(data_dir=self.root/'memory',data_stores=[DataStore('selected',self.home,type='codex')],personal=personal)
        self.store=AutobiographyStore(self.cfg.db_path,PersonalPrincipal('owner','agent',role='extractor',session_id='extractor'))
        self.addCleanup(self.store.close);self.extract=PersonalExtraction(self.cfg,self.store)

    def candidate(self,quote='I prefer concise updates.'):
        return {'segment':'People','title':'Communication preference','body':'The user stated a preference for concise updates.',
                'attribution':'user_statement','quote':quote}

    def test_independent_publication_is_deduplicated_and_never_authors_core(self):
        prepared=self.extract.prepare();self.assertEqual(len(prepared['parts']),1)
        part=prepared['parts'][0]
        first=self.extract.publish(part['id'],[self.candidate()])
        self.assertEqual(first,self.extract.publish(part['id'],[self.candidate()]))
        self.assertEqual(len(PersonalInbox(self.store).list()['candidates']),1)
        self.assertEqual(self.store.list_records()['records'],[])
        self.assertEqual(self.extract.prepare()['parts'],[])
        with self.assertRaises(PersonalError):self.extract.publish(part['id'],[])

    def test_initial_history_is_baselined_and_only_new_units_are_prepared(self):
        cfg=replace(self.cfg,personal=replace(self.cfg.personal,historical=False))
        extract=PersonalExtraction(cfg,self.store)
        self.assertEqual(extract.prepare()['parts'],[])
        append(self.path,wrapped('response_item',message('user','A new preference today.')))
        append(self.path,wrapped('response_item',message('assistant','Acknowledged.')))
        parts=extract.prepare()['parts']
        self.assertEqual(len(parts),1)
        self.assertIn('A new preference today.',parts[0]['text'])
        self.assertNotIn('I prefer concise updates.',parts[0]['text'])

    def test_source_edit_invalidates_pending_candidates_without_silent_overwrite(self):
        part=self.extract.prepare()['parts'][0];receipt=self.extract.publish(part['id'],[self.candidate()])
        self.path.write_text(self.path.read_text().replace('concise updates','detailed updates'))
        next_parts=self.extract.prepare()['parts']
        self.assertEqual(len(next_parts),1)
        stale=PersonalInbox(self.store).list(state='stale')['candidates']
        self.assertEqual(stale[0]['id'],receipt['candidate_ids'][0])
        self.assertEqual(stale[0]['proposal']['quote'],'I prefer concise updates.')

    def test_false_quote_and_wrong_attribution_roll_back_all_candidates(self):
        part=self.extract.prepare()['parts'][0]
        for candidate in [self.candidate('An invented statement.'),self.candidate('I will keep that in mind.')]:
            with self.assertRaises(PersonalError):self.extract.publish(part['id'],[self.candidate(),candidate])
        self.assertEqual(PersonalInbox(self.store).list()['candidates'],[])
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts').fetchone()[0],'pending')

    def test_changed_source_during_model_work_is_not_acknowledged(self):
        part=self.extract.prepare()['parts'][0]
        self.path.write_text(self.path.read_text().replace('concise updates','different updates'))
        with self.assertRaisesRegex(PersonalError,'prefix changed'):self.extract.publish(part['id'],[self.candidate()])
        self.assertEqual(PersonalInbox(self.store).list()['candidates'],[])

    def test_derived_context_tool_results_and_credentials_are_not_extracted(self):
        turns=[Turn('user','A genuine new correction',None),Turn('tool','mcp__personal__autobiography_get {}',None),
               Turn('tool_result','Old personal belief repeated',None),Turn('assistant','A new reflection',None),
               Turn('tool_result','Bearer ABCDEFGHIJKLMNOPQRSTUVWXYZ',None)]
        text,_=clean_unit(turns)
        self.assertIn('genuine new correction',text);self.assertNotIn('Old personal belief',text)
        self.assertNotIn('ABCDEFGHIJKLMNOPQRSTUVWXYZ',text)

    def test_persistent_provider_failures_are_retained_as_dead_letters(self):
        part=self.extract.prepare()['parts'][0];inbox=PersonalInbox(self.store)
        for _ in range(3):inbox.fail(part['id'],'ProviderTimeout')
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts').fetchone()[0],'failed')
        self.assertEqual(self.extract.prepare()['parts'],[])

    def test_disabled_extraction_does_not_initialize_producer_tables(self):
        config=replace(self.cfg,personal=replace(self.cfg.personal,extraction_enabled=False))
        self.assertFalse(PersonalExtraction(config,self.store).prepare()['enabled'])
        self.assertIsNone(self.store.db.execute("SELECT 1 FROM sqlite_master WHERE name='autobio_candidates'").fetchone())

    def test_bounded_scans_rotate_past_the_first_128_sources(self):
        for i in range(130):write_rollout(self.home,'session-'+str(i))
        self.extract.prepare();self.extract.prepare()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_extractions').fetchone()[0],131)

    def test_metadata_only_change_does_not_invalidate_verified_personal_evidence(self):
        home=self.root/'agent-core';home.mkdir()
        path=home/'session.json';path.write_text(json.dumps({'id':'session','updatedAt':'before','messages':[
            {'role':'user','content':'I prefer concise updates.'},{'role':'assistant','content':'Acknowledged.'}]}))
        cfg=replace(self.cfg,data_stores=[DataStore('selected',home,type='agent-core')])
        extraction=PersonalExtraction(cfg,self.store);part=extraction.prepare()['parts'][0]
        data=json.loads(path.read_text());data['updatedAt']='after';path.write_text(json.dumps(data))
        extraction.prepare()
        self.assertTrue(extraction.publish(part['id'],[self.candidate()])['published'])
