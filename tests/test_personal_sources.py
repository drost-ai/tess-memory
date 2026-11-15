"""Ticket52: canonical Tess sources, isolated backfill, bounded batches/cursors."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tess_memory.autobiography import AutobiographyStore, PersonalError
from tess_memory.autobiography_config import PersonalConfig
from tess_memory.autobiography_extract import PersonalExtraction, clean_unit
from tess_memory.config import Config, DataStore
from tess_memory.extraction_workflow import ExtractionWorkflow
from tess_memory.personal_operations import PersonalOperations
from tess_memory.personal_sources import SourceCursors, selected_sources


class PersonalSourceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve(); self.sessions=self.root/'tess'; self.sessions.mkdir()
        p=PersonalConfig(enabled=True,extraction_enabled=True,owner='sam',agent='assistant',
                         source_mode='native',source_paths=(str(self.sessions),))
        self.cfg=Config(data_dir=self.root/'memory',personal=p,quiet_seconds=0)
        self.store=AutobiographyStore(self.cfg.db_path,p.principal(role='extractor',session_id='test'))
        self.addCleanup(lambda:self.store.close())
        self.extract=PersonalExtraction(self.cfg,self.store)

    def session(self, name='main', messages=None, metadata=None):
        path=self.sessions/(name+'.json')
        path.write_text(json.dumps({'id':name,'cwd':'/any/channel/workspace',
                                    'metadata':metadata or {},'messages':[]}))
        folder=self.sessions/name; folder.mkdir(exist_ok=True)
        body=folder/'full-transcript-with-tools.jsonl'
        body.write_text('')
        self.append(body,messages or [('user','first'),('assistant','answer')])
        return path,body

    def append(self,body,messages,stamp='2026-04-01T12:00:00Z'):
        count=len(body.read_text().splitlines())
        with body.open('a') as file:
            for index, item in enumerate(messages,count):
                if isinstance(item,dict):
                    record=item
                else:
                    role,text=item
                    record={'type':'message','message_index':index,'role':role,'content':text,
                            'persisted_at':stamp,'message_id':f'm{index}'}
                    if role=='assistant':record['stop_reason']='end_turn'
                file.write(json.dumps(record)+'\n')

    def prepared(self):
        report=self.extract.prepare()
        self.assertEqual(report['errors'],[])
        return report['parts']

    def flow(self):
        ops=PersonalOperations(self.cfg,self.store); self.addCleanup(ops.close)
        flow=ExtractionWorkflow(self.cfg,self.store,ops)
        op=ops.claim('extractor',manual=True)
        return ops,flow,(op['operation_id'],op['fence'])

    def finalize(self,flow,auth,row):
        return flow.finalize(*auth,row['range_id'],row['lease_token'],'finish-'+row['range_id'])

    def test_existing_sources_backfill_without_coding_historical_flag(self):
        self.session()
        self.assertFalse(self.cfg.personal.historical)
        self.assertEqual(len(self.prepared()),1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_baseline_paths').fetchone()[0],0)

    def test_append_and_restart_only_clean_new_units(self):
        _,body=self.session()
        first=self.prepared()[0]; self.extract.publish(first['id'],[])
        self.store.close()
        self.store=AutobiographyStore(self.cfg.db_path,self.cfg.personal.principal(role='extractor'))
        self.extract=PersonalExtraction(self.cfg,self.store)
        self.append(body,[('user','second'),('assistant','new answer')])
        with patch('tess_memory.autobiography_extract.clean_unit',wraps=clean_unit) as clean:
            parts=self.prepared()
            self.assertEqual(clean.call_count,1)
        self.assertEqual(len(parts),1)
        self.assertNotIn('first',parts[0]['text'])
        self.assertNotEqual(parts[0]['id'],first['id'])
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts WHERE id=?',(first['id'],)).fetchone()[0],'published')

    def test_incomplete_snapshot_append_preserves_verified_prefix(self):
        _,body=self.session(messages=[('user','unfinished first request')])
        first=self.prepared()[0]; self.extract.publish(first['id'],[])
        self.append(body,[('assistant','now complete')])
        parts=self.prepared()
        self.assertEqual(len(parts),1)
        self.assertNotIn('unfinished first request',parts[0]['text'])
        self.assertIn('now complete',parts[0]['text'])

    def test_cutoff_is_explicit_inclusive_and_missing_timestamp_is_unknown(self):
        self.cfg=replace(self.cfg,personal=replace(self.cfg.personal,source_since='2026-04-01T12:00:00Z'))
        self.extract=PersonalExtraction(self.cfg,self.store)
        _,body=self.session(messages=[{'type':'message','message_index':0,'role':'user','content':'unknown date'},
                                      {'type':'message','message_index':1,'role':'assistant','content':'old','persisted_at':'2026-03-31T00:00:00Z'}])
        self.append(body,[('user','at boundary'),('assistant','eligible')])
        text=''.join(x['text'] for x in self.prepared())
        self.assertNotIn('unknown date',text); self.assertNotIn('old',text)
        self.assertIn('at boundary',text)
        for value in ('2026-04-01','garbage',123):
            with self.assertRaises(ValueError):PersonalConfig.parse({'source_since':value})

    def test_foreign_stores_do_not_affect_personal_scope_or_get_mutated(self):
        foreign=self.root/'foreign'; foreign.mkdir()
        stores=[DataStore('codex',foreign,type='codex'),DataStore('claude',foreign,type='claude-code'),
                DataStore('tess-code',foreign,type='tess-core')]
        cfg=replace(self.cfg,data_stores=stores)
        self.assertEqual([d.type for d in selected_sources(cfg)],['tess-core'])
        self.assertEqual(selected_sources(cfg)[0].path,self.sessions)
        self.session(); self.extract=PersonalExtraction(cfg,self.store)
        self.assertEqual(len(self.prepared()),1)
        self.assertEqual(cfg.data_stores,stores)
        self.assertEqual(list(foreign.iterdir()),[])

    def test_old_foreign_pending_work_not_claimed_in_strict_mode(self):
        from test_autobiography_extract import ExtractionTests
        foreign=ExtractionTests(); foreign.setUp()
        try:
            foreign.extract.prepare()
            cfg=replace(foreign.cfg,personal=replace(foreign.cfg.personal,source_mode='native',source_paths=(str(self.sessions),)))
            self.assertEqual(PersonalExtraction(cfg,foreign.store).prepare()['parts'],[])
        finally:foreign.doCleanups()

    def test_child_and_internal_origins_never_fabricate_channel(self):
        self.session('child',metadata={'parentSessionId':'parent'})
        self.session('extractor',metadata={'personalMemoryOrigin':'extractor'})
        parts=self.prepared(); self.assertEqual(len(parts),1)
        self.assertNotIn('first',parts[0]['text'])
        info=json.loads(self.store.db.execute('SELECT metadata FROM autobio_sources').fetchone()[0])
        self.assertEqual(info['source_origin']['kind'],'child')
        self.assertIsNone(info['source_origin']['channel'])
        self.assertFalse(any(r.get('human_evidence') for r in info['roles']))

    def test_replacement_fences_old_generation_and_cursor(self):
        _,body=self.session()
        ops,flow,auth=self.flow(); row=flow.claim(*auth)['ranges'][0]
        body.write_text(body.read_text().replace('first','changed'))
        with self.assertRaises(PersonalError):self.finalize(flow,auth,row)
        changed=self.prepared()[0]
        self.assertNotEqual(changed['generation'],row['generation'])
        self.assertEqual(self.store.db.execute('SELECT completed_units FROM autobio_source_cursors').fetchone()[0],0)
        with self.assertRaises(PersonalError):self.finalize(flow,auth,row)

    def test_out_of_order_units_do_not_advance_past_gap(self):
        self.session(messages=[('user','first'),('assistant','answer'),('user','second'),('assistant','answer2')])
        ops,flow,auth=self.flow(); rows=flow.claim(*auth)['ranges']
        rows.sort(key=lambda x:x['metadata']['unit'])
        result=self.finalize(flow,auth,rows[1])
        self.assertEqual(result['source_cursor']['completed_units'],0)
        result=self.finalize(flow,auth,rows[0])
        self.assertEqual(result['source_cursor']['completed_units'],2)
        self.assertTrue(result['source_cursor']['complete'])

    def test_parts_gap_and_draft_do_not_complete_unit(self):
        self.session(messages=[('user','x'*25000),('assistant','answer')])
        ops,flow,auth=self.flow(); rows=flow.claim(*auth)['ranges']; rows.sort(key=lambda x:x['start'])
        self.assertEqual(len(rows),3)
        self.assertEqual(self.finalize(flow,auth,rows[2])['source_cursor']['completed_units'],0)
        self.assertEqual(self.finalize(flow,auth,rows[0])['source_cursor']['completed_units'],0)
        self.assertEqual(self.finalize(flow,auth,rows[1])['source_cursor']['completed_units'],1)

    def test_manifest_is_bounded_immutable_and_pass_does_not_chase_append(self):
        _,body=self.session()
        ops,flow,auth=self.flow(); claim=flow.claim(*auth,limit=1)
        manifest=claim['source_manifest']; path=Path(manifest['snapshot_path']); raw=path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest['sha256'])
        self.assertLessEqual(len(raw),manifest['max_bytes'])
        row=claim['ranges'][0]; self.finalize(flow,auth,row)
        self.append(body,[('user','after snapshot'),('assistant','next pass')])
        self.assertEqual(flow.claim(*auth)['ranges'],[])
        self.assertEqual(path.read_bytes(),raw)
        self.assertNotIn(b'after snapshot',raw)
        # New pass preparation sees append, without touching finalized prefix.
        self.assertEqual(len(self.prepared()),1)

    def test_operation_can_drain_more_than_32_parts_with_bounded_claims(self):
        self.session(messages=[pair for i in range(40) for pair in [('user',f'u{i}'),('assistant',f'a{i}')]])
        ops,flow,auth=self.flow(); seen=set()
        while True:
            claim=flow.claim(*auth,limit=7)
            self.assertEqual(claim['batch']['range_count'],40)
            self.assertLessEqual(len(claim['ranges']),7)
            if not claim['ranges']:break
            for row in claim['ranges']:
                self.assertNotIn(row['range_id'],seen); seen.add(row['range_id'])
                self.finalize(flow,auth,row)
        self.assertEqual(len(seen),40)

    def test_symlink_outside_personal_root_is_rejected(self):
        foreign=self.root/'foreign.json'; foreign.write_text(json.dumps({'id':'outside','messages':[]}))
        (self.sessions/'escape.json').symlink_to(foreign)
        report=self.extract.prepare()
        self.assertEqual(report['parts'],[])
        self.assertEqual(len(report['errors']),1)

    def test_strict_config_requires_explicit_root_legacy_default_unchanged(self):
        self.assertEqual(PersonalConfig.parse({}).source_mode,'allowlist')
        with self.assertRaises(ValueError):PersonalConfig.parse({'source_mode':'native'})
        with self.assertRaises(ValueError):PersonalConfig.parse({'source_mode':'guess-from-name'})

    def test_source_index_gap_is_not_a_completed_source(self):
        self.session(messages=[{'type':'message','message_index':0,'role':'user','content':'first'},
            {'type':'message','message_index':2,'role':'assistant','content':'missing intervening record'}])
        report=self.extract.prepare()
        self.assertEqual(report['parts'],[])
        self.assertEqual(len(report['errors']),1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_source_cursors').fetchone()[0],0)

    def test_reconciliation_manifest_is_bounded_supporting_evidence_only(self):
        from tess_memory.dream_cycle import DreamCycle
        self.session(messages=[('user','I prefer concise updates.'),('assistant','Understood.')])
        part=self.prepared()[0]
        proposal={'segment':'People','title':'Preference','body':'Prefers concise updates.',
                  'attribution':'user_statement','quote':'I prefer concise updates.'}
        self.extract.publish(part['id'],[proposal])
        cfg=replace(self.cfg,personal=replace(self.cfg.personal,dream_enabled=True))
        principal=cfg.personal.principal(role='dream',session_id='reconcile-test')
        dream_store=AutobiographyStore(cfg.db_path,principal); self.addCleanup(dream_store.close)
        claim=DreamCycle(cfg,dream_store).prepare()
        manifest=claim['source_manifest']
        self.assertEqual(manifest['purpose'],'reconciliation_evidence')
        self.assertEqual(manifest['range_count'],1)
        payload=json.loads(Path(manifest['snapshot_path']).read_text())
        self.assertEqual(payload['ranges'][0]['text'],proposal['quote'])
        self.assertEqual(payload['ranges'][0]['cursor']['completed_units'],1)
        self.assertNotIn('source_revisions',payload['ranges'][0])
        self.assertNotIn('boundaries',payload['ranges'][0])

    def test_replace_then_restore_has_new_generation_not_old_lease(self):
        _,body=self.session()
        original=body.read_text()
        ops,flow,auth=self.flow(); old=flow.claim(*auth)['ranges'][0]
        body.write_text(original.replace('first','replacement'))
        second=self.prepared()[0]
        body.write_text(original)
        restored=self.prepared()[0]
        self.assertNotEqual(restored['generation'],old['generation'])
        self.assertNotEqual(restored['generation'],second['generation'])
        with self.assertRaises(PersonalError):self.finalize(flow,auth,old)
        self.extract.publish(restored['id'],[])
        self.assertEqual(self.store.db.execute('SELECT completed_units FROM autobio_source_cursors').fetchone()[0],1)

    def test_truncate_then_restore_does_not_resurrect_old_range(self):
        _,body=self.session(messages=[('user','first'),('assistant','answer'),('user','second'),('assistant','answer2')])
        original=body.read_text(); parts=self.prepared()
        second=next(p for p in parts if 'second' in p['text'])
        body.write_text('\n'.join(original.splitlines()[:2])+'\n')
        self.prepared()
        body.write_text(original)
        restored=next(p for p in self.prepared() if 'second' in p['text'])
        self.assertNotEqual(second['generation'],restored['generation'])

    def test_truncate_to_empty_invalidates_and_restore_is_new_generation(self):
        _,body=self.session(); original=body.read_text()
        old=self.prepared()[0]
        body.write_text('')
        self.assertEqual(self.prepared(),[])
        self.assertEqual(self.store.db.execute('SELECT state FROM autobio_extraction_parts WHERE id=?',(old['id'],)).fetchone()[0],'stale')
        body.write_text(original)
        restored=self.prepared()[0]
        self.assertNotEqual(old['generation'],restored['generation'])

    def test_gap_repair_in_body_retries_without_metadata_touch(self):
        path,body=self.session(messages=[{'type':'message','message_index':0,'role':'user','content':'first'},
            {'type':'message','message_index':2,'role':'assistant','content':'answer'}])
        stamp=path.stat().st_mtime_ns
        self.assertEqual(len(self.extract.prepare()['errors']),1)
        body.write_text(body.read_text().replace('"message_index": 2','"message_index": 1'))
        self.assertEqual(path.stat().st_mtime_ns,stamp)
        self.assertEqual(len(self.prepared()),1)

    def test_active_session_completed_prefix_not_starved_by_quiet_window(self):
        self.cfg=replace(self.cfg,quiet_seconds=900)
        self.extract=PersonalExtraction(self.cfg,self.store)
        _,body=self.session()
        self.append(body,[('user','persisted in-flight request')])
        with body.open('a') as file:file.write('{"type":"message","message_index":3,"content":"NOT YET COMMITTED')
        ops,flow,auth=self.flow(); claim=flow.claim(*auth)
        self.assertEqual(claim['errors'],[])
        text=''.join(row['text'] for row in claim['ranges'])
        self.assertIn('first',text)
        self.assertNotIn('NOT YET COMMITTED',text)
        self.assertTrue(all(row['pending_tail'] for row in claim['source_manifest']['ranges']))
        complete=next(row for row in claim['ranges'] if row['metadata']['unit_kind']=='turn_complete')
        result=self.finalize(flow,auth,complete)
        self.assertEqual(result['source_cursor']['completed_units'],1)
        self.assertFalse(result['source_cursor']['complete'])

    def test_finalize_claimed_prefix_after_live_append_and_metadata_refresh(self):
        self.cfg=replace(self.cfg,quiet_seconds=900)
        self.extract=PersonalExtraction(self.cfg,self.store)
        path,body=self.session()
        ops,flow,auth=self.flow(); claim=flow.claim(*auth)
        row=claim['ranges'][0]
        self.append(body,[('user','next request'),('assistant','next response')])
        metadata=json.loads(path.read_text());metadata['updatedAt']='2026-04-01T13:00:00Z'
        path.write_text(json.dumps(metadata))
        result=self.finalize(flow,auth,row)
        self.assertEqual(result['source_cursor']['completed_units'],1)
        self.assertEqual(result['source_cursor']['prepared_units'],1)
        self.assertEqual(flow.claim(*auth)['ranges'],[])
        next_parts=self.prepared()
        self.assertEqual(len(next_parts),1)
        self.assertIn('next request',next_parts[0]['text'])
        self.assertNotIn('first',next_parts[0]['text'])

    def test_metadata_refresh_does_not_bless_changed_claimed_prefix(self):
        path,body=self.session()
        ops,flow,auth=self.flow(); row=flow.claim(*auth)['ranges'][0]
        body.write_text(body.read_text().replace('first','altered prefix'))
        metadata=json.loads(path.read_text());metadata['updatedAt']='2026-04-01T13:00:00Z'
        path.write_text(json.dumps(metadata))
        with self.assertRaises(PersonalError):self.finalize(flow,auth,row)
        self.assertEqual(self.store.db.execute('SELECT completed_units FROM autobio_source_cursors').fetchone()[0],0)
