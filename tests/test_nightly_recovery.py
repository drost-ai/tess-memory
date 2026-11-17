from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from tess_memory.codex import CodexCatalog, CodexParseError
from tess_memory.config import Config, DataStore
from tess_memory.ingest import ingest
from tess_memory.readers import CompletedSourceReader
from tess_memory.scheduler_config import ScheduledStore, ServiceConfig, load_service
from tess_memory.scheduler import worker
from tess_memory.scheduler_queue import WorkQueue
from tess_memory.store import MemoryStore
from test_codex import RecordingEmbedder, append, message, wrapped, write_rollout
from test_incremental_ingest import Backend


PT = ZoneInfo('America/Los_Angeles')


class CalendarRecoveryTests(unittest.TestCase):
    def test_recovery_worker_uses_one_budget_for_sources_and_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();home=root/'codex'
            path=write_rollout(home,items=[message('user','An unfinished request')]);os.utime(path,(1,1))
            providers=root/'providers.json'
            providers.write_text(json.dumps({'providerProfiles':{'p':{'kind':'openai','apiMode':'codex'}}}))
            settings=root/'settings.toml'
            settings.write_text(f'[[data_stores]]\nname="codex"\ntype="codex"\npath="{home}"\n'
                f'[storage]\ndata_dir="{root / "memory"}"\n[llm]\nsettings="{providers}"\nprofile="p"\n')
            config_path=root/'scheduler.toml'
            config_path.write_text(f'[service]\ndata_dir="{root / "service"}"\n[[stores]]\nname="codex"\nsettings="{settings}"\n'
                'recovery_hour=22\nrecovery_timezone="America/Los_Angeles"\nrecovery_model_calls_per_run=2\n')
            cfg=load_service(config_path);queue=WorkQueue(cfg)
            run=queue.claim(instance='codex',kind='recovery',explicit=True)
            backend=Backend()
            with mock.patch('tess_memory.scheduler.EngineBackend',return_value=backend), \
                 mock.patch('tess_memory.scheduler.make_embedder',return_value=RecordingEmbedder()):
                self.assertEqual(worker(config_path,run['id']),0)
            result=queue.get_run(run['id']);stats=json.loads(result['stats'])
            self.assertEqual((stats['model_calls'],stats['recovered_partial']),(2,1))
            self.assertGreater(stats['budget_deferred'],0)  # dirty aggregate retained, not over budget
            due=next(r['due'] for r in queue.status()['jobs'] if r['kind']=='recovery')
            self.assertEqual(datetime.fromtimestamp(due,PT).hour,22)
            self.assertGreater(due,result['finished'])

    def test_calendar_tracks_dst_and_coalesces_sleep_without_catchup_storm(self):
        store = ScheduledStore('one', Path('/one.toml'), recovery_hour=22, recovery_timezone='America/Los_Angeles')
        for day, expected_hours in [('2026-03-07', 23), ('2026-10-31', 25), ('2026-09-09', 24)]:
            now = datetime.fromisoformat(day+'T22:00:00').replace(tzinfo=PT).timestamp()
            due = store.next_recovery(now)
            self.assertEqual((due-now)/3600, expected_hours)
            self.assertEqual(datetime.fromtimestamp(due, PT).hour, 22)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ServiceConfig(root/'config', root/'data', (replace(store, enabled=True),))
            now = datetime(2026, 9, 9, 12, tzinfo=PT).timestamp()
            with mock.patch('tess_memory.scheduler_queue.time.time', return_value=now):
                queue = WorkQueue(cfg)
                queue.set_enabled('one', True)
            due = next(r['due'] for r in queue.status()['jobs'] if r['kind']=='recovery')
            self.assertEqual(due, store.next_recovery(now))
            self.assertIsNone(queue.claim(instance='one', kind='recovery', now=due-1))
            wake = due + 3*86400
            reopened = WorkQueue(cfg)
            run = reopened.claim(instance='one', kind='recovery', now=wake)
            self.assertIsNotNone(run)
            self.assertIsNone(queue.claim(instance='one', kind='ingest', now=wake))
            queue.finish(run['id'], outcome='partial', remaining=100, now=wake+60)
            next_due = next(r['due'] for r in queue.status()['jobs'] if r['kind']=='recovery')
            self.assertEqual(next_due, store.next_recovery(wake+60))
            self.assertIsNone(queue.claim(instance='one', kind='recovery', now=wake+600))


class IncompleteRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.home = self.root/'codex'
        self.path = write_rollout(self.home)
        self.cfg = Config(data_stores=[DataStore('codex', self.home, type='codex', project='p')],
                          data_dir=self.root/'memory', incremental_ingestion=True,
                          segment_chars=3000, segment_overlap=0, llm_workers=2, segment_workers=2)
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.backend, self.embedder = Backend(), RecordingEmbedder()

    def run_ingest(self, recovery=False, **kwargs):
        cfg = replace(self.cfg, recovery_ingestion=recovery, recovery_min_age_seconds=1)
        if recovery:
            os.utime(self.path, (1, 1))
        return ingest(cfg, self.store, self.embedder, self.backend, refresh_aggregates=False, **kwargs)

    def test_nightly_snapshot_is_once_only_and_later_completion_reuses_its_prefix(self):
        self.run_ingest()
        append(self.path, wrapped('event_msg', {'type':'task_started'}))
        append(self.path, wrapped('response_item', message('user', 'Pending unique request')))
        before = self.path.read_bytes()
        self.assertEqual(self.run_ingest().model_calls, 0)
        first = self.run_ingest(True)
        self.assertEqual((first.distilled, first.recovered_partial), (1,1))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIn('Partial source recovery', self.store.get_session('codex:one')['outcome'])
        self.assertTrue(any('incomplete persisted-message' in p for p in self.backend.calls))
        count = len(self.backend.calls)
        self.assertEqual(self.run_ingest(True).model_calls, 0)
        self.assertEqual(self.run_ingest().model_calls, 0)
        append(self.path, wrapped('response_item', message('assistant','Newly completed result')))
        append(self.path, wrapped('event_msg', {'type':'task_complete'}))
        later = self.run_ingest()
        self.assertEqual((later.distilled,later.model_calls), (1,2))
        maps = [p for p in self.backend.calls[count:] if p.startswith(('A completed persisted-message', 'An incomplete persisted-message'))]
        self.assertEqual(len(maps), 1)
        self.assertNotIn('Pending unique request',maps[0])
        self.assertIn('Newly completed result',maps[0])
        self.assertFalse(json.loads(self.store.source_progress(str(self.path))['state'])['pending_tail'])

    def test_short_user_only_history_is_retained_and_hot_or_held_content_is_deferred(self):
        self.path = write_rollout(self.home, items=[message('user','Keep this unresolved request')])
        self.assertEqual(self.run_ingest().model_calls,0)
        self.assertEqual(self.run_ingest(True).recovered_partial,1)
        self.assertEqual(self.store.get_session('codex:one')['n_user'],1)
        append(self.path,wrapped('response_item',message('user','New hot data')))
        cfg=replace(self.cfg,recovery_ingestion=True,recovery_min_age_seconds=86400)
        self.assertEqual(ingest(cfg,self.store,self.embedder,self.backend,refresh_aggregates=False).model_calls,0)
        self.cfg=replace(self.cfg,ingestion_hold_texts=['Keep this unresolved request'])
        os.utime(self.path,(1,1))
        result=self.run_ingest(True)
        self.assertEqual(result.model_calls,0)
        self.assertEqual(self.store.source_progress(str(self.path))['status'],'held')

    def test_generic_recovery_keeps_committed_and_uncommitted_messages_with_partial_record(self):
        home=self.root/'tess';home.mkdir()
        path=home/'one.json'
        path.write_text(json.dumps({'id':'tess-one','messages':[]}))
        sidecar=home/'one.messages.jsonl'
        for i,role in enumerate(['user','assistant','user']):
            append(sidecar,{'type':'message','message_index':i,'message':{'role':role,'content':f'evidence {i}'}})
            if i==1:append(sidecar,{'type':'commit','message_count':2})
        with sidecar.open('ab') as f:f.write(b'{"incomplete":')
        before=sidecar.read_bytes()
        ds=DataStore('tess',home,type='tess-core')
        cfg=replace(self.cfg,data_stores=[ds],recovery_ingestion=True)
        snapshot=CompletedSourceReader(cfg).read(ds,path,'p')
        self.assertEqual([t.text for t in snapshot.transcript.turns],['evidence 0','evidence 1','evidence 2'])
        self.assertTrue(snapshot.pending_tail)
        self.assertTrue(any('commit' in n for n in snapshot.recovery_notes))
        self.assertTrue(any('final record' in n for n in snapshot.recovery_notes))
        self.assertEqual(sidecar.read_bytes(),before)

    def test_rolled_back_source_keeps_all_paid_rows_and_stops_futile_retries(self):
        self.run_ingest()
        before={t:[tuple(r) for r in self.store.db.execute('SELECT * FROM '+t)] for t in ('sessions','memories','embeddings','files')}
        write_rollout(self.home,items=[message('user','<environment_context><cwd>/p</cwd></environment_context>')])
        self.assertEqual(self.run_ingest().errors,1)
        recovered=self.run_ingest(True)
        self.assertEqual((recovered.model_calls,recovered.preserved),(0,1))
        self.assertEqual(self.store.source_progress(str(self.path))['status'],'retained_snapshot')
        after={t:[tuple(r) for r in self.store.db.execute('SELECT * FROM '+t)] for t in before}
        self.assertEqual(before,after)
        cfg=replace(self.cfg,source_retry_backoff=True)
        self.assertEqual(ingest(cfg,self.store,self.embedder,self.backend,refresh_aggregates=False).model_calls,0)

    def test_budgeted_recovery_waits_for_next_night_and_reuses_completed_requests(self):
        self.path=write_rollout(self.home,items=[message('user','unfinished request')])
        self.cfg=replace(self.cfg,max_model_calls_per_run=1,source_retry_backoff=True)
        result=self.run_ingest(True)
        self.assertEqual((result.model_calls,result.checkpointed_requests,result.budget_deferred),(1,1,1))
        self.assertEqual(self.run_ingest().model_calls,0)
        calls=len(self.backend.calls)
        resumed=self.run_ingest(True)
        self.assertEqual((resumed.model_calls,resumed.distilled),(1,1))
        self.assertFalse(any(p.startswith('An incomplete') for p in self.backend.calls[calls:]))

    def test_unknown_payload_stays_visible_and_original_remains_untouched(self):
        self.run_ingest()
        append(self.path,wrapped('response_item',{'type':'future_unknown','payload':'keep original evidence'}))
        before=self.path.read_bytes()
        result=self.run_ingest(True)
        self.assertEqual((result.errors,result.model_calls),(1,0))
        progress=self.store.source_progress(str(self.path))
        self.assertEqual(progress['status'],'error')
        self.assertTrue(json.loads(progress['state'])['recovery_only'])
        self.assertIn('future_unknown',progress['last_error'])
        self.assertEqual(self.path.read_bytes(),before)
        self.assertIsNotNone(self.store.get_session('codex:one'))

    def test_superseded_pending_path_does_not_keep_the_paid_queue_in_a_fast_loop(self):
        self.run_ingest()
        stale=self.home/'sessions'/'older-source.jsonl'
        stale.write_text('original bytes retained')
        self.store.record_source(path=str(stale),session_id='codex:one',project='p',source_type='codex',
                                 signature=None,status='pending_work',state={},now=1)
        self.store.commit()
        result=self.run_ingest()
        self.assertEqual((result.model_calls,result.sources_pending,result.sources_remaining),(0,0,0))
        self.assertEqual(stale.read_text(),'original bytes retained')
        self.assertIsNotNone(self.store.source_progress(str(stale)))


class CodexSegmentRecoveryTests(unittest.TestCase):
    def test_real_cycles_and_truncated_parents_remain_explicit_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);ds=DataStore('codex',home,type='codex')
            parent=write_rollout(home,'parent')
            child=write_rollout(home,'child',meta={'history_base':{'thread_id':'parent','end_byte_offset':parent.stat().st_size+100}})
            with self.assertRaisesRegex(CodexParseError,'available parent bytes'):
                CodexCatalog([ds]).inspect(child)
            write_rollout(home,'parent',meta={'history_base':{'thread_id':'child','end_byte_offset':1}})
            write_rollout(home,'child',meta={'history_base':{'thread_id':'parent','end_byte_offset':1}})
            # Use valid metadata byte boundaries; a cycle must fail by path,
            # never be mistaken for the recoverable same-session split case.
            for p in [parent,child]:
                rows=[json.loads(s) for s in p.read_text().splitlines()]
                rows[0]['payload']['history_base'].pop('end_byte_offset')
                rows[0]['payload']['history_base']['end_ordinal_exclusive']=0
                p.write_text(''.join(json.dumps(dict(r,ordinal=i))+'\n' for i,r in enumerate(rows)))
            with self.assertRaisesRegex(CodexParseError,'cyclic'):CodexCatalog([ds]).inspect(child)

    def test_same_session_segments_resolve_by_physical_id_without_dropping_earlier_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)
            sid='00000000-0000-0000-0000-000000000001'
            alias='00000000-0000-0000-0000-000000000002'
            original=write_rollout(home,sid,items=[message('user','original request'),message('assistant','original result')])
            original_size=original.stat().st_size
            second=write_rollout(home,'segment-two',meta={'id':sid,'history_base':{'thread_id':sid,'end_byte_offset':original_size}},
                items=[message('user','second request'),message('assistant','second result')])
            second=second.rename(second.with_name('rollout-'+sid+'_'+alias+'.jsonl'))
            third=write_rollout(home,'segment-three',meta={'id':sid,'history_base':{'thread_id':alias,'end_byte_offset':second.stat().st_size}},
                items=[message('user','third request'),message('assistant','third result')])
            third=third.rename(third.with_name('rollout-'+sid+'_00000000-0000-0000-0000-000000000003.jsonl'))
            for i,p in enumerate([original,second,third]):os.utime(p,(i+1,i+1))
            ds=DataStore('codex',home,type='codex')
            catalog=CodexCatalog([ds])
            st,_=catalog.inspect(third,min_chars=0)
            self.assertEqual([t.text for t in st.turns],['original request','original result','second request','second result','third request','third result'])
            self.assertIsNone(st.parent_session)
            cfg=Config(data_stores=[ds],data_dir=home/'memory')
            snapshot=CompletedSourceReader(cfg).read(ds,third,'p')
            self.assertEqual(len(snapshot.unit_keys()),3)

    def test_empty_legacy_compaction_placeholders_are_supported_but_unknown_payload_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);ds=DataStore('codex',home,type='codex')
            path=write_rollout(home)
            append(path,wrapped('compacted',{'replacement_history':[{'type':'other'}]}))
            st,report=CodexCatalog([ds]).inspect(path)
            self.assertEqual(st.n_user,1)
            self.assertEqual(report.ignored_records,1)
            append(path,wrapped('compacted',{'replacement_history':[{'type':'other','payload':'unknown evidence'}]}))
            with self.assertRaises(CodexParseError):CodexCatalog([ds]).inspect(path)
