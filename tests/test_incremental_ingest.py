from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from agentic_memory.checkpoints import DistillationCheckpoints
from agentic_memory.config import Config, DataStore, load_config
from agentic_memory.ingest import ingest
from agentic_memory.store import MemoryStore
from test_codex import RecordingEmbedder, append, message, wrapped, write_rollout


class Backend:
    name = 'synthetic-primary'
    cache_identity = 'synthetic-model-v1'
    def __init__(self):
        self.calls = []
        self.hook = None
        self.fail = False
    def complete(self, *, system, prompt):
        self.calls.append(prompt)
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        if self.fail:
            raise RuntimeError('synthetic provider failure')
        key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return json.dumps({'title': 'Validated work', 'summary': 'Summary ' + key, 'outcome': 'Observed work',
                           'memories': [{'content': 'A durable unchanged decision', 'kind': 'decision', 'importance': 9}]})


class IncrementalIngestTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.home = self.root / 'codex'
        self.path = write_rollout(self.home)
        self.cfg = Config(data_stores=[DataStore('codex', self.home, type='codex', project='p')],
                          data_dir=self.root / 'memory', incremental_ingestion=True,
                          segment_chars=3000, segment_overlap=0, llm_workers=2, segment_workers=2)
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(lambda: self.store.close())
        self.backend = Backend()
        self.embedder = RecordingEmbedder()

    def run_ingest(self, **kw):
        return ingest(self.cfg, self.store, self.embedder, self.backend, refresh_aggregates=False, **kw)

    def add_unit(self):
        append(self.path, wrapped('response_item', message('user', 'A second distinctive request'), 40))
        append(self.path, wrapped('response_item', message('assistant', 'A second distinctive answer'), 41))

    def paid_rows(self):
        return {table: [tuple(row) for row in self.store.db.execute('SELECT * FROM ' + table)]
                for table in ('sessions', 'memories', 'embeddings', 'files')}

    def test_append_reuses_prior_maps_vectors_and_memory_ids(self):
        self.assertEqual(self.run_ingest().distilled, 1)
        old = {r['content']: (r['id'], r['vector']) for r in self.store.session_rows_with_vectors('codex:one')}
        first_calls = len(self.backend.calls)
        first_embeds = len(self.embedder.calls)
        self.add_unit()
        result = self.run_ingest()
        self.assertEqual((result.distilled, result.model_calls), (1, 2))  # new map + root reduction
        new_prompts = self.backend.calls[first_calls:]
        maps = [p for p in new_prompts if p.startswith('A completed persisted-message')]
        self.assertEqual(len(maps), 1)
        self.assertNotIn('Remember the durable workspace decision', maps[0])
        self.assertGreater(len(self.embedder.calls), first_embeds)
        self.assertEqual(result.embedded_passages, 2)  # new summary and raw exchange
        for row in self.store.session_rows_with_vectors('codex:one'):
            if row['content'] in old:
                self.assertEqual((row['id'], row['vector']), old[row['content']])
        before = (len(self.backend.calls), len(self.embedder.calls))
        result = self.run_ingest()
        self.assertEqual((result.unchanged, result.model_calls, result.embedded_passages), (1, 0, 0))
        self.assertEqual(before, (len(self.backend.calls), len(self.embedder.calls)))

    def test_identical_sessions_share_model_work_but_keep_separate_provenance(self):
        other = write_rollout(self.home, 'two')
        result = self.run_ingest()
        self.assertEqual(result.distilled, 2)
        self.assertEqual(result.model_calls, 2)  # one shared map and one reduction
        self.assertGreaterEqual(result.shared_cache_hits, 2)
        first = self.store.get_session('codex:one')
        second = self.store.get_session('codex:two')
        self.assertEqual(first['source_path'], str(self.path))
        self.assertEqual(second['source_path'], str(other))
        a = {r['id'] for r in self.store.session_rows_with_vectors(first['id'])}
        b = {r['id'] for r in self.store.session_rows_with_vectors(second['id'])}
        self.assertTrue(a and b and a.isdisjoint(b))
        self.assertEqual(self.run_ingest().model_calls, 0)
        self.add_unit()
        result = self.run_ingest()
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(second, self.store.get_session('codex:two'))

    def test_cache_warming_preserves_memory_and_promotes_existing_requests_only(self):
        from agentic_memory.distill import SESSION_SYSTEM
        from agentic_memory.readers import CompletedSourceReader
        from agentic_memory.stable_distill import stable_map_prompts
        from agentic_memory.warm_checkpoints import warm_incremental_checkpoints
        other = write_rollout(self.home, 'two')
        reader = CompletedSourceReader(self.cfg)
        snapshot = reader.read(self.cfg.data_stores[0], self.path, 'p')
        prompt = next(stable_map_prompts(self.cfg, snapshot))
        cache = DistillationCheckpoints(self.cfg.data_dir/'distillation-cache.db')
        old_key = cache.key('incremental/codex/codex:one', self.backend.cache_identity, SESSION_SYSTEM, prompt)
        cache.put(old_key, {'summary': 'Previously paid answer', 'memories': []})
        for path, sid in ((self.path, 'one'), (other, 'two')):
            self.store.record_source(path=str(path), session_id='codex:'+sid, project='p', source_type='codex',
                                     signature=None, status='budget_pending', state={}, now=1)
        self.store.commit()
        before = self.paid_rows()
        progress = [tuple(r) for r in self.store.db.execute('SELECT * FROM ingest_sources')]
        self.backend.hook = lambda: self.fail('cache warming invoked a model')
        result = warm_incremental_checkpoints(self.cfg, self.backend)
        self.assertEqual((result['sessions'], result['cached_requests'], result['model_calls']), (2, 2, 0))
        self.assertEqual(result['shared_hits'], 1)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.embedder.calls, [])
        self.assertEqual(self.paid_rows(), before)
        self.assertEqual([tuple(r) for r in self.store.db.execute('SELECT * FROM ingest_sources')], progress)
        new_key = cache.key('incremental/codex/codex:two', self.backend.cache_identity, SESSION_SYSTEM, prompt)
        self.assertEqual(cache.get(old_key), cache.get(new_key))

    def test_cache_warming_refuses_to_overlap_an_ingestion_owner(self):
        from agentic_memory.locks import ingestion_lock, IngestionBusy
        from agentic_memory.warm_checkpoints import warm_incremental_checkpoints
        with ingestion_lock(self.cfg.data_dir):
            with self.assertRaises(IngestionBusy):
                warm_incremental_checkpoints(self.cfg, self.backend)

    def test_pending_tail_does_not_trigger_model_or_claim_whole_file_processed(self):
        self.run_ingest()
        before = len(self.backend.calls)
        append(self.path, wrapped('event_msg', {'type': 'task_started', 'turn_id': 'active'}))
        append(self.path, wrapped('response_item', message('user', 'Pending task')))
        result = self.run_ingest()
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(len(self.backend.calls), before)
        self.assertEqual(self.store.source_progress(str(self.path))['status'], 'pending_tail')
        self.assertNotEqual(self.store.file_state(str(self.path)), (self.path.stat().st_mtime, self.path.stat().st_size))
        self.assertEqual(self.run_ingest().model_calls, 0)

    def test_failed_map_preserves_paid_rows_and_retry_is_normal(self):
        self.run_ingest()
        before = self.paid_rows()
        self.add_unit()
        self.backend.fail = True
        failed = self.run_ingest()
        self.assertEqual((failed.errors, failed.preserved), (1, 1))
        self.assertEqual(before, self.paid_rows())
        self.backend.fail = False
        self.assertEqual(self.run_ingest().distilled, 1)

    def test_embedding_failure_resumes_successful_maps_and_reduction(self):
        self.run_ingest()
        before = self.paid_rows()
        self.add_unit()
        with mock.patch.object(self.embedder, 'embed_passages', side_effect=RuntimeError('embedding offline')):
            self.assertEqual(self.run_ingest().errors, 1)
        self.assertEqual(before, self.paid_rows())
        calls = len(self.backend.calls)
        self.assertEqual(self.run_ingest().distilled, 1)
        self.assertEqual(len(self.backend.calls), calls)

    def test_partial_publication_write_rolls_back_rows_and_progress(self):
        self.run_ingest()
        before = self.paid_rows()
        old_state = json.loads(self.store.source_progress(str(self.path))['state'])
        self.add_unit()
        replace = self.store.replace_session_rows
        def fail(*args):
            replace(*args)
            raise RuntimeError('injected publication failure')
        with mock.patch.object(self.store, 'replace_session_rows', side_effect=fail):
            self.assertEqual(self.run_ingest().errors, 1)
        self.store.commit()
        self.assertEqual(before, self.paid_rows())
        self.assertEqual(json.loads(self.store.source_progress(str(self.path))['state']), old_state)
        self.assertEqual(self.run_ingest().distilled, 1)

    def test_append_while_model_runs_publishes_only_verified_prefix(self):
        self.backend.hook = self.add_unit
        self.assertEqual(self.run_ingest().distilled, 1)
        progress = self.store.source_progress(str(self.path))
        self.assertEqual(progress['status'], 'pending_work')
        self.assertEqual(len(json.loads(progress['state'])['indexed_units']), 1)
        self.assertEqual(self.run_ingest().distilled, 1)
        self.assertEqual(len(json.loads(self.store.source_progress(str(self.path))['state'])['indexed_units']), 2)

    def test_rewrite_while_model_runs_keeps_old_memory(self):
        self.run_ingest()
        before = self.paid_rows()
        self.add_unit()
        self.backend.hook = lambda: write_rollout(self.home, items=[message('user', 'Rewritten request'), message('assistant', 'Rewritten answer')])
        failed = self.run_ingest()
        self.assertEqual(failed.deferred_active, 1)
        self.assertEqual(before, self.paid_rows())

    def test_existing_paid_session_bootstraps_without_model_or_embedding_calls(self):
        self.cfg.incremental_ingestion = False
        self.run_ingest()
        before = self.paid_rows()
        calls = len(self.backend.calls), len(self.embedder.calls)
        self.cfg.incremental_ingestion = True
        self.assertEqual(self.run_ingest().unchanged, 1)
        self.assertEqual(before, self.paid_rows())
        self.assertEqual((len(self.backend.calls), len(self.embedder.calls)), calls)
        self.add_unit()
        result = self.run_ingest()
        self.assertEqual((result.distilled, result.model_calls), (1, 2))

    def test_adoption_does_not_replace_paid_memory_with_a_shorter_partial_prefix(self):
        self.cfg.incremental_ingestion = False
        self.add_unit()
        self.path.write_bytes(self.path.read_bytes().rstrip(b'\n'))
        self.run_ingest()
        before = self.paid_rows()
        self.cfg.incremental_ingestion = True
        result = self.run_ingest()
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(result.preserved, 1)
        self.assertEqual(before, self.paid_rows())

    def incompatible_tracking(self):
        progress = self.store.source_progress(str(self.path))
        state = json.loads(progress['state'])
        state['indexed_units'] = ['obsolete-boundary'] * 4
        state['seed'] = {'units': state['indexed_units'], 'digest': {'summary': 'stale seed'}}
        self.store.db.execute("UPDATE ingest_sources SET state=?,status='budget_pending' WHERE path=?",
                              (json.dumps(state), str(self.path)))
        self.store.commit()

    def test_incompatible_tracking_readopts_verified_paid_content_without_calls(self):
        self.run_ingest()
        before = self.paid_rows()
        counts = len(self.backend.calls), len(self.embedder.calls)
        self.incompatible_tracking()
        result = self.run_ingest()
        self.assertEqual((result.unchanged, result.model_calls, result.embedded_passages), (1, 0, 0))
        self.assertEqual(before, self.paid_rows())
        self.assertEqual(counts, (len(self.backend.calls), len(self.embedder.calls)))
        state = json.loads(self.store.source_progress(str(self.path))['state'])
        self.assertEqual(state['seed']['units'], state['indexed_units'])
        self.add_unit()
        self.assertEqual(self.run_ingest().model_calls, 2)  # new unit plus merge

    def test_incompatible_tracking_does_not_adopt_changed_content_or_model_policy(self):
        self.run_ingest()
        self.incompatible_tracking()
        write_rollout(self.home, items=[message('user', 'Changed task ' * 20), message('assistant', 'Changed solution ' * 20)])
        self.assertGreater(self.run_ingest().model_calls, 0)
        self.incompatible_tracking()
        self.backend.cache_identity = 'explicitly-changed-model'
        self.assertGreater(self.run_ingest().model_calls, 0)

    def test_budget_deferred_run_resumes_without_repaying_for_completed_map(self):
        self.cfg.max_model_calls_per_run = 1
        first = self.run_ingest()
        self.assertEqual((first.budget_deferred, first.model_calls), (1, 1))
        self.assertEqual(first.checkpointed_requests, 1)
        self.assertIsNone(self.store.get_session('codex:one'))
        second = self.run_ingest()
        self.assertEqual((second.distilled, second.model_calls), (1, 1))
        self.assertEqual(len(self.backend.calls), 2)

    def test_bounded_scan_rotates_past_unfinished_large_source_across_restart(self):
        import time
        from agentic_memory.source_schedule import SCAN_CURSOR
        other = write_rollout(self.home, 'two', items=[
            message('user', 'A different durable question. ' * 20),
            message('assistant', 'A different durable answer. ' * 20)])
        items = []
        for i in range(8):
            items.extend([message('user', f'Long-session request {i}. ' * 20),
                          message('assistant', f'Long-session answer {i}. ' * 20)])
        large = write_rollout(self.home, 'zz-large', items=items)
        for path in (self.path, other):
            os.utime(path, (time.time()-3600, time.time()-3600))
        self.cfg.max_model_calls_per_run = 1
        self.run_ingest(limit=1)
        self.assertEqual(self.store.get_meta(SCAN_CURSOR), str(self.path))
        self.store.close()
        self.store = MemoryStore(self.cfg.db_path)
        self.run_ingest(limit=1)
        self.assertEqual(self.store.get_meta(SCAN_CURSOR), str(other))
        self.run_ingest(limit=1)
        self.assertEqual(self.store.get_meta(SCAN_CURSOR), str(large))
        self.assertTrue(all(self.store.source_progress(str(p)) is not None for p in (self.path, other, large)))
        self.run_ingest(limit=1)
        self.assertIsNotNone(self.store.get_session('codex:one'))

    def test_targeted_session_run_does_not_change_background_cursor(self):
        from agentic_memory.source_schedule import SCAN_CURSOR
        with self.store.write_transaction():
            self.store.set_meta(SCAN_CURSOR, '/retained/background/position')
        self.run_ingest(limit=1, session_ids={'one'})
        self.assertEqual(self.store.get_meta(SCAN_CURSOR), '/retained/background/position')

    def test_reader_failure_advances_background_cursor(self):
        from agentic_memory.source_schedule import SCAN_CURSOR
        from agentic_memory.source_journal import SourceReadError
        other = write_rollout(self.home, 'two')
        with mock.patch('agentic_memory.incremental.CompletedSourceReader.read', side_effect=SourceReadError('unreadable source')):
            self.assertEqual(self.run_ingest(limit=1).errors, 1)
        self.assertEqual(self.store.get_meta(SCAN_CURSOR), str(self.path))
        self.run_ingest(limit=1)
        self.assertIsNotNone(self.store.get_session('codex:two'))
        self.assertEqual(self.store.get_meta(SCAN_CURSOR), str(other))

    def test_project_move_and_source_archive_preserve_paid_data_and_queue_rollups(self):
        self.run_ingest()
        original_ids = [r['id'] for r in self.store.session_rows_with_vectors('codex:one')]
        calls = len(self.backend.calls)
        self.cfg.data_stores[0].project = 'new-project'
        self.assertEqual(self.run_ingest().unchanged, 1)
        self.assertEqual(self.store.get_session('codex:one')['project'], 'new-project')
        scopes = {(r['level'], r['scope']) for r in self.store.pending_aggregates(float('inf'))}
        self.assertTrue({('project', 'p'), ('project', 'new-project')} <= scopes)
        archive = self.home / 'archived_sessions' / self.path.name
        archive.parent.mkdir()
        self.path.rename(archive)
        self.assertEqual(self.run_ingest().unchanged, 1)
        self.assertEqual(self.store.get_session('codex:one')['source_path'], str(archive))
        self.assertEqual(len(self.backend.calls), calls)
        self.assertEqual([r['id'] for r in self.store.session_rows_with_vectors('codex:one')], original_ids)

    def test_retention_preserves_pinned_requests_and_old_writer_schema(self):
        self.run_ingest()
        cache = DistillationCheckpoints(self.cfg.data_dir / 'distillation-cache.db')
        cache.put('obsolete', {'summary': 'unused'})
        with cache._connect() as db:
            db.execute("INSERT INTO checkpoints VALUES ('legacy-writer','{}')")
        result = cache.prune(float('inf'))
        self.assertEqual(result['requests'], 1)
        self.assertIsNone(cache.get('obsolete'))
        self.assertIsNotNone(cache.get('legacy-writer'))

    def test_config_rejects_invalid_incremental_values(self):
        p = self.root / 'settings.toml'
        for value in ('incremental="yes"', 'incremental_reduce_fanout=1', 'cache_retention_days=0', 'max_model_calls_per_run=-1',
                      'incremental=true\nsegment_chars=10\nsegment_overlap=10'):
            p.write_text('[ingest]\n' + value)
            with self.assertRaises(ValueError):
                load_config(p)

    def test_scheduled_source_backoff_does_not_block_changed_input(self):
        self.cfg.source_retry_backoff = True
        self.backend.fail = True
        self.assertEqual(self.run_ingest().errors, 1)
        calls = len(self.backend.calls)
        self.assertEqual(self.run_ingest().retries_deferred, 1)
        self.assertEqual(len(self.backend.calls), calls)
        self.backend.fail = False
        self.add_unit()
        self.assertEqual(self.run_ingest().distilled, 1)

    def test_source_limit_reports_remaining_work_without_reprocessing_indexed_sources(self):
        write_rollout(self.home, 'two')
        write_rollout(self.home, 'three')
        for remaining in (2, 1, 0):
            stats = self.run_ingest(limit=1)
            self.assertEqual((stats.distilled, stats.source_reads, stats.sources_remaining), (1, 1, remaining))
        self.assertEqual(self.run_ingest().model_calls, 0)

    def test_automatic_poll_records_hot_source_without_calls_until_quiet(self):
        self.cfg.quiet_seconds = 900
        modified = self.path.stat().st_mtime
        with mock.patch('agentic_memory.incremental.time.time', return_value=modified + 1):
            first = self.run_ingest(limit=1)
        self.assertEqual((first.quiet_deferred, first.model_calls, first.embedded_passages,
                          first.source_reads, first.sources_remaining), (1, 0, 0, 0, 0))
        progress = self.store.source_progress(str(self.path))
        self.assertEqual(progress['status'], 'quiet_pending')
        self.assertAlmostEqual(json.loads(progress['state'])['quiet_until'], modified + 900, places=3)
        with mock.patch('agentic_memory.incremental.time.time', return_value=modified + 601):
            second = self.run_ingest(limit=1)
        self.assertEqual((second.quiet_deferred, second.model_calls), (1, 0))
        with mock.patch('agentic_memory.incremental.time.time', return_value=modified + 901):
            last = self.run_ingest(limit=1)
        self.assertEqual((last.distilled, last.model_calls, last.quiet_deferred), (1, 2, 0))
        self.assertEqual(self.store.source_progress(str(self.path))['status'], 'indexed')

    def test_active_indexed_source_keeps_paid_rows_until_quiet_then_ingests_suffix(self):
        self.run_ingest()
        before = self.paid_rows()
        counts = len(self.backend.calls), len(self.embedder.calls)
        self.add_unit()
        modified = self.path.stat().st_mtime
        self.cfg.quiet_seconds = 900
        with mock.patch('agentic_memory.incremental.time.time', return_value=modified + 1):
            deferred = self.run_ingest(limit=1)
        self.assertEqual((deferred.quiet_deferred, deferred.model_calls), (1, 0))
        self.assertEqual(before, self.paid_rows())
        self.assertEqual(counts, (len(self.backend.calls), len(self.embedder.calls)))
        self.assertEqual(self.store.source_progress(str(self.path))['status'], 'quiet_pending')
        with mock.patch('agentic_memory.incremental.time.time', return_value=modified + 901):
            completed = self.run_ingest(limit=1)
        self.assertEqual((completed.distilled, completed.model_calls), (1, 2))

    def test_explicit_targeted_ingest_bypasses_automatic_quiet_window(self):
        self.cfg.quiet_seconds = 900
        result = self.run_ingest(limit=1, session_ids={'one'})
        self.assertEqual((result.distilled, result.quiet_deferred), (1, 0))

    def test_new_activity_restarts_quiet_deadline(self):
        self.cfg.quiet_seconds = 900
        first_modified = self.path.stat().st_mtime
        with mock.patch('agentic_memory.incremental.time.time', return_value=first_modified + 1):
            self.run_ingest(limit=1)
        self.add_unit()
        second_modified = self.path.stat().st_mtime
        self.assertGreaterEqual(second_modified, first_modified)
        with mock.patch('agentic_memory.incremental.time.time', return_value=second_modified + 1):
            result = self.run_ingest(limit=1)
        self.assertEqual((result.quiet_deferred, result.model_calls), (1, 0))
        state = json.loads(self.store.source_progress(str(self.path))['state'])
        self.assertAlmostEqual(state['quiet_until'], second_modified + 900, places=3)

    def test_agent_core_body_mtime_drives_automatic_quiet_window(self):
        home = self.root / 'tess-quiet'
        home.mkdir()
        meta = home / 'native.json'
        meta.write_text(json.dumps({'id': 'native', 'cwd': '/work/tess'}))
        body = home / 'native' / 'full-transcript-with-tools.jsonl'
        body.parent.mkdir()
        body.write_text('')
        for i, (role, content) in enumerate([('user', 'First task'), ('assistant', 'First answer')]):
            append(body, {'type': 'message', 'message_index': i, 'role': role, 'content': content})
        self.cfg.data_stores = [DataStore('tess', home, type='agent-core', project='fixed-project')]
        self.cfg.min_session_chars = 0
        self.cfg.quiet_seconds = 900
        metadata_mtime = meta.stat().st_mtime
        modified = body.stat().st_mtime
        with mock.patch('agentic_memory.incremental.time.time', return_value=modified + 1):
            result = self.run_ingest(limit=1)
        self.assertEqual((result.quiet_deferred, result.model_calls, result.source_reads), (1, 0, 0))
        self.assertEqual(meta.stat().st_mtime, metadata_mtime)
        self.assertEqual(self.store.source_progress(str(meta))['status'], 'quiet_pending')

    def test_operator_export_hold_prevents_model_and_embedding_calls_until_released(self):
        self.cfg.ingestion_hold_texts = ['Remember the durable workspace decision.']
        stats = self.run_ingest()
        self.assertEqual((stats.model_calls, stats.embedded_passages), (0, 0))
        self.assertFalse(self.backend.calls)
        self.assertFalse(self.embedder.calls)
        self.assertEqual(self.store.source_progress(str(self.path))['status'], 'held')
        self.cfg.ingestion_hold_texts = []
        self.assertEqual(self.run_ingest().distilled, 1)

    def test_reduction_tree_append_only_rebuilds_the_rightmost_path(self):
        self.cfg.incremental_reduce_fanout = 2
        for i in range(15):
            append(self.path, wrapped('response_item', message('user', f'Distinct request {i}'), 10 + i * 2))
            append(self.path, wrapped('response_item', message('assistant', f'Distinct answer {i}'), 11 + i * 2))
        first = self.run_ingest()
        self.assertEqual(first.model_calls, 31)  # 16 leaves + 15 reduction nodes
        append(self.path, wrapped('response_item', message('user', 'Final unique request'), 50))
        append(self.path, wrapped('response_item', message('assistant', 'Final unique answer'), 51))
        second = self.run_ingest()
        self.assertEqual(second.distilled, 1)
        self.assertLessEqual(second.model_calls, 6)
        self.assertGreaterEqual(second.request_cache_hits, 30)

    def test_agent_core_ingestion_observes_body_changes_with_unchanged_metadata(self):
        home = self.root / 'tess'
        home.mkdir()
        meta = home / 'native.json'
        meta.write_text(json.dumps({'id': 'native', 'cwd': '/work/tess'}))
        body = home / 'native' / 'full-transcript-with-tools.jsonl'
        body.parent.mkdir()
        body.write_text('')
        for i, (role, content) in enumerate([('user', 'First task'), ('assistant', 'First answer')]):
            append(body, {'type': 'message', 'message_index': i, 'role': role, 'content': content})
        self.cfg.data_stores = [DataStore('tess', home, type='agent-core', project='fixed-project')]
        self.cfg.min_session_chars = 0
        self.assertEqual(self.run_ingest().distilled, 1)
        before = meta.read_bytes(), meta.stat().st_mtime_ns
        for i, (role, content) in enumerate([('user', 'Second task'), ('assistant', 'Second answer')], 2):
            append(body, {'type': 'message', 'message_index': i, 'role': role, 'content': content})
        self.assertEqual(self.run_ingest().model_calls, 2)
        self.assertEqual((meta.read_bytes(), meta.stat().st_mtime_ns), before)
        self.assertEqual(self.store.get_session('native')['n_user'], 2)
        self.assertEqual(self.run_ingest().model_calls, 0)


if __name__ == '__main__':
    unittest.main()
