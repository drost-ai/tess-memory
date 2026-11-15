import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from tess_memory import server
from tess_memory.freshness import describe, hit_freshness, rollup_freshness, session_freshness, store_status
from tess_memory.scheduler_config import ServiceConfig, ScheduledStore
from tess_memory.scheduler_queue import WorkQueue
from test_incremental_ingest import Backend
from test_codex import RecordingEmbedder, append, message, wrapped, write_rollout
from tess_memory.config import Config, DataStore
from tess_memory.ingest import ingest
from tess_memory.store import MemoryStore


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.home = self.root / 'codex'
        self.path = write_rollout(self.home)
        self.cfg = Config(data_stores=[DataStore('c', self.home, type='codex', project='p')],
                          data_dir=self.root / 'memory', incremental_ingestion=True,
                          segment_chars=3000, segment_overlap=0)
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.embedder, self.backend = RecordingEmbedder(), Backend()

    def ingest(self):
        return ingest(self.cfg, self.store, self.embedder, self.backend, refresh_aggregates=False)

    def test_filesystem_changes_are_visible_before_another_ingestion_poll(self):
        self.ingest()
        self.assertEqual(session_freshness(self.store, 'codex:one', self.cfg)['status'], 'indexed')
        append(self.path, wrapped('response_item', message('user', 'Later unfinished work')))
        fresh = session_freshness(self.store, 'codex:one', self.cfg)
        self.assertEqual(fresh['status'], 'source_changed')
        self.assertTrue(fresh['source_changed'])
        self.assertIn('source changed', describe(fresh))

    def test_pending_tail_and_dirty_rollup_are_not_reported_as_current(self):
        self.ingest()
        append(self.path, wrapped('response_item', message('user', 'Pending request')))
        self.ingest()
        fresh = session_freshness(self.store, 'codex:one', self.cfg)
        self.assertEqual(fresh['status'], 'pending_tail')
        self.assertTrue(fresh['pending_tail'])
        self.assertEqual(rollup_freshness(self.store, 'project', 'p')['status'], 'pending_refresh')

    def test_quiet_pending_status_exposes_deadline_without_state_blob(self):
        self.ingest()
        with self.store.write_transaction():
            self.store.record_source(path=str(self.path), session_id='codex:one', project='p', source_type='codex',
                                     signature='changed', status='quiet_pending',
                                     state={'quiet_until': 12345, 'private': 'do not expose'}, now=1,
                                     error='source is still inside the configured quiet window')
        fresh = session_freshness(self.store, 'codex:one', self.cfg)
        self.assertEqual((fresh['status'], fresh['quiet_until']), ('quiet_pending', 12345))
        self.assertIn('deferred until source is quiet', describe(fresh))
        status = store_status(self.store, self.cfg)
        problem = status['ingestion']['problems'][0]
        self.assertEqual(problem['quiet_until'], 12345)
        self.assertNotIn('state', problem)
        self.assertNotIn('do not expose', json.dumps(status))

    def test_legacy_snapshot_is_explicitly_unverified(self):
        self.cfg.incremental_ingestion = False
        self.ingest()
        fresh = session_freshness(self.store, 'codex:one', self.cfg)
        self.assertEqual(fresh['status'], 'legacy_unverified')
        self.assertIsNone(fresh['source_changed'])

    def test_status_preserves_legacy_count_fields_and_redacts_diagnostic_secrets(self):
        self.ingest()
        counts = self.store.counts()
        with self.store.write_transaction():
            self.store.db.execute("UPDATE ingest_sources SET status='error',last_error=?", ('Authorization: Bearer secret-credential password=hidden-value',))
        result = store_status(self.store, self.cfg)
        self.assertTrue(all(result[k] == v for k, v in counts.items()))
        text = json.dumps(result)
        self.assertNotIn('secret-credential', text)
        self.assertNotIn('hidden-value', text)
        self.assertEqual(result['ingestion']['source_statuses']['error'], 1)

    def test_mcp_context_and_status_expose_freshness_without_provider_calls(self):
        self.ingest()
        append(self.path, wrapped('response_item', message('user', 'Pending work')))
        with mock.patch.object(server, '_get_store', return_value=self.store), \
             mock.patch.object(server, 'CONFIG', self.cfg), \
             mock.patch.object(server, '_get_embedder', side_effect=AssertionError('no embeddings')):
            context = server.get_project_context('p')
            self.assertIn('Memory freshness: source changed', context)
            self.assertIn('ingestion', json.loads(server.memory_status()))
            self.assertIn('Memory freshness:', server.get_session('codex:one'))

    def test_empty_memory_scope_can_report_a_source_hold(self):
        self.cfg.ingestion_hold_texts = ['Remember the durable workspace decision.']
        self.ingest()
        with mock.patch.object(server, '_get_store', return_value=self.store), \
             mock.patch.object(server, 'CONFIG', self.cfg), \
             mock.patch.object(server, '_get_embedder', side_effect=AssertionError('no embeddings')):
            result = server.search_memory('history', project='p', mode='static')
        self.assertIn('No matching memories', result)
        self.assertIn('held=1', result)

    def test_supervisor_heartbeat_is_separate_from_source_publication(self):
        config = ServiceConfig(path=self.root / 'service.toml', data_dir=self.root / 'service',
                               stores=(ScheduledStore('p', self.root / 'settings.toml'),))
        queue = WorkQueue(config)
        queue.supervisor_heartbeat(os.getpid(), time.time())
        status = queue.status()
        self.assertTrue(status['supervisor']['process_alive'])
        self.assertTrue(all(job['last_success'] is None for job in status['jobs']))
        self.assertIsNone(store_status(self.store)['ingestion']['last_publication'])

    def test_legacy_rollups_get_tracking_once_without_requeueing_clean_scopes(self):
        self.store.upsert_session(id='legacy', project='p', ended_at='2026-08-10T00:00:00Z')
        self.store.commit()
        self.assertEqual(self.store.bootstrap_aggregate_tracking(('project', 'digest')), 2)
        with self.store.write_transaction():
            for row in self.store.pending_aggregates(float('inf')):
                self.store.acknowledge_aggregate(row['level'], row['scope'], row['revision'], time.time())
        self.assertEqual(self.store.bootstrap_aggregate_tracking(('project', 'digest')), 0)
        self.assertEqual(self.store.pending_aggregates(float('inf')), [])

    def test_untracked_legacy_projects_are_in_the_profile_denominator(self):
        self.store.upsert_session(id='legacy', project='p', ended_at='2026-08-10T00:00:00Z')
        self.store.commit()
        data = store_status(self.store)['ingestion']['aggregate_progress']['project']
        self.assertEqual((data['total'], data['current'], data['untracked'], data['percent_current']), (1, 0, 1, 0.0))
        self.assertEqual(self.store.pending_aggregates(float('inf')), [])
        self.store.bootstrap_aggregate_tracking(('project',))
        with self.store.write_transaction():
            self.store.acknowledge_aggregate('project', 'p', 1, time.time())
        self.assertEqual(store_status(self.store)['ingestion']['aggregate_progress']['project']['percent_current'], 100.0)
        self.store.upsert_session(id='new-project', project='q', ended_at='2026-09-07')
        self.store.commit()
        data = store_status(self.store)['ingestion']['aggregate_progress']['project']
        self.assertEqual((data['total'], data['current'], data['untracked'], data['percent_current']), (2, 1, 1, 50.0))

    def test_empty_store_does_not_claim_one_hundred_percent_aggregate_coverage(self):
        data = store_status(self.store)['ingestion']['aggregate_progress']
        self.assertIsNone(data['project']['percent_current'])
        self.assertIsNone(data['digest']['percent_current'])


if __name__ == '__main__':
    unittest.main()
