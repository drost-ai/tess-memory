import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tess_memory.config import Config, DataStore
from tess_memory.detection import detect_sources, detection_status
from tess_memory.locks import ingestion_lock
from tess_memory.scheduler import worker, serve
from tess_memory.scheduler_config import load_service
from tess_memory.scheduler_queue import WorkQueue
from tess_memory.store import MemoryStore
from test_codex import write_rollout, append, wrapped, message


class DetectionTests(unittest.TestCase):
    def test_real_detector_job_observes_changes_while_ingestion_owns_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            home = root/'source'
            source = write_rollout(home)
            settings = root/'memory.toml'
            settings.write_text(f'[[data_stores]]\nname="codex"\ntype="codex"\npath="{home}"\n'
                                f'[storage]\ndata_dir="{root / "memory"}"\n'
                                '[ingest]\nquiet_seconds=900\n'
                                '[llm]\nagent_core_path="/nonexistent/no-provider-needed"\n')
            service_path = root/'service.toml'
            service_path.write_text(f'[service]\ndata_dir="{root / "service"}"\n[[stores]]\n'
                                    f'name="codex"\nsettings="{settings}"\nenabled=true\n')
            service = load_service(service_path)
            queue = WorkQueue(service)
            paid = queue.claim(instance='codex', kind='ingest', explicit=True)
            queue.start(paid['id'], os.getpid())
            cfg = Config(data_stores=[DataStore('codex', home, type='codex')], data_dir=root/'memory', quiet_seconds=900)
            store = MemoryStore(cfg.db_path)
            try:
                with ingestion_lock(cfg.data_dir), store.write_transaction(), \
                     mock.patch('tess_memory.scheduler.AgentCoreBackend', side_effect=AssertionError('detector must not construct models')), \
                     mock.patch('tess_memory.scheduler.make_embedder', side_effect=AssertionError('detector must not construct embeddings')):
                    store.set_meta('writer-is-active', 'preserve me')
                    first = queue.claim(instance='codex', kind='detect', explicit=True)
                    self.assertEqual(worker(service_path, first['id']), 0)
                    self.assertEqual(queue.get_run(paid['id'])['outcome'], 'running')
                    status = detection_status(cfg)
                    self.assertEqual((status['sources'], status['hot_sources'], status['new_sources']), (1, 1, 1))
                    append(source, wrapped('response_item', message('user', 'new activity'), 30))
                    first_run = queue.get_run(first['id'])
                    second = queue.claim(instance='codex', kind='detect', now=first_run['started']+300)
                    self.assertIsNotNone(second)
                    self.assertEqual(worker(service_path, second['id']), 0)
                    self.assertEqual(detection_status(cfg)['changed_sources'], 1)
                    self.assertEqual(store.db.execute('SELECT COUNT(*) FROM ingest_sources').fetchone()[0], 0)
                    self.assertEqual(store.db.execute('SELECT COUNT(*) FROM memories').fetchone()[0], 0)
                    self.assertEqual(store.get_meta('writer-is-active'), 'preserve me')
                    self.assertEqual(queue.get_run(paid['id'])['outcome'], 'running')
            finally:
                store.close()

    def test_detector_includes_body_changes_and_retains_last_scan_when_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root/'tess'
            home.mkdir()
            meta = home/'one.json'
            meta.write_text(json.dumps({'id': 'one', 'cwd': '/project'}))
            body = home/'one.messages.jsonl'
            body.write_text('')
            cfg = Config(data_stores=[DataStore('tess', home, type='agent-core')], data_dir=root/'db', quiet_seconds=900)
            detect_sources(cfg)
            meta_stat = meta.stat()
            append(body, {'type': 'message', 'message': {'role': 'user', 'content': 'new data'}})
            result = detect_sources(cfg)
            self.assertEqual(result['changed_sources'], 1)
            self.assertEqual(meta.stat().st_mtime_ns, meta_stat.st_mtime_ns)
            previous = detection_status(cfg)['finished_at']
            with self.assertRaises(InterruptedError):
                detect_sources(cfg, stop_requested=lambda: True)
            self.assertEqual(detection_status(cfg)['finished_at'], previous)

    def test_supervisor_starts_detection_even_when_all_paid_capacity_is_full(self):
        from types import SimpleNamespace
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = root/'memory.toml'
            memory.write_text(f'[[data_stores]]\nname="a"\npath="{root}"\n[storage]\ndata_dir="{root / "db"}"')
            path = root/'service.toml'
            path.write_text(f'[service]\ndata_dir="{root / "service"}"\nparallel_stores=1\n[[stores]]\nname="a"\nsettings="{memory}"\nenabled=true')
            cfg = load_service(path)
            queue = WorkQueue(cfg)
            paid = queue.claim(instance='a', kind='ingest', explicit=True)
            queue.start(paid['id'], os.getpid())
            stop = threading.Event()
            launched = []
            def spawn(config, run):
                launched.append(run['kind'])
                stop.set()
                return SimpleNamespace(pid=os.getpid(), poll=lambda: None)
            with mock.patch('tess_memory.scheduler.owned_worker', return_value=True), \
                 mock.patch('tess_memory.scheduler.spawn_worker', side_effect=spawn):
                serve(path, stop_event=stop)
            self.assertEqual(launched, ['detect'])
            self.assertEqual(queue.get_run(paid['id'])['outcome'], 'running')
