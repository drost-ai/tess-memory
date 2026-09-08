from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sqlite3
import sys
import unittest
from unittest import mock

from agentic_memory.scheduler_config import ServiceConfig, ScheduledStore, load_service
from agentic_memory.scheduler_queue import WorkQueue
from agentic_memory.scheduler import worker, service_status, serve, owned_worker, stop_run, run_outcome
from agentic_memory.ingest import IngestStats
from agentic_memory.provider_limits import ProviderGate
from agentic_memory.scheduler_install import build_plist, owned_plist, service_control
from test_codex import RecordingEmbedder, write_rollout


class SchedulerQueueTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.config = ServiceConfig(path=self.root / 'config.toml', data_dir=self.root / 'queue',
                                    stores=tuple(ScheduledStore(n, self.root / (n + '.toml'), enabled=True) for n in ('a', 'b')),
                                    parallel_stores=2, retry_seconds=10, max_retry_seconds=30)
        self.queue = WorkQueue(self.config)

    def test_one_job_per_store_even_across_queue_connections(self):
        one = self.queue.claim(now=10**10)
        other = WorkQueue(self.config)
        two = other.claim(now=10**10)
        self.assertNotEqual(one['instance'], two['instance'])
        self.assertIsNone(other.claim(now=10**10))
        self.queue.finish(one['id'], now=10**10)
        self.assertEqual(other.claim(now=10**10)['instance'], one['instance'])

    def test_status_does_not_compete_for_a_coordinator_write_lock(self):
        ProviderGate(self.config.db_path)
        writer = self.queue.connect()
        self.addCleanup(writer.close)
        writer.execute('BEGIN IMMEDIATE')
        writer.execute("UPDATE scheduled_stores SET enabled=0 WHERE name='a'")
        before = self.config.db_path.stat().st_mtime_ns
        status = service_status(self.config)
        self.assertEqual(len(status['jobs']), 6)
        self.assertTrue(all(j['enabled'] for j in status['jobs']))
        self.assertEqual(self.config.db_path.stat().st_mtime_ns, before)
        self.assertEqual(status['providers']['active'], [])
        writer.rollback()

    def test_readonly_status_cannot_initialize_or_mutate_the_coordinator(self):
        readonly = WorkQueue(self.config, readonly=True)
        with self.assertRaises(sqlite3.OperationalError):
            readonly.set_enabled('a', False)
        absent = replace(self.config, data_dir=self.root/'missing')
        with self.assertRaises(sqlite3.OperationalError):
            service_status(absent)
        self.assertFalse(absent.data_dir.exists())

    def test_supervisor_retries_busy_heartbeat_without_releasing_work(self):
        import threading
        from contextlib import nullcontext
        stop = threading.Event()
        queue = mock.Mock()
        def recover():
            stop.set()
            return []
        queue.supervisor_heartbeat.side_effect = [sqlite3.OperationalError('database is locked'), None]
        queue.recover.side_effect = recover
        queue.status.return_value = {'jobs': []}
        for store in self.config.stores:
            store.settings.touch()
        with mock.patch('agentic_memory.scheduler.load_service', return_value=self.config), \
             mock.patch('agentic_memory.scheduler.WorkQueue', return_value=queue), \
             mock.patch('agentic_memory.scheduler._gate') as gate, \
             mock.patch('agentic_memory.scheduler.supervisor_lock', return_value=nullcontext()), \
             mock.patch('agentic_memory.scheduler.signal.signal'):
            serve(self.config.path, stop_event=stop)
        self.assertEqual(queue.supervisor_heartbeat.call_count, 2)
        queue.finish.assert_not_called()
        gate.return_value.release_owner.assert_not_called()

    def test_supervisor_retains_worker_and_leases_across_unknown_identity_probe(self):
        import threading
        import time
        from contextlib import nullcontext
        stop = threading.Event()
        queue = mock.Mock()
        ticks = 0
        def recover():
            nonlocal ticks
            ticks += 1
            if ticks == 2:
                stop.set()
            return []
        queue.recover.side_effect = recover
        queue.status.return_value = {'jobs': [{'state': 'running', 'pid': 123, 'run_id': 'owned', 'started': time.time()}]}
        queue.claim.return_value = None
        for store in self.config.stores:
            store.settings.touch()
        with mock.patch('agentic_memory.scheduler.load_service', return_value=self.config), \
             mock.patch('agentic_memory.scheduler.WorkQueue', return_value=queue), \
             mock.patch('agentic_memory.scheduler._gate') as gate, \
             mock.patch('agentic_memory.scheduler.supervisor_lock', return_value=nullcontext()), \
             mock.patch('agentic_memory.scheduler.signal.signal'), \
             mock.patch('agentic_memory.scheduler.owned_worker', side_effect=[None, True]) as probe, \
             mock.patch('agentic_memory.scheduler.stop_run') as terminate:
            serve(self.config.path, stop_event=stop)
        self.assertEqual(probe.call_count, 2)
        queue.finish.assert_not_called()
        gate.return_value.release_owner.assert_not_called()
        terminate.assert_not_called()

    def test_disable_persists_across_restart_and_preserves_active_jobs(self):
        run = self.queue.claim(instance='a', explicit=True)
        self.queue.set_enabled('a', False)
        reopened = WorkQueue(self.config)
        self.assertIsNone(reopened.claim(instance='a', now=10**10))
        self.assertEqual(reopened.get_run(run['id'])['outcome'], 'starting')
        self.queue.finish(run['id'])
        self.assertIsNone(reopened.claim(instance='a', now=10**10))
        self.assertIsNotNone(reopened.claim(instance='a', kind='ingest', explicit=True))

    def test_lost_worker_requeues_and_old_completion_cannot_clear_new_run(self):
        run = self.queue.claim(instance='a', kind='ingest', explicit=True)
        self.queue.start(run['id'], 99999999)
        with mock.patch('agentic_memory.scheduler_queue.process_alive', return_value=False):
            self.assertEqual(self.queue.recover(), [run['id']])
        new = self.queue.claim(instance='a', kind='ingest', explicit=True)
        self.assertFalse(self.queue.finish(run['id']))
        self.assertEqual(self.queue.get_run(new['id'])['outcome'], 'starting')

    def test_same_run_cannot_be_claimed_by_a_different_process(self):
        run = self.queue.claim(instance='a', explicit=True)
        self.queue.start(run['id'], 123)
        with self.assertRaisesRegex(ValueError, 'different worker'):
            self.queue.start(run['id'], 456)

    def test_retry_during_running_job_survives_completion(self):
        run = self.queue.claim(instance='a', kind='ingest', explicit=True)
        self.queue.request('a', 'ingest')
        self.queue.finish(run['id'], now=100)
        following = self.queue.claim(instance='a', kind='ingest', now=100)
        self.assertIsNotNone(following)
        self.assertTrue(self.queue.get_run(following['id'])['force_retry'])

    def test_failure_backoff_is_bounded_and_partial_passes_do_not_back_off_every_store(self):
        for i, expected in enumerate((10, 20, 30), 1):
            run = self.queue.claim(instance='a', kind='ingest', explicit=True)
            self.queue.finish(run['id'], outcome='error', error='temporary', now=100)
            row = next(r for r in self.queue.status()['jobs'] if r['instance'] == 'a' and r['kind'] == 'ingest')
            self.assertEqual(row['due'], 100 + expected)
        run = self.queue.claim(instance='a', kind='ingest', explicit=True)
        self.queue.finish(run['id'], outcome='partial', remaining=5, now=100)
        row = next(r for r in self.queue.status()['jobs'] if r['instance'] == 'a' and r['kind'] == 'ingest')
        self.assertEqual((row['attempts'], row['due']), (0, 115))

    def test_sleep_wake_clock_jump_coalesces_due_work_without_duplicate_jobs(self):
        # The service may be suspended for hours. It owes one current pass per
        # job kind, not one duplicate job for every missed five-minute tick.
        wake = 10**10
        one = self.queue.claim(now=wake)
        two = self.queue.claim(now=wake)
        self.assertNotEqual(one['instance'], two['instance'])
        self.assertIsNone(self.queue.claim(now=wake))
        self.assertEqual(len(self.queue.status()['jobs']), 6)


class WorkerOwnershipTests(unittest.TestCase):
    def test_retained_checkpoint_progress_is_partial_despite_an_unrelated_source_error(self):
        stats = IngestStats(errors=1, distilled=0, budget_deferred=15, checkpointed_requests=100)
        self.assertEqual(run_outcome(stats, 128, 'ingest'), 'partial')
        stats.checkpointed_requests = 0
        self.assertEqual(run_outcome(stats, 128, 'ingest'), 'error')
        stats.distilled = 1
        self.assertEqual(run_outcome(stats, 128, 'ingest'), 'partial')

    def test_probe_timeout_or_os_failure_is_unknown_instead_of_unowned(self):
        for error in (subprocess.TimeoutExpired('ps', 3), OSError('probe unavailable')):
            with mock.patch('agentic_memory.scheduler.process_alive', return_value=True), \
                 mock.patch('agentic_memory.scheduler.subprocess.run', side_effect=error):
                self.assertIsNone(owned_worker(123, 'run'))

    def test_live_unreadable_probe_is_unknown_and_dead_process_is_unowned(self):
        from types import SimpleNamespace
        for result in (SimpleNamespace(returncode=1, stdout=''), SimpleNamespace(returncode=0, stdout=''),
                       SimpleNamespace(returncode=0, stdout='"unfinished quotation')):
            with mock.patch('agentic_memory.scheduler.process_alive', return_value=True), \
                 mock.patch('agentic_memory.scheduler.subprocess.run', return_value=result):
                self.assertIsNone(owned_worker(123, 'run'))
        with mock.patch('agentic_memory.scheduler.process_alive', return_value=False), \
             mock.patch('agentic_memory.scheduler.subprocess.run') as probe:
            self.assertIs(owned_worker(123, 'run'), False)
            probe.assert_not_called()

    def test_positive_ownership_requires_the_expected_worker_command_and_run_id(self):
        from types import SimpleNamespace
        command = '/repo/.venv/bin/python -u -m agentic_memory.cli scheduler-worker --config /config --run-id mine'
        with mock.patch('agentic_memory.scheduler.process_alive', return_value=True), \
             mock.patch('agentic_memory.scheduler.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=command)):
            self.assertIs(owned_worker(123, 'mine'), True)
            self.assertIs(owned_worker(123, 'different'), False)

    def test_stop_run_never_signals_a_process_with_unconfirmed_ownership(self):
        with mock.patch('agentic_memory.scheduler.WorkQueue') as queue, \
             mock.patch('agentic_memory.scheduler.owned_worker', return_value=None), \
             mock.patch('agentic_memory.scheduler.os.killpg') as signal_group, \
             mock.patch('agentic_memory.scheduler.os.getpgid') as group_id:
            queue.return_value.get_run.return_value = {'pid': 123}
            self.assertFalse(stop_run(None, 'run', force=True))
            signal_group.assert_not_called()
            group_id.assert_not_called()


class SchedulerWorkerTests(unittest.TestCase):
    def test_real_cli_worker_process_completes_empty_poll_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / 'source'
            source.mkdir()
            core = root / 'core' / 'agent_core'
            core.mkdir(parents=True)
            core.joinpath('__init__.py').write_text('''async def unused(*a, **k):
    raise AssertionError("empty ingestion must not invoke a provider")
resolve_config = resolve_session = bootstrap_runtime = run_agent_prompt = unused
create_agent_runtime_bindings = is_task_supervisor_gateway = unused
''')
            providers = root / 'providers.json'
            providers.write_text(json.dumps({'providerProfiles': {'offline': {'kind': 'openai', 'defaultModel': 'unused'}}}))
            settings = root / 'memory.toml'
            settings.write_text(f'[[data_stores]]\nname="empty"\npath="{source}"\n'
                                f'[storage]\ndata_dir="{root / "memory"}"\n'
                                f'[llm]\nprofile="offline"\nagent_core_path="{core.parent}"\nsettings="{providers}"\n'
                                '[embedding]\nprovider="local"\n')
            config_path = root / 'scheduler.toml'
            config_path.write_text(f'[service]\ndata_dir="{root / "service"}"\n[[stores]]\nname="empty"\nsettings="{settings}"\n')
            result = subprocess.run([sys.executable, '-m', 'agentic_memory.cli', 'scheduler', '--config', str(config_path),
                                     'run', '--instance', 'empty'], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            completed = json.loads(result.stdout)
            self.assertEqual(completed['outcome'], 'success')
            self.assertEqual(completed['stats']['model_calls'], 0)
            self.assertEqual(completed['stats']['embedded_passages'], 0)

    def test_worker_uses_instance_settings_and_completes_durable_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            home = root / 'codex'
            write_rollout(home)
            profiles = root / 'providers.json'
            profiles.write_text(json.dumps({'providerProfiles': {'p': {'kind': 'openai', 'apiMode': 'codex'}}}))
            settings = root / 'memory.toml'
            settings.write_text(f'[[data_stores]]\nname="test"\ntype="codex"\npath="{home}"\n'
                                f'[storage]\ndata_dir="{root / "memory"}"\n[llm]\nsettings="{profiles}"\nprofile="p"\nworkers=2\n')
            config_path = root / 'scheduler.toml'
            config_path.write_text(f'[service]\ndata_dir="{root / "service"}"\n[[stores]]\nname="codex"\nsettings="{settings}"\nenabled=false\n')
            config = load_service(config_path)
            queue = WorkQueue(config)
            run = queue.claim(instance='codex', kind='ingest', explicit=True)
            class Backend:
                name = cache_identity = 'synthetic'
                def complete(self, **kw):
                    return json.dumps({'title': 'Test', 'summary': 'Useful history', 'outcome': 'Complete', 'memories': []})
            with mock.patch('agentic_memory.scheduler.AgentCoreBackend', return_value=Backend()) as ctor, \
                 mock.patch('agentic_memory.scheduler.make_embedder', return_value=RecordingEmbedder()):
                self.assertEqual(worker(config_path, run['id']), 0)
            self.assertFalse(ctor.call_args.kwargs['probe'])
            saved = WorkQueue(config).get_run(run['id'])
            self.assertEqual(saved['outcome'], 'success')
            self.assertEqual(json.loads(saved['stats'])['distilled'], 1)
            run2 = queue.claim(instance='codex', kind='ingest', explicit=True)
            with mock.patch('agentic_memory.scheduler.AgentCoreBackend', return_value=Backend()), \
                 mock.patch('agentic_memory.scheduler.make_embedder', return_value=RecordingEmbedder()):
                self.assertEqual(worker(config_path, run2['id']), 0)
            self.assertEqual(json.loads(queue.get_run(run2['id'])['stats'])['model_calls'], 0)

    def test_service_configuration_rejects_shared_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / 'memory.toml'
            settings.write_text(f'[[data_stores]]\nname="s"\npath="{root}"\n[storage]\ndata_dir="{root / "memory"}"\n')
            config = root / 'scheduler.toml'
            config.write_text(f'[service]\ndata_dir="{root / "service"}"\n'
                              f'[[stores]]\nname="a"\nsettings="{settings}"\n'
                              f'[[stores]]\nname="b"\nsettings="{settings}"\n')
            with self.assertRaisesRegex(ValueError, 'distinct'):
                load_service(config)

    def test_launch_agent_uses_persistent_service_and_rejects_unowned_plist(self):
        import plistlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ServiceConfig(path=root / 'config.toml', data_dir=root / 'service', stores=())
            value = build_plist(cfg, 'com.agentic-memory.test')
            self.assertTrue(value['RunAtLoad'])
            self.assertTrue(value['KeepAlive'])
            self.assertIn('serve', value['ProgramArguments'])
            path = root / 'agent.plist'
            path.write_bytes(plistlib.dumps(value))
            self.assertEqual(owned_plist(path)['Label'], 'com.agentic-memory.test')
            path.write_bytes(plistlib.dumps({'ProgramArguments': ['/bin/true']}))
            with self.assertRaisesRegex(ValueError, 'not owned'):
                owned_plist(path)

    def test_launchd_restart_retries_transient_teardown_race(self):
        from types import SimpleNamespace
        calls = []
        bootstraps = 0
        def command(args, **kwargs):
            nonlocal bootstraps
            calls.append(args[1])
            if args[1] == 'bootstrap':
                bootstraps += 1
                return SimpleNamespace(returncode=5 if bootstraps == 1 else 0)
            if args[1] == 'print':
                return SimpleNamespace(returncode=0 if len(calls) == 1 else 113)
            return SimpleNamespace(returncode=0)
        with mock.patch('agentic_memory.scheduler_install.sys.platform', 'darwin'), \
             mock.patch('agentic_memory.scheduler_install.owned_plist', return_value={'Label': 'com.agentic-memory.test'}), \
             mock.patch('agentic_memory.scheduler_install.subprocess.run', side_effect=command), \
             mock.patch('agentic_memory.scheduler_install.time.sleep'):
            self.assertTrue(service_control('restart', label='com.agentic-memory.test')['loaded'])
        self.assertEqual(bootstraps, 2)


if __name__ == '__main__':
    unittest.main()
