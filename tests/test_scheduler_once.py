"""Isolated finite-batch tests: temporary coordinators, synthetic subprocesses only."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agentic_memory import scheduler_once as batch
from agentic_memory.scheduler import supervisor_lock
from agentic_memory.scheduler_config import ScheduledStore, ServiceConfig
from agentic_memory.scheduler_queue import WorkQueue


class FiniteBatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = ServiceConfig(self.root / 'service.toml', self.root / 'queue',
            tuple(ScheduledStore(name, self.root / (name + '.toml'), enabled=name != 'disabled',
                                 poll_seconds=14400) for name in ('a', 'b', 'disabled')),
            parallel_stores=1, job_timeout_seconds=2, job_drain_seconds=0.1)
        self.children = []
        self.runs = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait()

    def spawn(self, config, run, *, outcome='success', exit_code=0):
        self.runs.append(run)
        # Terminal state may beat the parent's start() registration in production.
        child = subprocess.Popen([sys.executable, '-c', f'import time; time.sleep(.03); raise SystemExit({exit_code})'],
                                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        queue = WorkQueue(config)
        queue.start(run['id'], child.pid)
        queue.finish(run['id'], outcome=outcome, remaining=5,
                     stats={'source_text': 'PRIVATE', 'errors': 2})
        return child

    def invoke(self, spawn=None):
        output = io.StringIO()
        with patch.object(batch, 'load_service', return_value=self.config), \
             patch.object(batch, 'spawn_worker', side_effect=spawn or self.spawn), redirect_stdout(output):
            code = batch.ingest_once(self.config.path)
        return code, json.loads(output.getvalue())

    def backlog(self, left, *, fail_at=None):
        """A worker for store 'a' that reports ``left`` sources waiting, pass by pass."""
        left = list(left)
        def spawn(config, run):
            self.runs.append(run)
            child = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(0)'], start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.children.append(child)
            queue = WorkQueue(config)
            queue.start(run['id'], child.pid)
            waiting = left.pop(0) if run['instance'] == 'a' and left else 0
            failed = fail_at is not None and len(self.runs) == fail_at
            queue.finish(run['id'], outcome='error' if failed else 'success', remaining=waiting,
                         stats={'sources_remaining': waiting, 'sources_pending': 0, 'source_text': 'PRIVATE'})
            return child
        return spawn

    def catch_up(self, spawn):
        output = io.StringIO()
        self.config = replace(self.config, backlog_seconds=0.05)
        with patch.object(batch, 'load_service', return_value=self.config), \
             patch.object(batch, 'spawn_worker', side_effect=spawn), redirect_stdout(output):
            code = batch.ingest_once(self.config.path, until_caught_up=True)
        return code, json.loads(output.getvalue())

    def test_catch_up_runs_passes_until_no_source_waits(self):
        code, report = self.catch_up(self.backlog([40, 24, 8, 0]))
        self.assertEqual((code, report['outcome'], report['remaining']), (0, 'success', 0))
        self.assertEqual([r['instance'] for r in self.runs], ['a', 'b', 'a', 'a', 'a'])  # b had nothing waiting
        self.assertEqual(report['passes'], 4)
        self.assertEqual(report['counts']['success'], 5)
        self.assertNotIn('PRIVATE', json.dumps(report))

    def test_without_catch_up_a_backlog_still_runs_one_pass(self):
        code, report = self.invoke(self.backlog([40, 24, 8, 0]))
        self.assertEqual((code, report['passes'], len(self.runs)), (0, 1, 2))

    def test_catch_up_stops_at_a_failed_pass_and_says_what_waits(self):
        code, report = self.catch_up(self.backlog([40, 24, 8, 0], fail_at=3))
        self.assertEqual((code, report['outcome']), (1, 'error'))
        self.assertEqual(len(self.runs), 3)

    def test_catch_up_stops_when_passes_stop_making_progress(self):
        code, report = self.catch_up(self.backlog([40] * 50))
        self.assertEqual(code, 0)
        self.assertEqual(report['remaining'], 40)
        self.assertEqual(report['passes'], batch.STALLED_PASSES + 1)

    def test_one_batch_all_enabled_serial_capacity_no_backlog_loop(self):
        code, report = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(report['outcome'], 'success')
        self.assertEqual(report['counts'], {'success': 2})
        self.assertEqual({r['instance'] for r in self.runs}, {'a', 'b'})
        self.assertTrue(all(r['kind'] == 'ingest' for r in self.runs))
        self.assertTrue(all(p.poll() == 0 for p in self.children))
        self.assertNotIn('PRIVATE', json.dumps(report))
        self.assertEqual(len(self.runs), 2)

    def test_external_cadence_not_legacy_poll_and_no_force_retry(self):
        queue = WorkQueue(self.config)
        db = queue.connect()
        with db:
            db.execute('UPDATE jobs SET due=?', (time.time() + 14400,))
        db.close()
        code, report = self.invoke()
        self.assertEqual(report['counts'], {'success': 2})
        for run in self.runs:
            self.assertFalse(queue.get_run(run['id'])['force_retry'])
        self.assertEqual(code, 0)

    def test_backoff_preserved(self):
        queue = WorkQueue(self.config)
        db = queue.connect()
        with db:
            db.execute('UPDATE jobs SET due=?,attempts=1', (time.time() + 14400,))
        db.close()
        code, report = self.invoke()
        self.assertEqual((code, report['outcome']), (0, 'no_work'))
        self.assertEqual(report['counts'], {'backoff': 2})
        self.assertFalse(self.runs)

    def test_supervisor_overlap_is_busy_no_workers(self):
        with supervisor_lock(self.config):
            code, report = self.invoke()
        self.assertEqual((code, report['outcome']), (0, 'busy'))
        self.assertFalse(self.runs)

    def test_live_manual_lease_not_stolen(self):
        queue = WorkQueue(self.config)
        run = queue.claim(instance='a', kind='ingest', explicit=True)
        queue.start(run['id'], os.getpid())
        code, report = self.invoke()
        self.assertEqual((code, report['outcome']), (0, 'busy'))
        self.assertFalse(self.runs)
        self.assertEqual(queue.get_run(run['id'])['outcome'], 'running')

    def test_partial_truthfully_reported(self):
        code, report = self.invoke(lambda c, r: self.spawn(c, r, outcome='partial'))
        self.assertEqual((code, report['outcome']), (0, 'partial'))
        self.assertEqual(report['counts'], {'partial': 2})

    def test_error_and_nonzero_exit_are_failures(self):
        for outcome, exit_code in [('error', 0), ('success', 3)]:
            with self.subTest(outcome=outcome):
                self.config = replace(self.config, data_dir=self.root / outcome)
                code, report = self.invoke(lambda c, r: self.spawn(c, r, outcome=outcome, exit_code=exit_code))
                self.assertEqual((code, report['outcome']), (1, 'error'))

    def test_launch_failure_finalizes_claim(self):
        code, report = self.invoke(lambda *_: (_ for _ in ()).throw(OSError('PRIVATE')))
        self.assertEqual((code, report['outcome']), (1, 'error'))
        self.assertEqual(report['counts'], {'error': 2})
        self.assertTrue(all(j['state'] != 'running' for j in WorkQueue(self.config).status()['jobs']))
        self.assertNotIn('PRIVATE', json.dumps(report))

    def sleeping_spawn(self, config, run):
        self.runs.append(run)
        child = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'],
            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        return child

    def test_timeout_kills_reaps_and_releases_owned_lease(self):
        self.config = replace(self.config, job_timeout_seconds=0.1)
        start = time.monotonic()
        code, report = self.invoke(self.sleeping_spawn)
        self.assertEqual((code, report['outcome']), (1, 'error'))
        self.assertEqual(report['counts'], {'timeout': 2})
        self.assertLess(time.monotonic() - start, 3)
        self.assertTrue(all(c.poll() is not None for c in self.children))
        queue = WorkQueue(self.config)
        self.assertTrue(all(queue.get_run(r['id'])['outcome'] == 'timeout' for r in self.runs))

    def test_sigterm_reaps_workers_and_restores_handler(self):
        old = signal.getsignal(signal.SIGTERM)
        def spawn(config, run):
            child = self.sleeping_spawn(config, run)
            timer = threading.Timer(.15, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
            self.addCleanup(timer.join)
            return child
        code, report = self.invoke(spawn)
        self.assertEqual((code, report['outcome']), (143, 'cancelled'))
        self.assertEqual(len(self.children), 1)
        self.assertTrue(all(c.poll() is not None for c in self.children))
        self.assertEqual(signal.getsignal(signal.SIGTERM), old)

    def test_sigint_cleans_all_concurrent_workers(self):
        self.config = replace(self.config, parallel_stores=2)
        def spawn(config, run):
            child = self.sleeping_spawn(config, run)
            if len(self.children) == 2:
                timer = threading.Timer(.15, lambda: os.kill(os.getpid(), signal.SIGINT))
                timer.start()
                self.addCleanup(timer.join)
            return child
        code, report = self.invoke(spawn)
        self.assertEqual((code, report['outcome']), (130, 'cancelled'))
        self.assertEqual(report['counts'], {'cancelled': 2})
        self.assertTrue(all(c.poll() is not None for c in self.children))

    def test_unreported_worker_exit_is_not_success(self):
        def spawn(config, run):
            self.runs.append(run)
            child = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
            self.children.append(child)
            return child
        code, report = self.invoke(spawn)
        self.assertEqual((code, report['outcome']), (1, 'error'))
        self.assertEqual(report['counts'], {'worker_lost': 2})

    def test_group_cleanup_stops_worker_descendants(self):
        self.config = replace(self.config, stores=(self.config.stores[0],), job_timeout_seconds=.5)
        pid_file = self.root / 'grandchild.pid'
        grandchild = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'
        script = ("import pathlib,signal,subprocess,sys,time; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
                  f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(30)")
        def spawn(config, run):
            child = subprocess.Popen([sys.executable, '-c', script], start_new_session=True)
            self.children.append(child)
            return child
        code, report = self.invoke(spawn)
        self.assertEqual((code, report['counts']), (1, {'timeout': 1}))
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        # A killed descendant can briefly remain a zombie awaiting OS reaping.
        deadline = time.monotonic() + 3
        while True:
            state = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='],
                                   text=True, capture_output=True).stdout.strip()
            if not state or state.startswith('Z'):
                break
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                self.fail('worker descendant survived group cleanup')
            time.sleep(.05)

    def test_registration_exception_still_reaps_child(self):
        with patch.object(WorkQueue, 'start', side_effect=RuntimeError('PRIVATE')):
            code, report = self.invoke(self.sleeping_spawn)
        self.assertEqual((code, report['outcome']), (1, 'error'))
        self.assertTrue(all(c.poll() is not None for c in self.children))
        self.assertTrue(all(j['state'] != 'running' for j in WorkQueue(self.config).status()['jobs']))

    def test_invalid_config_safe_json_no_coordinator_write(self):
        output = io.StringIO()
        with patch.object(batch, 'load_service', side_effect=ValueError('PRIVATE')), redirect_stdout(output):
            code = batch.ingest_once(self.config.path)
        self.assertEqual(code, 1)
        self.assertNotIn('PRIVATE', output.getvalue())
        self.assertFalse(self.config.data_dir.exists())

    def test_real_cli_invalid_config_safe_json(self):
        result = subprocess.run([sys.executable, '-u', '-m', 'agentic_memory.cli', 'scheduler',
                                 '--config', str(self.root / 'absent.toml'), 'ingest-once'],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)['outcome'], 'error')
        self.assertEqual(result.stderr, '')

    def test_real_finite_cli_empty_worker_without_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / 'source'
            source.mkdir()
            providers = root / 'providers.json'
            providers.write_text(json.dumps({'providerProfiles': {'offline': {'kind': 'openai', 'defaultModel': 'unused', 'auth': {'kind': 'none'}}}}))
            settings = root / 'memory.toml'
            settings.write_text(f'[[data_stores]]\nname="empty"\npath="{source}"\n'
                                f'[storage]\ndata_dir="{root / "memory"}"\n'
                                f'[llm]\nprofile="offline"\nsettings="{providers}"\n'
                                '[embedding]\nprovider="local"\n')
            # Exercise installed modules without synthetic import-path shims.
            env = {key: value for key, value in os.environ.items()
                   if key not in {'PYTHONPATH', 'AGENT_CORE_PATH'}
                   and not key.startswith('AGENTIC_MEMORY_')}
            config_path = root / 'scheduler.toml'
            config_path.write_text(f'[service]\ndata_dir="{root / "service"}"\n[[stores]]\nname="empty"\nenabled=true\npoll_seconds=14400\nsettings="{settings}"\n')
            result = subprocess.run([sys.executable, '-m', 'agentic_memory.cli', 'scheduler', '--config', str(config_path),
                                     'ingest-once'], capture_output=True, text=True, timeout=15, env=env)
            diagnostic = result.stderr + result.stdout + ''.join(
                path.read_text() for path in (root / 'service' / 'runs').glob('*.log'))
            self.assertEqual(result.returncode, 0, diagnostic)
            completed = json.loads(result.stdout)
            self.assertEqual(completed['outcome'], 'success')
            self.assertEqual(completed['counts'], {'success': 1})
            self.assertEqual(result.stderr, '')
            self.assertEqual(len(result.stdout.splitlines()), 1)
            # Repeating immediately must not obey the old four-hour poll.
            again = subprocess.run([sys.executable, '-u', '-m', 'agentic_memory.cli', 'scheduler',
                                    '--config', str(config_path), 'ingest-once'],
                                   capture_output=True, text=True, timeout=15, env=env)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(json.loads(again.stdout)['counts'], {'success': 1})


if __name__ == '__main__':
    unittest.main()
