from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from agentic_memory.observation import operation_peaks, private_json, observe
from agentic_memory.provider_limits import ProviderGate
from agentic_memory.scheduler_config import ServiceConfig


class ObservationTests(unittest.TestCase):
    def test_receipts_measure_peak_overlap_and_detect_limit_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ServiceConfig(path=root/'config.toml', data_dir=root, stores=(), model_limit=2,
                                provider_limits={'p': 2})
            gate = ProviderGate(cfg.db_path)
            with gate.connect() as db:
                db.executemany('INSERT INTO call_receipts VALUES(?,?,?,?,?,?,?)', [
                    ('a', 'one', 'p', 'model', 10, 14, 'success'),
                    ('b', 'two', 'p', 'model', 11, 15, 'success'),
                    ('c', 'three', 'p', 'model', 15, 16, 'success')])
            result = operation_peaks(cfg, 10)
            self.assertEqual(result['global']['model'], 2)
            self.assertTrue(result['within_limits'])
            with gate.connect() as db:
                db.execute('INSERT INTO call_receipts VALUES(?,?,?,?,?,?,?)', ('d', 'four', 'p', 'model', 12, 13, 'success'))
            self.assertFalse(operation_peaks(cfg, 10)['within_limits'])

    def test_observation_report_is_written_privately(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'nested'/'status.json'
            private_json(path, {'status': 'observing'})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIn('observing', path.read_text())

    def run_observer(self, statuses):
        now = [100.0]
        def sleep(seconds):
            now[0] += seconds
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'status.json'
            with mock.patch('agentic_memory.observation.load_service', return_value=SimpleNamespace()), \
                 mock.patch('agentic_memory.observation.service_status', side_effect=statuses), \
                 mock.patch('agentic_memory.observation.operation_peaks', return_value={'within_limits': True}), \
                 mock.patch('agentic_memory.observation.time.time', side_effect=lambda: now[0]), \
                 mock.patch('agentic_memory.observation.time.sleep', side_effect=sleep):
                result = observe(Path('config'), path, seconds=1, interval=1)
            history = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
            return result, history

    @staticmethod
    def healthy(enabled=True):
        return {'jobs': [{'instance': 'a', 'kind': 'ingest', 'enabled': enabled, 'last_success': 101}],
                'supervisor': {'process_alive': True, 'heartbeat_age_seconds': 0},
                'stores': {'a': {'sessions': 1, 'ingestion': {}}}, 'providers': {'active': []}}

    def test_transient_busy_sample_is_recorded_and_observation_recovers(self):
        result, history = self.run_observer([sqlite3.OperationalError('database is locked'), self.healthy()])
        self.assertTrue(result['passed'])
        self.assertEqual(result['sampling_errors'], 1)
        self.assertEqual(history[0]['sampling_error'], 'database is locked')
        self.assertIsNone(history[0]['supervisor'])
        self.assertEqual(result['max_consecutive_unhealthy_samples'], 1)

    def test_disabling_a_store_cannot_make_final_acceptance_vacuously_pass(self):
        result, _ = self.run_observer([self.healthy(), self.healthy(enabled=False)])
        self.assertFalse(result['passed'])
        self.assertEqual(result['disabled_during_observation'], ['a'])

    def test_failed_final_sample_is_reported_as_failed_acceptance(self):
        result, _ = self.run_observer([self.healthy(), sqlite3.OperationalError('database is locked')])
        self.assertFalse(result['passed'])
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['operation_peaks'])
        self.assertFalse(result['supervisor_healthy_at_finish'])


if __name__ == '__main__':
    unittest.main()
