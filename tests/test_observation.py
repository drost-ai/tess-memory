from pathlib import Path
import tempfile
import unittest

from agentic_memory.observation import operation_peaks, private_json
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


if __name__ == '__main__':
    unittest.main()
