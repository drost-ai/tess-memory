import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agentic_memory.locks import ingestion_lock


class IngestionLockTests(unittest.TestCase):
    def test_another_process_is_excluded_only_from_the_same_store(self):
        code = """from pathlib import Path
import sys
from agentic_memory.locks import ingestion_lock, IngestionBusy
try:
    with ingestion_lock(Path(sys.argv[1])):
        pass
except IngestionBusy:
    raise SystemExit(23)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
            def check(path):
                return subprocess.run([sys.executable, '-c', code, str(path)], env=env, timeout=5,
                                      capture_output=True).returncode
            with ingestion_lock(root / 'one'):
                self.assertEqual(check(root / 'one'), 23)
                self.assertEqual(check(root / 'two'), 0)
            self.assertEqual(check(root / 'one'), 0)


if __name__ == '__main__':
    unittest.main()
