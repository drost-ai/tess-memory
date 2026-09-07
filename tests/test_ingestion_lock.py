import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agentic_memory.locks import ingestion_lock
from agentic_memory.store import MemoryStore


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

    def test_process_death_during_publication_keeps_previous_committed_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'memory.db'
            store = MemoryStore(path)
            store.upsert_session(id='one', project='p', summary='last good')
            store.commit()
            code = '''import os,sys
from pathlib import Path
from agentic_memory.store import MemoryStore
s=MemoryStore(Path(sys.argv[1]))
with s.write_transaction():
    s.upsert_session(id='one',project='wrong',summary='partial replacement')
    os._exit(9)
'''
            env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
            child = subprocess.run([sys.executable, '-c', code, str(path)], env=env, capture_output=True, timeout=5)
            self.assertEqual(child.returncode, 9)
            self.assertEqual(store.get_session('one')['summary'], 'last good')
            self.assertEqual(store.get_session('one')['project'], 'p')
            store.close()


if __name__ == '__main__':
    unittest.main()
