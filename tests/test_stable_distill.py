import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from agentic_memory.config import Config
from agentic_memory.distill import DistillationPool, SESSION_SYSTEM
from agentic_memory.readers import ParsedSource
from agentic_memory.stable_distill import stable_digest, stable_map_prompts
from agentic_memory.transcript import SessionTranscript, Turn


class StableReductionTests(unittest.TestCase):
    def test_parallel_reductions_keep_serial_request_keys_and_resume_successful_siblings(self):
        st = SessionTranscript('one', 'p', '/unused')
        st.turns = [Turn('user', f'Unique request {i}') for i in range(8)]
        snapshot = ParsedSource(st, list(range(1, 9)), ['turn_complete'] * 8, [], False, '', 0)
        cfg = Config(segment_chars=3000, segment_overlap=0, incremental_reduce_fanout=2,
                     segment_workers=4, llm_workers=4)
        barrier = threading.Barrier(4)
        class Backend:
            name = 'stub'
            calls = []
            failed = False
            parallel = True
            def complete(self, *, system, prompt):
                self.calls.append(prompt)
                if self.parallel and 'unit-' in prompt:
                    barrier.wait(timeout=3)
                    if 'unit-0' in prompt and not self.failed:
                        self.failed = True
                        raise RuntimeError('interrupted reduction')
                return json.dumps({'summary': hashlib.sha256(prompt.encode()).hexdigest(), 'memories': []})
        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.db'
            with DistillationPool(backend, workers=4, cache_path=path) as pool:
                calls = pool.session('one')
                for i, prompt in enumerate(stable_map_prompts(cfg, snapshot)):
                    pool.cache.put(pool.cache.key('one', backend.name, SESSION_SYSTEM, prompt),
                                   {'summary': f'unit-{i}', 'memories': []})
                with self.assertRaises(RuntimeError):
                    stable_digest(cfg, snapshot, calls)
                self.assertEqual(pool.checkpointed_requests, 3)
            backend.parallel = False
            with DistillationPool(backend, workers=4, cache_path=path) as pool:
                result = stable_digest(cfg, snapshot, pool.session('one'))
                self.assertEqual(pool.calls, 4)  # failed sibling plus two parents and root
            cfg.segment_workers = cfg.llm_workers = 1
            with DistillationPool(backend, workers=1, cache_path=path) as pool:
                serial = stable_digest(cfg, snapshot, pool.session('one'))
                self.assertEqual(pool.calls, 0)
                self.assertEqual(result.summary, serial.summary)


if __name__ == '__main__':
    unittest.main()
