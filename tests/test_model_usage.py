from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from agentic_memory.distill import DistillationPool
from agentic_memory.model_usage import UsageObserver, current_observer
from agentic_memory.provider_limits import LimitedBackend, ModelBudget, ProviderGate


class ModelUsageTests(unittest.TestCase):
    def test_wire_details_survive_normalization_and_failed_attempt_usage_is_not_doubled(self):
        records = []
        usage = UsageObserver(records.append)
        usage.configured({'model': 'requested', 'effort': 'xhigh'})
        usage.event('provider.request_started', {'attempt': 1})
        usage.event('provider.retrying', {'rawMessage': 'sensitive error'})
        usage.event('provider.request_started', {'attempt': 2})
        usage.event('provider.stream_event', {'stream_event': 'response.completed', 'payload': {'response': {
            'model': 'reported', 'output': 'private output',
            'usage': {'input_tokens': 100, 'output_tokens': 40, 'input_tokens_details': {'cached_tokens': 60},
                      'output_tokens_details': {'reasoning_tokens': 30}}}}})
        usage.event('provider.response_completed', {'model': 'runtime', 'usage': {'input_tokens': 100, 'output_tokens': 40}})
        usage.finish({'model': 'runtime', 'usage': {'input_tokens': 100, 'output_tokens': 40}, 'text': 'private output'})
        result = usage.data()
        self.assertEqual((result['input_tokens'], result['output_tokens'], result['reasoning_tokens'], result['cache_read_tokens']), (100, 40, 30, 60))
        self.assertEqual((result['wire_attempts'], result['transport_retries']), (2, 1))
        self.assertFalse(result['usage_complete'])  # first attempt usage is unknown
        self.assertEqual(result['response_model'], 'reported')
        self.assertNotIn('private output', json.dumps(records))
        self.assertNotIn('sensitive error', json.dumps(records))

    def test_concurrent_calls_keep_session_attribution_and_json_retry_accounting(self):
        class Backend:
            name = 'fake'
            cache_identity = 'model'
            lock = threading.Lock()
            failed = False
            def complete(self, *, system, prompt):
                observer = current_observer()
                observer.configured({'model': 'm', 'effort': 'xhigh'})
                observer.event('provider.request_started', {})
                result = {'model': 'm', 'usage': {'input_tokens': 10, 'output_tokens': 20}}
                observer.event('provider.response_completed', result)
                observer.finish(result)
                with self.lock:
                    if not self.failed:
                        self.failed = True
                        return 'invalid JSON'
                return '{"summary":"Valid", "memories":[]}'
        with tempfile.TemporaryDirectory() as tmp:
            gate = ProviderGate(Path(tmp)/'gate.db')
            budget = ModelBudget(10)
            backend = LimitedBackend(Backend(), gate, budget, provider='codex', owner='run', instance='store')
            with DistillationPool(backend, workers=2) as pool, ThreadPoolExecutor(2) as sessions:
                futures = [sessions.submit(lambda sid=sid: pool.session(sid).submit('system', sid, 'segment stable').result())
                           for sid in ('session-one', 'session-two')]
                for future in futures:
                    future.result(timeout=3)
            with gate.connect() as db:
                rows = db.execute('select * from model_requests').fetchall()
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['namespace'] for r in rows}, {'session-one', 'session-two'})
            self.assertEqual(sum(r['json_retry'] for r in rows), 1)
            self.assertTrue(all(r['phase'] == 'segment stable' and len(r['request_hash']) == 64 for r in rows))
            group = gate.usage(run_id='run')['groups'][0]
            self.assertEqual((group['calls'], group['input_tokens'], group['output_tokens'], group['json_retries']), (3, 30, 60, 1))
            self.assertEqual(group['calls_with_complete_usage'], 3)
            self.assertIsNone(current_observer())

    def test_missing_usage_remains_unknown_and_old_receipts_remain_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = ProviderGate(Path(tmp)/'gate.db')
            with gate.connect() as db:
                # Existing workers still use this seven-column insert.
                db.execute('INSERT INTO call_receipts VALUES(?,?,?,?,?,?,?)', ('old', 'store', 'p', 'model', 1, 2, 'success'))
            group = gate.usage()['groups'][0]
            self.assertEqual((group['calls'], group['attributed_calls'], group['calls_with_usage']), (1, 0, 0))
            self.assertIsNone(group['input_tokens'])

    def test_usage_groups_distinguish_auth_files_without_storing_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = ProviderGate(Path(tmp)/'gate.db')
            with gate.connect() as db:
                for index, path in enumerate(['/codex/auth.json', '/alt/auth.json']):
                    key = str(index)
                    db.execute('INSERT INTO call_receipts VALUES(?,?,?,?,?,?,?)', (key, 'codex', 'codex', 'model', 1, 2, 'success'))
                    db.execute('INSERT INTO model_requests VALUES(?,?,?,?,?,?,?,?)',
                               (key, 'run', 'session', 'profile', key, 10, 0,
                                json.dumps({'requested_model': 'm', 'auth_source': path, 'input_tokens': 10})))
            groups = gate.usage()['groups']
            self.assertEqual(len(groups), 2)
            self.assertEqual({g['auth_source'] for g in groups}, {'/codex/auth.json', '/alt/auth.json'})

    def test_accounting_failure_does_not_reissue_or_discard_provider_result(self):
        class Backend:
            name = 'fake'
            def complete(self, **kwargs):
                current_observer().finish({'usage': {'input_tokens': 10}, 'text': 'paid'})
                return 'paid'
        with tempfile.TemporaryDirectory() as tmp:
            gate = ProviderGate(Path(tmp)/'gate.db')
            budget = ModelBudget(10)
            backend = LimitedBackend(Backend(), gate, budget, provider='p', owner='r', instance='s')
            # A missing new accounting table does not affect legacy lease/receipt tables.
            with gate.connect() as db:
                db.execute('DROP TABLE model_requests')
            self.assertEqual(backend.complete(system='s', prompt='p'), 'paid')
            self.assertEqual(budget.calls, 1)
            self.assertGreater(budget.telemetry_errors, 0)
