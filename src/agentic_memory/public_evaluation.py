"""Provider-backed evidence selection over explicitly supplied public fixtures."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np

from . import retrieval
from .llm import AgentCoreBackend
from .provider_limits import LimitedBackend, ModelBudget, ProviderGate, provider_key
from .search_agent import agentic_search
from .store import MemoryRow, MemoryStore


class FixtureEmbedder:
    descriptor = 'public-fixture:4d'
    def embed_query(self, query):
        return np.array([1, 0, 0, 0], dtype=np.float32)


def evaluate_public(cfg, cases_path: Path, *, workers=4, limit=None):
    document = json.loads(cases_path.read_text())
    if not str(document.get('classification', '')).startswith('public synthetic'):
        raise ValueError('this evaluator requires an explicitly labelled public synthetic fixture file')
    cases = document['cases'][:limit] if limit else document['cases']
    if not cases:
        raise ValueError('no evaluation cases')
    effective = replace(cfg, search_max_rounds=2, search_per_query_limit=12, search_evidence_limit=12)
    backend = AgentCoreBackend.for_role(effective, 'search', probe=False)
    budget = ModelBudget(len(cases) * 4)
    rows = []
    previous = retrieval.CONFIG
    retrieval.CONFIG = effective
    try:
        with tempfile.TemporaryDirectory(prefix='agentic-memory-public-eval-') as tmp:
            root = Path(tmp)
            gate = ProviderGate(root / 'limits.db', model_limit=workers)
            model = LimitedBackend(backend, gate, budget, provider=provider_key(effective, effective.search_profile or effective.agent_core_profile),
                                   owner=uuid.uuid4().hex, instance='public-evaluation')
            def run(case, index):
                store = MemoryStore(root / str(index) / 'memory.db')
                try:
                    store.set_meta('embedder', FixtureEmbedder.descriptor)
                    labels = {}
                    for record in case['records']:
                        sid = None if record.get('level') == 'plan' else case['id'] + ':' + record['key']
                        ts = record.get('ts', '2026-09-05T10:00:00Z')
                        if sid:
                            store.upsert_session(id=sid, project=case['project'], title=record['key'], summary=record['content'],
                                                 outcome='public synthetic fixture', started_at=ts, ended_at=ts)
                            labels['session:' + sid] = record['key']
                        mid = store.add_memories([MemoryRow(level=record.get('level', 'memory'), kind=record['kind'],
                                                           content=record['content'], project=case['project'], session_id=sid,
                                                           ts=ts, importance=0.8, vector=np.array([1, 0, 0, 0], dtype=np.float32))])[0]
                        labels[str(mid)] = record['key']
                    store.commit()
                    started = time.monotonic()
                    actions = []
                    class CaseModel:
                        def complete(self, **kwargs):
                            response = model.complete(**kwargs)
                            actions.append(response)
                            return response
                    hits, trace = agentic_search(effective, store, FixtureEmbedder(), CaseModel(), case['query'], project=case['project'])
                    selected = {labels.get(str(h['id']), 'unrecognized') for h in hits}
                    required, allowed = set(case['required']), set(case['allowed'])
                    unsupported = selected - allowed
                    passed = (required <= selected and not unsupported and (bool(selected) if required else not selected)
                              and trace.model_finished and not trace.fallback_used)
                    return {'id': case['id'], 'category': case['category'], 'passed': passed,
                            'selected': sorted(selected), 'missing': sorted(required - selected),
                            'unsupported': sorted(unsupported), 'rounds': trace.rounds,
                            'seconds': round(time.monotonic() - started, 3), 'note': trace.note,
                            'fallback_used': trace.fallback_used, 'model_failures': trace.failures,
                            'model_actions': actions}
                finally:
                    store.close()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run, case, i) for i, case in enumerate(cases)]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    print(json.dumps({'case': row['id'], 'passed': row['passed'], 'rounds': row['rounds']}), flush=True)
    finally:
        retrieval.CONFIG = previous
    negatives = [r for r in rows if r['category'] == 'unsupported']
    return {'cases': len(rows), 'passed': sum(r['passed'] for r in rows), 'model': backend.name,
            'model_operations': budget.calls, 'private_history_sent': False,
            'embedding_provider_calls': 0, 'fixture_families': 5,
            'negative_false_positive_rate': sum(bool(r['selected']) for r in negatives) / len(negatives) if negatives else None,
            'temporal_passes': sum(r['passed'] for r in rows if r['category'] == 'temporal'),
            'results': sorted(rows, key=lambda r: r['id']),
            'scope': 'Public synthetic evidence-selection regression; not private-corpus semantic recall or answer generation.'}
