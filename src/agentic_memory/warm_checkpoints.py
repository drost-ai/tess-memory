"""Promote existing incremental map checkpoints without any provider calls."""
from contextlib import ExitStack
import json
from types import SimpleNamespace

from .attribution import source_adapter
from .distill import DistillationPool, SESSION_SYSTEM
from .locks import ingestion_lock
from .readers import CompletedSourceReader
from .stable_distill import stable_map_prompts
from .store import MemoryStore


def warm_incremental_checkpoints(cfg, primary, fallback=None, *, session_ids=None, report=lambda _: None):
    """Only reader/checkpoint caches may change; published memory stays read-only.

    Call with the same effective parser/model settings as the ingestion worker.
    By default, visit unfinished sources and published incremental sources with
    request keys. Sources never read by the incremental pipeline are not scanned.
    Missing requests stay missing: this function cannot construct a model call.
    """
    def identity_only(backend):
        if backend is None:
            return None
        return SimpleNamespace(name=backend.name, cache_identity=getattr(backend, 'cache_identity', backend.name))

    results = []
    with ExitStack() as stack:
        stack.enter_context(ingestion_lock(cfg.data_dir))
        if any(ds.type == 'codex' for ds in cfg.data_stores):
            from .codex import ingestion_lock as codex_lock
            stack.enter_context(codex_lock(cfg.data_dir))
        store = MemoryStore(cfg.db_path, readonly=True)
        stack.callback(store.close)
        progress = {r['path']: dict(r) for r in store.db.execute('SELECT * FROM ingest_sources')}
        reader = CompletedSourceReader(cfg)
        pool = stack.enter_context(DistillationPool(identity_only(primary), identity_only(fallback), workers=1,
                                  cache_path=cfg.data_dir/'distillation-cache.db', share_requests=True))
        for ds in cfg.data_stores:
            for path, project, parent in source_adapter(ds, codex_catalog=reader.catalog)[0]:
                old = progress.get(str(path), {})
                state = json.loads(old.get('state', '{}'))
                if old.get('status') == 'held':
                    continue
                if session_ids is not None:
                    if old.get('session_id') not in session_ids:
                        continue
                elif not state.get('request_keys') and old.get('status') not in ('processing', 'budget_pending', 'pending_work'):
                    continue
                report({'phase': 'reading', 'session_id': old.get('session_id')})
                snapshot = reader.read(ds, path, ds.resolve_project(project), parent)
                st = snapshot.transcript
                if st is None or any(pattern in t.text for pattern in cfg.ingestion_hold_texts for t in st.turns):
                    continue
                seed = state.get('seed') or {}
                units = seed.get('units', [])
                seed_units = len(units) if snapshot.unit_keys()[:len(units)] == units else 0
                calls = pool.session(f'incremental/{st.source_type}/{st.session_id}')
                checked = found = 0
                before_promotions, before_shared = pool.cache_promotions, pool.shared_cache_hits
                for prompt in stable_map_prompts(cfg, snapshot, seed_units=seed_units):
                    checked += 1
                    found += calls.cached_result(SESSION_SYSTEM, prompt, 'segment stable') is not None
                pool.cache.add_pins(calls.namespace, calls.keys)
                row = {'session_id': st.session_id, 'map_requests': checked, 'cached_requests': found,
                       'missing_requests': checked - found, 'promoted_requests': pool.cache_promotions - before_promotions,
                       'shared_hits': pool.shared_cache_hits - before_shared}
                results.append(row)
                report({'phase': 'warmed', **row})
        assert pool.calls == 0
        return {'sessions': len(results), 'map_requests': sum(r['map_requests'] for r in results),
                'cached_requests': sum(r['cached_requests'] for r in results),
                'promoted_requests': pool.cache_promotions, 'shared_hits': pool.shared_cache_hits,
                'model_calls': 0, 'embedding_calls': 0, 'results': results}
