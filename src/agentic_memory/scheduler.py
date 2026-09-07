"""Durable local scheduler and isolated worker entry point."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import signal
import shlex
import sqlite3
import subprocess
import sys
import threading
import time

from .config import load_config
from .embed import LocalEmbedder, OpenAICompatEmbedder, make_embedder
from .ingest import ingest
from .llm import AgentCoreBackend
from .locks import IngestionBusy
from .provider_limits import LimitedBackend, ModelBudget, ProviderGate, process_alive, provider_key
from .scheduler_config import load_service
from .scheduler_queue import WorkQueue
from .store import MemoryStore

ROOT = Path(__file__).resolve().parents[2]


def _gate(config, *, readonly=False):
    return ProviderGate(config.db_path, model_limit=config.model_limit, embedding_limit=config.embedding_limit,
                        provider_limits=config.provider_limits, readonly=readonly)


def service_status(config):
    from .freshness import safe_error, store_status
    queue = WorkQueue(config, readonly=True)
    result = {**queue.status(), 'providers': _gate(config, readonly=True).status(), 'stores': {}}
    for scheduled in config.stores:
        store = None
        try:
            cfg = load_config(scheduled.settings)
            import tomllib
            declared = Path(tomllib.loads(scheduled.settings.read_text())['storage']['data_dir']).expanduser()
            if cfg.data_dir.resolve() != declared.resolve():
                raise ValueError('environment overrides conflict with this store')
            if not cfg.db_path.exists():
                result['stores'][scheduled.name] = {'status': 'not_initialized'}
                continue
            store = MemoryStore(cfg.db_path, readonly=True)
            result['stores'][scheduled.name] = store_status(store, cfg)
        except Exception as exc:
            result['stores'][scheduled.name] = {'status': 'unavailable', 'error': safe_error(exc)}
        finally:
            if store:
                store.close()
    for row in result['jobs']:
        row['last_error'] = safe_error(row['last_error'])
    for row in result['recent_runs']:
        row['error'] = safe_error(row['error'])
    return result


def worker(config_path: Path, run_id: str) -> int:
    config = load_service(config_path)
    queue = WorkQueue(config)
    run = queue.get_run(run_id)
    if not run:
        raise ValueError('unknown scheduled run')
    scheduled = config.store(run['instance'])
    queue.start(run_id, os.getpid())
    gate = _gate(config)
    stop = threading.Event()
    def beat():
        while not stop.wait(5):
            try:
                queue.heartbeat(run_id)
            except Exception:
                pass
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    store = None
    try:
        cfg = load_config(scheduled.settings)
        # Keep each destination explicit even in a non-Codex instance.
        import tomllib
        declared = tomllib.loads(scheduled.settings.read_text())['storage']['data_dir']
        if cfg.data_dir.resolve() != Path(declared).expanduser().resolve():
            raise ValueError('environment overrides conflict with scheduled store settings')
        if any(ds.type == 'codex' for ds in cfg.data_stores):
            from .codex import load_dedicated_config
            cfg = load_dedicated_config(scheduled.settings)
        workers = scheduled.workers or cfg.llm_workers
        maximum = scheduled.model_calls_per_run if run['kind'] == 'ingest' else scheduled.aggregate_calls_per_run
        held = []
        if scheduled.hold_text_file:
            held = json.loads(scheduled.hold_text_file.read_text())
            if not isinstance(held, list) or any(not isinstance(p, str) or not p for p in held):
                raise ValueError('operator export holds must be a list of nonempty strings')
        cfg = replace(cfg, incremental_ingestion=True, distillation_cache=True,
                      llm_workers=workers, segment_workers=workers,
                      codex=replace(cfg.codex, batch_size=max(cfg.codex.batch_size, min(workers, scheduled.sources_per_run))),
                      max_model_calls_per_run=maximum, ingestion_hold_texts=held,
                      source_retry_backoff=not run['force_retry'])
        store = MemoryStore(cfg.db_path)
        embedder = make_embedder(cfg)
        if cfg.embed_provider != 'local' and isinstance(embedder, LocalEmbedder):
            raise ValueError('configured embedding provider is unavailable; scheduled workers do not change embedding spaces')
        if isinstance(embedder, OpenAICompatEmbedder):
            key = provider_key(cfg, cfg.embed_provider)
            embedder.request_guard = lambda: gate.acquire(key, 'embedding', owner=run_id, instance=scheduled.name)
        budget = ModelBudget(maximum)
        primary = AgentCoreBackend(cfg, probe=False)
        llm = LimitedBackend(primary, gate, budget, provider=provider_key(cfg, cfg.agent_core_profile), owner=run_id, instance=scheduled.name)
        fallback = None
        if cfg.agent_core_fallback_profile:
            backend = AgentCoreBackend.for_role(cfg, 'fallback')
            fallback = LimitedBackend(backend, gate, budget, provider=provider_key(cfg, cfg.agent_core_fallback_profile), owner=run_id, instance=scheduled.name)
        levels = None if run['kind'] == 'ingest' else ('project',) if run['kind'] == 'profiles' else ('digest',)
        if levels:
            store.bootstrap_aggregate_tracking(levels)
        stats = ingest(cfg, store, embedder, llm, fallback,
                       limit=scheduled.sources_per_run if run['kind'] == 'ingest' else None,
                       refresh_aggregates=False if run['kind'] == 'ingest' else True,
                       aggregates_only=run['kind'] != 'ingest', aggregate_levels=levels)
        result = asdict(stats)
        result['model_calls'] = budget.calls
        result['held_sources'] = store.db.execute("SELECT COUNT(*) FROM ingest_sources WHERE status='held'").fetchone()[0]
        result['failed_sources'] = store.db.execute("SELECT COUNT(*) FROM ingest_sources WHERE status='error'").fetchone()[0]
        if run['kind'] == 'ingest':
            remaining = stats.sources_remaining + store.db.execute("SELECT COUNT(*) FROM ingest_sources WHERE status IN ('pending_work','budget_pending','source_changed','processing')").fetchone()[0]
        else:
            remaining = sum(r['level'] in levels for r in store.pending_aggregates(float('inf')))
        outcome = 'partial' if stats.errors else 'success'
        if stats.errors and budget.calls and not stats.distilled and run['kind'] == 'ingest':
            outcome = 'error'
        queue.finish(run_id, outcome=outcome, stats=result, remaining=remaining,
                     error=f'{stats.errors} source/aggregate failures; see store status' if stats.errors else None)
        print(json.dumps({'run_id': run_id, 'instance': scheduled.name, 'kind': run['kind'],
                          'outcome': outcome, 'remaining': remaining, 'stats': result}), flush=True)
        return 0 if outcome in ('success', 'partial') else 1
    except IngestionBusy:
        queue.finish(run_id, outcome='busy', error='another ingester owns this store')
        return 0
    except BaseException as exc:
        queue.finish(run_id, outcome='error', error=f'{type(exc).__name__}: {str(exc)[:400]}')
        return 1
    finally:
        stop.set()
        thread.join(timeout=1)
        gate.release_owner(run_id)
        if store:
            store.close()


def _worker_environment():
    # Routing belongs to the instance file, while normal provider credential
    # environment variables remain available without being copied to reports.
    env = {k: v for k, v in os.environ.items() if not k.startswith('AGENTIC_MEMORY_') and k != 'AGENT_CORE_PATH'}
    env['PYTHONUNBUFFERED'] = '1'
    return env


def spawn_worker(config, run):
    directory = config.data_dir / 'runs'
    directory.mkdir(mode=0o700, exist_ok=True)
    log = directory / (run['id'] + '.log')
    with log.open('xb') as output:
        os.fchmod(output.fileno(), 0o600)
        process = subprocess.Popen([sys.executable, '-u', '-m', 'agentic_memory.cli', 'scheduler-worker',
                                    '--config', str(config.path), '--run-id', run['id']],
                                   cwd=ROOT, env=_worker_environment(), stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    return process


@contextmanager
def supervisor_lock(config):
    config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = config.data_dir / 'supervisor.lock'
    path.touch(mode=0o600, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IngestionBusy('the memory scheduler is already running') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def owned_worker(pid, identifier):
    if not process_alive(pid):
        return False
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True, timeout=3)
    try:
        args = shlex.split(result.stdout.strip())
        return (len(args) >= 9 and Path(args[0]).name.startswith('python')
                and args[1:5] == ['-u', '-m', 'agentic_memory.cli', 'scheduler-worker']
                and args[args.index('--run-id') + 1] == identifier)
    except (ValueError, IndexError):
        return False


def stop_run(config, identifier, *, force=False):
    queue = WorkQueue(config)
    run = queue.get_run(identifier)
    if not run or not run['pid'] or not owned_worker(run['pid'], identifier):
        return False
    if os.getpgid(run['pid']) != run['pid']:
        raise RuntimeError('worker is not the expected dedicated process-group leader')
    os.killpg(run['pid'], signal.SIGKILL if force else signal.SIGTERM)
    return True


def serve(config_path: Path, *, stop_event=None):
    config = load_service(config_path)
    queue = WorkQueue(config)
    gate = _gate(config)
    stop_event = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    active = {}
    modified = {s.name: s.settings.stat().st_mtime_ns for s in config.stores}
    started = time.time()
    last_heartbeat = 0
    with supervisor_lock(config):
        while not stop_event.is_set():
            try:
                if time.time() - last_heartbeat >= 5:
                    queue.supervisor_heartbeat(os.getpid(), started)
                    last_heartbeat = time.time()
                for identifier, process in list(active.items()):
                    if process.poll() is not None:
                        active.pop(identifier)
                        run = queue.get_run(identifier)
                        if run and run['outcome'] in ('starting', 'running'):
                            queue.finish(identifier, outcome='worker_lost', error=f'worker exited with code {process.returncode}')
                            gate.release_owner(identifier)
                for identifier in queue.recover():
                    gate.release_owner(identifier)
                for job in queue.status()['jobs']:
                    if job['state'] == 'running' and job['pid']:
                        if not owned_worker(job['pid'], job['run_id']):
                            queue.finish(job['run_id'], outcome='worker_lost', error='recorded worker ownership is no longer valid')
                            gate.release_owner(job['run_id'])
                        elif time.time() - job['started'] > config.job_timeout_seconds:
                            stop_run(config, job['run_id'], force=time.time() - job['started'] > config.job_timeout_seconds + 30)
                for instance in config.stores:
                    try:
                        current = instance.settings.stat().st_mtime_ns
                    except OSError:
                        continue
                    if modified[instance.name] != current:
                        modified[instance.name] = current
                        for kind in ('ingest', 'profiles', 'digests'):
                            queue.request(instance.name, kind)
                while len(active) < config.parallel_stores and not stop_event.is_set():
                    run = queue.claim()
                    if run is None:
                        break
                    try:
                        active[run['id']] = spawn_worker(config, run)
                        queue.start(run['id'], active[run['id']].pid, allow_finished=True)
                    except sqlite3.OperationalError:
                        # A spawned worker may already own this run. Keep its
                        # process and durable claim intact while the DB is busy.
                        raise
                    except Exception as exc:
                        queue.finish(run['id'], outcome='error', error=f'worker launch failed: {type(exc).__name__}')
            except sqlite3.OperationalError as exc:
                code = getattr(exc, 'sqlite_errorcode', None)
                if (code is not None and (code & 0xff) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)) or (code is None and str(exc) not in ('database is locked', 'database table is locked')):
                    raise
                print('[scheduler] coordinator busy; retrying without releasing owned work', file=sys.stderr, flush=True)
            stop_event.wait(1)
    # Children keep their own heartbeats and durable ownership. A graceful
    # supervisor stop drains existing work; restart can adopt those workers.


def run_once(config_path: Path, instance: str, kind='ingest') -> int:
    config = load_service(config_path)
    queue = WorkQueue(config)
    for identifier in queue.recover():
        _gate(config).release_owner(identifier)
    run = queue.claim(instance=instance, kind=kind, explicit=True)
    if run is None:
        raise IngestionBusy('this store already has active work')
    child = spawn_worker(config, run)
    queue.start(run['id'], child.pid, allow_finished=True)
    code = child.wait()
    state = queue.get_run(run['id'])
    if state['outcome'] in ('starting', 'running'):
        queue.finish(run['id'], outcome='worker_lost', error=f'worker exited with code {code}')
        _gate(config).release_owner(run['id'])
    result = queue.get_run(run['id'])
    if result['stats']:
        result['stats'] = json.loads(result['stats'])
    print(json.dumps(result, indent=2))
    return code
