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

from .config import load_config, load_manifest
from .embed import LocalEmbedder, OpenAICompatEmbedder, make_embedder
from .ingest import ingest
from .llm import AgentCoreBackend
from .locks import IngestionBusy
from .provider_limits import LimitedBackend, ModelBudget, ProviderGate, process_alive, provider_key
from .scheduler_config import load_service
from .scheduler_queue import WorkQueue
from .store import MemoryStore

# Legacy launchd installer metadata only; workers never use this as an import cwd.
ROOT = Path(__file__).resolve().parents[2]


def _scheduled_config(settings: Path):
    """Keep canonical embedded providers; instance TOML selects source/storage only."""
    manifest = os.environ.get('AGENTIC_MEMORY_CONFIG')
    if manifest:
        return load_manifest(Path(manifest), store_settings=settings)
    return load_config(settings)


def _gate(config, *, readonly=False):
    return ProviderGate(config.provider_gate_db or config.db_path, model_limit=config.model_limit, embedding_limit=config.embedding_limit,
                        provider_limits=config.provider_limits, readonly=readonly)


def service_status(config):
    from .freshness import safe_error, store_status
    queue = WorkQueue(config, readonly=True)
    result = {**queue.status(), 'providers': _gate(config, readonly=True).status(), 'stores': {}}
    for scheduled in config.stores:
        store = None
        try:
            cfg = _scheduled_config(scheduled.settings)
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
        if row['kind'] == 'recovery':
            scheduled = config.store(row['instance'])
            row['schedule'] = {'hour': scheduled.recovery_hour, 'timezone': scheduled.recovery_timezone,
                               'model_calls_per_night': scheduled.recovery_model_calls_per_run,
                               'source_reads_per_night': scheduled.recovery_sources_per_run,
                               'minimum_idle_seconds': scheduled.recovery_min_age_seconds,
                               'unfinished_work': 'retained for the next nightly pass'}
    for row in result['recent_runs']:
        row['error'] = safe_error(row['error'])
    return result


def run_outcome(stats, model_calls, kind):
    if not stats.errors:
        return 'success'
    # Retained map/reduction results are progress even before a large session
    # can publish. An unrelated malformed source must not turn that into a
    # total worker failure and exponentially postpone all remaining sources.
    if kind in ('ingest', 'recovery') and model_calls and not stats.distilled and not getattr(stats, 'checkpointed_requests', 0):
        return 'error'
    return 'partial'


def worker(config_path: Path, run_id: str) -> int:
    config = load_service(config_path)
    queue = WorkQueue(config)
    run = queue.get_run(run_id)
    if not run:
        raise ValueError('unknown scheduled run')
    scheduled = config.store(run['instance'])
    queue.start(run_id, os.getpid())
    gate = _gate(config)
    is_recovery = run['kind'] == 'recovery'
    is_ingest = run['kind'] in ('ingest', 'recovery')
    maximum = scheduled.recovery_model_calls_per_run if is_recovery else scheduled.model_calls_per_run if is_ingest else scheduled.aggregate_calls_per_run
    # Stop admission before the supervisor's soft deadline. A request already
    # running can finish and checkpoint; neither a model-budget boundary nor
    # SIGTERM should throw away its paid output.
    reserve = min(60, config.job_timeout_seconds / 10)
    remaining_time = run['started'] + config.job_timeout_seconds - reserve - time.time()
    budget = ModelBudget(maximum, deadline=time.monotonic() + max(0, remaining_time))
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.signal(sig, lambda *_: budget.draining.set())
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
        cfg = _scheduled_config(scheduled.settings)
        # Keep each destination explicit even in a non-Codex instance.
        import tomllib
        declared = tomllib.loads(scheduled.settings.read_text())['storage']['data_dir']
        if cfg.data_dir.resolve() != Path(declared).expanduser().resolve():
            raise ValueError('environment overrides conflict with scheduled store settings')
        if any(ds.type == 'codex' for ds in cfg.data_stores):
            from .codex import load_dedicated_config, validate_dedicated_database
            if cfg.embedded:
                if any(ds.type != 'codex' for ds in cfg.data_stores):
                    raise ValueError('Codex memory requires exclusively Codex sources')
                validate_dedicated_database(cfg.db_path)
            else:
                cfg = load_dedicated_config(scheduled.settings)
        workers = scheduled.workers or cfg.llm_workers
        if run['kind'] == 'detect':
            from .detection import detect_sources
            result = detect_sources(cfg, stop_requested=budget.draining.is_set)
            outcome = 'partial' if result['errors'] else 'success'
            queue.finish(run_id, outcome=outcome, stats=result)
            print(json.dumps({'run_id': run_id, 'instance': scheduled.name, 'kind': 'detect',
                              'outcome': outcome, 'stats': result}), flush=True)
            return 0
        held = []
        if scheduled.hold_text_file:
            held = json.loads(scheduled.hold_text_file.read_text())
            if not isinstance(held, list) or any(not isinstance(p, str) or not p for p in held):
                raise ValueError('operator export holds must be a list of nonempty strings')
        cfg = replace(cfg, incremental_ingestion=True, distillation_cache=True,
                      llm_workers=workers, segment_workers=workers,
                      codex=replace(cfg.codex, batch_size=max(cfg.codex.batch_size, min(workers, scheduled.sources_per_run))),
                      max_model_calls_per_run=maximum, ingestion_hold_texts=held,
                      source_retry_backoff=not run['force_retry'],
                      recovery_ingestion=is_recovery, recovery_min_age_seconds=scheduled.recovery_min_age_seconds)
        store = MemoryStore(cfg.db_path)
        embedder = make_embedder(cfg)
        if cfg.embed_provider != 'local' and isinstance(embedder, LocalEmbedder):
            raise ValueError('configured embedding provider is unavailable; scheduled workers do not change embedding spaces')
        if isinstance(embedder, OpenAICompatEmbedder):
            from .model_usage import check_source_admission
            key = provider_key(cfg, cfg.embed_provider)
            embedder.request_guard = lambda: gate.acquire(key, 'embedding', owner=run_id, instance=scheduled.name,
                                                         check_admission=check_source_admission)
        primary = AgentCoreBackend(cfg, probe=False)
        llm = LimitedBackend(primary, gate, budget, provider=provider_key(cfg, cfg.agent_core_profile), owner=run_id, instance=scheduled.name)
        fallback = None
        if cfg.agent_core_fallback_profile:
            backend = AgentCoreBackend.for_role(cfg, 'fallback')
            fallback = LimitedBackend(backend, gate, budget, provider=provider_key(cfg, cfg.agent_core_fallback_profile), owner=run_id, instance=scheduled.name)
        levels = None if is_ingest else ('project',) if run['kind'] == 'profiles' else ('digest',)
        if levels:
            store.bootstrap_aggregate_tracking(levels)
        from .model_usage import request_metadata
        with request_metadata(phase=run['kind']):
            stats = ingest(cfg, store, embedder, llm, fallback,
                           limit=scheduled.recovery_sources_per_run if is_recovery else scheduled.sources_per_run if is_ingest else None,
                           refresh_aggregates=False if is_ingest else True,
                           aggregates_only=not is_ingest, aggregate_levels=levels)
            if is_recovery:
                # Share the same nightly admission budget with aggregate repair.
                # Reserving separate scheduled aggregate jobs prevents a large
                # tail backlog from starving them if this budget is exhausted.
                from .ingest import _refresh_aggregates
                store.bootstrap_aggregate_tracking(('project', 'digest'))
                _refresh_aggregates(cfg, store, embedder, llm, fallback, stats, 0)
        result = asdict(stats)
        result['model_calls'] = budget.calls
        result['deadline_drained'] = budget.draining.is_set()
        usage = gate.usage(run_id=run_id)
        result['usage'] = usage
        result['usage_recording_errors'] = budget.telemetry_errors
        for name in ('input_tokens', 'output_tokens'):
            values = [g[name] for g in usage.get('groups', []) if g[name] is not None]
            result[name] = sum(values) if values else None
        result['held_sources'] = store.db.execute("SELECT COUNT(*) FROM ingest_sources WHERE status='held'").fetchone()[0]
        result['failed_sources'] = store.db.execute("SELECT COUNT(*) FROM ingest_sources WHERE status='error'").fetchone()[0]
        if is_ingest:
            remaining = stats.sources_remaining + stats.sources_pending
        else:
            remaining = sum(r['level'] in levels for r in store.pending_aggregates(float('inf')))
        outcome = run_outcome(stats, budget.calls, run['kind'])
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
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)


def _worker_environment():
    # Preserve only the explicitly selected embedded manifest, never ambient
    # memory overrides or source-checkout import paths. Credential refs remain
    # ordinary environment variables and are never copied into reports.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('AGENTIC_MEMORY_') and k not in {'AGENT_CORE_PATH', 'PYTHONPATH'}}
    manifest = os.environ.get('AGENTIC_MEMORY_CONFIG')
    if manifest:
        if not Path(manifest).is_absolute():
            raise ValueError('AGENTIC_MEMORY_CONFIG must be an absolute generated manifest path')
        env['AGENTIC_MEMORY_CONFIG'] = manifest
    env['PYTHONUNBUFFERED'] = '1'
    return env


def spawn_worker(config, run):
    # Validate metadata before opening the log or launching a worker. Installed
    # modules must work from the runtime home, not the memory source checkout.
    embedded = bool(os.environ.get('AGENTIC_MEMORY_CONFIG'))
    cfg = _scheduled_config(config.store(run['instance']).settings) if embedded else None
    cwd = cfg.runtime_home if embedded else config.path.parent
    directory = config.data_dir / 'runs'
    directory.mkdir(mode=0o700, exist_ok=True)
    log = directory / (run['id'] + '.log')
    with log.open('xb') as output:
        os.fchmod(output.fileno(), 0o600)
        process = subprocess.Popen([sys.executable, '-u', '-m', 'agentic_memory.cli', 'scheduler-worker',
                                    '--config', str(config.path), '--run-id', run['id']],
                                   cwd=cwd, env=_worker_environment(), stdin=subprocess.DEVNULL,
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


def owned_worker(pid, identifier) -> bool | None:
    """True confirms ownership; None means a live process cannot be checked."""
    if not process_alive(pid):
        return False
    try:
        result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True, timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None if process_alive(pid) else False
    try:
        args = shlex.split(result.stdout.strip())
    except ValueError:
        return None
    try:
        return (len(args) >= 9 and Path(args[0]).name.startswith('python')
                and args[1:5] == ['-u', '-m', 'agentic_memory.cli', 'scheduler-worker']
                and args[args.index('--run-id') + 1] == identifier)
    except (ValueError, IndexError):
        return False


def stop_run(config, identifier, *, force=False):
    queue = WorkQueue(config)
    run = queue.get_run(identifier)
    if not run or not run['pid'] or owned_worker(run['pid'], identifier) is not True:
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
    detectors = {}
    modified = {s.name: s.settings.stat().st_mtime_ns for s in config.stores}
    started = time.time()
    last_heartbeat = 0
    with supervisor_lock(config):
        while not stop_event.is_set():
            try:
                if time.time() - last_heartbeat >= 5:
                    queue.supervisor_heartbeat(os.getpid(), started)
                    last_heartbeat = time.time()
                for processes in (active, detectors):
                    for identifier, process in list(processes.items()):
                        if process.poll() is not None:
                            processes.pop(identifier)
                            run = queue.get_run(identifier)
                            if run and run['outcome'] in ('starting', 'running'):
                                queue.finish(identifier, outcome='worker_lost', error=f'worker exited with code {process.returncode}')
                                gate.release_owner(identifier)
                for identifier in queue.recover():
                    gate.release_owner(identifier)
                for job in queue.status()['jobs']:
                    if job['state'] == 'running' and job['pid']:
                        ownership = owned_worker(job['pid'], job['run_id'])
                        if ownership is None:
                            print(f"[scheduler] worker {job['pid']} identity check unavailable; retaining ownership and retrying", file=sys.stderr, flush=True)
                            continue
                        if ownership is False:
                            queue.finish(job['run_id'], outcome='worker_lost', error='recorded worker ownership is no longer valid')
                            gate.release_owner(job['run_id'])
                        elif time.time() - job['started'] > config.job_timeout_seconds:
                            stop_run(config, job['run_id'], force=time.time() - job['started'] > config.job_timeout_seconds + config.job_drain_seconds)
                for instance in config.stores:
                    try:
                        current = instance.settings.stat().st_mtime_ns
                    except OSError:
                        continue
                    if modified[instance.name] != current:
                        modified[instance.name] = current
                        for kind in ('ingest', 'profiles', 'digests', 'detect'):
                            queue.request(instance.name, kind)
                for kind, processes in (('detect', detectors), (None, active)):
                    while len(processes) < config.parallel_stores and not stop_event.is_set():
                        run = queue.claim(kind=kind) if kind else queue.claim()
                        if run is None:
                            break
                        try:
                            processes[run['id']] = spawn_worker(config, run)
                            queue.start(run['id'], processes[run['id']].pid, allow_finished=True)
                        except sqlite3.OperationalError:
                            # A spawned worker may already own this run. Keep
                            # its process and durable claim while the DB is busy.
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
