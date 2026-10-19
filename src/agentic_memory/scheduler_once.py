"""Finite, externally scheduled ingestion batch (not another scheduler)."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import signal
import threading
import time

from .locks import IngestionBusy
from .scheduler import _gate, spawn_worker, supervisor_lock
from .scheduler_config import load_service
from .scheduler_queue import WorkQueue

# Leave the host enough time to reap us after terminating our worker groups.
MAX_SHUTDOWN_GRACE = 60.0
POLL_SECONDS = 0.1
_TERMINAL = {'success', 'partial', 'busy', 'error', 'worker_lost', 'cancelled', 'timeout'}


def _signal_group(process, sig):
    # These are Popen handles created by this batch with start_new_session=True,
    # never PIDs read from an untrusted/stale coordinator row.
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _cleanup(active, grace):
    """Drain concurrently, then kill groups (including surviving descendants)."""
    for process, _ in active.values():
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while any(process.poll() is None for process, _ in active.values()):
        if time.monotonic() >= deadline:
            break
        time.sleep(POLL_SECONDS)
    for process, _ in active.values():
        _signal_group(process, signal.SIGKILL)
    for process, _ in active.values():
        process.wait()


def ingest_once(config_path: Path) -> int:
    """Run each enabled ingest job at most once and print a safe summary.

    Existing retry backoff, durable leases, worker caps, hold policy and
    provider gates remain authoritative. No explicit/retry override is used.
    """
    cancelled = []
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_signals[sig] = signal.signal(sig, lambda number, _: cancelled.append(number))
    results = []
    outcome, code = 'error', 1
    try:
        config = load_service(config_path)
        # The lock also excludes the legacy persistent supervisor. Queue claims
        # still arbitrate against manual workers and other store ingesters.
        with supervisor_lock(config):
            queue = WorkQueue(config)
            gate = _gate(config)
            for identifier in queue.recover():
                gate.release_owner(identifier)
            enabled = {store.name for store in config.stores if store.enabled}
            now = time.time()
            pending = []
            for job in queue.status()['jobs']:
                if job['instance'] not in enabled or job['kind'] != 'ingest':
                    continue
                if job['state'] == 'running':
                    results.append({'instance': job['instance'], 'status': 'busy'})
                elif job['attempts'] == 0 or job['due'] <= now:
                    pending.append(job['instance'])
                else:
                    results.append({'instance': job['instance'], 'status': 'backoff'})
            active = {}
            timed_out = set()
            grace = min(config.job_drain_seconds, MAX_SHUTDOWN_GRACE)
            try:
                while (pending or active) and not cancelled:
                    for identifier, (process, started) in list(active.items()):
                        elapsed = time.monotonic() - started
                        if process.poll() is None and elapsed >= config.job_timeout_seconds:
                            if identifier not in timed_out:
                                timed_out.add(identifier)
                                _signal_group(process, signal.SIGTERM)
                            if elapsed >= config.job_timeout_seconds + grace:
                                _signal_group(process, signal.SIGKILL)
                        if process.poll() is None:
                            continue
                        process.wait()
                        # A worker must not leave grandchildren behind on exit.
                        _signal_group(process, signal.SIGKILL)
                        state = queue.get_run(identifier)
                        status = state['outcome']
                        if identifier in timed_out:
                            status = 'timeout'
                        elif status not in _TERMINAL or process.returncode != 0:
                            status = 'worker_lost' if status in ('starting', 'running') else 'error'
                        if state['outcome'] in ('starting', 'running'):
                            queue.finish(identifier, outcome=status)
                        gate.release_owner(identifier)
                        results.append({'instance': state['instance'], 'status': status})
                        del active[identifier]
                    while pending and len(active) < config.parallel_stores and not cancelled:
                        instance = pending.pop(0)
                        run = queue.claim(instance=instance, kind='ingest', external_trigger=True)
                        if run is None:
                            results.append({'instance': instance, 'status': 'busy'})
                            continue
                        try:
                            process = spawn_worker(config, run)
                        except Exception:
                            queue.finish(run['id'], outcome='error', error='worker launch failed')
                            results.append({'instance': instance, 'status': 'error'})
                            continue
                        # Register immediately, before any fallible database operation.
                        active[run['id']] = (process, time.monotonic())
                        queue.start(run['id'], process.pid, allow_finished=True)
                    if active and not cancelled:
                        time.sleep(POLL_SECONDS)
                if cancelled:
                    outcome, code = 'cancelled', 128 + cancelled[0]
                elif any(r['status'] in ('error', 'worker_lost', 'timeout') for r in results):
                    outcome, code = 'error', 1
                elif any(r['status'] == 'partial' for r in results):
                    outcome, code = 'partial', 0
                elif any(r['status'] == 'busy' for r in results):
                    outcome, code = 'busy', 0
                else:
                    outcome, code = ('success' if any(r['status'] == 'success' for r in results) else 'no_work'), 0
            finally:
                # Also covers launch/start failures, DB exceptions, and interrupts.
                # Reap first; never release a durable lease while its owner lives.
                _cleanup(active, grace)
                for identifier in active:
                    state = queue.get_run(identifier)
                    status = 'cancelled' if cancelled else 'error'
                    if state['outcome'] in ('starting', 'running'):
                        queue.finish(identifier, outcome=status)
                    gate.release_owner(identifier)
                    results.append({'instance': state['instance'], 'status': status})
    except IngestionBusy:
        outcome, code = 'busy', 0
    except Exception:
        # Logs/exception strings can contain source text, credentials or paths.
        outcome, code = 'error', 1
    finally:
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
    if cancelled:
        outcome, code = 'cancelled', 128 + cancelled[0]
    print(json.dumps({'schema_version': 1, 'command': 'ingest-once',
                      'outcome': outcome, 'counts': dict(Counter(r['status'] for r in results)),
                      'stores': results}), flush=True)
    return code
