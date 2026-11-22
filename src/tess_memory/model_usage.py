"""Content-free per-request attribution and observed provider usage.

Never retain raw provider events, prompts, completions, reasoning, or credentials.
Missing usage stays unknown. Reasoning/cache tokens are detail fields, not extra
tokens to add to the provider's input/output totals.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_metadata = ContextVar('memory_request_metadata', default={})
_observer = ContextVar('memory_usage_observer', default=None)
_admission_guard = ContextVar('memory_source_admission_guard', default=None)


def source_namespace(source_type, session_id):
    """One formatter for ingestion, warming and source-level receipt audits."""
    if source_type == 'codex':
        session_id = 'codex:' + session_id.removeprefix('codex:')
    elif source_type in {'claude-code', 'tess-core', 'transcript', 'antigravity'}:
        # Existing generic-reader checkpoints use an empty source-type field.
        # Accept configured adapter names in audits without renaming paid keys.
        source_type = ''
    return f'incremental/{source_type}/{session_id}'


@contextmanager
def guard_source_requests(check):
    token = _admission_guard.set(check)
    try:
        yield
    finally:
        _admission_guard.reset(token)


def check_source_admission():
    check = _admission_guard.get()
    if check:
        check()


@contextmanager
def request_metadata(**values):
    token = _metadata.set({**_metadata.get(), **values})
    try:
        yield
    finally:
        _metadata.reset(token)


@contextmanager
def observe_usage(observer):
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def current_metadata():
    return dict(_metadata.get())


def current_observer():
    return _observer.get()


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _count(*values):
    return next((v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0), None)


def usage_counts(raw):
    return {
        'input_tokens': _count(field(raw, 'input_tokens'), field(raw, 'prompt_tokens')),
        'output_tokens': _count(field(raw, 'output_tokens'), field(raw, 'completion_tokens')),
        'reasoning_tokens': _count(field(raw, 'reasoning_tokens'),
                                  field(field(raw, 'output_tokens_details'), 'reasoning_tokens'),
                                  field(field(raw, 'completion_tokens_details'), 'reasoning_tokens')),
        'cache_read_tokens': _count(field(raw, 'cache_read_input_tokens'), field(raw, 'prompt_cache_hit_tokens'),
                                   field(field(raw, 'input_tokens_details'), 'cached_tokens'),
                                   field(field(raw, 'prompt_tokens_details'), 'cached_tokens')),
        'cache_write_tokens': _count(field(raw, 'cache_creation_input_tokens')),
    }


class UsageObserver:
    def __init__(self, persist):
        self.persist = persist
        self.observed = False
        self.attempts = []
        self.retries = 0
        self.runtime_model = self.response_model = None
        self.requested_model = self.requested_effort = None
        self.auth_source = None
        self.final_usage = {}
        self.cleanup_error = None

    def configured(self, config):
        self.observed = True
        self.requested_model = config.get('model')
        self.requested_effort = config.get('effort')
        self.auth_source = config.get('openaiCodexAuthPath') if config.get('openaiApiMode') == 'codex' else None
        self.persist(self.data())

    def event(self, event, data):
        if event == 'provider.request_started':
            self.observed = True
            self.attempts.append({})
        elif event == 'provider.retrying':
            self.retries += 1
        elif event == 'provider.stream_event':
            kind = data.get('stream_event')
            if kind not in {'response.completed', 'response.incomplete', 'response.failed', 'chat.completion.chunk'}:
                return
            payload = data.get('payload')
            response = field(payload, 'response', payload)
            raw = field(response, 'usage')
            if raw is None:
                return
            self._merge(raw)
            self.response_model = field(response, 'model') or self.response_model
        elif event == 'provider.response_completed':
            self._merge(data.get('usage'))
            self.runtime_model = data.get('model') or self.runtime_model
        else:
            return
        self.persist(self.data())

    def _merge(self, raw):
        if not self.attempts:
            self.attempts.append({})
        # Normalized events can omit details already observed on the wire.
        self.attempts[-1].update({k: v for k, v in usage_counts(raw).items() if v is not None})

    def finish(self, result):
        self.cleanup_error = result.get('memoryCleanupError')
        self.runtime_model = result.get('model') or self.runtime_model
        self.final_usage = usage_counts(result.get('usage'))
        self.persist(self.data())

    def data(self):
        known = [a for a in self.attempts if any(v is not None for v in a.values())]
        totals = {}
        for key in usage_counts(None):
            values = [a[key] for a in known if key in a]
            totals[key] = sum(values) if values else self.final_usage.get(key)
        return {**totals, 'requested_model': self.requested_model, 'requested_effort': self.requested_effort,
                'cleanup_error': self.cleanup_error,
                'auth_source': self.auth_source,
                'runtime_model': self.runtime_model, 'response_model': self.response_model,
                'wire_attempts': len(self.attempts) if self.observed else None,
                'transport_retries': self.retries if self.observed else None,
                'usage_complete': bool(self.attempts) and all('input_tokens' in a and 'output_tokens' in a for a in self.attempts),
                'attempt_usage': self.attempts}
