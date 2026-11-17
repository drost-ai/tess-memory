"""Explicit personal-memory enablement; absent settings are completely inert."""
from dataclasses import dataclass, fields
from .autobiography import PersonalError, PersonalPrincipal


# The source mode that trusts a host's own session folders is called "native".
# Records already stored carry an earlier identifier for it, and that identifier
# is hashed into their source policy. It is kept here, encoded, so those records
# stay valid; the configuration accepts either spelling. Do not change it.
NATIVE_SOURCE_MODE = bytes.fromhex('746573732d6167656e74').decode()
_SOURCE_MODES = {'allowlist': 'allowlist', 'native': NATIVE_SOURCE_MODE, NATIVE_SOURCE_MODE: NATIVE_SOURCE_MODE}


def is_native_mode(value) -> bool:
    return isinstance(value, str) and _SOURCE_MODES.get(value) == NATIVE_SOURCE_MODE


# Limits that apply only when configured (None = no limit).
OPTIONAL_LIMITS = ('max_candidates','max_edits','model_calls','daily_model_calls','daily_token_budget','max_output_tokens')


@dataclass(frozen=True)
class PersonalConfig:
    enabled: bool = False
    owner: str = ''
    agent: str = ''
    audience: str = 'owner'
    role: str = 'reader'
    session_id: str = 'unbound'
    context_budget: int = 6000
    context_subjects: tuple[str, ...] = ()
    extraction_enabled: bool = False
    dream_enabled: bool = False
    registers_enabled: bool = False
    care_interval: int = 1
    research_interval: int = 1
    register_reconciliation_interval: int = 1
    shadow: bool = True
    source_names: tuple[str, ...] = ()
    historical: bool = False
    # Explicit personal-only roots; never inherits the coding archive.
    source_mode: str = 'allowlist'
    source_paths: tuple[str, ...] = ()
    source_since: str | None = None
    # Explicit trust in a host's native session ledger; never auto-discovered.
    native_creation_ledger: str | None = None
    ai_name: str = ''
    user_name: str = ''
    dream_hour: int = 23
    timezone: str = 'UTC'
    # Work and spending limits are off unless configured: no daily call or
    # token budget, no per-job call allowance, no cap on a reflection's
    # candidates or edits, and the provider's own output limit. An operator
    # who wants bounds sets them.
    max_candidates: int | None = None
    max_edits: int | None = None
    model_calls: int | None = None
    max_seconds: int = 1800  # a job's deadline: a hang guard, not a budget
    lease_seconds: int = 600
    extraction_interval: int = 900
    reconciliation_interval: int = 900
    daily_model_calls: int | None = None
    daily_token_budget: int | None = None
    max_output_tokens: int | None = None
    service_config: str | None = None
    # One store, an audience per person a host serves. Off =
    # exactly one audience, as before.
    multi_audience: bool = False

    def __post_init__(self):
        if isinstance(self.source_mode, str) and self.source_mode in _SOURCE_MODES:
            object.__setattr__(self, 'source_mode', _SOURCE_MODES[self.source_mode])

    def principal(self, *, role=None, session_id=None, audience=None):
        """The principal for a session. Another audience than the configured
        (owner's) one needs multi_audience; it then reads its own records
        plus the agent's shared Self/Principles, and the owner reads all."""
        if not self.enabled:
            raise PersonalError('disabled','personal memory is disabled')
        audience = audience or self.audience
        if audience != self.audience and not self.multi_audience:
            raise PersonalError('forbidden','this store serves one audience (enable personal.multi_audience)')
        reads = '' if not self.multi_audience else '*' if audience == self.audience else self.audience
        return PersonalPrincipal(self.owner,self.agent,audience,role or self.role,session_id or self.session_id,reads)

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw,dict):raise ValueError('unknown personal configuration fields')
        raw=dict(raw)
        if 'reconciler_enabled' in raw:
            if 'dream_enabled' in raw and raw['dream_enabled']!=raw['reconciler_enabled']:
                raise ValueError('conflicting reconciler_enabled/dream_enabled')
            raw['dream_enabled']=raw.pop('reconciler_enabled')
        if set(raw)-{f.name for f in fields(cls)}:
            raise ValueError('unknown personal configuration fields')
        values=dict(raw)
        for key in ['enabled','extraction_enabled','dream_enabled','registers_enabled','shadow','historical','multi_audience']:
            if key in values and type(values[key]) is not bool:
                raise ValueError(f'personal.{key} must be a boolean')
        for key in ['context_subjects','source_names','source_paths']:
            if key in values:
                if not isinstance(values[key],list) or any(not isinstance(x,str) or not x for x in values[key]):
                    raise ValueError(f'personal.{key} must be a string array')
                values[key]=tuple(values[key])
        result=cls(**values)
        if result.source_mode not in ('allowlist',NATIVE_SOURCE_MODE):
            raise ValueError('personal.source_mode must be allowlist or native')
        if result.native_creation_ledger is not None:
            if not isinstance(result.native_creation_ledger, str) or not result.native_creation_ledger.strip():
                raise ValueError('personal.native_creation_ledger must be a nonempty path')
            if result.source_mode != NATIVE_SOURCE_MODE:
                raise ValueError('personal.native_creation_ledger requires native source mode')
        for key in ('ai_name','user_name'):
            if not isinstance(getattr(result,key),str):
                raise ValueError(f'personal.{key} must be a string')
        if result.source_since is not None:
            from .personal_sources import parse_boundary
            parse_boundary(result.source_since)
        if result.source_mode == NATIVE_SOURCE_MODE and not result.source_paths:
            raise ValueError('personal.source_paths must explicitly identify native session roots')
        for key in ['context_budget','max_seconds','lease_seconds','extraction_interval','reconciliation_interval','care_interval','research_interval','register_reconciliation_interval']:
            if type(getattr(result,key)) is not int or getattr(result,key)<=0:
                raise ValueError(f'personal.{key} must be a positive integer')
        for key in OPTIONAL_LIMITS:
            if getattr(result,key) is not None and (type(getattr(result,key)) is not int or getattr(result,key)<=0):
                raise ValueError(f'personal.{key} must be a positive integer when set')
        from .autobiography_context import MAX_CONTEXT_BYTES
        if not 256<=result.context_budget<=MAX_CONTEXT_BYTES:
            raise ValueError(f'personal.context_budget must be 256..{MAX_CONTEXT_BYTES} (UTF-8 bytes, ~4 per token)')
        if type(result.dream_hour) is not int or not 0<=result.dream_hour<=23:
            raise ValueError('personal.dream_hour must be 0..23')
        from zoneinfo import ZoneInfo
        ZoneInfo(result.timezone)
        if result.enabled:
            result.principal()
        return result


def session_audience(metadata, personal):
    """The audience a session belongs to: the person the host named in its
    metadata (personalAudience), else the configured (owner's) audience."""
    from .autobiography import identifier
    value = ((metadata or {}).get('metadata') or {}).get('personalAudience')
    try:
        return identifier(value, 'audience') if value else personal.audience
    except PersonalError:
        return personal.audience

