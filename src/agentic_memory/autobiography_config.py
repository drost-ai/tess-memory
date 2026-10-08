"""Explicit personal-memory enablement; absent settings are completely inert."""
from dataclasses import dataclass, fields
from .autobiography import PersonalError, PersonalPrincipal


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
    shadow: bool = True
    source_names: tuple[str, ...] = ()
    historical: bool = False
    dream_hour: int = 23
    timezone: str = 'UTC'
    max_candidates: int = 16
    max_edits: int = 8
    model_calls: int = 8
    max_seconds: int = 300
    lease_seconds: int = 600
    extraction_interval: int = 300
    daily_model_calls: int = 32
    daily_token_budget: int = 500000
    max_output_tokens: int = 4096
    service_config: str | None = None
    # Epic U: one store, an audience per person (the host's people). Off =
    # exactly one audience, as before.
    multi_audience: bool = False

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
        if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(cls)}:
            raise ValueError('unknown personal configuration fields')
        values=dict(raw)
        for key in ['enabled','extraction_enabled','dream_enabled','shadow','historical','multi_audience']:
            if key in values and type(values[key]) is not bool:
                raise ValueError(f'personal.{key} must be a boolean')
        for key in ['context_subjects','source_names']:
            if key in values:
                if not isinstance(values[key],list) or any(not isinstance(x,str) or not x for x in values[key]):
                    raise ValueError(f'personal.{key} must be a string array')
                values[key]=tuple(values[key])
        result=cls(**values)
        for key in ['context_budget','max_candidates','max_edits','model_calls','max_seconds','lease_seconds','extraction_interval','daily_model_calls','daily_token_budget','max_output_tokens']:
            if type(getattr(result,key)) is not int or getattr(result,key)<=0:
                raise ValueError(f'personal.{key} must be a positive integer')
        if not 256<=result.context_budget<=64000 or result.max_candidates>100 or result.max_edits>32:
            raise ValueError('personal context/candidate/edit budget exceeds its bound')
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

