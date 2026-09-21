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

    def principal(self, *, role=None, session_id=None):
        if not self.enabled:
            raise PersonalError('disabled','personal memory is disabled')
        return PersonalPrincipal(self.owner,self.agent,self.audience,role or self.role,session_id or self.session_id)

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw,dict) or set(raw)-{f.name for f in fields(cls)}:
            raise ValueError('unknown personal configuration fields')
        values=dict(raw)
        for key in ['enabled','extraction_enabled','dream_enabled','shadow','historical']:
            if key in values and type(values[key]) is not bool:
                raise ValueError(f'personal.{key} must be a boolean')
        for key in ['context_subjects','source_names']:
            if key in values:
                if not isinstance(values[key],list) or any(not isinstance(x,str) or not x for x in values[key]):
                    raise ValueError(f'personal.{key} must be a string array')
                values[key]=tuple(values[key])
        result=cls(**values)
        for key in ['context_budget','max_candidates','max_edits','model_calls','max_seconds','lease_seconds']:
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
