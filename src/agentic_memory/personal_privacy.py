"""Conservative credential handling for personal text, never general PII filtering."""
import re
from .autobiography import PersonalError

SECRET = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._-]{16,})\b|\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*[\"\x27]?[A-Za-z0-9+/_=.-]{16,}',re.I)


def redact_credentials(text):
    return SECRET.sub('[credential omitted]',text)


def require_no_credentials(text):
    if SECRET.search(text):
        raise PersonalError('sensitive_content','credentials belong in secret storage, not autobiographical records')
