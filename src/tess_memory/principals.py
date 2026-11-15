"""Who is reading, and what they may read: users, teams and visibility.

The scope model is org > team > user; the org is the instance for now. Every
coding-memory row has an owner user and a visibility, both nullable so that
rows written by older code stay correct:

- ``owner_user`` NULL means the instance's default owner (``meta.default_owner``,
  "owner" until an administrator names one).
- ``visibility`` NULL means the policy's default for the row's level and kind
  (``VisibilityPolicy``): with the owner's defaults, distilled knowledge
  (decisions, solutions, facts, problems, project profiles, plan items) is
  ``team``; raw chunks, session summaries, digests, preferences and task state
  are ``private``. An explicit value (a share or unshare) wins.

A row is readable by a principal when the principal is an administrator (the
default owner always is), owns it, or its visibility is ``org``, or ``team``
and the principal has access to its project (owns the project, or is in the
project's team). A project nobody mapped belongs to the default owner alone.

The caller's principal is bound when the server or command starts and never
comes from a model's arguments. An instance with no identity configured runs
as the default owner and reads everything exactly as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

DEFAULT_OWNER = "owner"
VISIBILITIES = ("private", "team", "org")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}")


def principal_id(value, name="user"):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid {name} id: {value!r} (letters, digits, '_', '.', ':', '@', '-'; at most 128)")
    return value


@dataclass(frozen=True)
class Principal:
    """The reader or writer. ``admin`` reads everything."""

    user: str
    teams: frozenset[str] = field(default_factory=frozenset)
    admin: bool = False

    @property
    def restricted(self) -> bool:
        return not self.admin

    def key(self) -> tuple:
        """Distinguishes cached reads of different principals."""
        return ("*",) if self.admin else (self.user, *sorted(self.teams))


@dataclass(frozen=True)
class VisibilityPolicy:
    """Default visibility by level, and by kind for level='memory'."""

    levels: dict[str, str] = field(default_factory=lambda: {
        "project": "team", "plan": "team", "chunk": "private", "session": "private", "digest": "private",
    })
    memory_kinds: dict[str, str] = field(default_factory=lambda: {
        "decision": "team", "solution": "team", "fact": "team", "problem": "team",
    })
    fallback: str = "private"

    def __post_init__(self):
        for value in (*self.levels.values(), *self.memory_kinds.values(), self.fallback):
            if value not in VISIBILITIES:
                raise ValueError(f"visibility must be one of {', '.join(VISIBILITIES)}, not {value!r}")

    def default_for(self, level: str, kind: str | None) -> str:
        if level == "memory":
            return self.memory_kinds.get(kind or "", self.fallback)
        return self.levels.get(level, self.fallback)

    def sql(self, alias: str = "m") -> tuple[str, list]:
        """The effective visibility of a row as an SQL expression."""
        cases, args = [], []
        for kind, value in self.memory_kinds.items():
            cases.append(f"WHEN {alias}.level='memory' AND {alias}.kind=? THEN ?")
            args += [kind, value]
        for level, value in self.levels.items():
            cases.append(f"WHEN {alias}.level=? THEN ?")
            args += [level, value]
        expression = f"COALESCE({alias}.visibility, CASE {' '.join(cases)} ELSE ? END)"
        return expression, args + [self.fallback]


def project_access_sql(principal: Principal, project_column: str, default_owner: str) -> tuple[str, list]:
    """Whether the principal may read a project's shared material."""
    teams = sorted(principal.teams)
    team_test = f" OR p.team_id IN ({','.join('?' for _ in teams)})" if teams else ""
    sql = (f"EXISTS(SELECT 1 FROM projects p WHERE p.key={project_column} "
           f"AND (COALESCE(p.owner_user, ?)=?{team_test}))")
    return sql, [default_owner, principal.user, *teams]


def memory_access_sql(principal: Principal, policy: VisibilityPolicy, default_owner: str, alias: str = "m") -> tuple[str, list]:
    """The predicate a memory row must pass for a restricted principal."""
    visibility, visibility_args = policy.sql(alias)
    project, project_args = project_access_sql(principal, f"{alias}.project", default_owner)
    sql = (f"(COALESCE({alias}.owner_user, ?)=? OR {visibility}='org' "
           f"OR ({visibility}='team' AND {project}))")
    return sql, [default_owner, principal.user, *visibility_args, *visibility_args, *project_args]


def session_access_sql(principal: Principal, default_owner: str, alias: str = "s") -> tuple[str, list]:
    """Session rows (title, summary, outcome, source) are their author's."""
    return f"COALESCE({alias}.owner_user, ?)=?", [default_owner, principal.user]
