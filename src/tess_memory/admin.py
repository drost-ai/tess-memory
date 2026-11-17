"""Administration and sharing: users, teams, project mappings, and an
author's share/unshare of their own memories.

Administration needs an administrator principal (the default owner always
is). Sharing needs the row's author or an administrator. Every visibility
change is recorded in ``visibility_events``.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .principals import VISIBILITIES, memory_access_sql, principal_id
from .store import MemoryStore

DISTILLED_LEVELS = ("memory", "session")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_admin(store: MemoryStore) -> None:
    if store.restricted:
        raise PermissionError("an administrator is required")


def set_default_owner(store: MemoryStore, user: str) -> None:
    """Name the user NULL owners stand for (e.g. sam instead of 'owner')."""
    _require_admin(store)
    with store.write_transaction():
        store.set_meta("default_owner", principal_id(user))


def add_user(store: MemoryStore, user: str, *, display_name: str | None = None, admin: bool = False) -> None:
    _require_admin(store)
    with store.write_transaction():
        store.db.execute(
            "INSERT INTO principals(id, display_name, admin, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET display_name=COALESCE(excluded.display_name, principals.display_name), "
            "admin=excluded.admin, disabled=0",
            (principal_id(user), display_name, int(admin), _now()))


def disable_user(store: MemoryStore, user: str) -> None:
    _require_admin(store)
    if user == store.default_owner():
        raise ValueError("the default owner cannot be disabled")
    with store.write_transaction():
        store.db.execute("INSERT INTO principals(id, disabled, created_at) VALUES(?,1,?) "
                         "ON CONFLICT(id) DO UPDATE SET disabled=1", (principal_id(user), _now()))


def add_team(store: MemoryStore, team: str, *, name: str | None = None) -> None:
    _require_admin(store)
    with store.write_transaction():
        store.db.execute("INSERT INTO teams(id, name, created_at) VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET "
                         "name=COALESCE(excluded.name, teams.name)", (principal_id(team, "team"), name or team, _now()))


def add_member(store: MemoryStore, team: str, user: str, *, role: str = "member") -> None:
    _require_admin(store)
    if role not in ("member", "admin"):
        raise ValueError("role is member or admin")
    with store.write_transaction():
        if store.db.execute("SELECT 1 FROM teams WHERE id=?", (team,)).fetchone() is None:
            raise ValueError(f"no team {team!r}")
        store.db.execute("INSERT INTO team_members(team_id, user_id, role) VALUES(?,?,?) "
                         "ON CONFLICT(team_id, user_id) DO UPDATE SET role=excluded.role", (team, principal_id(user), role))


def remove_member(store: MemoryStore, team: str, user: str) -> None:
    _require_admin(store)
    with store.write_transaction():
        store.db.execute("DELETE FROM team_members WHERE team_id=? AND user_id=?", (team, user))


def map_project(store: MemoryStore, project: str, *, team: str | None = None, owner: str | None = None) -> None:
    """Give a project to a team (its team-visible material becomes readable
    by the team) and/or name its owner."""
    _require_admin(store)
    with store.write_transaction():
        if team is not None and store.db.execute("SELECT 1 FROM teams WHERE id=?", (team,)).fetchone() is None:
            raise ValueError(f"no team {team!r}")
        store.db.execute(
            "INSERT INTO projects(key, owner_user, team_id, created_at) VALUES(?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "owner_user=COALESCE(excluded.owner_user, projects.owner_user), team_id=COALESCE(excluded.team_id, projects.team_id)",
            (project, store.owner_for(owner) if owner else None, team, _now()))
        store.mark_aggregate_dirty("project", project)  # its profile is now built for a team


def list_users(store: MemoryStore) -> list[dict]:
    _require_admin(store)
    owner = store.default_owner()
    teams: dict[str, list[str]] = {}
    for row in store.db.execute("SELECT user_id, team_id FROM team_members ORDER BY team_id"):
        teams.setdefault(row["user_id"], []).append(row["team_id"])
    users = [{"id": owner, "admin": True, "disabled": False, "default_owner": True, "teams": teams.get(owner, [])}]
    for row in store.db.execute("SELECT * FROM principals ORDER BY id"):
        if row["id"] != owner:
            users.append({"id": row["id"], "display_name": row["display_name"], "admin": bool(row["admin"]),
                          "disabled": bool(row["disabled"]), "default_owner": False, "teams": teams.get(row["id"], [])})
    known = {u["id"] for u in users}
    users += [{"id": user, "admin": False, "disabled": False, "default_owner": False, "teams": team_list}
              for user, team_list in teams.items() if user not in known]
    return users


def share(store: MemoryStore, target: str, visibility: str | None) -> dict:
    """Set the visibility of a memory (``memory:<id>`` or a bare id) or of a
    session's distilled output (``session:<id>``). ``None`` returns rows to
    the policy default. Only the author or an administrator may."""
    if visibility is not None and visibility not in VISIBILITIES:
        raise ValueError(f"visibility is one of {', '.join(VISIBILITIES)} (or none for the policy default)")
    kind, _, ident = target.partition(":") if ":" in target else ("memory", "", target)
    owner = store.default_owner()
    actor = store.principal.user if store.principal else owner
    with store.write_transaction():
        if kind == "memory":
            rows = store.db.execute("SELECT id, owner_user FROM memories WHERE id=?", (int(ident),)).fetchall()
        elif kind == "session":
            rows = store.db.execute(
                f"SELECT id, owner_user FROM memories WHERE session_id=? AND level IN ({','.join('?' for _ in DISTILLED_LEVELS)})",
                (ident, *DISTILLED_LEVELS)).fetchall()
        else:
            raise ValueError("target is memory:<id> or session:<id>")
        if not rows:
            raise LookupError(f"nothing to share for {target}")
        if store.restricted and any((row["owner_user"] or owner) != actor for row in rows):
            raise PermissionError("only the author or an administrator can change who sees this")
        store.db.executemany("UPDATE memories SET visibility=? WHERE id=?", [(visibility, row["id"]) for row in rows])
        store.db.execute("INSERT INTO visibility_events(ts, actor, target, visibility, detail) VALUES(?,?,?,?,?)",
                         (_now(), actor, target, visibility, f"{len(rows)} rows"))
    return {"target": target, "visibility": visibility or "policy default", "rows": len(rows)}


def who_can_see(store: MemoryStore, memory_id: int) -> dict:
    """The users (and teams) that may read one memory, for administrators."""
    _require_admin(store)
    row = store.db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    if row is None:
        raise LookupError(f"no memory {memory_id}")
    from .principals import Principal
    candidates = {u["id"] for u in list_users(store)}
    readers = []
    for user in sorted(candidates):
        principal = store.resolve_principal(user) if user != store.default_owner() else Principal(user, admin=True)
        if principal.admin:
            readers.append(user)
            continue
        sql, args = memory_access_sql(principal, store.policy, store.default_owner())
        if store.db.execute(f"SELECT 1 FROM memories m WHERE m.id=? AND {sql}", (memory_id, *args)).fetchone():
            readers.append(user)
    visibility, args = store.policy.sql("m")
    effective = store.db.execute(f"SELECT {visibility} FROM memories m WHERE m.id=?", (*args, memory_id)).fetchone()[0]
    return {"memory": memory_id, "owner": row["owner_user"] or store.default_owner(), "visibility": effective,
            "project": row["project"], "team": store.project_team(row["project"]), "readers": readers}
