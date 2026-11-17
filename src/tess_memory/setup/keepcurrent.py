"""Keeping the store current after setup: the background scheduler.

A memory that is only read once goes stale the same day, so setup turns the
scheduler on unless the person says otherwise. The rules it runs by:

- every 5 minutes it looks at each source for new or changed sessions;
- a session is read once it has been still for 15 minutes (the settings
  file's ``quiet_seconds``), so a conversation is not distilled mid-thought;
- project profiles refresh hourly and weekly digests daily;
- at 10 PM, in this machine's time zone, one pass retries what failed that day.

A run makes at most 128 model calls and then hands over; when work is left
the next run starts 15 seconds later. That bounds how much one failure can
waste and lets several stores share a provider. It does not slow a long
first read.

The scheduler is installed as a login service: launchd on macOS, a systemd
user service on Linux. Setup never replaces or removes a scheduler it did not
install.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .providers import SetupError
from .vault import write_private

CONFIG = "scheduler.toml"
UNIT = "tess-memory-scheduler.service"
RULES = {"poll_seconds": 300, "profile_seconds": 3600, "digest_seconds": 86400, "sources_per_run": 16,
         "model_calls_per_run": 128, "aggregate_calls_per_run": 4, "recovery_hour": 22,
         "recovery_sources_per_run": 256, "recovery_model_calls_per_run": 128, "recovery_min_age_seconds": 900}


def local_timezone() -> str:
    """This machine's time zone by name, for the nightly pass; UTC if unknown."""
    from zoneinfo import ZoneInfo
    candidates = [os.environ.get("TZ") or ""]
    try:
        target = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in target:
            candidates.append(target.split("zoneinfo/", 1)[1])
    except OSError:
        pass
    try:
        candidates.append(Path("/etc/timezone").read_text(encoding="utf-8").strip())
    except OSError:
        pass
    for name in candidates:
        try:
            if name and "/" in name:
                ZoneInfo(name)
                return name
        except Exception:  # noqa: BLE001 (not a zone this Python knows)
            continue
    return "UTC"


def config_text(home: Path, settings: Path, *, timezone: str | None = None) -> str:
    quoted = lambda value: json.dumps(str(value))  # noqa: E731
    lines = ["# Written by `tess-memory setup`: keeps this store current in the background.",
             "[service]", f"data_dir = {quoted(home / 'service')}", "backlog_seconds = 15", "",
             "[[stores]]", 'name = "main"', f"settings = {quoted(settings)}", "enabled = true"]
    lines += [f"{key} = {value}" for key, value in RULES.items()]
    lines += [f"recovery_timezone = {quoted(timezone or local_timezone())}", ""]
    return "\n".join(lines)


def _unit_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / UNIT


def _unit_text(config: Path) -> str:
    return "\n".join([
        "[Unit]", "Description=Tess Memory scheduler: keeps memory current", "",
        "[Service]", f"ExecStart={sys.executable} -u -m tess_memory.cli scheduler --config {config} serve",
        "Restart=always", "RestartSec=30", "Environment=PYTHONUNBUFFERED=1", "",
        "[Install]", "WantedBy=default.target", ""])


def _platform() -> str | None:
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        return "systemd"
    return None


def status(home: Path) -> dict[str, Any]:
    """Whether a scheduler is installed here, whether it is this store's, and
    whether it is running."""
    config = Path(home) / CONFIG
    command = f"{sys.executable} -m tess_memory.cli scheduler --config {config} serve"
    kind = _platform()
    result: dict[str, Any] = {"supported": kind is not None, "kind": kind, "installed": False, "ours": False,
                              "running": False, "command": command, "timezone": local_timezone()}
    if kind == "launchd":
        from ..scheduler_install import DEFAULT_LABEL, agent_path
        plist = agent_path(DEFAULT_LABEL)
        if plist.exists():
            result["installed"] = True
            try:
                import plistlib
                result["ours"] = str(config) in (plistlib.loads(plist.read_bytes()).get("ProgramArguments") or [])
            except Exception:  # noqa: BLE001 (unreadable: treated as someone else's)
                pass
    elif kind == "systemd":
        unit = _unit_path()
        if unit.exists():
            result["installed"] = True
            try:
                result["ours"] = str(config) in unit.read_text(encoding="utf-8")
            except OSError:
                pass
    if result["ours"]:
        result.update(_activity(config))
    return result


def _activity(config: Path) -> dict[str, Any]:
    """Whether our scheduler's supervisor is alive, and what it last did."""
    try:
        from ..scheduler_config import load_service
        from ..scheduler_queue import WorkQueue
        queue = WorkQueue(load_service(config), readonly=True).status()
        supervisor = queue.get("supervisor") or {}
        ingest = next((job for job in queue.get("jobs") or [] if job.get("kind") == "ingest"), {})
        stats = ingest.get("last_stats") or {}
        return {"running": bool(supervisor.get("process_alive")), "state": ingest.get("state"),
                "pending": (stats.get("sources_remaining") or 0) + (stats.get("sources_pending") or 0) if stats else None,
                "last_error": bool(ingest.get("last_error"))}
    except Exception:  # noqa: BLE001 (not started yet, or its files are not there yet)
        return {"running": False}


def install(home: Path, settings: Path, *, run: Any = subprocess.run) -> dict[str, Any]:
    """Write this store's scheduler configuration and start it as a login service."""
    current = status(home)
    if not current["supported"]:
        raise SetupError("scheduler_unsupported")
    if current["installed"] and not current["ours"]:
        raise SetupError("scheduler_already_installed")
    config = Path(home) / CONFIG
    write_private(config, config_text(Path(home), settings))
    try:
        from ..scheduler_config import load_service
        load_service(config)  # a configuration the scheduler itself accepts
        if current["kind"] == "launchd":
            from ..scheduler_install import install as install_agent
            install_agent(config)
        else:
            unit = _unit_path()
            unit.parent.mkdir(parents=True, exist_ok=True)
            unit.write_text(_unit_text(config), encoding="utf-8")
            for command in (["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", UNIT]):
                result = run(command, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout or "systemctl failed").strip()[:200])
    except SetupError:
        raise
    except Exception as error:  # noqa: BLE001
        raise SetupError("scheduler_install_failed", f"{type(error).__name__}: {str(error)[:200]}") from None
    return status(home)


def remove(home: Path, *, run: Any = subprocess.run) -> dict[str, Any]:
    """Stop this store's scheduler and take it out of the login services.
    The store, its settings and the scheduler's own records are kept."""
    current = status(home)
    if not current["installed"]:
        return current
    if not current["ours"]:
        raise SetupError("scheduler_already_installed")
    try:
        if current["kind"] == "launchd":
            from ..scheduler_install import DEFAULT_LABEL, agent_path, service_control
            service_control("stop", label=DEFAULT_LABEL)
            agent_path(DEFAULT_LABEL).unlink(missing_ok=True)
        else:
            run(["systemctl", "--user", "disable", "--now", UNIT], capture_output=True, text=True, timeout=60)
            _unit_path().unlink(missing_ok=True)
            run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True, timeout=60)
    except Exception as error:  # noqa: BLE001
        raise SetupError("scheduler_remove_failed", f"{type(error).__name__}: {str(error)[:200]}") from None
    return status(home)
