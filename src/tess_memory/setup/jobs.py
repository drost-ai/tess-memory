"""The first ingestion, run as the ordinary command.

It runs in a child process with the settings file setup has just written, so
what the person sees working here is exactly what the scheduler will run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .providers import SetupError

_PUBLISHED = re.compile(r"^\[(ingest|incremental)\] .*(distilled|fallback|extractive|published|preserv)")
_COUNTED = re.compile(r"^\[ingest\] \((\d+)/(\d+)\)")

class FirstIngest:
    """The ingestion setup starts. It is its own process, writing to files in
    the store's folder and not to setup: it carries on when setup is finished
    or closed, which is what "you can close this page" promises. Setup only
    reads its log to show progress."""

    LOG, OUT = "setup-ingest.log", "setup-ingest.json"

    def __init__(self, settings: Path) -> None:
        self.settings = settings
        self.status: dict[str, Any] = {"state": "idle"}
        self.after: Any = None  # called once a run has finished by itself
        self.command: list[str] | None = None  # a test seam; the real command by default
        self._process: subprocess.Popen[bytes] | None = None

    def _environment(self) -> dict[str, str]:
        return {**os.environ, "TESS_MEMORY_SETTINGS": str(self.settings), "PYTHONUNBUFFERED": "1"}

    def start(self, limit: int | None) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            raise SetupError("ingest_running")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise SetupError("invalid_limit")
        command = self.command or [sys.executable, "-m", "tess_memory.cli", "ingest", *(["--limit", str(limit)] if limit else [])]
        home = self.settings.parent
        home.mkdir(parents=True, exist_ok=True)
        self.status = {"state": "running", "limit": limit, "done": 0, "total": limit, "lines": []}
        with open(home / self.LOG, "wb") as log, open(home / self.OUT, "wb") as out:
            # Files, not pipes, and a session of its own: nothing ties it to setup's lifetime.
            self._process = subprocess.Popen(command, env=self._environment(), stdin=subprocess.DEVNULL, stdout=out, stderr=log,
                                             start_new_session=True)
        threading.Thread(target=self._follow, args=(self._process, home / self.LOG, home / self.OUT),
                         name="tess-memory-first-ingest", daemon=True).start()
        return self.status

    def _follow(self, process: subprocess.Popen[bytes], log: Path, out: Path) -> None:
        status = self.status
        with open(log, "r", encoding="utf-8", errors="replace") as lines:
            pending = ""
            while True:
                chunk = lines.read()
                if chunk:
                    pending += chunk
                    *complete, pending = pending.split("\n")
                    for line in complete:
                        self._line(status, line.rstrip())
                elif process.poll() is not None:
                    if pending.strip():
                        self._line(status, pending.rstrip())
                    break
                else:
                    time.sleep(0.25)
        code = process.wait()
        try:
            status["stats"] = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status["stats"] = None
        if status["state"] == "running":
            status["state"] = "finished" if code == 0 or (status["stats"] or {}).get("distilled") else "failed"
        status["exit_code"] = code
        if status["state"] == "finished" and self.after is not None:
            try:
                self.after()
            except Exception:  # noqa: BLE001 (what follows a run must not undo it)
                pass

    @staticmethod
    def _line(status: dict[str, Any], line: str) -> None:
        if not line:
            return
        counted = _COUNTED.match(line)
        if counted:
            status["done"], status["total"] = int(counted.group(1)), int(counted.group(2))
        elif _PUBLISHED.match(line):
            status["done"] += 1
        status["lines"] = (status["lines"] + [line[:240]])[-12:]

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self.status["state"] = "stopped"
            self._process.terminate()
