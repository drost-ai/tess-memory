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
from pathlib import Path
from typing import Any

from .providers import SetupError

_PUBLISHED = re.compile(r"^\[(ingest|incremental)\] .*(distilled|fallback|extractive|published|preserv)")
_COUNTED = re.compile(r"^\[ingest\] \((\d+)/(\d+)\)")

class FirstIngest:
    def __init__(self, settings: Path) -> None:
        self.settings = settings
        self.status: dict[str, Any] = {"state": "idle"}
        self.after: Any = None  # called once a run has finished by itself
        self._process: subprocess.Popen[str] | None = None

    def _environment(self) -> dict[str, str]:
        return {**os.environ, "TESS_MEMORY_SETTINGS": str(self.settings), "PYTHONUNBUFFERED": "1"}

    def start(self, limit: int | None) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            raise SetupError("ingest_running")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise SetupError("invalid_limit")
        command = [sys.executable, "-m", "tess_memory.cli", "ingest", *(["--limit", str(limit)] if limit else [])]
        self.status = {"state": "running", "limit": limit, "done": 0, "total": limit, "lines": []}
        self._process = subprocess.Popen(command, env=self._environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self._follow, args=(self._process,), name="tess-memory-first-ingest", daemon=True).start()
        return self.status

    def _follow(self, process: subprocess.Popen[str]) -> None:
        status = self.status
        output: list[str] = []
        reader = threading.Thread(target=lambda: output.append(process.stdout.read()), daemon=True)
        reader.start()
        for line in process.stderr:
            line = line.rstrip()
            if not line:
                continue
            counted = _COUNTED.match(line)
            if counted:
                status["done"], status["total"] = int(counted.group(1)), int(counted.group(2))
            elif _PUBLISHED.match(line):
                status["done"] += 1
            status["lines"] = (status["lines"] + [line[:240]])[-12:]
        code = process.wait()
        reader.join(timeout=5)
        try:
            status["stats"] = json.loads("".join(output))
        except ValueError:
            status["stats"] = None
        if status["state"] == "running":
            status["state"] = "finished" if code == 0 or (status["stats"] or {}).get("distilled") else "failed"
        status["exit_code"] = code
        if status["state"] == "finished" and self.after is not None:
            try:
                self.after()
            except Exception:  # noqa: BLE001 (what follows a run must not undo it)
                pass

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self.status["state"] = "stopped"
            self._process.terminate()
