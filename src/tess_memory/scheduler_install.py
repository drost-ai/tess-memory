"""Install/control only this package's launchd agent; never remove memory data."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import tempfile
import time

from .scheduler import ROOT
from .scheduler_config import load_service

DEFAULT_LABEL = 'com.tess-memory.scheduler'


def agent_path(label):
    if not re.fullmatch(r'com\.tess-memory\.[A-Za-z0-9_.-]+', label):
        raise ValueError('launchd label must start with com.tess-memory.')
    return Path.home() / 'Library/LaunchAgents' / (label + '.plist')


def owned_plist(path):
    value = plistlib.loads(path.read_bytes())
    args = value.get('ProgramArguments', [])
    if ('tess_memory.cli' not in args or 'scheduler' not in args or 'serve' not in args
            or value.get('WorkingDirectory') != str(ROOT)):
        raise ValueError('refusing to alter a launch agent not owned by this checkout')
    return value


def build_plist(config, label):
    return {'Label': label,
            'ProgramArguments': [sys.executable, '-u', '-m', 'tess_memory.cli', 'scheduler',
                                 '--config', str(config.path), 'serve'],
            'WorkingDirectory': str(ROOT),
            'EnvironmentVariables': {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'PYTHONUNBUFFERED': '1'},
            'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 30,
            'StandardOutPath': str(config.data_dir / 'supervisor.log'),
            'StandardErrorPath': str(config.data_dir / 'supervisor.log')}


def service_control(action, *, label=DEFAULT_LABEL):
    if sys.platform != 'darwin':
        raise ValueError('launchd management is available on macOS; scheduler serve is portable')
    path = agent_path(label)
    value = owned_plist(path)
    if value.get('Label') != label:
        raise ValueError('launchd label mismatch')
    domain = 'gui/' + str(os.getuid())
    target = domain + '/' + label
    loaded = subprocess.run(['/bin/launchctl', 'print', target], capture_output=True).returncode == 0
    if action in ('stop', 'restart') and loaded:
        subprocess.run(['/bin/launchctl', 'bootout', target], check=True)
        for attempt in range(40):
            if subprocess.run(['/bin/launchctl', 'print', target], capture_output=True).returncode != 0:
                break
            if attempt == 39:
                raise RuntimeError('launchd has not finished unloading the owned scheduler')
            time.sleep(0.25)
        loaded = False
    if action in ('start', 'restart') and not loaded:
        # bootout can return while launchd is still tearing down the old job.
        # Retry the owned agent briefly; error 5 here does not imply root is
        # required, and repeated start must not create duplicate supervisors.
        for attempt in range(10):
            result = subprocess.run(['/bin/launchctl', 'bootstrap', domain, str(path)], capture_output=True, text=True)
            if result.returncode == 0 or subprocess.run(['/bin/launchctl', 'print', target], capture_output=True).returncode == 0:
                loaded = True
                break
            if attempt == 9:
                raise RuntimeError(f'launchd bootstrap failed after teardown retries (exit {result.returncode})')
            time.sleep(min(1, 0.2 * (attempt + 1)))
    return {'label': label, 'loaded': loaded, 'plist': str(path)}


def install(config_path: Path, *, label=DEFAULT_LABEL, start=True):
    if sys.platform != 'darwin':
        raise ValueError('use scheduler serve with your platform service manager')
    config = load_service(config_path)
    path = agent_path(label)
    value = build_plist(config, label)
    backup = None
    if path.exists():
        owned_plist(path)
        service_control('stop', label=label)
        directory = config.data_dir / 'backups'
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        backup = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.plist')
        with backup.open('xb') as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(path.read_bytes())
    config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (config.data_dir / 'supervisor.log').touch(mode=0o600, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + label + '-', dir=path.parent)
    with os.fdopen(fd, 'wb') as handle:
        os.fchmod(handle.fileno(), 0o600)
        plistlib.dump(value, handle)
    os.replace(temporary, path)
    result = service_control('start', label=label) if start else {'label': label, 'loaded': False, 'plist': str(path)}
    return {**result, 'backup': str(backup) if backup else None}
