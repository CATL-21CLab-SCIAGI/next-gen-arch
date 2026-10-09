"""Supervise one command and publish its actual lifecycle for external tracking.

This process does not restart jobs or interpret training metrics. The parent
observes exit even if the child is killed before its own cleanup can run.
"""

import argparse
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from archlab.artifacts import atomic_write_json


def supervise(command, status_path, interval=30):
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    started = datetime.now(timezone.utc).isoformat()
    original_handlers = {}

    def forward(signum, frame):
        del frame
        if process.poll() is None:
            process.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        original_handlers[signum] = signal.signal(signum, forward)
    try:
        while True:
            code = process.poll()
            atomic_write_json(status_path, {
                'state': 'running' if code is None else ('finished' if code == 0 else 'failed'),
                'pid': process.pid, 'supervisor_pid': os.getpid(), 'hostname': socket.gethostname(),
                'started_at_utc': started, 'observed_at_epoch': time.time(), 'exit_code': code,
            })
            if code is not None:
                return code
            try:
                process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    if args.status.exists():
        parser.error('status already exists; use a fresh run')
    raise SystemExit(supervise(command, args.status))


if __name__ == '__main__':
    main()
