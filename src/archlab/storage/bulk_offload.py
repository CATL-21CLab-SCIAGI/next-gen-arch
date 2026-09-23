"""Resumable verified copies, optionally replacing immutable NAS files with links."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    info = path.stat()
    return info.st_ino, info.st_size, info.st_mtime_ns


def assert_not_in_use(source):
    # Atomic replacement is NOT safe for mmap users on other NFS clients.
    # A persistent guard must remain until all readers of the corpus are stopped.
    for parent in Path(source).absolute().parents:
        if (parent/'.archlab-storage-in-use.json').exists() or any((parent/'.checkpoint-readers').glob('*.json')):
            raise ValueError(f'Active readers protect this storage tree: {parent}')


def transfer(source, destination, link=False, expected_sha=None):
    source, destination = Path(source), Path(destination)
    if not source.is_symlink() and source.resolve() == destination.resolve():
        raise ValueError('Source and destination must be different files')
    if source.is_symlink():
        if source.resolve() != destination.resolve():
            raise ValueError('Source is an unrelated symlink')
        checksum = sha(destination)
        if expected_sha is not None and checksum != expected_sha:
            raise ValueError('Existing linked payload differs from its sealed manifest')
        return {'bytes': destination.stat().st_size, 'sha256': checksum, 'linked': True}
    if not source.is_file():
        raise ValueError('Source must be a regular file')
    if link:
        assert_not_in_use(source)
    original = identity(source)
    for parent in (destination, *destination.parents):
        if parent.is_symlink():
            raise ValueError('Destination contains a symlink')
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = None
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != original[1]:
            raise ValueError('Existing destination has different size/type')
        checksum = sha(source)
        if sha(destination) != checksum:
            raise ValueError('Existing destination has different bytes')
    else:
        partial = destination.with_name('.' + destination.name + '.copy-' + uuid.uuid4().hex)
        digest = hashlib.sha256()
        try:
            with source.open('rb') as inp, partial.open('xb') as out:
                while chunk := inp.read(8 * 1024 * 1024):
                    out.write(chunk)
                    digest.update(chunk)
                out.flush()
                os.fsync(out.fileno())
            checksum = digest.hexdigest()
            if expected_sha is not None and checksum != expected_sha:
                raise ValueError('Payload differs from its sealed manifest')
            if partial.stat().st_size != original[1] or sha(partial) != checksum:
                raise ValueError('Copied payload checksum mismatch')
            if identity(source) != original:
                raise ValueError('Source changed during copy')
            if destination.exists():
                if sha(destination) != checksum:
                    raise ValueError('Destination appeared with different content')
            else:
                # Each manifest owns a distinct destination and the transfer
                # database is locked against concurrent executions of the plan.
                partial.rename(destination)
        finally:
            partial.unlink(missing_ok=True)
    if identity(source) != original:
        raise ValueError('Source changed after verification')
    if expected_sha is not None and checksum != expected_sha:
        raise ValueError('Payload differs from its sealed manifest')
    if link:
        assert_not_in_use(source)
        temporary = source.with_name('.' + source.name + '.oss-link-' + uuid.uuid4().hex)
        try:
            temporary.symlink_to(destination)
            if identity(source) != original:
                raise ValueError('Source changed before link replacement')
            os.replace(temporary, source)
        finally:
            temporary.unlink(missing_ok=True)
        if not source.is_symlink() or source.resolve() != destination.resolve():
            raise ValueError('Published symlink is incorrect')
    return {'bytes': original[1], 'sha256': checksum, 'linked': link}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('Use 1–32 transfer workers')
    args.database.parent.mkdir(parents=True, exist_ok=True)
    with args.database.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(args.database)
        db.execute('CREATE TABLE IF NOT EXISTS transfers (source TEXT PRIMARY KEY, destination TEXT NOT NULL, link INTEGER NOT NULL, bytes INTEGER, sha256 TEXT, state TEXT NOT NULL, error TEXT)')
        plan = [json.loads(line) for line in args.plan.read_text().split('\n') if line.strip()]
        if len({item['source'] for item in plan}) != len(plan) or len({item['destination'] for item in plan}) != len(plan):
            raise ValueError('Plan contains duplicate sources or destinations')
        for item in plan:
            old = db.execute('SELECT destination,link FROM transfers WHERE source=?', (item['source'],)).fetchone()
            if old and old != (item['destination'], int(item.get('link', False))):
                raise ValueError('Transfer plan changed under the existing journal')
            db.execute('INSERT OR IGNORE INTO transfers VALUES (?,?,?,?,?,?,?)', (item['source'], item['destination'], int(item.get('link', False)), item.get('bytes'), None, 'pending', None))
        db.commit()
        pending = [item for item in plan if db.execute('SELECT state FROM transfers WHERE source=?', (item['source'],)).fetchone()[0] != 'verified']
        began = time.monotonic()
        last_report = began
        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(transfer, item['source'], item['destination'], item.get('link', False), item.get('sha256')): item for item in pending}
            for completed, task in enumerate(as_completed(tasks), 1):
                item = tasks[task]
                try:
                    result = task.result()
                    db.execute('UPDATE transfers SET bytes=?,sha256=?,state=?,error=NULL WHERE source=?', (result['bytes'], result['sha256'], 'verified', item['source']))
                except Exception as error:
                    errors += 1
                    db.execute('UPDATE transfers SET state=?,error=? WHERE source=?', ('error', repr(error), item['source']))
                    print(json.dumps({'error': str(error), 'source': item['source']}), flush=True)
                if completed % 32 == 0:
                    db.commit()
                if time.monotonic() - last_report > 20:
                    count, size = db.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM transfers WHERE state='verified'").fetchone()
                    print(json.dumps({'verified_files': count, 'verified_bytes': size, 'total_files': len(plan), 'seconds': time.monotonic()-began, 'errors': errors}), flush=True)
                    last_report = time.monotonic()
        db.commit()
        count, size = db.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM transfers WHERE state='verified'").fetchone()
        summary = {'complete': count == len(plan), 'verified_files': count, 'total_files': len(plan), 'verified_bytes': size, 'errors': errors, 'seconds': time.monotonic()-began}
        args.database.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)
        if errors:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
