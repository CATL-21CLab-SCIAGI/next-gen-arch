"""Select completed bulk result payloads without traversing immutable Git repos."""
import argparse
import json
import os
from pathlib import Path

EXTENSIONS = {'.pt', '.pth', '.safetensors', '.bin', '.idx', '.npy', '.npz', '.arrow', '.parquet', '.gguf', '.u32', '.u16'}


def build(source, destination, minimum_bytes=4*1024*1024):
    source, destination = source.absolute(), destination.absolute()
    tasks, pending, repos = [], [], []
    folders = [source]
    while folders:
        folder = folders.pop()
        with os.scandir(folder) as scan:
            entries = list(scan)
        names = {entry.name for entry in entries}
        if '.git' in names:
            repos.append(str(folder))
            continue
        if folder.name.startswith('step-') and folder.parent.name == 'checkpoints' and 'COMPLETE.json' not in names:
            pending.append(str(folder))
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if not entry.name.startswith('.retention-delete-'):
                    folders.append(path)
            elif path.suffix.lower() in EXTENSIONS and entry.is_file(follow_symlinks=False):
                info = entry.stat(follow_symlinks=False)
                if info.st_size >= minimum_bytes:
                    tasks.append({'source': str(path), 'destination': str(destination/path.relative_to(source)), 'bytes': info.st_size, 'link': True})
    return tasks, {'files': len(tasks), 'bytes': sum(t['bytes'] for t in tasks),
                   'incomplete_checkpoints_skipped': pending, 'source_repositories_skipped': len(repos),
                   'metadata_stays_on_nas': True, 'minimum_bytes': minimum_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks, summary = build(args.source, args.destination)
    args.output.write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
