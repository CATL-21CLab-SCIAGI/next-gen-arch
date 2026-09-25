"""Canonical Miles V4.1 launcher: resolve a contract, then delegate to train.py.

See docs/MILES_BASELINE.md. Render requires only the CPU project environment.
Check/train require the pinned runtime. This module never orchestrates RL steps.
"""

import argparse
import importlib.metadata
import json
import os
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from archlab.megatron.miles_v41_stock_config import digest, resolve


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def verify_runtime(plan, miles, image_manifest):
    """Check identities before importing GPU libraries or connecting to Ray."""
    revision = git(miles, 'rev-parse', 'HEAD')
    if revision != plan['upstream']['revision'] or git(miles, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Miles must be a clean checkout of the contract revision')
    image = json.loads(image_manifest.read_text())
    if plan['runtime_expected']['image'] not in image.get('repo_digests', []):
        raise ValueError('runtime image manifest does not match the contract')
    versions = {name: importlib.metadata.version(name) for name in plan['runtime_expected']['packages']}
    if versions != plan['runtime_expected']['packages']:
        raise ValueError(f'runtime package versions differ from contract: {versions}')
    model_dir = Path(plan['bindings']['MODEL_DIR'])
    model = json.loads((model_dir / 'config.json').read_text())
    if model.get('archlab', {}).get('variant') != 'normal':
        raise ValueError('this contract is for the normal finetuned checkpoint')
    if model.get('quantization_config') != {
        'quant_method': 'fp8', 'activation_scheme': 'dynamic',
        'fmt': 'e4m3', 'weight_block_size': [128, 128],
    }:
        raise ValueError('model config must select native FP8 block128 rollout')
    parent = model['archlab']['full_checkpoint']
    if not Path(parent).is_dir():
        raise ValueError(f'parent checkpoint is unavailable: {parent}')
    if digest(Path(parent) / 'COMPLETE.json') != model['archlab']['complete_sha256']:
        raise ValueError('parent checkpoint completion identity differs from model config')
    scratch = Path(plan['bindings']['RUN_ROOT']) / 'offload'
    filesystem = subprocess.check_output(
        ['findmnt', '-T', str(scratch), '-n', '-o', 'FSTYPE'], text=True).strip()
    if not scratch.is_dir() or filesystem not in ('overlay', 'ext4', 'xfs'):
        raise ValueError('optimizer scratch must be mounted on node-local disk inside the runtime')
    data = Path(plan['bindings']['RUN_ROOT']) / 'train.jsonl'
    if not data.is_file():
        raise ValueError(f'preformatted prompt data is unavailable: {data}')
    return {'miles_revision': revision, 'packages': versions, 'image': image['repo_digests'],
            'image_manifest_sha256': digest(image_manifest), 'model_config_sha256': digest(model_dir / 'config.json'),
            'parent_checkpoint': parent, 'parent_complete_sha256': model['archlab']['complete_sha256'],
            'prompt_data_sha256': digest(data), 'optimizer_filesystem': filesystem}


def write_record(path, record):
    # Never overwrite an earlier launch identity, including a failed attempt.
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('render', 'check', 'train'))
    parser.add_argument('--config', type=Path, default=Path('recipes/experiments/deepseek_v41_stock_fp8.yaml'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, help='Prepared tokenizer/config bundle; defaults to RUN_ROOT/model')
    parser.add_argument('--miles', type=Path, required=True)
    parser.add_argument('--address', required=True)
    parser.add_argument('--image-manifest', type=Path, help='IMAGE_RUNTIME.json from the pinned runtime deployment')
    cli = parser.parse_args()
    root, miles = cli.run_root.resolve(), cli.miles.resolve()
    plan = resolve(cli.config, run_root=root, model_dir=cli.model_dir or root / 'model', address=cli.address)
    plan['command'] = [sys.executable, str(miles / 'train.py'), *plan['argv']]
    if cli.action == 'render':
        print(json.dumps(plan, indent=2))
        return
    if cli.image_manifest is None:
        parser.error('--image-manifest is required for check/train')
    record_path = root / 'resolved-launch.json'
    if cli.action == 'train' and (record_path.exists() or (root / 'checkpoints').exists()):
        raise ValueError('use a fresh run root; this contract starts fresh Adam from the parent, not an RL resume')
    plan['runtime_observed'] = verify_runtime(plan, miles, cli.image_manifest)
    project = Path(__file__).resolve().parents[3]
    plan['project'] = {'revision': git(project, 'rev-parse', 'HEAD'),
                       'dirty': bool(git(project, 'status', '--porcelain', '--untracked-files=no'))}
    plan['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    # Capture only runtime controls, never the full environment (which may hold credentials).
    inherited = {key: os.environ[key] for key in ('PYTHONPATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
                 'CUDA_HOME') if key in os.environ}
    plan['environment'] = inherited | plan['environment']
    if os.environ.get('EVERGREENTREE_WEIGHT_CACHE_DIR') or os.environ.get('ARCHLAB_RL_OFFLOAD_POLICY'):
        raise ValueError('historical cache/offload environment conflicts with this contract')
    os.environ.update(plan['environment'])
    sys.path.insert(0, str(miles))
    import miles as miles_package

    if not Path(miles_package.__file__).resolve().is_relative_to(miles):
        raise ValueError('imported Miles does not match --miles')
    sys.argv = [str(miles / 'train.py'), *plan['argv']]
    if cli.action == 'check':
        from miles.utils.arguments import parse_args

        parse_args()
        print('STOCK_MILES_ARGUMENTS_ACCEPTED', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    write_record(record_path, plan)
    (root / 'launch-argv.json').write_text(json.dumps(plan['argv'], indent=2) + '\n')
    import ray

    ray.init(address=cli.address, runtime_env={'env_vars': plan['environment']})
    runpy.run_path(str(miles / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
