"""CPU contracts, not distributed numerical qualification."""

import json
from pathlib import Path

import pytest

from archlab.megatron.miles_v41_stock_config import resolve
from archlab.megatron.miles_v41_stock_launch import main, verify_runtime, write_record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'recipes/experiments/deepseek_v41_stock_fp8.yaml'


def effective(argv):
    result = {}
    for token in argv:
        if token.startswith('--'):
            key = token
            result[key] = []
        else:
            result[key].append(token)
    return result


def plan(tmp_path):
    return resolve(CONFIG, run_root=tmp_path, model_dir=tmp_path / 'model', address='head:17379')


def test_matches_qualified_effective_arguments(tmp_path):
    previous = json.loads((ROOT / 'tests/fixtures/miles_v41_qualified_argv.json').read_text())['argv']
    bindings = {'${RUN_ROOT}': str(tmp_path), '${MODEL_DIR}': str(tmp_path / 'model')}
    for key, value in bindings.items():
        previous = [arg.replace(key, value) for arg in previous]
    resolved = plan(tmp_path)
    assert effective(resolved['argv']) == effective(previous)
    flags = [arg for arg in resolved['argv'] if arg.startswith('--')]
    assert len(flags) == len(set(flags))
    assert '--apply-chat-template' not in flags
    assert '--load' not in flags  # Fresh optimizer contract, never an accidental resume.
    assert resolved['environment']['ARCHLAB_RL_FREEZE_ENGRAM'] == '1'
    assert resolved['environment']['ARCHLAB_MILES_RESIDENT_POLICY'] == '1'


def test_paths_with_spaces_and_rollout_placeholder_are_literal(tmp_path):
    root = tmp_path / 'run $(no-shell) with spaces'
    resolved = plan(root)
    assert str(root / 'rollout-{rollout_id}.pt') in resolved['argv']
    assert str(root / 'model') in resolved['argv']
    assert not root.exists()


def test_contract_identity_tracks_numerical_changes_not_machine_paths(tmp_path):
    first = plan(tmp_path)
    assert first['contract_sha256'] == plan(tmp_path / 'other')['contract_sha256']
    changed = tmp_path / 'contract.yaml'
    changed.write_text(CONFIG.read_text().replace('--lr: 1e-6', '--lr: 2e-6'))
    assert resolve(changed, run_root=tmp_path, model_dir=tmp_path, address='x')['contract_sha256'] != first['contract_sha256']


@pytest.mark.parametrize('replacement', ['  --lr: 2e-6\n', ''])
def test_duplicate_and_unknown_bindings_are_rejected(tmp_path, replacement):
    changed = tmp_path / 'contract.yaml'
    text = CONFIG.read_text()
    if replacement:
        text = text.replace('  --lr: 1e-6\n', '  --lr: 1e-6\n' + replacement)
    else:
        text = text.replace('${MODEL_DIR}', '${TYPO}')
    changed.write_text(text)
    with pytest.raises(ValueError, match='duplicate|unknown'):
        resolve(changed, run_root=tmp_path, model_dir=tmp_path, address='x')


def test_launch_record_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'resolved-launch.json'
    write_record(path, {'experiment': 'first'})
    with pytest.raises(FileExistsError):
        write_record(path, {'experiment': 'second'})
    assert json.loads(path.read_text()) == {'experiment': 'first'}


def test_render_has_no_runtime_or_filesystem_side_effects(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'absent'
    monkeypatch.setattr('sys.argv', ['launch', 'render', '--config', str(CONFIG), '--run-root', str(root),
                                   '--miles', '/missing/miles', '--address', 'head:17379'])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['command'][1] == '/missing/miles/train.py'
    assert not root.exists()


def test_revision_mismatch_fails_before_runtime_import(tmp_path, monkeypatch):
    monkeypatch.setattr('archlab.megatron.miles_v41_stock_launch.git', lambda *a: 'wrong')
    with pytest.raises(ValueError, match='clean checkout'):
        verify_runtime(plan(tmp_path), tmp_path, tmp_path / 'absent.json')


def test_train_refuses_existing_run_before_runtime_import(tmp_path, monkeypatch):
    (tmp_path / 'checkpoints').mkdir()
    monkeypatch.setattr('sys.argv', ['launch', 'train', '--config', str(CONFIG), '--run-root', str(tmp_path),
                                   '--miles', '/missing/miles', '--address', 'head:17379',
                                   '--image-manifest', '/missing/image.json'])
    with pytest.raises(ValueError, match='fresh run root'):
        main()


@pytest.fixture
def runtime_inputs(tmp_path, monkeypatch):
    from archlab.megatron.miles_v41_stock_config import digest

    resolved = plan(tmp_path)
    (tmp_path / 'model').mkdir()
    (tmp_path / 'parent').mkdir()
    (tmp_path / 'parent/COMPLETE.json').write_text('{}')
    (tmp_path / 'offload').mkdir()
    (tmp_path / 'train.jsonl').write_text('{"prompt":"prepared","label":"1"}\n')
    model = {'archlab': {'variant': 'normal', 'full_checkpoint': str(tmp_path / 'parent'),
                        'complete_sha256': digest(tmp_path / 'parent/COMPLETE.json')},
             'quantization_config': {'quant_method': 'fp8', 'activation_scheme': 'dynamic',
                                     'fmt': 'e4m3', 'weight_block_size': [128, 128]}}
    (tmp_path / 'model/config.json').write_text(json.dumps(model))
    image = tmp_path / 'image.json'
    image.write_text(json.dumps({'repo_digests': [resolved['runtime_expected']['image']]}))
    monkeypatch.setattr('archlab.megatron.miles_v41_stock_launch.git',
                        lambda path, *args: resolved['upstream']['revision'] if args[0] == 'rev-parse' else '')
    monkeypatch.setattr('importlib.metadata.version', resolved['runtime_expected']['packages'].__getitem__)
    monkeypatch.setattr('subprocess.check_output', lambda *args, **kwargs: 'ext4\n')
    return resolved, image


def test_runtime_identity_and_input_checksums(tmp_path, runtime_inputs):
    resolved, image = runtime_inputs
    observed = verify_runtime(resolved, tmp_path, image)
    assert observed['packages'] == resolved['runtime_expected']['packages']
    assert len(observed['prompt_data_sha256']) == 64
    assert observed['optimizer_filesystem'] == 'ext4'


@pytest.mark.parametrize('mutation,match', [
    ('image', 'image manifest'), ('version', 'package versions'),
    ('variant', 'normal finetuned'), ('parent', 'completion identity'),
    ('scratch', 'node-local disk'), ('quantization', 'FP8'),
])
def test_runtime_rejects_wrong_contract(tmp_path, runtime_inputs, monkeypatch, mutation, match):
    resolved, image = runtime_inputs
    if mutation == 'image':
        image.write_text('{"repo_digests": ["wrong"]}')
    elif mutation == 'version':
        monkeypatch.setattr('importlib.metadata.version', lambda name: 'wrong')
    elif mutation == 'parent':
        (tmp_path / 'parent/COMPLETE.json').write_text('{"changed":true}')
    elif mutation == 'scratch':
        monkeypatch.setattr('subprocess.check_output', lambda *args, **kwargs: 'nfs4\n')
    else:
        config = tmp_path / 'model/config.json'
        model = json.loads(config.read_text())
        if mutation == 'variant':
            model['archlab']['variant'] = 'simplicial'
        else:
            model.pop('quantization_config')
        config.write_text(json.dumps(model))
    with pytest.raises(ValueError, match=match):
        verify_runtime(resolved, tmp_path, image)


def test_train_records_identity_before_delegating_to_upstream(tmp_path, monkeypatch):
    import os
    import sys
    from types import ModuleType

    miles = ModuleType('miles')
    miles.__file__ = str(tmp_path / 'miles/__init__.py')
    ray = ModuleType('ray')
    events = []

    def init(**kwargs):
        record = json.loads((tmp_path / 'resolved-launch.json').read_text())
        assert record['argv'] == sys.argv[1:]
        assert record['environment'] == kwargs['runtime_env']['env_vars']
        assert 'SECRET_TOKEN' not in record['environment']
        events.append('ray')

    def run(path, *, run_name):
        assert path == str(tmp_path / 'miles/train.py')
        assert run_name == '__main__'
        events.append('upstream_driver')

    ray.init = init
    monkeypatch.setitem(sys.modules, 'miles', miles)
    monkeypatch.setitem(sys.modules, 'ray', ray)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(os, 'environ', {'PYTHONPATH': '/prepared/runtime', 'SECRET_TOKEN': 'not-logged'})
    monkeypatch.setattr('archlab.megatron.miles_v41_stock_launch.verify_runtime', lambda *a: {})
    monkeypatch.setattr('archlab.megatron.miles_v41_stock_launch.git', lambda *a: 'revision')
    monkeypatch.setattr('runpy.run_path', run)
    monkeypatch.setattr(sys, 'argv', ['launch', 'train', '--config', str(CONFIG), '--run-root', str(tmp_path),
                                   '--miles', str(tmp_path / 'miles'), '--address', 'head:17379',
                                   '--image-manifest', '/prepared/image.json'])
    main()
    assert events == ['ray', 'upstream_driver']


def test_2simplicial_contract_restores_sampling_and_length_budget(tmp_path):
    config = ROOT / 'recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml'
    resolved = resolve(config, run_root=tmp_path, model_dir=tmp_path / 'model', address='head:17379')
    args = effective(resolved['argv'])
    assert resolved['semantics']['variant'] == 'simplicial'
    assert args['--rollout-max-response-len'] == ['4096']
    assert args['--sglang-context-length'] == ['5120']
    assert args['--dynamic-sampling-filter-path'] == ['miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std']
    assert '--use-tis' in args
    assert args['--over-sampling-batch-size'] == ['512']
    assert args['--eval-max-response-len'] == ['4096']
    assert args['--eval-prompt-data'] == ['heldout', str(tmp_path / 'heldout-32.jsonl')]
    assert args['--eval-interval'] == args['--save-interval'] == ['20']
    # Serving requests fit the fixed KV token budget at the configured full context.
    assert int(args['--sglang-max-running-requests'][0]) * int(args['--sglang-context-length'][0]) <= int(args['--sglang-max-total-tokens'][0])


def test_simplicial_runtime_requires_matching_variant_and_lineage(tmp_path, runtime_inputs):
    resolved, image = runtime_inputs
    resolved['semantics']['variant'] = 'simplicial'
    config = tmp_path / 'model/config.json'
    model = json.loads(config.read_text())
    with pytest.raises(ValueError, match='simplicial finetuned'):
        verify_runtime(resolved, tmp_path, image)
    model['archlab']['variant'] = 'simplicial'
    config.write_text(json.dumps(model))
    verify_runtime(resolved, tmp_path, image)
    resolved['semantics']['parent_complete_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='lineage'):
        verify_runtime(resolved, tmp_path, image)


def test_source_checkout_megatron_version_is_supported(tmp_path, monkeypatch):
    import importlib.metadata
    from types import SimpleNamespace

    from archlab.megatron.miles_v41_stock_launch import runtime_version

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    package = tmp_path / 'megatron'
    (package / 'core').mkdir(parents=True)
    (package / 'core/package_info.py').write_text("__version__ = '0.19.0+651dd72'\n")
    monkeypatch.setattr('importlib.metadata.version', missing)
    monkeypatch.setattr('importlib.util.find_spec', lambda name: SimpleNamespace(submodule_search_locations=[str(package)]))
    monkeypatch.setattr('archlab.megatron.miles_v41_stock_launch.git', lambda *args: '')
    assert runtime_version('megatron-core') == '0.19.0+651dd72'
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        runtime_version('torch')


def test_simplicial_rejects_deterministic_mode_before_launch(tmp_path):
    import yaml

    config = ROOT / 'recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml'
    contract = yaml.safe_load(config.read_text())
    assert '--deterministic-mode' not in contract['arguments']
    contract['arguments']['--deterministic-mode'] = None
    incompatible = tmp_path / 'incompatible.yaml'
    incompatible.write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match='simplicial atomic backward'):
        resolve(incompatible, run_root=tmp_path, model_dir=tmp_path / 'model', address='localhost:6379')
