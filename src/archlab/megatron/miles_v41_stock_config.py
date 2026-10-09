"""Portable argument resolution for the single supported Miles V4.1 experiment."""

import hashlib
import json
import re
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    """Reject accidental YAML overrides instead of silently changing an experiment."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(config, *, run_root, model_dir, address):
    """Resolve explicit paths; never import upstream, evaluate shell, or mutate state."""
    contract = yaml.load(Path(config).read_text(), Loader=UniqueLoader)
    if contract.get('schema_version') != 1:
        raise ValueError('unsupported Miles contract schema')
    if (contract.get('semantics', {}).get('variant') == 'simplicial'
            and '--deterministic-mode' in contract['arguments']):
        raise ValueError('simplicial atomic backward is incompatible with --deterministic-mode')
    bindings = {'RUN_ROOT': str(Path(run_root).resolve()),
                'MODEL_DIR': str(Path(model_dir).resolve()), 'RAY_ADDRESS': address}

    def expand(value):
        if not isinstance(value, str):
            raise ValueError(f'argument/environment values must be quoted strings: {value!r}')

        def replacement(match):
            name = match.group(1)
            if name not in bindings:
                raise ValueError(f'unknown path binding: {name}')
            return bindings[name]

        return re.sub(r'\$\{([^}]+)\}', replacement, value)

    argv = []
    for key, value in contract['arguments'].items():
        if not isinstance(key, str) or not key.startswith('--') or '=' in key:
            raise ValueError(f'invalid Miles option: {key!r}')
        argv.append(key)
        values = [] if value is None else value if isinstance(value, list) else [value]
        argv.extend(expand(v) for v in values)
    environment = {key: expand(value) for key, value in contract['environment'].items()}
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return {'schema_version': 1, 'experiment': contract['name'], 'contract_sha256': identity,
            'config_file_sha256': digest(config), 'upstream': contract['upstream'],
            'runtime_expected': contract['runtime'], 'semantics': contract['semantics'],
            'bindings': bindings, 'argv': argv, 'environment': environment}
