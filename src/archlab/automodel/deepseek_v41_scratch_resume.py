"""Narrow admission of the recorded TileLang metadata/qualification correction."""
from __future__ import annotations

import copy


TILELANG_PARENT = '375711979eb99df5d3d2916ea36e85c8e531bc36'


def maintenance_resume_changes(parent, current):
    """Return None for other lineages; reject any math/data change in this one."""
    if parent == current:
        return []
    if parent.get('project_commit') != TILELANG_PARENT:
        return None
    if parent.get('variant') not in ('normal', 'simplicial'):
        raise ValueError('maintenance continuation is only for the recorded TileLang pair')
    old, new = copy.deepcopy(parent), copy.deepcopy(current)
    for value in (old, new):
        value.pop('project_commit')
        value.pop('implementation_sha256')
        value['runtime'].pop('resolved_kernel_packages', None)
        value['runtime']['sparse_precision'].pop('sink_gradient_reduction', None)
    if old != new:
        raise ValueError('maintenance continuation changed training, data, geometry, or runtime semantics')
    mutable = {'automodel/deepseek_v41_scratch_training.py',
               'automodel/deepseek_v41_scratch_high_mfu.py'}
    additions = {'automodel/deepseek_v41_scratch_resume.py',
                 'automodel/deepseek_v41_scratch_sparse_probe.py',
                 'automodel/deepseek_v41_sparse_qualification.py',
                 'automodel/deepseek_v41_official_adapter.py',
                 'architectures/linsimp_attention.py',
                 'architectures/deepseek_v41_linsimp_adapter.py'}
    before, after = parent['implementation_sha256'], current['implementation_sha256']
    if before.keys() - after.keys() or after.keys() - before.keys() - additions:
        raise ValueError('maintenance continuation changed the implementation inventory')
    for name, digest in before.items():
        if name not in mutable and digest != after[name]:
            raise ValueError(f'maintenance continuation changed protected implementation: {name}')
    return ['independent sparse forward/dq/dkv/dsink qualification',
            'record resolved container kernel modules', 'record sink-gradient atomics']
