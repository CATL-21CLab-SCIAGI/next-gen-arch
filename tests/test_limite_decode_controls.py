from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from archlab.automodel.limite_decode_controls import (
    cache_equal,
    cache_selection_equal,
    decode_reference,
)


def fixed_cache():
    cache = SimpleNamespace(
        global_layers={0}, position=torch.tensor([7]),
        buffers=[[torch.arange(24).reshape(2, 1, 3, 4).float(), torch.ones(2, 1, 3, 4)]],
        archlab_short={0: [torch.ones(2, 3, 1, 4), torch.zeros(2, 3, 1, 4)]},
        get_seq_length=lambda: 7,
    )
    cache.archlab_native_preludes = SimpleNamespace(
        global_layers={0}, position=torch.tensor([7]),
        buffers=[[torch.ones(2, 1, 3, 4), torch.zeros(2, 1, 3, 4)]],
        archlab_short={}, get_seq_length=lambda: 7,
    )
    return cache


@pytest.mark.parametrize("changed", ["backbone", "prelude", "short", "position", "dtype", "topology"])
def test_cache_control_detects_every_owned_history(changed):
    expected = fixed_cache()
    actual = deepcopy(expected)
    assert cache_equal(actual, expected)
    if changed == "backbone":
        actual.buffers[0][0][0, 0, 0, 0] += 1
    elif changed == "prelude":
        actual.archlab_native_preludes.buffers[0][1][0, 0, 0, 0] += 1
    elif changed == "short":
        actual.archlab_short[0][0][0, 0, 0, 0] += 1
    elif changed == "position":
        actual.position.add_(1)
    elif changed == "dtype":
        actual.buffers[0][0] = actual.buffers[0][0].to(torch.bfloat16)
    else:
        del actual.archlab_native_preludes
    assert not cache_equal(actual, expected)


@pytest.mark.parametrize("simplicial", [False, True])
def test_decode_reference_matches_explicit_per_head_pair_sum(simplicial):
    generator = torch.Generator().manual_seed(194)
    q = torch.randn(2, 1, 4, 3, generator=generator)
    k, v = [torch.randn(2, 5, 2, 3, generator=generator) for _ in range(2)]
    short = tuple(torch.randn(2, 3, 2, 3, generator=generator) for _ in range(2)) if simplicial else None
    actual = decode_reference(q, k, v, scaling=.4, short=short)
    expected = torch.empty_like(actual)
    for batch in range(2):
        for head in range(4):
            kv_head = head // 2
            if short is None:
                scores = k[batch, :, kv_head] @ q[batch, 0, head] * .4
                expected[batch, 0, head] = scores.softmax(0) @ v[batch, :, kv_head]
            else:
                scores, values = [], []
                for short_index in range(3):
                    for long_index in range(5):
                        scores.append((q[batch, 0, head] * short[0][batch, short_index, kv_head]
                                       * k[batch, long_index, kv_head]).sum() * .4)
                        values.append(short[1][batch, short_index, kv_head] * v[batch, long_index, kv_head])
                expected[batch, 0, head] = torch.stack(scores).softmax(0) @ torch.stack(values)
    torch.testing.assert_close(actual, expected)


def test_migration_control_checks_original_row_mapping_and_unused_tail():
    source = fixed_cache()
    selected = deepcopy(source)
    source.buffers[0] = [torch.arange(40).reshape(2, 1, 5, 4).float(), torch.ones(2, 1, 5, 4)]
    selected.buffers[0] = [value[[1], :, :3].clone() for value in source.buffers[0]]
    selected.buffers[0] = [torch.cat((value, torch.zeros(1, 1, 2, 4)), dim=2) for value in selected.buffers[0]]
    selected.archlab_short = {0: [value[[1]].clone() for value in source.archlab_short[0]]}
    selected.get_seq_length = lambda: 3
    selected.position = torch.tensor([3])
    del source.archlab_native_preludes
    del selected.archlab_native_preludes
    assert cache_selection_equal(source, selected, torch.tensor([1]), length=3)
    assert not cache_selection_equal(source, selected, torch.tensor([0]), length=3)
    selected.buffers[0][0][0, 0, -1, 0] = 17
    assert not cache_selection_equal(source, selected, torch.tensor([1]), length=3)
