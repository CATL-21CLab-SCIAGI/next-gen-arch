"""Bind sequential oracle outputs to exact weights, batches and EP ownership."""

from copy import deepcopy

import pytest
import torch

from archlab.automodel.deepseek_v41_official_qualification import (
    batch_digest, tensor_digest, validate_prepared_reference,
)


def test_tensor_digest_binds_exact_values_shape_and_dtype():
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    original = tensor_digest(value)
    assert tensor_digest(value.clone()) == original
    assert tensor_digest(value.T.contiguous().T) == original
    for changed in (value.reshape(6, 4), value.double(), value + 1):
        assert tensor_digest(changed) != original


def test_batch_digest_binds_both_tokens_and_supervised_mask():
    inputs = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[-100, 2, 3]])
    original = batch_digest(inputs, labels)
    assert batch_digest(inputs.clone(), labels.clone()) == original
    assert batch_digest(inputs + 1, labels) != original
    assert batch_digest(inputs, torch.tensor([[1, 2, 3]])) != original


def _prepared():
    return {"rank": 9, "contexts": [128, 2048, 16384], "head_sha256": "verified-head-bytes",
            "outputs": {128: {}, 2048: {}, 16384: {}},
            "reference_loading": {"expert_owner_ranks": list(range(8, 16)),
                                  "engram_owner_ranks": list(range(32))}}


def test_reference_requires_same_rank_contexts_and_native_expert_batches():
    prepared = _prepared()
    validate_prepared_reference(prepared, rank=9, contexts=(128, 2048, 16384))
    changes = [
        {"rank": 8}, {"contexts": [128]}, {"outputs": {128: {}}}, {"head_sha256": ""},
        {"reference_loading": {"expert_owner_ranks": list(range(32)), "engram_owner_ranks": list(range(32))}},
        {"reference_loading": {"expert_owner_ranks": list(range(8, 16)), "engram_owner_ranks": list(range(8))}},
    ]
    for change in changes:
        candidate = deepcopy(prepared)
        candidate.update(change)
        with pytest.raises(ValueError, match="matching EP8/Engram32"):
            validate_prepared_reference(candidate, rank=9, contexts=(128, 2048, 16384))
