import copy

import pytest
import torch

from archlab.megatron.miles_v41_inplace_restore import load_muown_in_place
from archlab.optimizers.muown import Muown


@torch.no_grad()
def test_restore_preserves_fp16_history_and_all_storage_then_resumes_exactly():
    torch.manual_seed(4)
    parameter = torch.nn.Parameter(torch.randn(8, 12) * .02)
    optimizer = Muown([parameter], lr=3e-6, momentum_dtype=torch.float16,
                      orthogonalization_dtype=torch.float32)
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    weights = parameter.detach().clone()
    addresses = {key: value.data_ptr() for key, value in optimizer.state[parameter].items()
                 if isinstance(value, torch.Tensor)}
    for value in optimizer.state[parameter].values():
        if isinstance(value, torch.Tensor):
            value.zero_()
    optimizer.state[parameter]["step"] = 0
    load_muown_in_place(optimizer, saved)
    for key, value in optimizer.state[parameter].items():
        expected = saved["state"][0][key]
        if isinstance(value, torch.Tensor):
            assert value.data_ptr() == addresses[key]
            assert value.dtype == expected.dtype
            assert torch.equal(value, expected)
        else:
            assert value == expected
    reference_parameter = torch.nn.Parameter(weights.clone())
    reference = Muown([reference_parameter], lr=3e-6, momentum_dtype=torch.float16,
                       orthogonalization_dtype=torch.float32)
    reference._init_group(reference.param_groups[0], skip_non_grad_params=False)
    load_muown_in_place(reference, saved)
    for _ in range(3):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        optimizer.step()
        reference.step()
        assert torch.equal(parameter, reference_parameter)


@torch.no_grad()
def test_restore_rejects_dtype_change_before_modifying_storage():
    parameter = torch.nn.Parameter(torch.randn(8, 12))
    optimizer = Muown([parameter], momentum_dtype=torch.float16)
    optimizer._init_group(optimizer.param_groups[0], skip_non_grad_params=False)
    saved = copy.deepcopy(optimizer.state_dict())
    before = optimizer.state[parameter]["g"].clone()
    saved["state"][0]["g"].zero_()
    saved["state"][0]["momentum_buffer"] = saved["state"][0]["momentum_buffer"].float()
    with pytest.raises(ValueError, match="tensor contract"):
        load_muown_in_place(optimizer, saved)
    assert torch.equal(optimizer.state[parameter]["g"], before)
