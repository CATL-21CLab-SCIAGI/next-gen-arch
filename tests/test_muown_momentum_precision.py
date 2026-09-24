import pytest
import torch

from archlab.optimizers.muown import Muown


@pytest.mark.parametrize("gradient_scale", [1e-12, 1., 1e4])
def test_scaled_fp16_retains_history_across_gradient_ranges(gradient_scale):
    torch.manual_seed(719)
    parameter = torch.nn.Parameter(torch.randn(16, 24) * .02)
    reference = torch.nn.Parameter(parameter.detach().clone())
    directions = []
    selected = Muown([parameter], momentum_dtype=torch.float16,
                     orthogonalization_dtype=torch.float32,
                     direction_observer=lambda x: directions.append(x.clone()))
    oracle = Muown([reference], momentum_dtype=torch.float32,
                   orthogonalization_dtype=torch.float32,
                   direction_observer=lambda x: directions.append(x.clone()))
    for _ in range(12):
        parameter.grad = torch.randn_like(parameter) * gradient_scale
        reference.grad = parameter.grad.clone()
        selected.step()
        oracle.step()
        actual, expected = directions
        assert (actual-expected).norm() / expected.norm().clamp_min(1e-20) < .01
        assert torch.isfinite(parameter).all()
        directions.clear()
    state = selected.state[parameter]
    assert state["momentum_buffer"].dtype == torch.float16
    assert state["momentum_scale"].dtype == torch.float32
    assert state["momentum_buffer"].abs().max() > 0
