import pytest
import torch

from archlab.rl.limite_update_oracle import first_adam_updates


def test_first_step_oracle_matches_clipped_native_gradients_and_torch_fp32_adam_masters():
    native = {
        "matrix": torch.nn.Parameter(torch.tensor([[.1, -.2], [.3, -.4]], dtype=torch.bfloat16)),
        "scale": torch.nn.Parameter(torch.tensor([1., -.00001, .01], dtype=torch.float32)),
        "unused": torch.nn.Parameter(torch.tensor(2., dtype=torch.float32)),
    }
    gradients = {
        "matrix": torch.tensor([[2., -3.], [1e-7, -1e-8]], dtype=torch.bfloat16),
        "scale": torch.tensor([1e-6, -1e-5, 0.]), "unused": None,
    }
    before = {key: value.detach().clone() for key, value in native.items()}
    saved_gradients = {key: None if value is None else value.clone() for key, value in gradients.items()}
    actual, evidence = first_adam_updates(before, gradients)
    for key, parameter in native.items():
        parameter.grad = None if gradients[key] is None else gradients[key].clone()
    norm = torch.nn.utils.clip_grad_norm_(native.values(), 1., foreach=False)
    masters = {key: torch.nn.Parameter(value.float().clone()) for key, value in before.items()}
    optimizer = torch.optim.Adam(masters.values(), lr=1e-5, betas=(.9, .95), eps=1e-8, foreach=False)
    for key, parameter in masters.items():
        parameter.grad = None if native[key].grad is None else native[key].grad.float()
    optimizer.step()
    assert evidence["gradient_norm"] == float(norm) and evidence["clip_coefficient"] < 1
    for key, parameter in masters.items():
        if gradients[key] is None:
            assert actual[key] is None
        else:
            torch.testing.assert_close(actual[key], parameter.detach() - before[key].float(), rtol=1e-5, atol=2e-9)
        assert torch.equal(before[key], native[key].detach())
        if gradients[key] is not None:
            assert torch.equal(gradients[key], saved_gradients[key])


def test_missing_gradients_are_a_no_step_and_dtype_changes_fail():
    parameters = {"scale": torch.ones(2)}
    updates, evidence = first_adam_updates(parameters, {"scale": None})
    assert updates == {"scale": None} and evidence["no_step"]
    with pytest.raises(ValueError, match="native"):
        first_adam_updates(parameters, {"scale": torch.ones(2, dtype=torch.bfloat16)})
    with pytest.raises(FloatingPointError, match="nonfinite"):
        first_adam_updates(parameters, {"scale": torch.tensor([float("nan"), 1.])})
