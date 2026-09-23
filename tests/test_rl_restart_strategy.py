import copy

import pytest
import torch
from torch import nn

from archlab.automodel.deepseek_v41_rl_memory import (
    configure_gpu_budget, input_offload_statistics, install_checkpoint_input_offload,
)
from archlab.automodel.deepseek_v41_rl_model import configure_rl_trainability
from archlab.rl.objectives import group_relative_policy_loss
from archlab.rl.regularization import DEFAULT_LENGTH_PENALTY, successful_length_rewards


def test_prompt_mean_weights_each_group_after_its_own_token_normalization():
    logp = torch.tensor([[[-.2, -.3, -.4], [-.5, -.6, -.7]],
                         [[-.4, -.5, -.6], [-.7, -.8, -.9]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[1, 1, 0], [1, 0, 0]], [[1, 1, 1], [1, 1, 1]]], dtype=torch.bool)
    advantages = torch.tensor([[1., -1.], [.5, -.5]], dtype=torch.float64)
    result = group_relative_policy_loss(logp, mask, advantages, normalization="prompt_token_mean")
    oracle = -(((logp[0, 0, :2].sum() - logp[0, 1, 0]) / 3) +
               ((.5 * logp[1, 0].sum() - .5 * logp[1, 1].sum()) / 6)) / 2
    torch.testing.assert_close(result.loss, oracle)
    torch.testing.assert_close(torch.autograd.grad(result.loss, logp, retain_graph=True)[0],
                               torch.autograd.grad(oracle, logp)[0])


def test_length_penalty_preserves_failures_and_difficult_groups():
    rewards = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.], [1., 1., 0., 0.]])
    lengths = torch.tensor([[400, 500, 600, 700], [1000, 10, 10, 10], [100, 300, 50, 50]])
    config = {**DEFAULT_LENGTH_PENALTY, "enabled": True}
    adjusted, receipt = successful_length_rewards(rewards, lengths, config)
    torch.testing.assert_close(adjusted[:2], rewards[:2])
    torch.testing.assert_close(adjusted[2], torch.tensor([1., 1 - .1 * .4 / .9, 0., 0.]))
    assert receipt["eligible_groups"] == 1 and receipt["penalized_successes"] == 1
    assert bool((adjusted[rewards == 1] > 0).all())
    assert bool((adjusted[rewards == 0] == 0).all())
    untouched, _ = successful_length_rewards(rewards, lengths, DEFAULT_LENGTH_PENALTY)
    torch.testing.assert_close(untouched, rewards)


def test_router_freeze_preserves_adapter_gates_and_other_weights():
    model = nn.Module()
    model.block = nn.Module()
    model.block.ffn = nn.Module()
    model.block.ffn.gate = nn.Linear(4, 3)
    model.block.ffn.expert = nn.Linear(4, 4)
    model.block.output_gate = nn.Linear(4, 4)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    receipt = configure_rl_trainability(model, freeze_router=True)
    assert receipt["frozen_parameter_names"] == ['block.ffn.gate.bias', 'block.ffn.gate.weight']
    for name, p in model.named_parameters():
        assert p.requires_grad == ('.ffn.gate.' not in name)
        torch.testing.assert_close(p, original[name])


def test_memory_policy_rejects_invalid_budgets_and_noncheckpointed_models():
    assert configure_gpu_budget(None) == {"enabled": False}
    for value in (False, -1, 0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            configure_gpu_budget(value)
    with pytest.raises(ValueError, match='checkpoint wrappers'):
        install_checkpoint_input_offload(nn.Linear(4, 4))


def test_shared_gpu_admission_requires_measured_headroom():
    from archlab.automodel.deepseek_v41_rl_training import admit_qualification, digest
    contract = {"recipe": {"context_limit": 2048, "evaluation_reserve_gib": 64}}
    receipt = {"passed": True, "contract_digest": digest(contract),
               "kind": "online-policy-numerical-v1", "synthetic_optimizer_updates": 0}
    with pytest.raises(ValueError, match='memory admission'):
        admit_qualification(receipt, contract)
    receipt['memory_admission'] = {"passed": True, "required_evaluation_reserve_gib": 64,
                                   "context_limit": 2048, "minimum_driver_free_gib": 63}
    with pytest.raises(ValueError, match='memory admission'):
        admit_qualification(receipt, contract)
    receipt['memory_admission']['minimum_driver_free_gib'] = 64
    admit_qualification(receipt, contract)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires the existing B300 runtime')
def test_checkpoint_input_offload_preserves_gradients_and_never_copies_weights():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    torch.manual_seed(155)
    reference = nn.Sequential(checkpoint_wrapper(nn.Sequential(nn.Linear(32, 64), nn.GELU(),
                                                                nn.Linear(64, 32)))).cuda()
    candidate = copy.deepcopy(reference)
    before = {name: id(p) for name, p in candidate.named_parameters()}
    install_checkpoint_input_offload(candidate)
    x = torch.randn(2, 17, 32, device='cuda', requires_grad=True)
    y = x.detach().clone().requires_grad_()
    first, second = reference(x), candidate(y)
    first.square().mean().backward()
    second.square().mean().backward()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        assert q.device.type == 'cuda' and id(q) == before[name]
    stats = input_offload_statistics(candidate)
    assert stats == {"tensor_copies": 1, "copied_bytes": y.numel() * y.element_size()}
