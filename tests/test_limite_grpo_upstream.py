"""Run the qualified upstream DAPO reduction against a global-gradient oracle.

Set ARCHLAB_TRL_RUNTIME to the pinned TRL overlay to run this optional contract
test without importing its GPU runtime into the CPU numerical oracle.
"""

import ast
import os
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from archlab.rl.limite_rollout import with_behavior_logprobs


def upstream_loss():
    root = os.environ.get("ARCHLAB_TRL_RUNTIME")
    if not root:
        pytest.skip("set ARCHLAB_TRL_RUNTIME to the qualified TRL overlay")
    source = Path(root) / "trl/trainer/grpo_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    loss = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss")
    module = ast.Module(body=[loss], type_ignores=[])
    namespace = dict(torch=torch, nanmin=torch.min, nanmax=torch.max)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_compute_loss"]


def test_upstream_dapo_microbatches_world_mean_and_clipped_gradients_equal_global_oracle():
    compute_loss = upstream_loss()
    current = torch.nn.Parameter(torch.tensor([
        [-.5, -1.1, -.9], [-1.5, -.9, -1.1],
        [-.95, -1.05, -.8], [-1.1, -.5, -1.5],
    ], dtype=torch.float64))
    behavior = torch.full_like(current, -1.)
    masks = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
    advantages = torch.tensor([1., -1., .5, -.5], dtype=torch.float64)
    denominator = masks.sum()
    ratios = (current - behavior).exp()
    oracle = -(torch.minimum(ratios * advantages[:, None],
                             ratios.clamp(.8, 1.2) * advantages[:, None]) * masks).sum() / denominator
    expected_gradients = torch.autograd.grad(oracle, current, retain_graph=True)[0]
    rank_losses = []
    for rank in range(2):
        trainer = SimpleNamespace(
            model=SimpleNamespace(training=True), beta=0., loss_type="dapo",
            scale_rewards="group", num_iterations=1, importance_sampling_level="token",
            top_entropy_quantile=1., off_policy_mask_threshold=None, use_vllm=False,
            epsilon_low=.2, epsilon_high=.2, args=SimpleNamespace(delta=None),
            current_gradient_accumulation_steps=2,
            accelerator=SimpleNamespace(num_processes=2, gather=lambda value: value.unsqueeze(0)),
            _metrics={"train": defaultdict(list)},
        )

        def scores(self, model, input_ids, *args, **kwargs):
            values = model[input_ids[:, 1]]
            return values, torch.ones_like(values)

        trainer._get_per_token_logps_and_entropies = MethodType(scores, trainer)
        microbatch_losses = []
        for row in range(rank * 2, rank * 2 + 2):
            inputs = dict(prompt_ids=torch.ones(1, 1, dtype=torch.long), prompt_mask=torch.ones(1, 1),
                          completion_ids=torch.full((1, 3), row, dtype=torch.long),
                          completion_mask=masks[row:row + 1], advantages=advantages[row:row + 1],
                          sampling_per_token_logps=behavior[row:row + 1], num_items_in_batch=denominator)
            microbatch_losses.append(compute_loss(trainer, current, with_behavior_logprobs(inputs)))
        rank_losses.append(sum(microbatch_losses))
    actual = sum(rank_losses) / 2  # DDP / deferred synchronization uses a world mean.
    torch.testing.assert_close(actual, oracle)
    torch.testing.assert_close(torch.autograd.grad(actual, current)[0], expected_gradients)
    # Positive over-high and negative over-low ratios have exactly zero gradient.
    assert expected_gradients[0, 0] == expected_gradients[1, 0] == 0
