import copy
import hashlib
import json
import multiprocessing
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from archlab.automodel.deepseek_v41_rl_head import install_rl_head
from archlab.automodel.deepseek_v41_rl_update import (
    policy_gradient_step,
    reconstruct_prefix,
    select_prefix_times,
)
from archlab.rl.rollout import RolloutBatch


class ToyActor(nn.Module):
    def __init__(self, *, include_branch=True):
        super().__init__()
        self.embedding = nn.Embedding(7, 3)
        self.extra = nn.Parameter(torch.tensor([.1, -.2, .3]))
        self.selector = nn.Linear(3, 3, bias=False)
        self.selector._archlab_auxiliary_backward_scale = .1
        self.selector._archlab_auxiliary = None
        self.router = nn.Module()
        self.router.aux_loss_coeff = .2
        self.router.bias_update_factor = .01
        self.router._track_load_balance = True
        self.router.register_buffer("routing_bias", torch.ones(3))
        self.lm_head = nn.Linear(3, 7, bias=False)
        self.include_branch = include_branch
        self.forward_receipts = []
        install_rl_head(self)

    def forward(self, input_ids, attention_mask=None, return_hidden_states=True):
        self.forward_receipts.append((self.training, self.selector._archlab_auxiliary_backward_scale,
                                      self.router.aux_loss_coeff, self.router.bias_update_factor))
        hidden = self.embedding(input_ids).cumsum(1)
        if self.include_branch:
            hidden = hidden + self.extra
        if self.training:
            hidden = hidden + self.selector.weight.sum() * self.selector._archlab_auxiliary_backward_scale
            hidden = hidden + self.router.aux_loss_coeff * self.router.routing_bias
        return SimpleNamespace(hidden_states=hidden)


class RecordingSGD(torch.optim.Optimizer):
    def __init__(self, params, *, no_change=False):
        super().__init__(params, {"lr": .001})
        self.last_metrics = {}
        self.steps = 0
        self.zero_calls = 0
        self.no_change = no_change
        self.gradients = []

    def zero_grad(self, set_to_none=True):
        self.zero_calls += 1
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        self.steps += 1
        changed = 0
        updated = 0
        self.gradients = []
        for group in self.param_groups:
            for parameter in group["params"]:
                self.gradients.append(None if parameter.grad is None else parameter.grad.clone())
                if parameter.grad is None:
                    continue
                previous = parameter.clone()
                if not self.no_change:
                    parameter.add_(parameter.grad, alpha=-group["lr"])
                changed += int((previous != parameter).sum())
                updated += 1
        self.last_metrics = {"changed_local_elements": changed, "updated_parameter_tensors": updated}


def make_rollout(model, *, rank=0, world=1, generated=None):
    generated = [[3, 4], [4]] if generated is None else generated
    prompts = [[1, 2], [1, 2]]
    ids = torch.zeros(2, 5, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    for row, tokens in enumerate(generated):
        all_tokens = prompts[row] + tokens
        ids[row, :len(all_tokens)] = torch.tensor(all_tokens)
        attention[row, :len(all_tokens)] = True
        labels[row, 1:1 + len(tokens)] = torch.tensor(tokens)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    with torch.no_grad():
        hidden = model(ids, attention).hidden_states
        policy = model.lm_head.rl_log_probs(hidden, labels)
    for module, training in modes:
        module.training = training
    receipt = {"rank": rank, "world_size": world, "policy_version": "fixture-policy-v0",
               "temperature": 1., "top_p": 1., "on_policy_sampling": True,
               "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
               "generated_tokens": sum(map(len, generated))}
    return RolloutBatch(ids, attention, labels, labels != -100, policy, policy.clone(),
                        generated, [2, 2], ["stop", "stop"], receipt)


def dense_oracle(model, rollout, *, denominator=2):
    model.eval()
    hidden = model(rollout.input_ids, rollout.attention_mask).hidden_states
    logits = torch.nn.functional.linear(hidden, model.lm_head.weight)
    selected = logits.log_softmax(-1).gather(-1, rollout.labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    selected = selected.masked_fill(~rollout.response_mask, 0)
    loss = -(selected[0].sum() - selected[1].sum()) / denominator
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    return float(loss.detach()), gradients


def test_prompt_token_mean_packed_and_all_prefixes_match_dense_gradient():
    for mode in ('packed', 'sampled-prefix'):
        torch.manual_seed(981)
        actor = ToyActor()
        rollout = make_rollout(actor)
        rollout.receipt.update(forward_count=2, forward_shapes=[[2, 4], [2, 4]],
                               pad_token_id=0, eos_token_ids=[4])
        oracle_actor = copy.deepcopy(actor)
        expected_loss, expected_gradients = dense_oracle(oracle_actor, rollout, denominator=3)
        optimizer = RecordingSGD(actor.parameters())
        metric = policy_gradient_step(actor, optimizer, [actor.selector], rollout,
            torch.tensor([[1., 0.]]), lr=.001, group_size=2, max_grad_norm=1e6,
            replay_mode=mode, replay_prefixes=2, loss_normalization='prompt_token_mean')
        assert abs(metric['policy_loss'] - expected_loss) < 1e-6
        assert metric['normalization'] == 'global-prompt-mean-of-group-token-means'
        for observed, expected in zip(optimizer.gradients, expected_gradients):
            if expected is None:
                assert observed is None
            else:
                torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


class BatchDependentActor(ToyActor):
    """Model the numerical dependence of compacted expert GEMMs on valid rows."""

    def forward(self, input_ids, attention_mask=None, return_hidden_states=True):
        result = super().forward(input_ids, attention_mask, return_hidden_states)
        return SimpleNamespace(hidden_states=result.hidden_states + attention_mask.sum() * self.extra)


def make_prefix_rollout(model, *, rank=0, world=1, generated=None, global_steps=None):
    generated = [[3, 4, 5], [6]] if generated is None else generated
    prompts = [[1, 2], [1, 2]]
    steps = max(map(len, generated)) if global_steps is None else global_steps
    shapes = [[2, ((2 + step + 3) // 4) * 4] for step in range(steps)]
    final_canvas = ((2 + steps + 3) // 4) * 4
    ids = torch.zeros(2, final_canvas, dtype=torch.long)
    attention = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    scores = torch.zeros_like(ids, dtype=torch.float32)
    for row, tokens in enumerate(generated):
        sequence = prompts[row] + tokens
        ids[row, :len(sequence)] = torch.tensor(sequence)
        attention[row, :len(sequence)] = True
        labels[row, 1:1 + len(tokens)] = torch.tensor(tokens)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    with torch.no_grad():
        for step, (_batch, canvas) in enumerate(shapes):
            prefix = torch.zeros(2, canvas, dtype=torch.long)
            mask = torch.zeros_like(prefix, dtype=torch.bool)
            for row, tokens in enumerate(generated):
                current = prompts[row] + tokens[:step]
                prefix[row, :len(current)] = torch.tensor(current)
                mask[row, :len(current)] = True
            hidden = model(prefix, mask).hidden_states
            last = torch.tensor([1 + min(step, len(tokens)) for tokens in generated])
            logp = model.lm_head(hidden[torch.arange(2), last]).log_softmax(-1)
            for row, tokens in enumerate(generated):
                if step < len(tokens):
                    scores[row, 1 + step] = logp[row, tokens[step]]
    for module, training in modes:
        module.training = training
    receipt = {"rank": rank, "world_size": world, "policy_version": "fixture-policy-v0",
               "temperature": 1., "top_p": 1., "on_policy_sampling": True,
               "prompt_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
               "generated_tokens": sum(map(len, generated)), "forward_count": steps,
               "forward_shapes": shapes, "pad_token_id": 0, "eos_token_ids": [6]}
    reasons = ["stop" if tokens[-1] == 6 else "length" for tokens in generated]
    return RolloutBatch(ids, attention, labels, labels != -100, scores, scores.clone(),
                        generated, [2, 2], reasons, receipt)


def independent_prefix_oracle(model, rollout, *, times=None, denominator=2):
    """Dense score-function oracle; reconstruct from raw fixture prompts/actions."""
    model.eval()
    total_steps = rollout.receipt["forward_count"]
    times = tuple(range(total_steps)) if times is None else tuple(times)
    loss = torch.zeros(())
    for step in times:
        sequences = [[1, 2] + tokens[:step] for tokens in rollout.generated_ids]
        canvas = rollout.receipt["forward_shapes"][step][1]
        inputs = torch.zeros((2, canvas), dtype=torch.long)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            inputs[row, :len(sequence)] = torch.tensor(sequence)
            mask[row, :len(sequence)] = True
        hidden = model(inputs, mask).hidden_states
        selected = hidden[torch.arange(2), torch.tensor([len(s) - 1 for s in sequences])]
        logp = torch.nn.functional.linear(selected, model.lm_head.weight).log_softmax(-1)
        for row, advantage in enumerate((1., -1.)):
            if step < len(rollout.generated_ids[row]):
                loss = loss - logp[row, rollout.generated_ids[row][step]] * advantage / denominator
    loss = loss * (total_steps / len(times))
    return float(loss.detach()), torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)


def seed_for_time(total, chosen):
    return next(seed for seed in range(10000) if select_prefix_times(total, 1, seed) == (chosen,))


def prefix_distributed_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    try:
        for prefix_count in (3, 1):
            torch.manual_seed(271)
            actor = ToyActor(include_branch=rank == 1)
            initial = copy.deepcopy(actor.state_dict())
            generated = [[6], [6]] if rank == 0 else [[3, 4, 5], [6]]
            rollout = make_prefix_rollout(actor, rank=rank, world=2, generated=generated, global_steps=3)
            optimizer = RecordingSGD(actor.parameters())
            rewards = torch.tensor([[0., 0.]]) if rank == 0 else torch.tensor([[1., 0.]])
            seed = seed_for_time(3, 2)
            metric = policy_gradient_step(actor, optimizer, [actor.selector], rollout, rewards,
                lr=.01, group_size=2, max_grad_norm=100, audit=True,
                replay_mode="sampled-prefix", replay_prefixes=prefix_count, replay_seed=seed)
            oracle = ToyActor(include_branch=True)
            oracle.load_state_dict(initial)
            oracle_rollout = make_prefix_rollout(oracle)
            times = (0, 1, 2) if prefix_count == 3 else (2,)
            _, expected = independent_prefix_oracle(oracle, oracle_rollout, times=times, denominator=4)
            for (name, parameter), gradient in zip(actor.named_parameters(), expected, strict=True):
                wanted = initial[name] if gradient is None else initial[name] - .01 * gradient
                torch.testing.assert_close(parameter, wanted, rtol=2e-6, atol=2e-7)
            assert metric["updated"] and metric["replay_selected_times"] == list(times)
            assert metric["replay_max_abs_error"] <= 1e-6
            assert metric["replay_scored_tokens"] == (6 if prefix_count == 3 else 1)
        (Path(output_dir) / f"rank-{rank}.json").write_text(json.dumps({"passed": True}))
    finally:
        dist.destroy_process_group()


def distributed_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        actor = ToyActor(include_branch=rank == 1)
        initial = copy.deepcopy(actor.state_dict())
        rollout = make_rollout(actor, rank=rank, world=2)
        optimizer = RecordingSGD(actor.parameters())
        rewards = torch.tensor([[0., 0.]]) if rank == 0 else torch.tensor([[1., 0.]])
        metric = policy_gradient_step(actor, optimizer, [actor.selector], rollout, rewards,
                                      lr=.01, group_size=2, max_grad_norm=100, audit=True)
        oracle = ToyActor(include_branch=True)
        oracle.load_state_dict(initial)
        _, expected = dense_oracle(oracle, make_rollout(oracle), denominator=4)
        norm = torch.sqrt(sum(gradient.double().square().sum() for gradient in expected if gradient is not None))
        for (name, parameter), gradient in zip(actor.named_parameters(), expected, strict=True):
            wanted = initial[name] if gradient is None else initial[name] - .01 * gradient
            torch.testing.assert_close(parameter, wanted, rtol=2e-6, atol=2e-7)
        assert abs(metric["gradient_norm_before_clip"] - float(norm)) < 1e-6
        assert metric["updated"] and metric["nonflat_prompt_groups"] == 1
        extra = next(entry for entry in metric["audit"]["coverage"] if entry["name"] == "extra")
        assert extra["gradient_present_ranks"] == 1
        # Every-rank-flat skips forward and every optimizer operation.
        flat = make_rollout(actor, rank=rank, world=2)
        before = (len(actor.forward_receipts), optimizer.steps, optimizer.zero_calls)
        skipped = policy_gradient_step(actor, optimizer, [actor.selector], flat, torch.ones(1, 2), lr=.01, group_size=2)
        assert skipped["skip_reason"] == "all_groups_flat"
        assert before == (len(actor.forward_receipts), optimizer.steps, optimizer.zero_calls)
        # One malformed rank must fail collectively before the other enters forward.
        invalid = make_rollout(actor, rank=rank, world=2)
        if rank == 0:
            invalid.behavior_log_probs[0, 1] += .001
        try:
            policy_gradient_step(actor, optimizer, [actor.selector], invalid, torch.tensor([[1., 0.]]), lr=.01, group_size=2)
        except ValueError:
            collective_rejection = True
        else:
            collective_rejection = False
        assert collective_rejection
        (Path(output_dir) / f"rank-{rank}.json").write_text(json.dumps({"passed": True, "norm": metric["gradient_norm_before_clip"], "replay": metric["replay_max_abs_error"]}))
    finally:
        dist.destroy_process_group()


class RLUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.model = ToyActor()
        self.optimizer = RecordingSGD(self.model.parameters())
        self.rollout = make_rollout(self.model)
        self.rewards = torch.tensor([[1., 0.]])

    def update(self, **kwargs):
        return policy_gradient_step(self.model, self.optimizer, [self.model.selector], self.rollout,
                                    self.rewards, lr=.01, group_size=2, **kwargs)

    def test_actual_signed_policy_gradient_matches_dense_and_masks(self):
        oracle = copy.deepcopy(self.model)
        loss, gradients = dense_oracle(oracle, self.rollout)
        norm = float(torch.sqrt(sum(g.double().square().sum() for g in gradients if g is not None)))
        metric = self.update(max_grad_norm=100, audit=True, head_chunk_size=1)
        self.assertTrue(metric["updated"])
        self.assertTrue(metric["optimizer_step_applied"])
        self.assertFalse(metric["update_skipped"])
        self.assertAlmostEqual(metric["policy_loss"], loss, places=6)
        self.assertAlmostEqual(metric["gradient_norm_before_clip"], norm, places=6)
        self.assertLessEqual(metric["replay_max_abs_error"], 1e-6)
        for actual, expected in zip(self.optimizer.gradients, gradients, strict=True):
            if expected is None:
                self.assertIsNone(actual)
            else:
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        self.assertEqual(metric["globally_missing_hard_selector_gradients"], 1)
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.forward_receipts[-1], (False, 0., 0., 0.))
        self.assertEqual(self.model.selector._archlab_auxiliary_backward_scale, .1)
        self.assertEqual(self.model.router.aux_loss_coeff, .2)

    def test_all_flat_groups_do_not_touch_optimizer_or_forward(self):
        self.rewards.fill_(1)
        calls = len(self.model.forward_receipts)
        metric = self.update(audit=True)
        self.assertEqual(metric["skip_reason"], "all_groups_flat")
        self.assertFalse(metric["updated"])
        self.assertFalse(metric["replay_verified"])
        self.assertEqual(self.optimizer.steps, 0)
        self.assertEqual(self.optimizer.zero_calls, 0)
        self.assertEqual(len(self.model.forward_receipts), calls)

    def test_audit_only_is_backward_qualification_without_learning(self):
        initial = copy.deepcopy(self.model.state_dict())
        rate = self.optimizer.param_groups[0]["lr"]
        metric = self.update(audit=True, audit_only=True)
        self.assertTrue(metric["numerical_qualification_passed"])
        self.assertEqual(metric["skip_reason"], "audit_only")
        self.assertGreater(metric["gradient_norm_before_clip"], 0)
        self.assertFalse(metric["updated"])
        self.assertEqual(self.optimizer.steps, 0)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], rate)
        self.assertFalse(self.optimizer.state)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_global_norm_clipping_and_no_change_are_reported(self):
        metric = self.update(max_grad_norm=.01)
        self.assertLess(metric["gradient_clip_scale"], 1)
        norm = torch.sqrt(sum(g.double().square().sum() for g in self.optimizer.gradients if g is not None))
        self.assertAlmostEqual(float(norm), .01, places=7)
        self.rollout = make_rollout(self.model)
        self.optimizer.no_change = True
        unchanged = self.update()
        self.assertEqual(unchanged["skip_reason"], "no_weight_change")
        self.assertTrue(unchanged["optimizer_step_applied"])
        self.assertFalse(unchanged["updated"])

    def test_zero_gradient_does_not_claim_an_update(self):
        self.rollout = make_rollout(self.model, generated=[[3], [3]])
        # Make this the exact-zero branch rather than relying on cancellation
        # order in FP32 vocabulary matrix reductions for two equal trajectories.
        for parameter in self.model.parameters():
            parameter.register_hook(lambda gradient: gradient * 0)
        metric = self.update()
        self.assertEqual(metric["skip_reason"], "zero_gradient")
        self.assertEqual(self.optimizer.steps, 0)
        self.assertFalse(metric["updated"])

    def test_used_rollout_stale_policy_and_replay_mismatch_rejected(self):
        self.update()
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.update()
        self.rollout = make_rollout(self.model)
        self.model._archlab_rl_policy_version = "different-policy"
        with self.assertRaisesRegex(ValueError, "different current policy"):
            self.update()
        del self.model._archlab_rl_policy_version
        self.rollout.policy_log_probs[self.rollout.response_mask] -= .1
        self.rollout.behavior_log_probs.copy_(self.rollout.policy_log_probs)
        with self.assertRaisesRegex(ValueError, "replay exceeds"):
            self.update(replay_tolerance=.001)
        self.assertTrue(self.model.training)
        self.assertEqual(self.optimizer.steps, 1)

    def test_behavior_and_next_token_alignment_rejected_before_forward(self):
        self.rollout.receipt["temperature"] = .5
        with self.assertRaisesRegex(ValueError, "temperature=1"):
            self.update()
        self.rollout = make_rollout(self.model)
        self.rollout.labels[0, 1] = 1
        with self.assertRaisesRegex(ValueError, "sampled continuation"):
            self.update()
        self.rollout = make_rollout(self.model)
        self.rollout.input_ids[1, 0] = 5
        with self.assertRaisesRegex(ValueError, "same prompt"):
            self.update()
        self.assertEqual(self.optimizer.steps, 0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_two_rank_sum_matches_global_oracle_with_one_flat_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            init_file = str(Path(temporary) / "rendezvous")
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=distributed_worker, args=(rank, init_file, temporary)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(40)
                self.assertTrue(all(not process.is_alive() for process in processes), "distributed collective hung")
                self.assertEqual([process.exitcode for process in processes], [0, 0])
                self.assertTrue(all(json.loads((Path(temporary) / f"rank-{rank}.json").read_text())["passed"] for rank in range(2)))
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)

    def test_uniform_single_prefix_gradient_average_equals_full_trajectory(self):
        initial = copy.deepcopy(self.model.state_dict())
        oracle = ToyActor()
        oracle.load_state_dict(initial)
        rollout = make_prefix_rollout(oracle)
        _, full_gradients = independent_prefix_oracle(oracle, rollout)
        captured = []
        previous_rng = random.getstate()
        for step in range(3):
            model = ToyActor()
            model.load_state_dict(initial)
            optimizer = RecordingSGD(model.parameters(), no_change=True)
            metric = policy_gradient_step(model, optimizer, [model.selector], make_prefix_rollout(model),
                self.rewards, lr=.01, group_size=2, max_grad_norm=100,
                replay_mode="sampled-prefix", replay_prefixes=1, replay_seed=seed_for_time(3, step))
            self.assertEqual(metric["replay_selected_times"], [step])
            self.assertEqual(metric["replay_time_weight"], 3)
            captured.append(optimizer.gradients)
        self.assertEqual(random.getstate(), previous_rng)
        for index, expected in enumerate(full_gradients):
            if expected is None:
                self.assertTrue(all(rows[index] is None for rows in captured))
            else:
                observed = torch.stack([rows[index] for rows in captured]).mean(0)
                torch.testing.assert_close(observed, expected, rtol=3e-6, atol=3e-7)

    def test_batch_dependent_packed_future_fails_but_exact_prefix_passes(self):
        torch.manual_seed(513)
        model = BatchDependentActor()
        optimizer = RecordingSGD(model.parameters())
        rollout = make_prefix_rollout(model)
        initial = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "replay exceeds"):
            policy_gradient_step(model, optimizer, [model.selector], rollout, self.rewards,
                                 lr=.01, group_size=2, replay_mode="packed")
        self.assertEqual(optimizer.steps, 0)
        metric = policy_gradient_step(model, optimizer, [model.selector], rollout, self.rewards,
            lr=.01, group_size=2, max_grad_norm=100, replay_mode="sampled-prefix", replay_prefixes=3)
        self.assertTrue(metric["updated"])
        self.assertLessEqual(metric["replay_max_abs_error"], 1e-6)
        oracle = BatchDependentActor()
        oracle.load_state_dict(initial)
        _, expected = independent_prefix_oracle(oracle, make_prefix_rollout(oracle))
        for observed, wanted in zip(optimizer.gradients, expected, strict=True):
            if wanted is None:
                self.assertIsNone(observed)
            else:
                torch.testing.assert_close(observed, wanted, rtol=3e-6, atol=3e-7)

    def test_early_eos_retains_prefix_and_keeps_full_head_batch(self):
        rollout = make_prefix_rollout(self.model, generated=[[3, 4, 5, 3], [6]])
        inputs, mask, positions, labels, active, recorded = reconstruct_prefix(rollout, 3)
        self.assertEqual(tuple(inputs.shape), (2, 8))
        self.assertEqual(inputs[0, :5].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(inputs[1, :3].tolist(), [1, 2, 6])
        self.assertEqual(mask.sum(-1).tolist(), [5, 3])
        self.assertEqual(positions.tolist(), [4, 2])
        self.assertEqual(labels.tolist(), [[3], [0]])
        self.assertEqual(active.tolist(), [[True], [False]])
        self.assertEqual(float(recorded[1, 0]), 0)
        calls = []
        original = self.model.lm_head.rl_log_probs

        def observe(hidden, targets, chunk_size=128):
            calls.append((tuple(hidden.shape), targets.tolist(), chunk_size))
            return original(hidden, targets, chunk_size)

        self.model.lm_head.rl_log_probs = observe
        metric = policy_gradient_step(self.model, self.optimizer, [self.model.selector], rollout,
            self.rewards, lr=.01, group_size=2, head_chunk_size=1, replay_mode="sampled-prefix",
            replay_prefixes=1, replay_seed=seed_for_time(4, 3))
        self.assertEqual(calls, [((2, 1, 3), [[3], [0]], 2)])
        self.assertEqual(metric["replay_scored_tokens"], 1)

    def test_prefix_metadata_and_seed_are_checked_before_forward(self):
        for mutation in ("canvas", "pad", "stops", "length"):
            rollout = make_prefix_rollout(self.model)
            if mutation == "canvas":
                rollout.receipt["forward_shapes"][1] = [2, 1]
            elif mutation == "pad":
                rollout.receipt["pad_token_id"] = 999
            elif mutation == "stops":
                rollout.finish_reasons[1] = "length"
            else:
                rollout.receipt["forward_count"] = 9
            before = len(self.model.forward_receipts)
            with self.assertRaises(ValueError):
                policy_gradient_step(self.model, self.optimizer, [self.model.selector], rollout,
                    self.rewards, lr=.01, group_size=2, replay_mode="sampled-prefix")
            self.assertEqual(len(self.model.forward_receipts), before)
        self.assertEqual(self.optimizer.steps, 0)
        with self.assertRaises(ValueError):
            select_prefix_times(3, 1, "not-an-integer")

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_two_rank_prefix_normalization_and_locally_inactive_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            init_file = str(Path(temporary) / "rendezvous")
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=prefix_distributed_worker, args=(rank, init_file, temporary)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(40)
                self.assertTrue(all(not process.is_alive() for process in processes), "prefix collective hung")
                self.assertEqual([process.exitcode for process in processes], [0, 0])
                self.assertTrue(all(json.loads((Path(temporary) / f"rank-{rank}.json").read_text())["passed"] for rank in range(2)))
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)


if __name__ == "__main__":
    unittest.main()
