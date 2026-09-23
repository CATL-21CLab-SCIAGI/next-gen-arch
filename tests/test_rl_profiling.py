import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from archlab.rl.profiling import profile_rollout_batches


class RolloutProfileTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.lm_head = nn.Linear(3, 8, bias=False)
        self.model.child = nn.Dropout(.2)
        self.model.train()
        self.model.child.eval()
        self.options = dict(policy_version="fixture", max_new_tokens=4, context_limit=16,
                            eos_token_ids=[7], pad_token_id=0, seed=29)

    @staticmethod
    def result(batch, length):
        return SimpleNamespace(
            generated_ids=[[3] * length for _ in range(batch)],
            response_mask=torch.ones(batch, length, dtype=torch.bool),
            receipt={"generated_tokens": batch * length, "forward_count": length,
                     "forward_shapes": [[batch, 8]] * length},
        )

    def test_actual_tokens_and_forward_work_exclude_warmup(self):
        calls = []

        def sampler(model, prompts, **kwargs):
            calls.append((len(prompts), kwargs))
            self.assertTrue(all(prompt == [1, 2] for prompt in prompts))
            self.assertEqual(kwargs["temperature"], 1.)
            self.assertEqual(kwargs["top_p"], 1.)
            # Warmups generate four tokens; measured batches hit EOS earlier.
            length = {1: 4, 2: 2, 3: 4, 4: 1}[len(calls)]
            return self.result(len(prompts), length)

        with patch("archlab.rl.profiling.sample_rollouts", sampler), patch("archlab.rl.profiling.time.perf_counter", side_effect=[0., 2., 10., 11.]):
            report = profile_rollout_batches(self.model, [1, 2], **self.options)
        self.assertEqual([size for size, _ in calls], [1, 1, 4, 4])
        first, fourth = report["batches"]
        self.assertEqual(first["actual_generated_tokens_global"], 2)
        self.assertEqual(fourth["actual_generated_tokens_global"], 4)
        self.assertEqual(first["input_tokens_processed_global"], 16)
        self.assertEqual(fourth["input_tokens_processed_global"], 32)
        self.assertEqual(first["rank_forward_calls_sum"], 2)
        self.assertEqual(fourth["rank_forward_calls_sum"], 1)
        self.assertEqual(first["generated_tokens_per_second"], 1.)
        self.assertEqual(fourth["generated_tokens_per_second"], 4.)
        self.assertEqual(report["batch4_over_batch1_throughput_ratio"], 4.)
        self.assertEqual(report["measurement_device"], "cpu")
        self.assertFalse(report["warmup_included_in_measurements"])
        self.assertIn("no MFU percentage claimed", report["scope"])

    def test_repeats_use_total_tokens_over_total_duration(self):
        lengths = iter([1, 4])

        def sampler(model, prompts, **kwargs):
            return self.result(len(prompts), next(lengths))

        with patch("archlab.rl.profiling.sample_rollouts", sampler), patch("archlab.rl.profiling.time.perf_counter", side_effect=[0., 1., 10., 13.]):
            report = profile_rollout_batches(self.model, [1, 2], batch_sizes=(2,), warmup=0, repeats=2, **self.options)
        self.assertEqual(report["batches"][0]["actual_generated_tokens_global"], 10)
        self.assertEqual(report["batches"][0]["generated_tokens_per_second"], 2.5)
        self.assertIsNone(report["batch4_over_batch1_throughput_ratio"])

    def test_distributed_throughput_uses_global_actual_tokens_and_slowest_rank(self):
        local = self.result(1, 2)
        # The local sequence stops at two tokens; rank1 remains active for three
        # forwards, so every rank still performs the same three collective calls.
        local.receipt.update(forward_count=3, forward_shapes=[[1, 8]] * 3)

        def gather(packets, packet):
            packets[:] = [packet, packet]

        def reduce(value, op):
            if value.numel() == 3:
                value.add_(torch.tensor([3, 3, 24]))
            elif value.dtype == torch.float64:
                value.fill_(max(float(value), 4.))
            else:
                value.fill_(max(int(value), 3))

        with patch("archlab.rl.profiling.dist.is_initialized", return_value=True), patch("archlab.rl.profiling.dist.get_world_size", return_value=2), patch("archlab.rl.profiling.dist.all_gather_object", gather), patch("archlab.rl.profiling.dist.barrier"), patch("archlab.rl.profiling.dist.all_reduce", reduce), patch("archlab.rl.profiling.sample_rollouts", return_value=local), patch("archlab.rl.profiling.time.perf_counter", side_effect=[0., 2.]):
            report = profile_rollout_batches(self.model, [1, 2], batch_sizes=(1,), warmup=0, **self.options)
        batch = report["batches"][0]
        self.assertEqual(batch["global_batch_size"], 2)
        self.assertEqual(batch["actual_generated_tokens_global"], 5)
        self.assertEqual(batch["generated_tokens_per_second"], 1.25)
        self.assertEqual(batch["rank_forward_calls_sum"], 6)
        self.assertEqual(batch["input_tokens_processed_global"], 48)

    def test_rng_modes_and_weights_are_preserved(self):
        random.seed(14)
        np.random.seed(15)
        torch.manual_seed(16)
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        modes = [module.training for module in self.model.modules()]
        weights = [parameter.detach().clone() for parameter in self.model.parameters()]

        def sampler(model, prompts, **kwargs):
            random.random()
            np.random.rand()
            torch.rand(3)
            model.eval()
            model.child.train()
            return self.result(len(prompts), 1)

        with patch("archlab.rl.profiling.sample_rollouts", sampler):
            profile_rollout_batches(self.model, [1, 2], **self.options)
        self.assertEqual(py_state, random.getstate())
        after_np = np.random.get_state()
        self.assertEqual(np_state[0], after_np[0])
        np.testing.assert_array_equal(np_state[1], after_np[1])
        self.assertEqual(np_state[2:], after_np[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        for old, current in zip(weights, self.model.parameters(), strict=True):
            self.assertTrue(torch.equal(old, current))

    def test_exception_restores_rng_and_modes(self):
        state = torch.get_rng_state()
        modes = [module.training for module in self.model.modules()]

        def fail(model, prompts, **kwargs):
            torch.rand(9)
            model.eval()
            raise RuntimeError("injected sampler failure")

        with patch("archlab.rl.profiling.sample_rollouts", fail), self.assertRaisesRegex(RuntimeError, "injected"):
            profile_rollout_batches(self.model, [1, 2], **self.options)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_cuda_rng_states_restored_without_test_gpu_work(self):
        states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
        with patch("torch.cuda.is_initialized", return_value=True), patch("torch.cuda.get_rng_state_all", return_value=states), patch("torch.cuda.set_rng_state_all") as restore, patch("archlab.rl.profiling.sample_rollouts", side_effect=lambda model, prompts, **kwargs: self.result(len(prompts), 1)):
            profile_rollout_batches(self.model, [1, 2], batch_sizes=(1,), warmup=0, **self.options)
        restore.assert_called_once_with(states)

    def test_bad_counts_budget_and_timing_fail_closed(self):
        bad = self.result(1, 1)
        bad.receipt["generated_tokens"] = 999
        with patch("archlab.rl.profiling.sample_rollouts", return_value=bad), self.assertRaisesRegex(ValueError, "actual generated"):
            profile_rollout_batches(self.model, [1, 2], batch_sizes=(1,), warmup=0, **self.options)
        with patch("archlab.rl.profiling.sample_rollouts", return_value=self.result(1, 1)), patch("archlab.rl.profiling.time.perf_counter", side_effect=[2., 2.]), self.assertRaisesRegex(ValueError, "duration"):
            profile_rollout_batches(self.model, [1, 2], batch_sizes=(1,), warmup=0, **self.options)
        with self.assertRaisesRegex(ValueError, "unique integers"):
            profile_rollout_batches(self.model, [1, 2], batch_sizes=(1, 1), **self.options)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            profile_rollout_batches(self.model, [1] * 16, **self.options)

    def test_retained_backend_is_recorded_for_each_separate_sampler_call(self):
        self.model._archlab_rl_retain_weights = True
        calls = []

        def sampler(model, prompts, **kwargs):
            calls.append(len(prompts))
            rollout = self.result(len(prompts), 1)
            rollout.receipt.update(backend="resident-model-full-prefix-retained-weights", retained_weights=True)
            return rollout

        with patch("archlab.rl.profiling.sample_rollouts", sampler):
            report = profile_rollout_batches(self.model, [1, 2], **self.options)
        self.assertEqual(calls, [1, 1, 4, 4])
        self.assertTrue(report["retained_weights"])
        self.assertTrue(report["residency_setup_and_cleanup_included"])
        for batch in report["batches"]:
            self.assertEqual(batch["samples"][0]["rollout_backend"], "resident-model-full-prefix-retained-weights")


if __name__ == "__main__":
    unittest.main()
