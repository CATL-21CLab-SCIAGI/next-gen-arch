import math
import unittest

import torch

from archlab.rl.objectives import group_relative_advantages, group_relative_policy_loss


class GroupRelativeObjectiveTests(unittest.TestCase):
    def test_standardized_and_leave_one_out_oracles(self):
        rewards = torch.tensor([[0., 1., 1., 0.], [1., 1., 1., 1.]], requires_grad=True)
        standardized = group_relative_advantages(rewards)
        torch.testing.assert_close(standardized, torch.tensor([[-1., 1., 1., -1.], [0., 0., 0., 0.]]))
        loo = group_relative_advantages(rewards, baseline="leave_one_out")
        expected = torch.tensor([[-2 / 3, 2 / 3, 2 / 3, -2 / 3], [0., 0., 0., 0.]])
        torch.testing.assert_close(loo, expected)
        self.assertFalse(loo.requires_grad)
        self.assertFalse(standardized.requires_grad)

    def test_rloo_matches_explicit_other_sample_baseline(self):
        rewards = torch.tensor([[1., 0., 0., 0.], [-3., -2., 2., 4.]], dtype=torch.float64)
        actual = group_relative_advantages(rewards, baseline="leave_one_out")
        expected = torch.empty_like(rewards)
        for b in range(2):
            for g in range(4):
                others = torch.cat((rewards[b, :g], rewards[b, g + 1:]))
                expected[b, g] = rewards[b, g] - others.mean()
        torch.testing.assert_close(actual, expected)

    def test_large_finite_rewards_do_not_overflow_moments(self):
        rewards = torch.tensor([[-1e30, 1e30], [1e30, 1e30]])
        torch.testing.assert_close(group_relative_advantages(rewards), torch.tensor([[-1., 1.], [0., 0.]]))

    def test_mask_and_three_normalization_gradient_oracles(self):
        mask = torch.tensor([[[False, True, True], [False, True, False]]])
        advantages = torch.tensor([[1., -1.]], dtype=torch.float64, requires_grad=True)
        for normalization in ("sequence_sum", "sequence", "token"):
            with self.subTest(normalization=normalization):
                logp = torch.tensor([[[float("nan"), -.2, -.4], [float("-inf"), -.7, float("nan")]]], dtype=torch.float64, requires_grad=True)
                result = group_relative_policy_loss(logp, mask, advantages, normalization=normalization)
                result.loss.backward()
                if normalization == "sequence_sum":
                    expected = torch.tensor([[[0., -.5, -.5], [0., .5, 0.]]], dtype=torch.float64)
                    self.assertAlmostEqual(float(result.loss.detach()), -.05)
                elif normalization == "sequence":
                    expected = torch.tensor([[[0., -.25, -.25], [0., .5, 0.]]], dtype=torch.float64)
                    self.assertAlmostEqual(float(result.loss.detach()), -.2)
                else:
                    expected = torch.tensor([[[0., -1 / 3, -1 / 3], [0., 1 / 3, 0.]]], dtype=torch.float64)
                    self.assertAlmostEqual(float(result.loss.detach()), -1 / 30)
                torch.testing.assert_close(logp.grad, expected)
                self.assertIsNone(advantages.grad)
                self.assertEqual(result.completion_tokens, 3)
                self.assertEqual(result.sequences, 2)
                self.assertTrue(result.has_policy_signal)
                self.assertFalse(result.skip_update)
                self.assertEqual(float(result.mean_ratio), 1.)
                self.assertFalse(result.reference_kl_available)
                self.assertNotIn("sampled_kl", result.metrics())

    def test_rloo_gradient_is_actual_sampled_trajectory_reinforce(self):
        logits = torch.tensor([.3, -.1], dtype=torch.float64, requires_grad=True)
        # Two independently sampled trajectories from one prompt, lengths 2 and 1.
        lp = logits.log_softmax(0)
        logp = torch.stack((torch.stack((lp[0], lp[0])), torch.stack((lp[1], lp[0])))).unsqueeze(0)
        mask = torch.tensor([[[True, True], [True, False]]])
        advantages = group_relative_advantages(torch.tensor([[1., 0.]]), baseline="leave_one_out")
        result = group_relative_policy_loss(logp, mask, advantages, normalization="sequence_sum")
        actual = torch.autograd.grad(result.loss, logits, retain_graph=True)[0]
        oracle = -(2 * lp[0] - lp[1]) / 2
        expected = torch.autograd.grad(oracle, logits)[0]
        torch.testing.assert_close(actual, expected)

    def test_old_policy_clipping_and_detachment(self):
        old = torch.full((1, 2, 1), -2., dtype=torch.float64, requires_grad=True)
        current = torch.tensor([[[-2 + math.log(1.5)], [-2 + math.log(.5)]]], dtype=torch.float64, requires_grad=True)
        result = group_relative_policy_loss(current, torch.ones_like(current), torch.tensor([[1., -1.]]), old_logprobs=old, clip_epsilon=.2, normalization="sequence_sum")
        self.assertAlmostEqual(float(result.loss.detach()), -.2)
        self.assertEqual(float(result.clip_fraction), 1.)
        result.loss.backward()
        torch.testing.assert_close(current.grad, torch.zeros_like(current))
        self.assertIsNone(old.grad)

    def test_importance_gradient_at_behavior_equals_score_function(self):
        advantages = torch.tensor([[1., -1.]])
        current = torch.tensor([[[-1., -2.], [-3., -4.]]], requires_grad=True)
        mask = torch.ones_like(current, dtype=torch.bool)
        direct = group_relative_policy_loss(current, mask, advantages, normalization="sequence_sum")
        ratio = group_relative_policy_loss(current, mask, advantages, old_logprobs=current.detach(), normalization="sequence_sum")
        torch.testing.assert_close(torch.autograd.grad(direct.loss, current)[0], torch.autograd.grad(ratio.loss, current)[0])

    def test_kl_oracle_stability_and_reference_detachment(self):
        current = torch.tensor([[[-1.], [-2.]]], dtype=torch.float64, requires_grad=True)
        reference = (current.detach() + torch.tensor([[[1e-8]], [[0.]]], dtype=torch.float64).reshape(1, 2, 1)).requires_grad_()
        result = group_relative_policy_loss(current, torch.ones_like(current), torch.zeros(1, 2), reference_logprobs=reference, kl_beta=.1)
        self.assertGreater(float(result.kl.detach()), 0.)
        self.assertAlmostEqual(float(result.kl.detach()), 2.5e-17, delta=1e-23)
        self.assertTrue(result.reference_kl_available)
        self.assertIn("sampled_kl", result.metrics())
        self.assertFalse(result.skip_update)
        result.loss.backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(current.grad)

    def test_flat_groups_legitimately_skip_without_signal(self):
        rewards = torch.ones(2, 4)
        advantages = group_relative_advantages(rewards, baseline="leave_one_out")
        current = torch.full((2, 4, 3), -1., requires_grad=True)
        result = group_relative_policy_loss(current, torch.ones_like(current), advantages, normalization="sequence_sum")
        self.assertTrue(result.skip_update)
        self.assertFalse(result.has_policy_signal)
        result.loss.backward()
        torch.testing.assert_close(current.grad, torch.zeros_like(current))

    def test_invalid_inputs_and_overflow_rejected(self):
        current = torch.full((1, 2, 2), -1.)
        mask = torch.ones_like(current)
        advantages = torch.tensor([[1., -1.]])
        for kwargs in ({"clip_epsilon": .2}, {"kl_beta": .1}, {"normalization": "ambiguous"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                group_relative_policy_loss(current, mask, advantages, **kwargs)
        with self.assertRaises(ValueError):
            group_relative_policy_loss(current, mask * .5, advantages)
        with self.assertRaises(ValueError):
            group_relative_policy_loss(current, torch.zeros_like(mask), advantages)
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.zeros(1, 1))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.tensor([[float("nan"), 0.]]))
        with self.assertRaises(FloatingPointError):
            group_relative_policy_loss(current, mask, advantages, old_logprobs=torch.full_like(current, -1000.))
        with self.assertRaises(FloatingPointError):
            group_relative_policy_loss(current, mask, advantages, reference_logprobs=torch.full_like(current, -1000.), kl_beta=.1)
        with self.assertRaises(ValueError):
            group_relative_policy_loss(-current, mask, advantages)


if __name__ == "__main__":
    unittest.main()
