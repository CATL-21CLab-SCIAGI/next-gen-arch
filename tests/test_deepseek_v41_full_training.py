import unittest
import torch
from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn
from archlab.architectures.deepseek_v41_indexer_objective import attention_indexer_kl
from archlab.automodel.deepseek_v41_full_boundaries import NativeTrainableHC, TrainableHeadCrossEntropy
from archlab.optimizers.sharded_adafactor import ShardedAdafactor, stochastic_bfloat16


class FullTrainingTests(unittest.TestCase):
    def test_hc_all_derivatives(self):
        torch.manual_seed(2)
        args = (torch.randn(2, 3, 24, requires_grad=True), torch.randn(3, requires_grad=True),
                torch.randn(24, requires_grad=True))
        expected = hc_split_sinkhorn(*args)
        actual = NativeTrainableHC.apply(*args, 4, 20, 1e-6, hc_split_sinkhorn)
        grad = tuple(torch.randn_like(x) for x in actual)
        want = torch.autograd.grad(expected, args, grad)
        got = torch.autograd.grad(actual, args, grad)
        for a, b in zip(want, got):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(a.abs().sum(), 0)

    def test_trainable_head_against_full_logits(self):
        torch.manual_seed(3)
        hidden = torch.randn(2, 9, 7, requires_grad=True)
        weight = torch.randn(23, 7, requires_grad=True)
        labels = torch.randint(0, 23, (2, 9))
        labels[:, :3] = -100
        want = torch.nn.functional.cross_entropy(torch.nn.functional.linear(hidden, weight).flatten(0, 1), labels.flatten(), reduction='sum')
        got = TrainableHeadCrossEntropy.apply(hidden, labels, weight, 4)
        torch.testing.assert_close(got, want)
        for a, b in zip(torch.autograd.grad(got * .25, (hidden, weight)), torch.autograd.grad(want * .25, (hidden, weight))):
            torch.testing.assert_close(a, b)
        self.assertEqual(float(TrainableHeadCrossEntropy.apply(hidden, torch.full_like(labels, -100), weight, 4)), 0)

    def test_indexer_mask_and_teacher_detachment(self):
        torch.manual_seed(4)
        scores = torch.randn(2, 4, 5, requires_grad=True)
        query = torch.randn(2, 4, 3, 7, requires_grad=True)
        keys = torch.randn(2, 5, 7, requires_grad=True)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, 0] = False
        mask[:, 1:, -1] = False
        loss = attention_indexer_kl(scores, query, keys, mask, scale=7 ** -.5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(query.grad)
        self.assertIsNone(keys.grad)
        self.assertEqual(float(scores.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(scores.grad[:, 1:, -1].abs().sum()), 0)
        self.assertGreater(float(scores.grad.abs().sum()), 0)

    def test_adafactor_matches_dense_equations(self):
        torch.manual_seed(5)
        for shape in [(7,), (9, 7), (3, 9, 7)]:
            initial = torch.randn(shape)
            p = torch.nn.Parameter(initial.clone())
            opt = ShardedAdafactor([p], lr=.01, chunk_elements=14, stochastic_rounding=False)
            expected = initial.clone()
            r = c = v = None
            for step in (1, 2, 3):
                grad = torch.randn_like(p)
                p.grad = grad.clone()
                beta = step ** -.8
                if len(shape) > 1:
                    rm, cm = grad.square().add(1e-30).mean(-1, keepdim=True), grad.square().add(1e-30).mean(-2, keepdim=True)
                    r = rm * beta if r is None else r.lerp(rm, beta)
                    c = cm * beta if c is None else c.lerp(cm, beta)
                    variance = r / r.mean(-2, keepdim=True) * c
                    axes = (-2, -1)
                else:
                    v = grad.square().add(1e-30) * beta if v is None else v.lerp(grad.square().add(1e-30), beta)
                    variance, axes = v, (-1,)
                update = grad / variance.clamp_min(1e-30).sqrt()
                rms = expected.square().mean(axes, keepdim=True).sqrt().clamp_min(1e-3)
                clip = update.square().mean(axes, keepdim=True).sqrt().clamp_min(1.)
                expected = expected - .01 * rms * update / clip
                opt.step()
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                torch.testing.assert_close(p, expected, rtol=2e-6, atol=2e-7)

    def test_stochastic_rounding_unbiased_and_replayable(self):
        values = torch.tensor([1.001, -1.001, 0.0, 0.0000123]).repeat(200000, 1)
        rng = torch.get_rng_state()
        output = stochastic_bfloat16(values)
        torch.set_rng_state(rng)
        self.assertTrue(torch.equal(output, stochastic_bfloat16(values)))
        torch.testing.assert_close(output.float().mean(0), values[0], rtol=0, atol=2e-5)
        exact = torch.randn(100).bfloat16().float()
        self.assertTrue(torch.equal(stochastic_bfloat16(exact).float(), exact))


if __name__ == '__main__':
    unittest.main()
