import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from archlab.optimizers.router_balance import balance_routers,auxiliary_scale,bias_correction


class RouterBalanceTests(unittest.TestCase):
    def test_global_load_and_checkpointed_bias(self):
        bias=torch.zeros(4,dtype=torch.float32)
        gate=SimpleNamespace(_cumulative_expert_load=torch.tensor([9,0,1,0]),e_score_correction_bias=bias,e_score_correction_bias_master=torch.ones(4))
        def add_other_rank(value):value.add_(torch.tensor([[0,7,0,3]]))
        with patch('archlab.optimizers.router_balance.dist.all_reduce',side_effect=add_other_rank):
            report=balance_routers([gate],.1)
        torch.testing.assert_close(bias,torch.tensor([-.1,-.1,.1,.1]),rtol=0,atol=0)
        self.assertIs(gate.e_score_correction_bias,bias)
        self.assertIsNone(gate._cumulative_expert_load)
        self.assertIsNone(gate.e_score_correction_bias_master)
        self.assertEqual(report['router_dead_fraction_max'],0.)

    def test_low_precision_bias_rejected(self):
        gate=SimpleNamespace(_cumulative_expert_load=torch.tensor([2,0]),e_score_correction_bias=torch.zeros(2,dtype=torch.bfloat16))
        with patch('archlab.optimizers.router_balance.dist.all_reduce'):
            with self.assertRaisesRegex(ValueError,'FP32'):balance_routers([gate],.1)


class RouterAuxiliaryScaleTests(unittest.TestCase):
    def test_variable_length_token_weighted_gradient(self):
        value=torch.tensor([.2,.4],requires_grad=True)
        first=torch.tensor([[1.,0.],[2.,1.],[0.,2.]])
        second=torch.tensor([[4.,2.],[1.,3.]])
        loss=auxiliary_scale(3,5)*(first@value).mean()+auxiliary_scale(2,5)*(second@value).mean()
        expected=(torch.cat([first,second])@value).mean()
        torch.testing.assert_close(torch.autograd.grad(loss,value)[0],torch.autograd.grad(expected,value)[0])
        self.assertEqual(auxiliary_scale(0,5),0.)
        with self.assertRaises(ValueError):auxiliary_scale(1,0)


class ProportionalBiasTests(unittest.TestCase):
    def test_magnitude_centering_and_balanced_fixed_point(self):
        change=bias_correction(torch.tensor([9.,7.,1.,3.]),.1,proportional=True)
        torch.testing.assert_close(change,torch.tensor([-.08,-.04,.08,.04]),rtol=0,atol=1e-8)
        self.assertAlmostEqual(float(change.sum()),0.)
        self.assertTrue(torch.equal(bias_correction(torch.ones(8),.01,proportional=True),torch.zeros(8)))
        concentrated=bias_correction(torch.tensor([100.]+[0.]*63),.01,proportional=True)
        self.assertLess(float(concentrated[0]),-.04)
        self.assertLess(float(concentrated.max()-concentrated.min()),.061)

    def test_rolling_coverage_tracks_actual_expert_use(self):
        gate=SimpleNamespace(_cumulative_expert_load=torch.tensor([2,0]),e_score_correction_bias=torch.zeros(2),e_score_correction_bias_master=None)
        with patch('archlab.optimizers.router_balance.dist.all_reduce'):
            first=balance_routers([gate],.01,proportional=True)
            self.assertEqual(first['router_unused_fraction_window'],.5)
            gate._cumulative_expert_load=torch.tensor([0,2])
            second=balance_routers([gate],.01,proportional=True)
            self.assertEqual(second['router_unused_fraction_window'],0.)
            self.assertEqual(second['router_usage_window_updates'],2)
