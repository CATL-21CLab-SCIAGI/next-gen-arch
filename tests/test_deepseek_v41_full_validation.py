import copy
import random
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from archlab.automodel.deepseek_v41_full_validation import (
    make_plan, capped_batch, head_totals, evaluation_state, admit_evaluation_upgrade,
    LEGACY_TRAINER_SHA256,
)


class ValidationTests(unittest.TestCase):
    def test_exact_budget_plan_and_mask(self):
        labels = torch.tensor([[-100, 1, 2, -100, 3, 4]])
        pilot = SimpleNamespace(windows=[{'targets':4}, {'targets':4}], order_seed=7,
            manifest={'windows_sha256':'windows','source_ready_sha256':'source'},
            batch=lambda index,device:(torch.zeros_like(labels), labels, 4))
        plan = make_plan(pilot, 6)
        self.assertEqual(plan['windows'], [{'index':0,'targets':4},{'index':1,'targets':2}])
        self.assertEqual(plan, make_plan(pilot,6))
        _, got, count = capped_batch(pilot, plan['windows'][-1], 'cpu')
        self.assertEqual(count, 2)
        self.assertEqual(got.tolist(), [[-100,1,2,-100,-100,-100]])
        self.assertEqual(labels.tolist(), [[-100,1,2,-100,3,4]])
        with self.assertRaises(ValueError): make_plan(pilot,9)

    def test_metrics_match_dense_oracle(self):
        torch.manual_seed(12)
        hidden, weight = torch.randn(2,7,5), torch.randn(13,5)
        labels = torch.randint(0,13,(2,7)); labels[0,:3] = -100
        logits = torch.nn.functional.linear(hidden,weight)[labels!=-100]
        targets = labels[labels!=-100]
        logp = logits.log_softmax(-1)
        expected = torch.tensor([len(targets),
            torch.nn.functional.cross_entropy(logits,targets,reduction='sum'),
            (logits.argmax(-1)==targets).sum(),
            (logits.topk(5,-1).indices==targets[:,None]).any(-1).sum(),
            -(logp.exp()*logp).sum()],dtype=torch.float64)
        torch.testing.assert_close(head_totals(hidden,labels,weight,chunk_size=3),expected)
        self.assertEqual(float(head_totals(hidden,torch.full_like(labels,-100),weight).sum()),0)

    def test_rng_modes_and_parameters_preserved_on_error(self):
        model = torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout())
        model.train();model[1].eval()
        before = copy.deepcopy(model.state_dict())
        cpu = torch.get_rng_state();py = random.getstate();npstate = np.random.get_state()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with evaluation_state(model):
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(all(not m.training for m in model.modules()))
                torch.rand(5);random.random();np.random.rand(3)
                raise RuntimeError('fixture')
        self.assertTrue(model.training);self.assertFalse(model[1].training)
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());np.testing.assert_equal(npstate,np.random.get_state())
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_upgrade_rejects_model_or_runtime_drift(self):
        old={'project_commit':'old','runtime':{'cuda':'same'},'implementation_sha256':{
            'automodel/deepseek_v41_full_training.py':LEGACY_TRAINER_SHA256,'model.py':'model'}}
        new=copy.deepcopy(old);new['project_commit']='new'
        new['implementation_sha256']['automodel/deepseek_v41_full_validation.py']='new'
        new['implementation_sha256']['model.py']='changed'
        with self.assertRaisesRegex(ValueError,'protected'): admit_evaluation_upgrade(old,new,lambda:None)
        new['runtime']['cuda']='other'
        with self.assertRaisesRegex(ValueError,'runtime'): admit_evaluation_upgrade(old,new,lambda:None)


if __name__ == '__main__': unittest.main()
