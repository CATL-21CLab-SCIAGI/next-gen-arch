import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from archlab.rl.weight_residency import retained_fsdp_weights

GIB = 1024 ** 3


class FakeFSDP(nn.Module):
    def __init__(self, *, policy=True, automatic=False):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
        self.original = self.weight
        mesh = SimpleNamespace(shard_mesh_size=2)
        parameter = SimpleNamespace(sharded_param=self.weight, offload_to_cpu=False,
                                    mesh_info=mesh, _sharded_param_data=torch.empty(6))
        self.group = SimpleNamespace(mesh_info=mesh, post_forward_mesh_info=mesh if policy else None,
                                     mp_policy=SimpleNamespace(param_dtype=torch.float32),
                                     fsdp_params=[parameter], _sharded_state=SimpleNamespace(name="SHARDED"))
        self.state = SimpleNamespace(_auto_reshard_after_forward=automatic, _fsdp_param_groups=[self.group])
        self.events = []

    def _get_fsdp_state(self):
        return self.state

    def set_reshard_after_forward(self, value, recurse=True):
        self.events.append(("set", value))
        self.state._auto_reshard_after_forward = False
        self.group.post_forward_mesh_info = self.group.mesh_info if value else None

    def unshard(self):
        if self.group._sharded_state.name == "SHARDED":
            self.events.append(("gather",))
            self.weight = nn.Parameter(self.original.detach().clone(), requires_grad=self.original.requires_grad)
            self.group._sharded_state.name = "UNSHARDED"

    def reshard(self):
        self.events.append(("reshard",))
        self.weight = self.original
        self.group._sharded_state.name = "SHARDED"

    def forward(self, hidden):
        self.unshard()
        return hidden @ self.weight.T


class WeightResidencyTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.body = FakeFSDP(policy=True)
        self.model.lm_head = FakeFSDP(policy=False)
        self.fsdp = patch("archlab.rl.weight_residency.FSDPModule", FakeFSDP)
        self.memory = patch("archlab.rl.weight_residency._free_memory", return_value=100 * GIB)
        self.fsdp.start()
        self.memory.start()
        self.addCleanup(self.fsdp.stop)
        self.addCleanup(self.memory.stop)

    def test_lazy_gather_head_retention_and_original_policies_restore(self):
        original = {name: parameter for name, parameter in self.model.named_parameters()}
        values = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        with retained_fsdp_weights(self.model) as residency:
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(any(event[0] == "gather" for event in self.model.body.events))
            self.model.body(torch.ones(1, 3))
            self.model.body(torch.ones(1, 3))
            for _ in range(3):
                with residency.inference_head(self.model.lm_head) as head:
                    torch.nn.functional.linear(torch.ones(1, 3), head.weight)
            self.assertEqual(self.model.body.events.count(("gather",)), 1)
            self.assertEqual(self.model.lm_head.events.count(("gather",)), 1)
            self.assertNotIn(("reshard",), self.model.lm_head.events)
        self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(residency.receipt["cleanup_verified"])
        self.assertIs(self.model.body.group.post_forward_mesh_info, self.model.body.group.mesh_info)
        self.assertIsNone(self.model.lm_head.group.post_forward_mesh_info)
        for name, parameter in self.model.named_parameters():
            self.assertIs(parameter, original[name])
            self.assertTrue(torch.equal(parameter, values[name]))
            self.assertTrue(parameter.requires_grad)
        self.assertFalse(self.model.body._forward_pre_hooks)
        self.assertFalse(hasattr(self.model, "_archlab_weight_residency_active"))

    def test_exception_cleans_all_modules_and_rejects_dead_callback(self):
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with retained_fsdp_weights(self.model) as residency:
                self.model.body(torch.ones(1, 3))
                with residency.inference_head(self.model.lm_head):
                    raise RuntimeError("intentional")
        self.assertTrue(residency.receipt["cleanup_verified"])
        self.assertEqual(self.model.body.group._sharded_state.name, "SHARDED")
        self.assertEqual(self.model.lm_head.group._sharded_state.name, "SHARDED")
        with self.assertRaisesRegex(RuntimeError, "active no-grad"):
            with residency.inference_head(self.model.lm_head):
                pass

    def test_initial_memory_reserve_fails_before_mutation(self):
        with patch("archlab.rl.weight_residency._free_memory", return_value=16 * GIB), self.assertRaises(MemoryError):
            with retained_fsdp_weights(self.model):
                pass
        self.assertEqual(self.model.body.events, [])
        self.assertEqual(self.model.lm_head.events, [])

    def test_memory_checked_again_before_lazy_gather(self):
        with patch("archlab.rl.weight_residency._free_memory", side_effect=[100 * GIB, 15 * GIB]), self.assertRaises(MemoryError):
            with retained_fsdp_weights(self.model) as residency:
                self.model.body(torch.ones(1, 3))
        self.assertTrue(residency.receipt["cleanup_verified"])
        self.assertNotIn(("gather",), self.model.body.events)

    def test_consecutive_context_reclaims_cached_buffers_only_on_entry(self):
        memory = {"free": 100 * GIB, "cache": 0}

        def release(device):
            memory["free"] += memory["cache"]
            memory["cache"] = 0

        with patch("archlab.rl.weight_residency._free_memory", side_effect=lambda device: memory["free"]), patch("archlab.rl.weight_residency._unused_allocator_cache", side_effect=lambda device: memory["cache"]), patch("archlab.rl.weight_residency._release_unused_cache", side_effect=release) as empty:
            with retained_fsdp_weights(self.model) as first:
                self.model.body(torch.ones(1, 3))
            self.assertFalse(first.receipt["entry_cache_release_attempted"])
            # Freed gathered buffers remain in PyTorch's unused reserved pool.
            memory.update(free=16 * GIB, cache=40 * GIB)
            with retained_fsdp_weights(self.model) as second:
                self.model.body(torch.ones(1, 3))
                with second.inference_head(self.model.lm_head):
                    pass
            self.assertTrue(second.receipt["cleanup_verified"])
            self.assertEqual(second.receipt["entry_cache_reclaimed_bytes"], 40 * GIB)
            empty.assert_called_once()

    def test_insufficient_cache_release_does_not_relax_memory_bound(self):
        with patch("archlab.rl.weight_residency._free_memory", return_value=16 * GIB), patch("archlab.rl.weight_residency._unused_allocator_cache", return_value=80 * GIB), patch("archlab.rl.weight_residency._release_unused_cache") as empty, self.assertRaises(MemoryError):
            with retained_fsdp_weights(self.model):
                pass
        empty.assert_called_once()
        self.assertEqual(self.model.body.events, [])

    def test_gradients_and_unsupported_policy_are_rejected(self):
        self.model.body.weight.grad = torch.ones_like(self.model.body.weight)
        with self.assertRaisesRegex(ValueError, "release all gradients"):
            with retained_fsdp_weights(self.model):
                pass
        self.model.body.weight.grad = None
        self.model.body.state._auto_reshard_after_forward = True
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            with retained_fsdp_weights(self.model):
                pass
        self.model.body.state._auto_reshard_after_forward = False
        self.model.body.group.post_forward_mesh_info = SimpleNamespace(shard_mesh_size=1)
        with self.assertRaisesRegex(ValueError, "partial/integer"):
            with retained_fsdp_weights(self.model):
                pass

    def test_nested_context_and_backward_mode_are_rejected(self):
        with retained_fsdp_weights(self.model):
            with self.assertRaisesRegex(ValueError, "nested"):
                with retained_fsdp_weights(self.model):
                    pass
            with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "no-grad"):
                self.model.body(torch.ones(1, 3))

    def test_gradient_flag_change_cannot_pass_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, "gradient flags|requires_grad"):
            with retained_fsdp_weights(self.model):
                self.model.body.weight.requires_grad_(False)


if __name__ == "__main__":
    unittest.main()
