"""Run in the frozen container with the pinned AutoModel source on PYTHONPATH."""

import importlib.util
import unittest

import torch

AVAILABLE = importlib.util.find_spec("nemo_automodel") is not None


@unittest.skipUnless(AVAILABLE, "requires the pinned upstream NeMo AutoModel source")
class AutoModelSimplicialTests(unittest.TestCase):
    def model_and_config(self):
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.qwen3_8_flash_next.config import (
            Qwen3_8_FlashNextConfig,
            Qwen3_8_FlashNextTextConfig,
        )
        from nemo_automodel.components.models.qwen3_8_flash_next.model import (
            Qwen3_8_FlashNextForConditionalGeneration,
        )

        from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig

        text = Qwen3_8_FlashNextTextConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=["full_attention"] * 2, full_attention_interval=1,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=2, num_experts_per_tok=1, hc_count=4, hc_lowrank=8,
            ple_layer_ids=[], indexer_budget=4, indexer_n_heads=2,
            indexer_head_dim=16, indexer_compress_ratio=2,
            max_position_embeddings=128, dtype="float32",
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .25},
        )
        model = Qwen3_8_FlashNextForConditionalGeneration(
            Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True),
            backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32",
                                  experts="torch", dispatcher="torch", enable_hf_state_dict_adapter=False),
        )
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
        adapter = SimplicialAdapterConfig(hidden_size=32, query_heads=2, kv_heads=1, head_dim=16,
                                         residual_streams=4, residual_low_rank=8,
                                         short_window=2, long_window=8, rope_theta=10000.)
        return model, adapter

    def test_identity_keys_gradients_and_roundtrip(self):
        from archlab.automodel.simplicial import ADAPTER_MARKER, install_simplicial_modules

        torch.manual_seed(57)
        model, config = self.model_and_config()
        tokens = torch.randint(2, 64, (1, 11))  # exercise sparse routing beyond budget
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            before = model(input_ids=tokens, output_hidden_states=True)
        rng = torch.get_rng_state().clone()
        adapters = install_simplicial_modules(model, config, backend="reference")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(adapters), 2)
        after = model(input_ids=tokens, output_hidden_states=True)
        self.assertTrue(torch.equal(before.logits, after.logits))
        for a, b in zip(before.hidden_states, after.hidden_states, strict=True):
            self.assertTrue(torch.equal(a, b))
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(sum(p.numel() for p in trainable), 2 * config.parameter_count())
        optimizer = torch.optim.AdamW(trainable, lr=.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(input_ids=tokens).logits.square().mean().backward()
            for name, parameter in model.named_parameters():
                if ADAPTER_MARKER not in name:
                    self.assertIsNone(parameter.grad, name)
                else:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if step == 1:
                        self.assertGreater(parameter.grad.abs().max().item(), 0, name)
            optimizer.step()
        for name, tensor in base.items():
            self.assertTrue(torch.equal(dict(model.named_parameters())[name], tensor), name)
        restored, _ = self.model_and_config()
        install_simplicial_modules(restored, config, backend="reference")
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(restored(input_ids=tokens).logits, model(input_ids=tokens).logits))
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_simplicial_modules(model, config)

    def test_batch_and_geometry_fail_closed(self):
        from dataclasses import replace

        from archlab.automodel.simplicial import install_simplicial_modules

        model, config = self.model_and_config()
        with self.assertRaisesRegex(ValueError, "geometry"):
            install_simplicial_modules(model, replace(config, hidden_size=64))
        install_simplicial_modules(model, config, backend="reference")
        tokens = torch.ones(1, 5, dtype=torch.long)
        for kwargs in ({"attention_mask": torch.ones_like(tokens)}, {"use_cache": True},
                       {"position_ids": torch.ones_like(tokens)}, {"cu_seqlens": torch.tensor([0, 5])}):
            with self.assertRaises(ValueError):
                model(input_ids=tokens, **kwargs)

    def test_nonzero_first_backward_preserves_frozen_base_and_roundtrips(self):
        from dataclasses import replace

        from archlab.automodel.simplicial import ADAPTER_MARKER, install_simplicial_modules

        torch.manual_seed(58)
        model, config = self.model_and_config()
        config = replace(config, output_initialization="normal")
        tokens = torch.randint(2, 64, (1, 11))
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            original = model(input_ids=tokens).logits.clone()
        install_simplicial_modules(model, config, backend="reference")
        logits = model(input_ids=tokens).logits
        self.assertFalse(torch.equal(original, logits))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        logits.backward(torch.randn_like(logits))
        for name, parameter in model.named_parameters():
            if ADAPTER_MARKER in name:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, expected in base.items():
            self.assertTrue(torch.equal(dict(model.named_parameters())[name], expected), name)
        restored, _ = self.model_and_config()
        install_simplicial_modules(restored, config, seed=88, backend="reference")
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(input_ids=tokens).logits, model(input_ids=tokens).logits,
                                   rtol=0, atol=0)

    def test_checkpoint_wrapped_decoder_executes_added_module(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        from archlab.automodel.simplicial import install_simplicial_modules

        model, config = self.model_and_config()
        model.requires_grad_(False)
        layers = model.model.language_model.layers
        for name, layer in list(layers.items()):
            layers[name] = checkpoint_wrapper(layer)
        adapters = install_simplicial_modules(model, config, backend="reference")
        tokens = torch.randint(2, 64, (1, 11))
        loss = model(input_ids=tokens).logits.square().mean()
        self.assertTrue(loss.requires_grad)
        loss.backward()
        for adapter in adapters.values():
            self.assertIsNotNone(adapter.output.weight.grad)
            self.assertGreater(adapter.output.weight.grad.abs().max().item(), 0)

    def test_meta_rope_uses_original_constructor(self):
        from archlab.automodel.loading import rebuild_nonpersistent_buffers

        model, _ = self.model_and_config()
        rotary = model.model.language_model.rotary_emb
        expected = {name: value.clone() for name, value in rotary.named_buffers()}
        with torch.device("meta"):
            model.model.language_model.rotary_emb = type(rotary)(config=rotary.config)
        rebuild_nonpersistent_buffers(model, torch.device("cpu"))
        for name, value in model.model.language_model.rotary_emb.named_buffers():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        model.register_buffer("unknown", torch.ones(1), persistent=False)
        with self.assertRaisesRegex(ValueError, "unaccounted"):
            rebuild_nonpersistent_buffers(model, torch.device("cpu"))

    def test_missing_load_destination_fails(self):
        from archlab.automodel.loading import (
            assert_loaded_weights_finite,
            poison_weights_before_load,
        )

        model = torch.nn.Linear(4, 4)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        poison_weights_before_load(model)
        with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
            assert_loaded_weights_finite(model)
        model.load_state_dict(original)
        self.assertEqual(assert_loaded_weights_finite(model), 20)


if __name__ == "__main__":
    unittest.main()
