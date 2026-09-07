"""Inference-only checkpoint subset and original PLE offload regression tests."""

from pathlib import Path
import json
import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp

from archlab.automodel.sample import SamplingConfig, attach_cpu_lookup_hook, restore_adapters


class SamplingTests(unittest.TestCase):
    def test_config_rejects_invalid_values(self):
        for kwargs in ({"max_new_tokens": 0}, {"temperature": float("nan")},
                       {"top_p": 0}, {"gpu_headroom_gib": -1}):
            with self.assertRaises(ValueError):
                SamplingConfig(**kwargs)

    def test_adapter_subset_load_exact_and_fail_closed(self):
        from archlab.automodel.checkpointing import state_digest

        adapters = {"3": torch.nn.Linear(5, 7), "7": torch.nn.Linear(5, 7)}
        saved = {name: {key: t.clone() for key, t in m.state_dict().items()} for name, m in adapters.items()}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            dcp.save({"adapters": saved, "optimizer": {"unused": torch.ones(23)}}, checkpoint_id=path / "state")
            digest = restore_adapters(adapters, path)
            self.assertEqual(digest, state_digest(saved))
            for name, module in adapters.items():
                for key, actual in module.state_dict().items():
                    torch.testing.assert_close(actual, saved[name][key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "coverage"):
                restore_adapters({"3": adapters["3"]}, path)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                restore_adapters({"3": torch.nn.Linear(5, 8), "7": adapters["7"]}, path)

    def test_upstream_single_owner_base_loader_exact_roundtrip(self):
        from safetensors.torch import save_file
        from nemo_automodel.components.checkpoint.config import CheckpointingConfig
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
        from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
        from archlab.automodel.execution import tiny_config
        from archlab.automodel.loading import poison_weights_before_load, assert_loaded_weights_finite

        config = tiny_config(8)
        config.text_config.ple_layer_ids = [2]
        text = config.text_config
        table = Qwen3_8_FlashNextEngramTableConfig(
            256, text.ple_embed_dim // ((text.ngram_size - 1) * text.heads_per_ngram))
        model = Qwen3_8_FlashNextForConditionalGeneration(
            config, engram_table_config=table,
            backend=BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32", experts="torch_mm",
                                  dispatcher="torch", rope_fusion=False, gate_precision="float32",
                                  enable_hf_state_dict_adapter=True),
            moe_overrides={"aux_loss_coeff": 0.0})
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
        expected = {name: value.detach().clone() for name, value in model.named_parameters()}
        hf = {name: value.contiguous().clone() for name, value in model.state_dict_adapter.to_hf(model.state_dict()).items()
              if isinstance(value, torch.Tensor)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            save_file(hf, path / "model.safetensors")
            (path / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {}, "weight_map": {name: "model.safetensors" for name in hf}}))
            poison_weights_before_load(model)
            loader = CheckpointingConfig(checkpoint_dir="", model_repo_id=str(path), save_consolidated=False,
                                         dequantize_base_checkpoint=False).build(dp_rank=0, tp_rank=0, pp_rank=0)
            loader.load_base_model(model, torch.device("cpu"), None, str(path))
            assert_loaded_weights_finite(model)
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), "requires GPU for CPU/GPU offload equivalence")
    def test_original_ple_cpu_lookup_matches_gpu_bitwise(self):
        from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig

        table = Qwen3_8_FlashNextEngramTableConfig(256, 160).build(process_group=None, dtype=torch.bfloat16)
        with torch.no_grad():
            table.weight.normal_()
        ids = torch.tensor([[0, 255, 23, 23], [255, 0, 2, 19]], device="cuda")
        expected = torch.nn.functional.embedding(ids, table.weight.to("cuda"))
        keys = set(table.state_dict())
        attach_cpu_lookup_hook(table)
        actual = table(ids)
        self.assertEqual(actual.device, ids.device)
        self.assertEqual(table.weight.device.type, "cpu")
        self.assertEqual(set(table.state_dict()), keys)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires the existing DSW sampling GPU")
    def test_upstream_gdn_qsa_moe_and_added_branch_uncached_forward(self):
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
        from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig
        from archlab.automodel.execution import tiny_config
        from archlab.automodel.runtime import configure_frozen_gdn_runtime
        from archlab.automodel.simplicial import install_simplicial_modules

        configure_frozen_gdn_runtime()
        model = Qwen3_8_FlashNextForConditionalGeneration(
            tiny_config(8),
            backend=BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32", experts="torch_mm",
                                  dispatcher="torch", rope_fusion=False, gate_precision="float32",
                                  enable_hf_state_dict_adapter=False),
            moe_overrides={"aux_loss_coeff": 0.0})
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
        config = SimplicialAdapterConfig(hidden_size=256, query_heads=4, kv_heads=2, head_dim=64,
                                         residual_low_rank=32, short_window=4, long_window=32)
        adapters = install_simplicial_modules(model, config, dtype=torch.bfloat16)
        with torch.no_grad():
            for adapter in adapters.values():
                adapter.output.weight.normal_(std=0.01)
        model.requires_grad_(False).eval().to("cuda")
        tokens = torch.tensor([[5, 17, 92, 43, 16, 87, 67]], device="cuda")
        with torch.inference_mode():
            for length in (6, 7, 6):
                result = model(input_ids=tokens[:, :length], logits_to_keep=1, use_cache=False,
                               output_hidden_states=False).logits
                self.assertEqual(tuple(result.shape), (1, 1, 1024))
                self.assertTrue(torch.isfinite(result).all())

    @unittest.skipUnless(torch.cuda.is_available(), "requires GPU for FlexAttention dispatch parity")
    def test_short_prefix_flex_unfused_matches_compiled(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def causal(batch, head, query, key):
            return query >= key

        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        generator = torch.Generator(device="cuda").manual_seed(43)
        with torch.inference_mode():
            for length in (17, 129):
                q = torch.randn(1, 24, length, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
                k, v = [torch.randn(1, 2, length, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
                        for _ in range(2)]
                mask = create_block_mask(causal, 1, None, length, length, device="cuda")
                expected = compiled(q, k, v, block_mask=mask, enable_gqa=True)
                actual = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
