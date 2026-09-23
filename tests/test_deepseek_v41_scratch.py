import json
import unittest

from archlab.architectures.deepseek_v41_scratch import (
    CHANNEL_FIELDS,
    adapter_layers,
    scaled_scratch_config,
)
from archlab.automodel.deepseek_v41_scratch_training import (
    build_batches,
    canonical_contract,
    learning_rate,
    training_runtime,
)


class ScratchContractTests(unittest.TestCase):
    def test_scaling_preserves_categorical_counts(self):
        base = {
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "moe_intermediate_size": 2304,
                "head_dim": 512,
                "qk_rope_head_dim": 64,
                "q_lora_rank": 1280,
                "o_lora_rank": 1024,
                "index_head_dim": 128,
                "engram_head_dim": 256,
                "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
                "kv_source_layer_ids": [2, 8, 14, 20],
                "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
                "engram_layer_ids": [1, 14],
                "candidate_source_layer_id": 20,
                "n_routed_experts": 384,
                "num_experts_per_tok": 6,
                "engram_num_embeddings": [384006168, 384016682],
                "num_attention_heads": 64,
            }
        }
        out = scaled_scratch_config(base)
        t = out["text_config"]
        self.assertEqual((t["hidden_size"], t["num_hidden_layers"]), (640, 20))
        for field in CHANNEL_FIELDS:
            self.assertEqual(t[field] * 8, base["text_config"][field])
        for field in [
            "n_routed_experts",
            "num_experts_per_tok",
            "engram_num_embeddings",
            "num_attention_heads",
        ]:
            self.assertEqual(t[field], base["text_config"][field])
        self.assertEqual(len(t["compress_ratios"]), 20)
        self.assertEqual(t["kv_source_layer_ids"], [1, 4, 7, 10])
        self.assertEqual(t["engram_layer_ids"], [0, 7])
        self.assertEqual(adapter_layers(), (2, 4, 7, 9, 12, 14, 17, 19))
        self.assertNotIn("quantization_config", out)
        self.assertEqual(base["text_config"]["hidden_size"], 5120)

    def test_learning_rate_endpoints_and_resume(self):
        self.assertEqual(learning_rate(0, 0), 0.0001)
        self.assertEqual(learning_rate(99, 2_000_000), 0.01)
        self.assertAlmostEqual(learning_rate(1000, 5_000_000_000), 0.0055)
        self.assertEqual(learning_rate(2000, 10_000_000_000), 0.001)
        self.assertEqual(
            learning_rate(1000, 5_000_000_000),
            learning_rate(**json.loads('{"step":1000,"tokens":5000000000}')),
        )


class ScratchSerializationTests(unittest.TestCase):
    def test_checkpoint_contract_json_round_trip(self):
        raw = {"geometry": {"id2label": {0: "LABEL_0", 1: "LABEL_1"}}, "axes": ("fsdp", "ep")}
        normalized = canonical_contract(raw)
        self.assertEqual(normalized, json.loads(json.dumps(normalized)))
        self.assertEqual(normalized["geometry"]["id2label"], {"0": "LABEL_0", "1": "LABEL_1"})
        self.assertEqual(raw["geometry"]["id2label"], {0: "LABEL_0", 1: "LABEL_1"})

    def test_padding_opt_in_and_boundaries(self):
        import torch

        from archlab.automodel.deepseek_v41_official_adapter import _check_text_window

        tokens = torch.tensor([[0, 12, 13, 2], [0, 11, 2, 2]])
        good = torch.tensor([[True, True, True, False], [True, True, False, False]])
        kwargs = {"input_ids": tokens, "attention_mask": good}
        with self.assertRaises(ValueError):
            _check_text_window(None, (), kwargs)
        _check_text_window(None, (), kwargs, allow_right_padding=True)
        for bad in [
            torch.tensor([[False, True, True, False], [True, True, False, False]]),
            torch.tensor([[True, False, True, False], [True, True, False, False]]),
        ]:
            with self.assertRaisesRegex(ValueError, "contiguous"):
                _check_text_window(
                    None, (), {"input_ids": tokens, "attention_mask": bad}, allow_right_padding=True
                )


class ScratchRankContractTests(unittest.TestCase):
    def test_uneven_shard_storage_does_not_change_contract(self):
        rank0 = {
            "parameters": 29055310008,
            "local_parameter_gib": 6.7929026037454605,
            "world_size": 8,
        }
        rank7 = {
            "parameters": 29055310008,
            "local_parameter_gib": 6.7928976863622665,
            "world_size": 8,
        }
        self.assertEqual(training_runtime(rank0), training_runtime(rank7))
        self.assertNotIn("local_parameter_gib", training_runtime(rank0))


class ScratchMicrobatchTests(unittest.TestCase):
    def test_microbatch_change_preserves_effective_batch(self):
        class Data:
            def batch(self, indices, device):
                return list(indices)

        old = [
            i
            for rank in range(8)
            for group in build_batches(Data(), 640, rank, microbatch=2, accumulation=4)
            for i in group
        ]
        new = [i for rank in range(8) for group in build_batches(Data(), 640, rank) for i in group]
        self.assertEqual(sorted(old), list(range(640, 704)))
        self.assertEqual(sorted(new), sorted(old))
