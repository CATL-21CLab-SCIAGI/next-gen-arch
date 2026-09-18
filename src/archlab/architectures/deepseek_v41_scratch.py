"""Explicit width/depth scaling contract for a randomly initialized V4.1 text model."""

from copy import deepcopy

CHANNEL_FIELDS = (
    "hidden_size",
    "moe_intermediate_size",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "index_head_dim",
    "engram_head_dim",
)


def scaled_scratch_config(base):
    text = deepcopy(base["text_config"])
    if (text["hidden_size"], text["num_hidden_layers"]) != (5120, 40):
        raise ValueError("expected released V4.1 Flash geometry")
    for key in CHANNEL_FIELDS:
        if text[key] % 8:
            raise ValueError(f"{key} cannot be divided by8")
        text[key] //= 8
    text["num_hidden_layers"] = 20
    text["compress_ratios"] = text["compress_ratios"][:40:2]
    for key in ["kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"]:
        text[key] = [x // 2 for x in text[key]]
    text["candidate_source_layer_id"] //= 2
    text.update(
        num_nextn_predict_layers=0,
        dspark_target_layer_ids=[],
        dspark_block_size=0,
        max_position_embeddings=16384,
        rope_scaling={"rope_type": "default"},
        use_cache=False,
    )
    return {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "text_config": text,
        "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 0},
    }


def adapter_layers():
    return tuple(i // 2 for i in (4, 9, 14, 19, 24, 29, 34, 39))
