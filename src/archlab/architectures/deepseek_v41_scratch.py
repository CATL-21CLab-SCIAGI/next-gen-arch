"""Explicit width/depth scaling contract for a randomly initialized V4.1 text model."""

from copy import deepcopy
from fractions import Fraction

SCRATCH_WIDTHS = (128, 384, 640, 1280)
SWEEP_WIDTHS = (128, 256, 384)
SCALING_EXPERT_WIDTH = 128
SCALING_SHARED_EXPERTS = 1
SCALING_ACTIVE_WIDTH_RATIO = Fraction(3, 1)

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


def _aligned_channel(value, width, *, multiple=1, power_of_two=False):
    scaled = Fraction(value * width, 5120)
    rounded = max(multiple, -(-scaled.numerator // (scaled.denominator * multiple)) * multiple)
    return 1 << (rounded - 1).bit_length() if power_of_two else rounded


def scratch_adapter_head_dim(width, *, sweep_geometry=False):
    if width not in (SWEEP_WIDTHS if sweep_geometry else SCRATCH_WIDTHS):
        raise ValueError(f"scratch width must be one of {SCRATCH_WIDTHS}")
    return _aligned_channel(128, width, multiple=16, power_of_two=True)


def scaling_experts_per_token(width, *, sweep_geometry=False):
    """Routed top-k; the native always-active shared expert also consumes width."""
    active = SCALING_ACTIVE_WIDTH_RATIO * width / SCALING_EXPERT_WIDTH
    allowed = SWEEP_WIDTHS if sweep_geometry else SCRATCH_WIDTHS
    if type(width) is not int or width not in allowed or active.denominator != 1:
        raise ValueError(f"scratch width must be one of {SCRATCH_WIDTHS}")
    return int(active) - SCALING_SHARED_EXPERTS


def scaled_scratch_config(base, *, width=640, scaling_study=False, depth=20, sweep_geometry=False,
                          scale_engram=False, engram_anchor_width=None):
    """Scale channels at fixed depth, preserving the original d640 geometry.

    Round nonintegral channels upward to the declared kernel alignments. Sparse
    head dimensions are powers of two, RoPE is even, matrix channels are aligned
    to 16, and legacy Engram channels to 8. With scale_engram, bucket count
    follows the model-width ratio and head channels round upward to multiples of 8.
    An explicit Engram anchor fixes both lookup dimensions across model sizes;
    its projections still map into the requested model width.
    """
    allowed = SWEEP_WIDTHS if sweep_geometry else SCRATCH_WIDTHS
    if type(width) is not int or width not in allowed:
        raise ValueError(f"scratch width must be one of {SCRATCH_WIDTHS}")
    if type(depth) is not int or not 10 <= depth <= 45:
        raise ValueError("scratch depth must be an integer in [10, 45]")
    text = deepcopy(base["text_config"])
    if (text["hidden_size"], text["num_hidden_layers"]) != (5120, 40):
        raise ValueError("expected released V4.1 Flash geometry")
    alignments = {
        "hidden_size": (1, False),
        "moe_intermediate_size": (16, False),
        "head_dim": (16, True),
        "qk_rope_head_dim": (2, False),
        "q_lora_rank": (16, False),
        "o_lora_rank": (16, False),
        "index_head_dim": (4, True),
        "engram_head_dim": (8, False),
    }
    for key in CHANNEL_FIELDS:
        multiple, power_of_two = alignments[key]
        text[key] = _aligned_channel(text[key], width, multiple=multiple, power_of_two=power_of_two)
    if engram_anchor_width is not None and not scale_engram:
        raise ValueError("fixed Engram geometry requires scale_engram")
    if scale_engram:
        from archlab.architectures.engram_scaling import scaled_engram_geometry

        text.update(scaled_engram_geometry(
            base["text_config"], width, anchor_width=engram_anchor_width
        ))
    if scaling_study:
        if text.get("n_shared_experts", 1) != SCALING_SHARED_EXPERTS:
            raise ValueError("scaling study requires the native single shared expert")
        text["n_shared_experts"] = SCALING_SHARED_EXPERTS
        text["moe_intermediate_size"] = SCALING_EXPERT_WIDTH
        text["num_experts_per_tok"] = scaling_experts_per_token(
            width, sweep_geometry=sweep_geometry
        )
    text["num_hidden_layers"] = depth
    text["compress_ratios"] = [text["compress_ratios"][i * 40 // depth] for i in range(depth)]
    # Source ownership starts at the first layer in each compression region.
    for key in ("kv_source_layer_ids", "index_source_layer_ids"):
        text[key] = sorted({(x * depth + 39) // 40 for x in text[key]})
    text["engram_layer_ids"] = [x * depth // 40 for x in text["engram_layer_ids"]]
    text["candidate_source_layer_id"] = (text["candidate_source_layer_id"] * depth + 39) // 40
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


def adapter_layers(depth=20):
    return tuple(i * depth // 40 for i in (4, 9, 14, 19, 24, 29, 34, 39))
