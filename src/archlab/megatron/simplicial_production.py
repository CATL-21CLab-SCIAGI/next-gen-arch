"""Opt-in attention replacement for the unchanged W320 production trainer.

The data iterator, loss, optimizer, schedule and native training loop are owned
by qwen38_flash_next_full_train. This boundary only changes the selected core.
"""

from __future__ import annotations

import hashlib
import json

GLOBAL_ATTENTION = "global"
SIMPLICIAL_ATTENTION = "simplicial-rope-16x128"
CHANGED_LAYERS = (8, 16, 24, 32, 40, 48)
EXTRA_PARAMETERS = len(CHANGED_LAYERS) * (2 * 64 * 320 + 32)


def attention_variant_contract(options):
    variant = getattr(options, "attention_variant", GLOBAL_ATTENTION)
    if variant == GLOBAL_ATTENTION:
        return None
    if variant != SIMPLICIAL_ATTENTION:
        raise ValueError("unknown production attention variant")
    if options.model_variant != "w320-e32-depth48-no-mtp" or options.parallelism != "dp-only":
        raise ValueError("simplicial production requires the W320/E32 DP-only model")
    return {
        "name": variant, "changed_layers_1based": list(CHANGED_LAYERS),
        "unchanged_global_layers_1based": [4, 12, 20, 28, 36, 44],
        "short_window": 16, "long_window": 128,
        "positional_encoding": "ordinary-partial-RoPE-on-Q-K1-K2",
        "retained": ["output-gate", "QK-RMSNorm", "four-residual-streams", "GDN", "MoE", "PLE"],
        "extra_parameters": EXTRA_PARAMETERS,
        "initialization": "fresh-common-weights-plus-isolated-extra-projection-RNG",
        "data_order": "unchanged-production-DP-rank-prefix-partition",
        "not_pilot_strided_data_order": True,
    }


def validate_attention_resume(previous, current):
    # Historical global runs did not need an attention-variant field.
    if previous.get("attention_variant") != current.get("attention_variant"):
        raise RuntimeError("attention variant changed; use a fresh run directory and weights")


def install_production_attention(model, options):
    """Preserve and hash every common parameter before native optimizer setup."""
    import torch

    from archlab.megatron.simplicial_attention import install_pilot_attention, parameter_hashes

    contract = attention_variant_contract(options)
    common = parameter_hashes(model)
    if contract is not None:
        install_pilot_attention(model, "C", seed=options.seed, short_window=16, long_window=128)
    if parameter_hashes(model, common_only=True) != common:
        raise RuntimeError("attention installation changed a common baseline weight")
    reference = getattr(options, "initialization_reference", None)
    if reference is not None:
        expected = json.loads(reference.read_text())["common_parameter_sha256"]
        if common != expected:
            raise RuntimeError("production common initialization differs from its baseline reference")
    all_hashes = parameter_hashes(model)
    digest = hashlib.sha256(json.dumps(all_hashes, sort_keys=True).encode()).hexdigest()
    replicas = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(replicas, digest)
    if any(value != digest for value in replicas):
        raise RuntimeError("initial production weights differ across DP replicas")
    return {"common_parameter_sha256": common, "all_parameter_sha256": all_hashes,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "matches_reference": reference is not None, "all_dp_replicas_equal": True,
            "attention_variant": contract}
