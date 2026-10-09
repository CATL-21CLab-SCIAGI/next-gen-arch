# ruff: noqa: I001  # Preserve the checkpoint-qualified import order.
"""Byte-exact selector and peak-memory oracle, including full 16K geometry."""

import argparse
from pathlib import Path
import torch
from archlab.automodel.deepseek_v41_full_indexer_memory import bind_memory_efficient_selector


def qualify_indexer_memory(config, *, sequence):
    from nemo_automodel.components.models.deepseek_v41.attention import (
        _Indexer,
        _RotaryEmbedding,
        DeepseekV41AttentionState,
    )

    reports = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(105)
        inputs = torch.randn(1, sequence, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        queries = torch.randn(1, sequence, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(sequence, device="cuda")[None]
        angles = _RotaryEmbedding(config, compressed=True)(positions)
        candidate_state = None
        selected_layers = [
            config.kv_source_layer_ids[0],
            config.candidate_source_layer_id,
            next(
                (x for x in config.index_source_layer_ids if x > config.candidate_source_layer_id),
                None,
            ),
        ]
        for layer in dict.fromkeys(x for x in selected_layers if x is not None and x >= 0):
            module = _Indexer(config, layer_idx=layer, dtype=torch.bfloat16).cuda()
            ratio = config.compress_ratios[layer]
            width = sequence // ratio
            latent = torch.randn(1, width, config.head_dim, device="cuda", dtype=torch.bfloat16)
            state = DeepseekV41AttentionState(
                compressed_kv=latent.clone(),
                compression_ratio=ratio,
                index_keys=None if module.owns_keys else candidate_state.index_keys,
                candidates=None if not module.uses_candidates else candidate_state.candidates,
            )
            outputs, peaks = [], []
            for fn in (module.forward, bind_memory_efficient_selector(module)):
                torch.cuda.reset_peak_memory_stats()
                outputs.append(
                    fn(
                        inputs,
                        query_latent=queries,
                        latent=latent if module.owns_keys else None,
                        angles=angles,
                        compressed_angles=angles[:, : width * ratio : ratio],
                        state=state,
                    )
                )
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated() / 2**30)
            for field in ("compressed_kv", "index_keys", "topk_indices", "candidates"):
                a, b = getattr(outputs[0], field), getattr(outputs[1], field)
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    raise AssertionError(f"layer {layer}: selector field {field} differs")
            reports.append(
                {
                    "layer": layer,
                    "sequence": sequence,
                    "byte_exact": True,
                    "original_peak_gib": peaks[0],
                    "optimized_peak_gib": peaks[1],
                }
            )
            if module.is_candidate_source:
                candidate_state = outputs[1]
            del outputs, module
        return {"passed": True, "cases": reports}


def main():
    import json
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path("/usr/local/lib/python3.12/dist-packages"))
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = DeepseekV41Config.from_pretrained(args.weights, local_files_only=True).text_config
    torch.set_num_threads(4)
    result = qualify_indexer_memory(config, sequence=16384)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
