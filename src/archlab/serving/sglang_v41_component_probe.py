"""GPU comparison of cached adapters against their trained production kernels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("normal", "simplicial"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
    from archlab.architectures.deepseek_v41_incremental import IncrementalV41Adapter
    from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter
    from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    inventory = V41CheckpointInventory(args.checkpoint)
    if inventory.marker["contract"]["variant"] != args.variant:
        raise ValueError("wrong checkpoint variant")
    cls = V41NormalAttentionAdapter if args.variant == "normal" else V41SimplicialAdapter
    backend = "flash-attn-deterministic" if args.variant == "normal" else "deterministic"
    adapter = cls(V41AdapterConfig(), backend=backend).eval().requires_grad_(False)
    prefix = "model.layers.4._checkpoint_wrapped_module.attn_hc.simplicial_adapter."
    adapter.load_state_dict({name: inventory.read_tensor(prefix + name)
                             for name in adapter.state_dict()}, strict=True)
    adapter.cuda()
    torch.manual_seed(83)
    streams = torch.randn(1, 514, 4, 5120, device="cuda", dtype=torch.bfloat16)
    branches = []
    hook = adapter.output.register_forward_hook(
        lambda module, inputs, value: branches.append(value.detach().float().cpu()))
    with torch.inference_mode():
        reference = adapter(streams).float().cpu()
        reference_branch = branches.pop()
        cache = IncrementalV41Adapter(adapter)
        actual = torch.cat([cache(streams[:, :257], start_position=0),
                            cache(streams[:, 257:513], start_position=257),
                            cache(streams[:, 513:], start_position=513)], dim=1).float().cpu()
    hook.remove()
    branch = torch.cat(branches, dim=1)
    denominator = float(reference_branch.norm())
    if denominator == 0:
        raise ValueError("zero trained branch cannot qualify a cached attention implementation")
    relative = float((branch - reference_branch).norm()) / denominator
    finite = bool(actual.isfinite().all() and branch.isfinite().all())
    maximum = float((actual - reference).abs().max())
    passed = finite and relative <= 0.02 and maximum <= 0.03125
    report = dict(passed=passed, variant=args.variant, cursor=inventory.marker["cursor"],
                  tokens=514, layer=4, precision="CUDA BF16 projections, FP32 attention arithmetic",
                  reference_backend=backend, output_max_abs_error=maximum,
                  branch_relative_l2_error=relative, branch_reference_l2=denominator,
                  thresholds=dict(branch_relative_l2=0.02, output_max_abs=0.03125),
                  cache_lengths=cache.cache_lengths, gpu="NVIDIA B300",
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  full_model_qualified=False)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit("cached adapter did not pass GPU component qualification")


if __name__ == "__main__":
    main()
