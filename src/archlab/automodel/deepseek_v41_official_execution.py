"""Pinned official AutoModel V4.1 construction in the existing container.

The upstream wrapper owns model construction, FSDP/EP placement and DCP base
loading. Project boundaries retain native HC coefficient and expert projection/
sum precision, then add separately optimized adapters after loading the base.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import os
import socket
import subprocess
from pathlib import Path

UPSTREAM_COMMIT = "f7ccd6f7902634af34c2f31b3294ac250dc97670"
MODEL_REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"


def configure_official_reproducibility():
    """Stabilize library math while retaining the measured native adapter core."""
    import torch

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("export CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    # The unchanged simplicial core explicitly rejects global deterministic
    # mode because its FP32 atomic backward has a measured replay envelope.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The frozen container cannot sentinel-fill packed FP4 allocations. These
    # checkpoint/kernel buffers are populated before use; algorithm selection
    # remains deterministic without this optional debug fill.
    torch.utils.deterministic.fill_uninitialized_memory = False


def official_core_hashes():
    from archlab.artifacts import sha256_file

    root = Path(__file__).parents[1]
    names = [
        "architectures/deepseek_v41_normal_adapter.py",
        "architectures/local_attention.py",
        "architectures/deepseek_v41_adapter.py",
        "architectures/deepseek_v41_math.py",
        "architectures/deepseek_v41_torch.py",
        "architectures/simplicial_attention.py",
        "architectures/simplicial_kernels.py",
        "optimizers/headwise_muon.py",
        "architectures/simplicial_deterministic.py",
        "architectures/ordered_reduction.py",
        *[
            f"automodel/deepseek_v41_{name}.py"
            for name in (
                "data",
                "loss",
                "training",
                "execution",
                "loading",
                "parallel",
                "runtime",
                "autograd",
                "native_quantization",
                "pytorch",
                "official_execution",
                "official_adapter",
                "official_reference",
                "official_training",
                "official_moe",
                "official_hc",
                "official_sparse",
            )
        ],
    ]
    return {name: sha256_file(root / name) for name in names}


def runtime_identity():
    import nemo_automodel
    import torch

    root = Path(nemo_automodel.__file__).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"official AutoModel source must be pinned to {UPSTREAM_COMMIT}: {commit}")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("the pinned official AutoModel checkout must remain clean")
    return {
        "automodel_root": str(root),
        "automodel_commit": commit,
        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "transformer-engine",
                "megatron-core",
                "triton",
                "flash-attn",
            )
        },
        "cuda": torch.version.cuda,
        "nccl": list(torch.cuda.nccl.version()),
        "reproducibility": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def tiny_official_config(assets: Path, *, experts=16):
    from nemo_automodel.components.models.deepseek_v41.config import (
        DeepseekV41Config,
        DeepseekV41TextConfig,
        DeepseekV41VisionConfig,
    )

    text = DeepseekV41TextConfig(
        vocab_size=129280,
        hidden_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=64,
        o_lora_rank=64,
        o_groups=2,
        n_routed_experts=experts,
        num_experts_per_tok=2,
        compress_ratios=[0, 0, 2, 2, 1, 1],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4],
        index_n_heads=4,
        index_head_dim=64,
        index_topk=16,
        candidate_source_layer_id=4,
        candidate_topk_blocks=4,
        candidate_block_size=4,
        engram_layer_ids=[1],
        engram_num_embeddings=[8192],
        engram_head_dim=32,
        engram_n_heads=2,
        engram_vocab_size=100,
        engram_compressed_vocab_size=99092,
        max_position_embeddings=16384,
    )
    config = DeepseekV41Config(
        text_config=text, vision_config=DeepseekV41VisionConfig(num_hidden_layers=0)
    )
    config.name_or_path = str(assets)
    return config


def build_official_base(
    *, weights: Path, assets: Path, ep_size=8, activation_checkpointing=True, tiny=False
):
    import torch
    import torch.distributed as dist
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.distributed.config import (
        DistributedSetup,
        FSDP2Config,
        MoEParallelizerConfig,
    )
    from nemo_automodel.components.distributed.mesh import ParallelismSizes
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor

    from archlab.automodel.deepseek_v41_official_hc import install_official_native_hc
    from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe
    from archlab.automodel.deepseek_v41_official_sparse import install_official_deterministic_sparse
    from archlab.automodel.deepseek_v41_training import emit

    configure_official_reproducibility()
    identity = runtime_identity()
    precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=precision, reshard_after_forward=True, sequence_parallel=False
        ),
        parallelism_sizes=ParallelismSizes(tp_size=1, pp_size=1, cp_size=1, ep_size=ep_size),
        moe_parallel_config=MoEParallelizerConfig(
            mp_policy=precision,
            lm_head_precision=torch.float32,
            reshard_after_forward=True,
            wrap_outer_model=True,
        ),
        activation_checkpointing=activation_checkpointing,
        world_size=dist.get_world_size(),
    )
    mesh = setup.mesh_context
    group = mesh.moe_mesh["ep"].get_group()
    hosts = [None] * ep_size
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        raise ValueError("the experiment requires node-local expert groups")
    emit(
        "official_mesh",
        world_ranks=mesh.device_mesh.mesh.tolist(),
        world_axes=list(mesh.device_mesh.mesh_dim_names),
        moe_ranks=mesh.moe_mesh.mesh.tolist(),
        moe_axes=list(mesh.moe_mesh.mesh_dim_names),
    )
    if tiny:
        config = tiny_official_config(assets, experts=max(16, ep_size * 2))
    else:
        config = DeepseekV41Config.from_pretrained(weights, local_files_only=True)
        config.name_or_path = str(weights)
        config.vision_config.num_hidden_layers = 0
        text = config.text_config
        if (
            text.hidden_size,
            text.num_hidden_layers,
            text.n_routed_experts,
            text.num_experts_per_tok,
            text.hc_mult,
        ) != (5120, 40, 384, 6, 4):
            raise ValueError("checkpoint is not the selected DeepSeek V4.1 Flash backbone")
    backend = BackendConfig(
        attn="tilelang",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        gate_precision="float32",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )
    emit("official_construct_load_start", checkpoint=str(weights), tiny=tiny)
    model = NeMoAutoModelForCausalLM.from_config(
        config,
        load_base_model=not tiny,
        backend=backend,
        distributed_setup=setup,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        force_hf=False,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        freeze_config={"freeze_modules": [{"glob": "*"}]},
    )
    model.requires_grad_(False)
    moe_precision = install_official_fp32_moe(model)
    hc_precision = install_official_native_hc(model, assets)
    sparse_precision = install_official_deterministic_sparse(model)
    local_bytes = 0
    nonfinite = []
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        local_bytes += local.numel() * local.element_size()
        # Bounded scans avoid a whole expert group's temporary boolean mask.
        for chunk in local.detach().reshape(-1).split(4 * 1024 * 1024):
            if not bool(chunk.isfinite().all()):
                nonfinite.append(name)
                break
    if nonfinite:
        raise ValueError(f"nonfinite official base parameters after loading: {nonfinite[:8]}")
    report = {
        **identity,
        "backend": "official-automodel-deepseek-v41",
        "tiny": tiny,
        "world_size": dist.get_world_size(),
        "ep_size": ep_size,
        "expert_fsdp_size": dist.get_world_size() // ep_size,
        "engram_owners": dist.get_world_size(),
        "base_fsdp": True,
        "local_parameter_gib": local_bytes / 2**30,
        "attention": "tilelang",
        "sinkhorn": "native-tilelang",
        "hc_precision": hc_precision,
        "sparse_backward": sparse_precision,
        "experts": "torch_mm",
        "dispatcher": "torch",
        "linear_compute": "bfloat16",
        "moe_precision": moe_precision,
        "kv_index_quantization_retained": True,
        "lm_head_precision": "float32",
    }
    emit("official_construct_load_complete", **report)
    return model, setup, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(args.container_kernel_packages)
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=120),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    try:
        model, setup, report = build_official_base(
            weights=args.weights, assets=args.assets, ep_size=args.ep_size, tiny=args.tiny
        )
        tokens = torch.randint(0, 1000, (1, 128), device="cuda")
        with torch.no_grad():
            output = model(tokens, return_hidden_states=True)
        hidden = output.hidden_states
        report["hidden_shape"] = list(hidden.shape)
        report["forward_finite"] = bool(hidden.isfinite().all())
        if not report["forward_finite"]:
            raise FloatingPointError("official forward produced nonfinite hidden states")
        atomic_write_json(args.output / f"rank{dist.get_rank()}.json", report)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
