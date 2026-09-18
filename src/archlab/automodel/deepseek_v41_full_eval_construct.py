"""Construct the trained16-rank model layout before restoring every full weight.

Uses the same public AutoModel/FSDP construction contract as the qualified
trainer. Loading a second copy of the released quantized base would create a
redundant complete dequantized state dictionary. The full-checkpoint evaluator
instead initializes the sharded layout and then verifies/restores every tensor.
No inference is allowed before that complete restore and forward qualification.
"""

import socket
from pathlib import Path


def build_eval_shell(*, weights: Path, assets: Path, tiny: bool):
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

    from archlab.automodel.deepseek_v41_official_execution import (
        configure_official_reproducibility,
        runtime_identity,
        tiny_official_config,
    )
    from archlab.automodel.deepseek_v41_official_hc import install_official_native_hc
    from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe
    from archlab.automodel.deepseek_v41_official_sparse import install_official_deterministic_sparse
    from archlab.automodel.deepseek_v41_training import emit

    configure_official_reproducibility()
    identity = runtime_identity()
    if dist.get_world_size() != 16:
        raise ValueError("full evaluation uses the trained16-rank layout")
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
        parallelism_sizes=ParallelismSizes(tp_size=1, pp_size=1, cp_size=1, ep_size=8),
        moe_parallel_config=MoEParallelizerConfig(
            mp_policy=precision,
            lm_head_precision=torch.float32,
            reshard_after_forward=True,
            wrap_outer_model=True,
        ),
        activation_checkpointing=True,
        world_size=16,
    )
    group = setup.mesh_context.moe_mesh["ep"].get_group()
    hosts = [None] * 8
    dist.all_gather_object(hosts, socket.gethostname(), group=group)
    if len(set(hosts)) != 1:
        raise ValueError("expert groups must be node local")
    if tiny:
        config = tiny_official_config(assets, experts=16)
    else:
        config = DeepseekV41Config.from_pretrained(weights, local_files_only=True)
        config.name_or_path = str(weights)
        config.vision_config.num_hidden_layers = 0
        t = config.text_config
        if (
            t.hidden_size,
            t.num_hidden_layers,
            t.n_routed_experts,
            t.num_experts_per_tok,
            t.hc_mult,
        ) != (5120, 40, 384, 6, 4):
            raise ValueError("wrong full V4.1 backbone geometry")
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
    emit("eval_model_layout_start", tiny=tiny)
    model = NeMoAutoModelForCausalLM.from_config(
        config,
        load_base_model=False,
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
    moe = install_official_fp32_moe(model)
    hc = install_official_native_hc(model, assets)
    sparse = install_official_deterministic_sparse(model)
    local_bytes = sum(
        (p.to_local() if isinstance(p, DTensor) else p).numel() * p.element_size()
        for p in model.parameters()
    )
    report = {
        **identity,
        "world_size": 16,
        "ep_size": 8,
        "expert_fsdp_size": 2,
        "engram_owners": 16,
        "construction": "from_config, initialize after sharding, full-checkpoint restore required",
        "released_base_weights_loaded": False,
        "local_parameter_gib": local_bytes / 2**30,
        "tiny": tiny,
        "moe_precision": moe,
        "hc_precision": hc,
        "sparse_precision": sparse,
    }
    emit("eval_model_layout_complete", local_parameter_gib=local_bytes / 2**30, tiny=tiny)
    return model, setup, report
