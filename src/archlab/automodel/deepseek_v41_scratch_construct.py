"""Random initialization using the pinned official V4.1 implementation."""

import json
import socket
from pathlib import Path


def construct_scratch(
    *,
    base_config,
    assets,
    variant,
    tiny=False,
    width=640,
    sparse_backend="deterministic",
    scaling_study=False,
    sweep_cell=None,
    grouped_experts=False,
    scale_engram=False,
    engram_anchor_width=None,
    retain_activations=False,
    expert_dispatcher="torch",
    simplicial_backend="deterministic",
    matched_mixer=None,
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
    from nemo_automodel.components.moe.layers import Gate
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.architectures.deepseek_v41_scratch import (
        adapter_layers,
        scaled_scratch_config,
        scratch_adapter_head_dim,
    )
    from archlab.automodel.deepseek_v41_full_boundaries import install_full_training_boundaries
    from archlab.automodel.deepseek_v41_full_indexer import install_trainable_indexers
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_official_execution import (
        configure_official_reproducibility,
        runtime_identity,
        tiny_official_config,
    )
    from archlab.automodel.deepseek_v41_official_hc import install_official_native_hc
    from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe
    from archlab.automodel.deepseek_v41_official_sparse import install_official_deterministic_sparse
    from archlab.automodel.deepseek_v41_training import emit
    from archlab.optimizers.sharded_adafactor import local_tensor

    if simplicial_backend not in ("deterministic", "triton"):
        raise ValueError("unsupported simplicial training backend")
    if expert_dispatcher not in ("torch", "deepep"):
        raise ValueError("unsupported expert dispatcher")
    if expert_dispatcher == "deepep" and not grouped_experts:
        raise ValueError("DeepEP requires the grouped scratch contract")
    if expert_dispatcher == "deepep" and width % 16:
        raise ValueError("DeepEP requires an aligned model width; transport padding is prohibited")
    depth = 20 if matched_mixer is None else matched_mixer["backbone"]["depth"]
    if matched_mixer is not None and (tiny or scaling_study or sweep_cell is not None
                                     or width != matched_mixer["backbone"]["width"]):
        raise ValueError("matched mixer requires its complete registered scratch geometry")
    if sweep_cell is not None:
        from archlab.automodel.loop_regularization import validate_cell

        validate_cell(sweep_cell)
        if width != sweep_cell["hidden_size"] or variant != "normal" or tiny or not scaling_study:
            raise ValueError(
                "the repeated-data sweep requires its registered normal-attention geometry"
            )
        depth = sweep_cell["stored_layers"]
    world = 16 if scaling_study else 8
    if dist.get_world_size() != world:
        raise ValueError(f"scratch contract requires {world} GPUs per variant")
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    if len(set(hosts)) != world // 8 or any(hosts.count(host) != 8 for host in hosts):
        raise ValueError("each scratch node must supply exactly eight GPUs")
    if len(set(hosts[:8])) != 1 or len(set(hosts[8:])) > 1:
        raise ValueError("expert parallel groups must remain within each node")
    configure_official_reproducibility()
    identity = runtime_identity()
    if matched_mixer is not None:
        from archlab.architectures.triadic_attention import official_dependency_contract

        identity["triadic_dependency"] = official_dependency_contract()
    precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=precision, reshard_after_forward=not retain_activations, sequence_parallel=False
        ),
        parallelism_sizes=ParallelismSizes(tp_size=1, pp_size=1, cp_size=1, ep_size=8),
        moe_parallel_config=MoEParallelizerConfig(
            mp_policy=precision,
            lm_head_precision=torch.float32,
            reshard_after_forward=not retain_activations,
            wrap_outer_model=True,
        ),
        activation_checkpointing=not retain_activations,
        world_size=world,
    )
    config = (
        tiny_official_config(Path(assets), experts=16)
        if tiny
        else DeepseekV41Config(
            **scaled_scratch_config(
                json.loads(Path(base_config).read_text()),
                width=width,
                scaling_study=scaling_study,
                depth=depth,
                sweep_geometry=sweep_cell is not None,
                scale_engram=scale_engram,
                engram_anchor_width=engram_anchor_width,
            )
        )
    )
    config.name_or_path = str(assets)
    backend = BackendConfig(
        attn="tilelang",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher=expert_dispatcher,
        gate_precision="float32",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )
    emit("scratch_construct_start", variant=variant, tiny=tiny)
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
    if expert_dispatcher == "torch":
        moe = install_official_fp32_moe(model)
    else:
        moe = {"transport_padding": False, "implementation": "upstream-deepep-grouped-experts", "dispatcher": "deepep",
               "experts": "torch_mm", "combination": "upstream-BF16"}
    if grouped_experts and expert_dispatcher == "torch":
        from nemo_automodel.components.moe.experts import GroupedExperts

        for module in model.modules():
            if isinstance(module, GroupedExperts):
                module._archlab_scratch_grouped_gemm = True
        moe.update(implementation="upstream-grouped-gate-up-down-fp32-combine-v1",
                   gate_up_compute="upstream-torch-grouped-mm",
                   expert_accumulation_order="upstream-fp32-scatter-add")
    hc = install_official_native_hc(model, Path(assets))
    if sparse_backend == "batched":
        from archlab.automodel.deepseek_v41_scratch_high_mfu import install_high_mfu_sparse

        sparse = install_high_mfu_sparse(model)
    elif sparse_backend == "deterministic":
        sparse = install_official_deterministic_sparse(model)
    else:
        raise ValueError(f"unsupported scratch sparse backend: {sparse_backend}")
    if variant in ("simplicial",):
        adapter_backend = simplicial_backend
    elif variant in ("normal",):
        adapter_backend = "flash-attn-deterministic"
    elif variant in ("gdn", "triadic"):
        if matched_mixer is None:
            raise ValueError("GDN/Triadic require a matched mixer experiment contract")
        adapter_backend = "official"
    else:
        adapter_backend = "reference"
    adapter_config = (V41AdapterConfig(**matched_mixer["adapter_config"])
                      if matched_mixer is not None else
                      V41AdapterConfig(width=256, head_dim=16) if tiny else V41AdapterConfig(
                          width=width, head_dim=scratch_adapter_head_dim(width, sweep_geometry=sweep_cell is not None)))
    adapters = install_official_adapters(
        model,
        adapter_config,
        layer_indices=tuple(matched_mixer["layers"]) if matched_mixer is not None else
                      (1, 3, 5) if tiny else adapter_layers(depth),
        device="cuda",
        variant=variant,
        backend=adapter_backend,
        allow_right_padding=True,
        matched_mixer=matched_mixer,
    )
    boundaries = install_full_training_boundaries(model)
    indexers = install_trainable_indexers(model)
    gates = [m for m in model.modules() if isinstance(m, Gate)]
    for gate in gates:
        gate.bias_update_factor = 0.01
        gate.aux_loss_coeff = 0.01
        gate._track_load_balance = True
    loop = None
    if sweep_cell is not None:
        from archlab.architectures.prelude_loop_coda import balanced_layout
        from archlab.automodel.deepseek_v41_loop import install_loop, set_repetitions

        loop = install_loop(model, layout=balanced_layout(depth))
        set_repetitions(model, sweep_cell["recursions"])
    logical = sum(
        local_tensor(p).numel() / (world if not hasattr(p, "placements") else 1)
        for p in model.parameters()
    )
    value = torch.tensor(logical, device="cuda", dtype=torch.float64)
    dist.all_reduce(value)
    report = {
        **identity,
        "geometry": config.to_dict(),
        "variant": variant,
        "router_auxiliary_loss_coefficient": 0.01,
        "right_padding_masked": True,
        "random_initialization": True,
        "pretrained_weights_loaded": False,
        "activation_checkpointing": not retain_activations,
        "reshard_after_forward": not retain_activations,
        "world_size": world,
        "ep_size": 8,
        "expert_fsdp_size": world // 8,
        "engram_owners": world,
        "all_parameters_unfrozen": all(p.requires_grad for p in model.parameters()),
        "parameters": int(value),
        "local_parameter_gib": sum(
            local_tensor(p).numel() * p.element_size() for p in model.parameters()
        )
        / 2**30,
        "moe_precision": moe,
        "hc_precision": hc,
        "sparse_precision": sparse,
        "boundaries": boundaries,
        "adapter_layers": list(adapters),
    }
    if loop is not None:
        report["loop"] = loop
    if matched_mixer is not None:
        report["matched_mixer"] = matched_mixer
        report["matched_parameter_contracts"] = {
            str(index): adapter.parameter_contract() for index, adapter in adapters.items()
        }
        report["matched_initialization_contracts"] = {
            str(index): adapter.initialization_contract() for index, adapter in adapters.items()
        }
        # Count the actual distributed model, excluding only vocabulary/lookup
        # tables. Engram projections and every mixer/compensation weight count.
        categories = torch.zeros(2, device="cuda", dtype=torch.float64)
        for name, parameter in model.named_parameters():
            size = local_tensor(parameter).numel() / (
                world if not hasattr(parameter, "placements") else 1
            )
            if ".simplicial_adapter." in name:
                categories[0] += size
            if not name.endswith(("embed_tokens.weight", "lm_head.weight", "engram.embed.weight")):
                categories[1] += size
        dist.all_reduce(categories)
        report["branch_parameters"] = int(categories[0])
        report["nonembedding_parameters"] = int(categories[1])
        report["nonembedding_definition"] = "exclude only token input/output vocabulary and Engram lookup tables"
    emit(
        "scratch_construct_complete",
        variant=variant,
        parameters=report["parameters"],
        local_parameter_gib=report["local_parameter_gib"],
    )
    return model, indexers, gates, report
