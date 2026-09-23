"""Eight-GPU random initialization using the pinned official V4.1 implementation."""
from pathlib import Path
import json
import socket


def construct_scratch(*,base_config,assets,variant,tiny=False):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.distributed.config import DistributedSetup,FSDP2Config,MoEParallelizerConfig
    from nemo_automodel.components.distributed.mesh import ParallelismSizes
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
    from nemo_automodel.components.moe.layers import Gate
    from archlab.architectures.deepseek_v41_scratch import scaled_scratch_config,adapter_layers
    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_official_execution import configure_official_reproducibility,runtime_identity,tiny_official_config
    from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe
    from archlab.automodel.deepseek_v41_official_hc import install_official_native_hc
    from archlab.automodel.deepseek_v41_official_sparse import install_official_deterministic_sparse
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_full_boundaries import install_full_training_boundaries
    from archlab.automodel.deepseek_v41_full_indexer import install_trainable_indexers
    from archlab.automodel.deepseek_v41_training import emit
    from archlab.optimizers.sharded_adafactor import local_tensor
    if dist.get_world_size()!=8:raise ValueError('scratch contract requires one8-GPU node per variant')
    hosts=[None]*8;dist.all_gather_object(hosts,socket.gethostname())
    if len(set(hosts))!=1:raise ValueError('each variant must occupy exactly one node')
    configure_official_reproducibility();identity=runtime_identity()
    precision=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,output_dtype=None,cast_forward_inputs=False)
    setup=DistributedSetup.build(strategy=FSDP2Config(mp_policy=precision,reshard_after_forward=True,sequence_parallel=False),parallelism_sizes=ParallelismSizes(tp_size=1,pp_size=1,cp_size=1,ep_size=8),moe_parallel_config=MoEParallelizerConfig(mp_policy=precision,lm_head_precision=torch.float32,reshard_after_forward=True,wrap_outer_model=True),activation_checkpointing=True,world_size=8)
    config=tiny_official_config(Path(assets),experts=16) if tiny else DeepseekV41Config(**scaled_scratch_config(json.loads(Path(base_config).read_text())))
    config.name_or_path=str(assets)
    backend=BackendConfig(attn='tilelang',linear='torch',rms_norm='torch_fp32',experts='torch_mm',dispatcher='torch',gate_precision='float32',rope_fusion=False,fake_balanced_gate=False,enable_hf_state_dict_adapter=True)
    emit('scratch_construct_start',variant=variant,tiny=tiny)
    model=NeMoAutoModelForCausalLM.from_config(config,load_base_model=False,backend=backend,distributed_setup=setup,torch_dtype=torch.bfloat16,trust_remote_code=False,force_hf=False,use_liger_kernel=False,use_sdpa_patching=False,freeze_config={'freeze_modules':[{'glob':'*'}]})
    moe=install_official_fp32_moe(model);hc=install_official_native_hc(model,Path(assets));sparse=install_official_deterministic_sparse(model)
    if variant in ('simplicial',):
        adapter_backend='deterministic'
    elif variant in ('normal',):
        adapter_backend='flash-attn-deterministic'
    else:
        adapter_backend='reference'
    adapters=install_official_adapters(model,V41AdapterConfig(width=256,head_dim=16) if tiny else V41AdapterConfig(width=640,head_dim=16),layer_indices=(1,3,5) if tiny else adapter_layers(),device='cuda',variant=variant,backend=adapter_backend,allow_right_padding=True)
    boundaries=install_full_training_boundaries(model);indexers=install_trainable_indexers(model)
    gates=[m for m in model.modules() if isinstance(m,Gate)]
    for gate in gates:gate.bias_update_factor=.01;gate.aux_loss_coeff=.01;gate._track_load_balance=True
    logical=sum(local_tensor(p).numel()/(8 if not hasattr(p,'placements') else 1) for p in model.parameters());value=torch.tensor(logical,device='cuda',dtype=torch.float64);dist.all_reduce(value)
    report={**identity,'geometry':config.to_dict(),'variant':variant,'router_auxiliary_loss_coefficient':.01,'right_padding_masked':True,'random_initialization':True,'pretrained_weights_loaded':False,'world_size':8,'ep_size':8,'expert_fsdp_size':1,'engram_owners':8,'all_parameters_unfrozen':all(p.requires_grad for p in model.parameters()),'parameters':int(value),'local_parameter_gib':sum(local_tensor(p).numel()*p.element_size() for p in model.parameters())/2**30,'moe_precision':moe,'hc_precision':hc,'sparse_precision':sparse,'boundaries':boundaries,'adapter_layers':list(adapters)}
    emit('scratch_construct_complete',variant=variant,parameters=report['parameters'],local_parameter_gib=report['local_parameter_gib'])
    return model,indexers,gates,report
