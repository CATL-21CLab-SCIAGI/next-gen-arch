"""Compare local native arithmetic with archived real layer-zero tensors."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','weights','boundaries','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages,load_native_reference
    select_container_kernel_packages(Path('/usr/local/lib/python3.12/dist-packages'))
    import torch
    from torch.nn import functional as F
    from safetensors import safe_open
    from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight
    from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
    from archlab.artifacts import atomic_write_json
    torch.set_grad_enabled(False);torch.set_num_threads(8);torch.utils.deterministic.fill_uninitialized_memory=False
    native=load_native_reference(a.assets,module_name='_archlab_local_leaf_precision')
    install_native_row_padding(native);native.linear=F.linear;native.default_dtype=torch.bfloat16
    args=native.ModelArgs(**json.loads((a.assets/'inference/config.json').read_text()));args.max_batch_size=8;args.max_seq_len=128;args.vision_n_layers=0
    saved=[torch.load(a.boundaries/f'rank{i}-boundaries.pt',map_location='cpu',mmap=True,weights_only=True)['native'] for i in range(8)]
    mapping=json.loads((a.weights/'model.safetensors.index.json').read_text())['weight_map'];files={}
    def tensor(name):
        filename=mapping[name]
        if filename not in files:files[filename]=safe_open(a.weights/filename,framework='pt',device='cpu')
        value=files[filename].get_tensor(name)
        if value.dtype==torch.float8_e4m3fn:
            scale=tensor(name.removesuffix('weight')+'scale')
            return dequantize_frozen_weight(value.cuda(),scale)
        return value.cuda()
    def measure(x,y):
        delta=x.float()-y.float()
        return {'equal':torch.equal(x,y),'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/y.float().norm()),'dtype':str(x.dtype)}
    dummy=SimpleNamespace(norm_eps=args.norm_eps,hc_mult=args.hc_mult,hc_sinkhorn_iters=args.hc_sinkhorn_iters,hc_eps=args.hc_eps)
    fn,scale,base=(tensor('layers.0.hc_attn_'+name) for name in ('fn','scale','base'))
    x=torch.cat([r['layers.0.attn_mix.input'] for r in saved]).cuda()
    expected_mix=[torch.cat([r['layers.0.attn_mix.'+name] for r in saved]).cuda() for name in ('pre','post','comb')]
    gate=native.Gate(0,args).cuda()
    for name,_ in gate.named_parameters():gate._parameters[name]=torch.nn.Parameter(tensor('layers.0.ffn.gate.'+name),requires_grad=False)
    gx=torch.cat([r['layers.0.ffn.gate.input'] for r in saved]).cuda()
    gw=torch.cat([r['layers.0.ffn.gate.weights'] for r in saved]).cuda();gi=torch.cat([r['layers.0.ffn.gate.indices'] for r in saved]).cuda()
    results={}
    for tf32 in (False,True):
        torch.backends.cuda.matmul.allow_tf32=tf32
        for split in (False,True):
            label=f'tf32={tf32},split={split}'
            if split:
                pieces=[native.Block.hc_mixes(dummy,row,fn,scale,base) for row in x.split(1)]
                mixes=[torch.cat([piece[i] for piece in pieces]) for i in range(3)]
                gp=[gate(part) for part in gx.split(128)];weights=torch.cat([v[0] for v in gp]);indices=torch.cat([v[1] for v in gp])
            else:mixes=native.Block.hc_mixes(dummy,x,fn,scale,base);weights,indices=gate(gx)
            results[label]={'mhc':[measure(v,e) for v,e in zip(mixes,expected_mix,strict=True)],'gate_weights':measure(weights,gw),'gate_indices_equal':torch.equal(indices,gi)}
    with native.set_dtype(torch.bfloat16),torch.device('cpu'):attn=native.Attention(0,args).requires_grad_(False)
    for name,parameter in list(attn.named_parameters()):
        owner,key=name.rsplit('.',1) if '.' in name else ('',name)
        value=tensor('layers.0.attn.'+name).to(parameter.dtype)
        attn.get_submodule(owner)._parameters[key]=torch.nn.Parameter(value,requires_grad=False)
    for module in attn.modules():
        for name,buffer in list(module.named_buffers(recurse=False)):module._buffers[name]=buffer.cuda()
    ax=torch.cat([r['layers.0.attn.input'] for r in saved]).cuda();expected=torch.cat([r['layers.0.attn.output'] for r in saved]).cuda()
    cpu_freq=attn.freqs_cis.clone()
    atomic_write_json(a.output,results,allow_nan=False)
    native.precompute_freqs_cis.cache_clear()
    with torch.device('cuda'):
        gpu_freq=native.precompute_freqs_cis(args.rope_head_dim,128,0,args.rope_theta,args.rope_factor,args.beta_fast,args.beta_slow)
    results['rope_cpu_vs_gpu']=measure(cpu_freq.view(torch.float32),gpu_freq.view(torch.float32))
    for tf32 in (False,True):
        torch.backends.cuda.matmul.allow_tf32=tf32
        for split in (False,True):
            for device,freq in (('cpu',cpu_freq),('gpu',gpu_freq)):
                attn.freqs_cis=freq
                with native.set_dtype(torch.bfloat16),torch.device('cuda'),torch.inference_mode():
                    value=torch.cat([attn(part,0) for part in ax.split(1)]) if split else attn(ax,0)
                results[f'attn tf32={tf32},split={split},rope={device}']=measure(value,expected)
    atomic_write_json(a.output,results,allow_nan=False)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
