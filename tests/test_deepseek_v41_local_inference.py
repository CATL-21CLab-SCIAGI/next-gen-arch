"""Transport and cached-adapter oracles for one-B300 evaluation."""

from types import SimpleNamespace
import copy

import unittest
import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig,V41SimplicialAdapter
from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight
from archlab.automodel.deepseek_v41_local_inference import (DecodedExpertCache,_offloaded_expert,cpu_engram_lookup,_adapter_block)

requires_gpu=unittest.skipUnless(torch.cuda.is_available(),'requires local B300')


@requires_gpu
def test_cpu_engram_rows_match_gpu_lookup_byte_exact():
    values=torch.linspace(-4,4,9*64).reshape(9,64).to(torch.float8_e4m3fn)
    scales=torch.tensor([[.5,2.]]*9).to(torch.float8_e8m0fnu)
    table=SimpleNamespace(weight=values,scale=scales,num_embeddings=9,dim=64,block_size=32)
    ids=torch.tensor([[[8,0,8],[2,4,1]]],device='cuda')
    actual=cpu_engram_lookup(table,ids)
    expected=(F.embedding(ids,values.cuda()).float().unflatten(-1,(-1,32))*F.embedding(ids,scales.cuda()).float().unsqueeze(-1)).flatten(-2).bfloat16()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    with unittest.TestCase().assertRaises(ValueError):cpu_engram_lookup(table,torch.tensor([9],device='cuda'))


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('w1','w2','w3'):
            module=nn.Module();module.weight=nn.Parameter(torch.randint(-128,128,(32,16),dtype=torch.int8),requires_grad=False)
            module.scale=nn.Parameter(torch.full((32,1),.03125).to(torch.float8_e8m0fnu),requires_grad=False)
            setattr(self,name,module)
        self._cpu_weights={n:(getattr(self,n).weight,getattr(self,n).scale) for n in ('w1','w2','w3')}
        self._original_forward=self.forward
    def forward(self,x,weights=None):
        h=F.silu(F.linear(x,self.w1.weight).float())*F.linear(x,self.w3.weight).float()
        if weights is not None:h=h*weights
        return F.linear(h.to(x.dtype),self.w2.weight)


@requires_gpu
@torch.no_grad()
def test_expert_offload_matches_decoding_and_restores_cpu_storage():
    torch.manual_seed(83)
    expert=_Expert();expert._weight_cache=DecodedExpertCache(6144);expert._cache_key='first'
    original={n:getattr(expert,n).weight for n in ('w1','w2','w3')}
    x=torch.randn(7,32,device='cuda',dtype=torch.bfloat16);weights=torch.rand(7,1,device='cuda')
    decoded={n:dequantize_frozen_weight(w.cuda(),s.cuda()) for n,(w,s) in expert._cpu_weights.items()}
    h=F.silu(F.linear(x,decoded['w1']).float())*F.linear(x,decoded['w3']).float()*weights
    expected=F.linear(h.bfloat16(),decoded['w2'])
    for _ in range(2):torch.testing.assert_close(_offloaded_expert(expert,x,weights),expected,rtol=0,atol=0)
    assert expert._weight_cache.misses==1 and expert._weight_cache.hits==1
    assert all(getattr(expert,n).weight is p and p.device.type=='cpu' for n,p in original.items())
    expert._cache_key='second';_offloaded_expert(expert,x,weights)
    assert list(expert._weight_cache.entries)==['second']
    assert expert._weight_cache.bytes==6144


class _Block:
    def __init__(self,adapter):
        self.simplicial_adapter=adapter;self._adapter_mask=torch.tensor([False,True],device='cuda');self._adapter_history=None
        self.hc_attn_fn=self.hc_attn_scale=self.hc_attn_base=None
        self.hc_ffn_fn=self.hc_ffn_scale=self.hc_ffn_base=None
        self.attn_norm=self.ffn_norm=lambda x:x
        self.attn=lambda x,*args:x*.2
        self.ffn=lambda x,*args:x*.3
    def hc_mixes(self,*args):return None,None,None
    def hc_pre(self,x,mix):return x.mean(-2)
    def hc_post(self,x,residual,*args):return residual+x.unsqueeze(-2)


@requires_gpu
@torch.no_grad()
def test_cached_adapter_decode_matches_full_prefill_and_preserves_base_rows():
    torch.manual_seed(22)
    config=V41AdapterConfig(width=16,query_heads=2,kv_heads=1,head_dim=16,short_window=2,long_window=4)
    a=V41SimplicialAdapter(config,backend='reference').cuda()
    a.output.weight.normal_(std=.08)
    full,decoded=_Block(a),_Block(copy.deepcopy(a))
    x=torch.randn(2,13,4,16,device='cuda')
    expected=_adapter_block(full,x,0,None,None)[0]
    chunks=[_adapter_block(decoded,x[:,:6],0,None,None)[0]]
    for index in range(6,13):chunks.append(_adapter_block(decoded,x[:,index:index+1],index,None,None)[0])
    torch.testing.assert_close(torch.cat(chunks,1),expected,rtol=2e-6,atol=2e-6)
    base=_Block(a);base._adapter_mask.zero_()
    original=_adapter_block(base,x,0,None,None)[0]
    torch.testing.assert_close(expected[0],original[0],rtol=0,atol=0)
    assert not torch.equal(expected[1],original[1])


if __name__ == "__main__":
    suite=unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in list(globals().items()) if name.startswith("test_"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
