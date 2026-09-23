"""One-GPU BF16 inference through the pinned released V4.1 structure.

CPU-mapped frozen experts/Engram avoid changing quantization to fit memory.
Only transport, decoded-weight caching, and the reviewed additive branch are
owned here. The released model still owns attention, routing, mHC and caches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight
from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
from archlab.automodel.deepseek_v41_runtime import load_native_reference


def _replace_parameter(model, name, value):
    owner, leaf = name.rsplit('.', 1) if '.' in name else ('', name)
    model.get_submodule(owner)._parameters[leaf] = nn.Parameter(value, requires_grad=False)


def cpu_engram_lookup(module, indices):
    """Gather exact encoded rows on CPU; dequantize only those rows on GPU."""
    shape = indices.shape
    ids = indices.reshape(-1).cpu()
    if bool(((ids < 0) | (ids >= module.num_embeddings)).any()):
        raise ValueError('Engram hash outside the original table')
    # CPU gather kernels need integer storage views for these float8 formats.
    values = module.weight.view(torch.uint8).index_select(0, ids).to(indices.device).view(module.weight.dtype)
    scales = module.scale.view(torch.uint8).index_select(0, ids).to(indices.device).view(module.scale.dtype)
    decoded = values.float().unflatten(-1, (-1, module.block_size)) * scales.float().unsqueeze(-1)
    return decoded.flatten(-2).to(torch.bfloat16).reshape(*shape, module.dim)


class DecodedExpertCache:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key, expert):
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        required = sum(expert._cpu_weights[name][0].numel() * 4 for name in ('w1', 'w2', 'w3'))
        while self.entries and self.bytes + required > self.max_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= sum(x.numel() * x.element_size() for x in removed.values())
        decoded = {}
        for name in ('w1', 'w2', 'w3'):
            weight, scale = expert._cpu_weights[name]
            decoded[name] = dequantize_frozen_weight(weight.cuda(non_blocking=True), scale.cuda(non_blocking=True))
        size = sum(x.numel() * x.element_size() for x in decoded.values())
        if size <= self.max_bytes:
            self.entries[key] = decoded
            self.bytes += size
        return decoded


def _offloaded_expert(self, x, weights=None):
    decoded = self._weight_cache.get(self._cache_key, self)
    original = {name: getattr(self, name).weight for name in decoded}
    try:
        for name, value in decoded.items():
            getattr(self, name)._parameters['weight'] = nn.Parameter(value, requires_grad=False)
        return self._original_forward(x, weights)
    finally:
        for name, value in original.items():
            getattr(self, name)._parameters['weight'] = value


def _adapter_block(self, x, start_pos, pre_mix, image_mask, *attn_args):
    residual = x
    attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    attended = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), start_pos, *attn_args)
    x = self.hc_post(attended, residual, attn_post, attn_comb)
    if self._adapter_mask is not None and bool(self._adapter_mask.any()):
        # The branch has no positional transform; a rolling 512-position raw
        # stream is exactly sufficient for its 32x512 causal pair window.
        history = x if start_pos == 0 else torch.cat((self._adapter_history, x), dim=1)
        adapted = self.simplicial_adapter(history)[:, -x.shape[1]:]
        self._adapter_history = history[:, -(self.simplicial_adapter.config.long_window - 1):].clone()
        x = torch.where(self._adapter_mask[:, None, None, None], adapted, x)
    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    ffn = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)), image_mask)
    return self.hc_post(ffn, residual, ffn_post, ffn_comb), ffn_pre


class LocalV41Inference:
    def __init__(self, *, assets: Path, weights: Path, adapter_checkpoint: Path,
                 context=2048, max_batch_size=8, expert_cache_gib=96, resident_cpu_experts=False):
        started = time.monotonic()
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
            raise ValueError("disable the local forced-TF32 override before starting evaluation")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(False)
        torch.utils.deterministic.fill_uninitialized_memory = False
        self.cpu_experts = None
        self.assets, self.weights = Path(assets), Path(weights)
        self.readers = ExitStack()
        self.files = {}
        self.mapping = json.loads((self.weights / 'model.safetensors.index.json').read_text())['weight_map']
        self.loaded = set()
        if resident_cpu_experts:
            from archlab.automodel.deepseek_v41_cpu_store import ResidentExpertStore
            self.cpu_experts = ResidentExpertStore(self.weights)
        self.reference = load_native_reference(self.assets, module_name='_archlab_local_v41_reference')
        native = self.reference
        self.quantization = install_native_row_padding(native)
        torch.set_grad_enabled(False)
        torch.set_num_threads(8)
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.utils.deterministic.fill_uninitialized_memory = False
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.assets, local_files_only=True)
        args = native.ModelArgs(**json.loads((self.assets / 'inference/config.json').read_text()))
        args.max_batch_size, args.max_seq_len = max_batch_size, context
        args.vision_n_layers = 0
        args.n_mtp_layers, args.dspark_block_size, args.dspark_target_layer_ids = 0, 0, ()
        # Empty frozen CPU tensors reserve address space only, then are replaced
        # by read-only safetensors mappings. Constant/cache buffers retain their
        # native initialization and are moved to the GPU below.
        with native.set_dtype(torch.bfloat16), torch.device('cpu'):
            self.model = native.Transformer(args, tokenizer=self.tokenizer).requires_grad_(False).eval()
        expected = dict(self.model.named_parameters())
        missing = expected.keys() - self.mapping.keys()
        if missing:
            raise ValueError(f'missing original checkpoint keys: {sorted(missing)[:10]}')
        print(json.dumps({'event': 'local_model_constructed', 'parameter_tensors': len(expected)}), flush=True)
        for number, (name, target) in enumerate(expected.items()):
            value = self.tensor(name)
            if name.endswith('.attn.wo_a.weight'):
                value = dequantize_frozen_weight(value.cuda(), self.tensor(name.removesuffix('weight') + 'scale').cuda())
            elif target.dtype == torch.float4_e2m1fn_x2:
                if value.dtype != torch.int8:
                    raise ValueError(f'wrong packed expert encoding: {name}')
                value = value.view(torch.float4_e2m1fn_x2)
            elif value.dtype != target.dtype:
                if value.dtype != torch.bfloat16 or target.dtype != torch.float32:
                    raise ValueError(f'unapproved frozen dtype conversion: {name}')
                value = value.float()
            if value.shape != target.shape:
                raise ValueError(f'wrong frozen shape: {name}')
            _replace_parameter(self.model, name, value)
            if number and number % 10000 == 0:
                print(json.dumps({'event': 'mapped_parameters', 'count': number}), flush=True)
        del expected
        allowed = re.compile(r'^(vision\.|aligner\.|mtp\.|image_)|\.ffn\.gate\.bias_vl$')
        unused = [name for name in self.mapping if name not in self.loaded and not allowed.search(name)]
        if unused:
            raise ValueError(f'unaccounted frozen checkpoint tensors: {unused[:10]}')
        self.cache = DecodedExpertCache(int(expert_cache_gib * 2**30))
        for name, module in list(self.model.named_modules()):
            if re.fullmatch(r'layers\.\d+\.ffn\.experts\.\d+', name):
                module._cpu_weights = {p: (getattr(module, p).weight, getattr(module, p).scale) for p in ('w1', 'w2', 'w3')}
                module._weight_cache, module._cache_key = self.cache, name
                module._original_forward = module.forward
                module.forward = MethodType(_offloaded_expert, module)
            elif name.endswith('.engram.embed'):
                module.forward = MethodType(cpu_engram_lookup, module)
            elif isinstance(module, native.Linear) and not re.search(r'\.ffn\.experts\.\d+\.', name):
                if module.weight.dtype == torch.float8_e4m3fn:
                    target = torch.empty(module.weight.shape, dtype=torch.bfloat16, device='cuda')
                    for first in range(0, target.shape[0], 1024):
                        last = min(first + 1024, target.shape[0])
                        target[first:last].copy_(dequantize_frozen_weight(module.weight[first:last].cuda(), module.scale[first//32:(last+31)//32].cuda()))
                    module.weight = nn.Parameter(target, requires_grad=False)
                    module.register_parameter('scale', None)
        for name, value in list(self.model.named_parameters()):
            if '.ffn.experts.' not in name and '.engram.embed.' not in name and value.device.type != 'cuda':
                _replace_parameter(self.model, name, value.cuda())
        for module in self.model.modules():
            for name, value in list(module.named_buffers(recurse=False)):
                module._buffers[name] = value.cuda()
        # Exactly the BF16 dense/expert compute used by the qualified reference;
        # direct KV/index quantization calls remain native and unchanged.
        native.linear = F.linear
        marker = json.loads((adapter_checkpoint / 'COMPLETE.json').read_text())
        checkpoint_file = adapter_checkpoint / 'adapter-state.pt'
        if checkpoint_file.stat().st_size != marker['state_bytes']:
            raise ValueError('incomplete adapter checkpoint')
        with checkpoint_file.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != marker['state_sha256']:
                raise ValueError('adapter checksum mismatch')
        payload = torch.load(checkpoint_file, map_location='cpu', weights_only=True, mmap=True)
        if payload['contract'] != marker['contract'] or payload['cursor'] != marker['cursor']:
            raise ValueError('adapter manifest/payload mismatch')
        self.adapters = {}
        for key, state in payload['adapters'].items():
            index = int(key)
            adapter = V41SimplicialAdapter(V41AdapterConfig(), seed=42+index, backend='deterministic')
            adapter.load_state_dict(state, strict=True)
            adapter = adapter.cuda().requires_grad_(False).eval()
            layer = self.model.layers[index]
            layer.simplicial_adapter = adapter
            layer._adapter_mask, layer._adapter_history = None, None
            layer.forward = MethodType(_adapter_block, layer)
            self.adapters[index] = adapter
        if tuple(sorted(self.adapters)) != (4, 9, 14, 19, 24, 29, 34, 39):
            raise ValueError('incorrect adapted layers')
        self.context, self.max_batch_size = context, max_batch_size
        self.report = {'hardware': 'NVIDIA B300 (NVML L20D mislabel)', 'python': os.sys.executable,
                       'packages': {p: importlib.metadata.version(p) for p in ('torch','triton','tilelang','transformers','safetensors')},
                       'container': {k: os.environ.get(k) for k in ('NVIDIA_PRODUCT_NAME','NVIDIA_BUILD_ID','NVIDIA_PYTORCH_VERSION','CUDA_VERSION')},
                       'cuda': torch.version.cuda, 'allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'float32_matmul_precision': torch.get_float32_matmul_precision(), 'checkpoint': str(adapter_checkpoint), 'cursor': marker['cursor'],
                       'adapter_sha256': marker['state_sha256'], 'trainable_checkpoint_parameters': marker['contract']['trainable_parameters'],
                       'frozen_tensors_loaded': len(self.loaded), 'inactive_tensors': len(self.mapping)-len(self.loaded),
                       'expert_cache_gib': expert_cache_gib, 'cpu_expert_store': None if self.cpu_experts is None else self.cpu_experts.report, 'context': context, 'max_batch_size': max_batch_size,
                       'construction_seconds': time.monotonic()-started, 'resident_gpu_gib': torch.cuda.memory_allocated()/2**30}
        print(json.dumps({'event':'local_inference_ready',**self.report}), flush=True)

    def tensor(self, name):
        if self.cpu_experts is not None and name in self.cpu_experts:
            self.loaded.add(name)
            return self.cpu_experts.get(name)
        filename = self.mapping[name]
        if Path(filename).name != filename:
            raise ValueError('weight shard path escapes the verified model')
        if filename not in self.files:
            self.files[filename] = self.readers.enter_context(safe_open(self.weights/filename,framework='pt',device='cpu'))
        self.loaded.add(name)
        with torch.device('cpu'):
            return self.files[filename].get_tensor(name)

    @torch.inference_mode()
    def forward(self, ids, *, adapted, start_pos=0, return_hidden=False):
        if ids.ndim != 2 or ids.shape[0] > self.max_batch_size or start_pos + ids.shape[1] > self.context:
            raise ValueError('evaluation input outside the qualified local geometry')
        if ids.device.type != 'cuda':
            ids = ids.cuda()
        mask = torch.as_tensor(adapted, device='cuda', dtype=torch.bool).expand(ids.shape[0])
        if start_pos == 0:
            self.reference.shared_attn = self.reference.SharedAttentionRuntime()
            for module in self.model.modules():
                for name, buffer in module.named_buffers(recurse=False):
                    if 'cache' in name or name in ('kv_state','score_state'):
                        buffer.zero_()
            for layer in self.model.layers:
                if hasattr(layer, '_adapter_history'):
                    layer._adapter_history = None
        for layer in self.model.layers:
            if hasattr(layer, '_adapter_mask'):
                layer._adapter_mask = mask
        captures = []
        handle = self.model.norm.register_forward_hook(lambda m,a,o: captures.append(o)) if return_hidden else None
        try:
            with self.reference.set_dtype(torch.bfloat16), torch.device('cuda'):
                _, logits, _ = self.model(ids, start_pos=start_pos)
        finally:
            if handle is not None:handle.remove()
        if not bool(logits.isfinite().all()):
            raise FloatingPointError('nonfinite local evaluation logits')
        return (logits, captures[0]) if return_hidden else logits
