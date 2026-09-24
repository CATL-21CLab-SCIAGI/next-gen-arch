"""Transactional Miles refreshes for the fully trained V4.1 serving extension."""

import logging
import re

import torch


def update(model, weights, native_loader):
    from sglang.srt.model_loader.weight_utils import default_weight_loader
    parameters = dict(model.named_parameters())
    ordinary = []
    end_version = None
    for name, value in weights:
        if name == "archlab_update_begin":
            if getattr(model, "_archlab_live_update", None) is not None:
                raise ValueError("nested policy update")
            model._archlab_live_update = {"version": int(value.item()), "loaded": set(),
                                         "expert_slices": set(), "compressor_pairs": {}}
            model._archlab_loaded = False
            for layer in model.model.layers:
                attention = layer.self_attn
                if attention.compress_ratio in (4, 128):
                    attention.compressor.ape_converted = False
                if attention.compress_ratio == 4:
                    attention.indexer.compressor.ape_converted = False
            for cache in model.archlab_cache_states.values():
                cache.caches.clear()
            for layer in model.config.engram_layer_ids:
                table = model.model.layers[layer].engram.embed
                table._pieces = []
                table._loaded = False
                table.transaction_open = True
        elif name == "archlab_update_end":
            end_version = int(value.item())
        else:
            state = getattr(model, "_archlab_live_update", None)
            if state is None:
                raise ValueError("weight packet outside a policy transaction")
            row = re.fullmatch(r"layers.(\d+).engram.embed.weight.rows.(\d+)", name)
            adapter = re.fullmatch(r"layers.(\d+).attn_hc.simplicial_adapter.(.+)", name)
            if row:
                layer, first = map(int, row.groups())
                model.model.layers[layer].engram.embed.load_piece(first, value)
                state["loaded"].add(f"model.layers.{layer}.engram.embed.weight")
            elif adapter:
                key = f"model.layers.{adapter[1]}.archlab_adapter.{adapter[2]}"
                p = parameters[key]
                if p.shape != value.shape or value.dtype != p.dtype:
                    raise ValueError(f"adapter refresh mismatch: {name}")
                p.data.copy_(value)
                state["loaded"].add(key)
            else:
                ordinary.extend(_stage_compressor(state, name, value, parameters,
                    model.remap_weight_name_to_dpsk_hf_format))
    if ordinary:
        originals = {name: getattr(p, "weight_loader", None) for name, p in parameters.items()}
        for name, p in parameters.items():
            loader = originals[name] or default_weight_loader

            def tracked(*args, _loader=loader, _name=name, **kwargs):
                result = _loader(*args, **kwargs)
                model._archlab_live_update["loaded"].add(_name)
                if "expert_id" in kwargs:
                    model._archlab_live_update["expert_slices"].add(
                        (_name, kwargs["expert_id"], kwargs["shard_id"]))
                return result

            p.weight_loader = tracked
        try:
            # Partial-bucket warnings are expected here. End-of-transaction
            # parameter coverage remains mandatory and raises on omissions.
            logger = logging.getLogger("sglang.srt.models.deepseek_v4")
            warning_filter = _PartialBucketWarningFilter()
            logger.addFilter(warning_filter)
            try:
                native_loader(ordinary, is_nextn=False)
            finally:
                logger.removeFilter(warning_filter)
        finally:
            for name, p in parameters.items():
                if originals[name] is None:
                    del p.weight_loader
                else:
                    p.weight_loader = originals[name]
    if end_version is not None:
        _finish(model, parameters, end_version)
    return set()


class _PartialBucketWarningFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Some weights are not initialized from checkpoints:")


def _stage_compressor(state, name, value, parameters, remap):
    match = re.fullmatch(r"(.+\.compressor)\.(wkv|wgate)\.weight", name)
    if match is None:
        return [(name, value)]
    prefix, part = match.groups()
    fused_name = prefix + ".wkv_gate.weight"
    if remap(fused_name) not in parameters:
        return [(name, value)]
    pending = state["compressor_pairs"]
    pieces = pending.setdefault(prefix, {})
    if part in pieces:
        raise ValueError(f"duplicate compressor shard: {name}")
    # IPC bucket storage is recycled after the RPC; retain only incomplete pairs.
    pieces[part] = value.detach().clone()
    if len(pieces) == 2:
        fused = torch.cat([pieces["wkv"], pieces["wgate"]], dim=0)
        del pending[prefix]
        return [(fused_name, fused)]
    retained_bytes = sum(t.numel() * t.element_size() for pair in pending.values() for t in pair.values())
    if retained_bytes > 128 * 2**20:
        raise ValueError("incomplete compressor pairs exceed bounded streaming workspace")
    return []


def _finish(model, parameters, version):
    state = model._archlab_live_update
    if state is None or state["version"] != version:
        raise ValueError("policy transaction version mismatch")
    if state["compressor_pairs"]:
        raise ValueError("policy transaction has incomplete compressor pairs")
    missing = {name for name in parameters if name not in state["loaded"]
               and not any(s in name for s in ("attn_mqa.k_scale", "attn_mqa.v_scale", "blockscale_swizzled"))}
    if missing:
        raise ValueError(f"incomplete policy update: {sorted(missing)}")
    _verify_expert_slices(parameters, state["expert_slices"], model.archlab_tp_rank)
    for layer in model.config.engram_layer_ids:
        table = model.model.layers[layer].engram.embed
        table.transaction_open = False
        table.finish_load(str(layer))
    model.finalize_live_weights()
    model.validate_derived_state()
    if any(p.device.type != "cuda" for p in parameters.values()):
        raise ValueError("all inference weights must remain resident on GPU")
    model.requires_grad_(False)
    model.eval()
    torch.cuda.synchronize()
    model._archlab_live_update = None
    model._archlab_loaded = True
    print(f"ARCHLAB policy update {version} complete: {len(state['loaded'])} tensors", flush=True)


def _verify_expert_slices(parameters, loaded, rank):
    for name, parameter in parameters.items():
        if not name.endswith((".experts.w13_weight", ".experts.w2_weight")):
            continue
        count = parameter.shape[0]
        parts = ("w1", "w3") if name.endswith("w13_weight") else ("w2",)
        for expert in range(rank * count, (rank + 1) * count):
            for part in parts:
                if (name, expert, part) not in loaded:
                    raise ValueError(f"incomplete expert refresh: {name}, expert={expert}, part={part}")
