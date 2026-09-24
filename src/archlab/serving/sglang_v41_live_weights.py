"""Transactional Miles refreshes for the fully trained V4.1 serving extension."""

import re

import torch
from sglang.srt.model_loader.weight_utils import default_weight_loader


def update(model, weights, native_loader):
    parameters = dict(model.named_parameters())
    ordinary = []
    end_version = None
    for name, value in weights:
        if name == "archlab_update_begin":
            if getattr(model, "_archlab_live_update", None) is not None:
                raise ValueError("nested policy update")
            model._archlab_live_update = {"version": int(value.item()), "loaded": set()}
            model._archlab_loaded = False
            for cache in model.archlab_cache_states.values():
                cache.caches.clear()
            for layer in model.config.engram_layer_ids:
                table = model.model.layers[layer].engram.embed
                table._pieces = []
                table._loaded = False
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
                ordinary.append((name, value))
    if ordinary:
        originals = {name: getattr(p, "weight_loader", None) for name, p in parameters.items()}
        for name, p in parameters.items():
            loader = originals[name] or default_weight_loader

            def tracked(*args, _loader=loader, _name=name, **kwargs):
                result = _loader(*args, **kwargs)
                model._archlab_live_update["loaded"].add(_name)
                return result

            p.weight_loader = tracked
        try:
            native_loader(ordinary, is_nextn=False)
        finally:
            for name, p in parameters.items():
                if originals[name] is None:
                    del p.weight_loader
                else:
                    p.weight_loader = originals[name]
    if end_version is not None:
        _finish(model, parameters, end_version)
    return set()


def _finish(model, parameters, version):
    state = model._archlab_live_update
    if state is None or state["version"] != version:
        raise ValueError("policy transaction version mismatch")
    missing = {name for name in parameters if name not in state["loaded"]
               and not any(s in name for s in ("attn_mqa.k_scale", "attn_mqa.v_scale", "blockscale_swizzled"))}
    if missing:
        raise ValueError(f"incomplete policy update: {sorted(missing)}")
    for layer in model.config.engram_layer_ids:
        model.model.layers[layer].engram.embed.finish_load(str(layer))
    model.validate_derived_state()
    if any(p.device.type != "cuda" for p in parameters.values()):
        raise ValueError("all inference weights must remain resident on GPU")
    model.requires_grad_(False)
    model.eval()
    torch.cuda.synchronize()
    model._archlab_live_update = None
    model._archlab_loaded = True
    print(f"ARCHLAB policy update {version} complete: {len(state['loaded'])} tensors", flush=True)
