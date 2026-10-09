"""Restore existing Muown storage without generic optimizer dtype conversion."""
from types import MethodType

import torch


@torch.no_grad()
def load_muown_in_place(optimizer, saved):
    groups = optimizer.param_groups
    if len(groups) != len(saved["param_groups"]):
        raise ValueError("Muown checkpoint group coverage changed")
    pairs = []
    for live_group, saved_group in zip(groups, saved["param_groups"], strict=True):
        if len(live_group["params"]) != len(saved_group["params"]):
            raise ValueError("Muown checkpoint parameter coverage changed")
        pairs.extend(zip(live_group["params"], saved_group["params"], strict=True))
    if {key for _, key in pairs} != saved["state"].keys():
        raise ValueError("Muown checkpoint state coverage changed")
    # Validate the complete payload before changing any tensor.
    for parameter, key in pairs:
        live, source = optimizer.state[parameter], saved["state"][key]
        if live.keys() != source.keys():
            raise ValueError("Muown checkpoint state fields changed")
        for name, value in source.items():
            current = live[name]
            if isinstance(current, torch.Tensor):
                if (not isinstance(value, torch.Tensor) or current.shape != value.shape
                        or current.dtype != value.dtype):
                    raise ValueError(f"Muown checkpoint tensor contract changed: {name}")
            elif type(current) is not type(value):
                raise ValueError(f"Muown checkpoint scalar contract changed: {name}")
    for parameter, key in pairs:
        live = optimizer.state[parameter]
        for name, value in saved["state"][key].items():
            if isinstance(value, torch.Tensor):
                live[name].copy_(value)
            else:
                live[name] = value
    for live_group, saved_group in zip(groups, saved["param_groups"], strict=True):
        live_group.update({key: value for key, value in saved_group.items() if key != "params"})


def install_for_resident(resident):
    """Also cover workers already initialized before this checkpoint boundary."""
    for entry in resident.entries.values():
        if entry["kind"] == "muown":
            entry["optimizer"].load_state_dict = MethodType(load_muown_in_place, entry["optimizer"])
