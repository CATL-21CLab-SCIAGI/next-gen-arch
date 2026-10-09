"""Load the verified, unmodified upstream Limite implementation as a package."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def upstream_classes(snapshot):
    # Direct package import avoids Transformers 5.8's incomplete recursive
    # relative-import cache for this local checkpoint; no runtime is patched.
    import hashlib

    snapshot = Path(snapshot).resolve()
    receipt = json.loads((snapshot / "DOWNLOAD_VERIFIED.json").read_text())
    if receipt["repo"] not in {"paradigma-inc/limite-1b-base", "paradigma-inc/limite-1b-violetto"}:
        raise ValueError("expected a verified Limite base or Violetto actor snapshot")
    for spec in receipt["files"]:
        if spec["path"].endswith((".py", ".json", ".jinja")):
            if hashlib.sha256((snapshot / spec["path"]).read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError("upstream snapshot changed")
    name = "archlab_upstream_limite_" + receipt["revision"][:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(snapshot)]
        sys.modules[name] = package
    config = importlib.import_module(name + ".configuration_limite").LimiteConfig
    model = importlib.import_module(name + ".modeling_limite").LimiteForCausalLM
    return config, model


def load_model(snapshot, **kwargs):
    import torch
    from safetensors import safe_open

    config_class, model_class = upstream_classes(snapshot)
    config = config_class.from_pretrained(snapshot, local_files_only=True)
    kwargs["dtype"] = torch.bfloat16
    model = model_class.from_pretrained(snapshot, config=config, local_files_only=True, **kwargs)
    # The checkpoint has BF16 matrices but FP32 gates/scales/MUDD tensors.
    # A global FP32 cast changes dtype-dependent RMSNorm epsilon; a global
    # BF16 cast loses the original small-parameter precision. Restore exact
    # source dtypes before constructing the optimizer.
    parameters = dict(model.named_parameters())
    with safe_open(
        str(Path(snapshot) / "model.safetensors"), framework="pt", device="cpu"
    ) as weights:
        for name in weights.keys():
            if weights.get_slice(name).get_dtype() == "F32":
                if name not in parameters:
                    raise ValueError("unmapped FP32 checkpoint parameter: " + name)
                parameter = parameters[name]
                parameter.data = weights.get_tensor(name).to(parameter.device)
    model.train(model.training)
    return model
