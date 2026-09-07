"""Isolated installed-kernel diagnostic using captured pretrained GDN inputs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch


def metrics(actual, reference):
    difference = (actual - reference).float()
    return {"equal": torch.equal(actual, reference), "max_abs": difference.abs().max().item(),
            "relative_l2": (difference.norm() / reference.float().norm().clamp_min(1e-30)).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--sequence-length", type=int, default=257)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--safe-runtime", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if args.inputs:
        payload = torch.load(args.inputs, weights_only=True, map_location="cuda")
    else:
        torch.manual_seed(1234)
        shape = (1, args.sequence_length, 48, 128)
        payload = {"args": tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)),
                   "kwargs": {"g": -torch.rand(shape[:-1], device="cuda"),
                              "beta": torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16),
                              "initial_state": None, "output_final_state": False,
                              "use_qk_l2norm_in_kernel": True}}
    chunk = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    if args.safe_runtime:
        from archlab.automodel.runtime import configure_frozen_gdn_runtime

        print(json.dumps({"stage": "runtime", **configure_frozen_gdn_runtime()}), flush=True)
    saved, counts = {}, {}
    iteration = 0
    originals = {}

    def trace(original, name):
        def wrapped(*inputs, **kwargs):
            index = counts.get(name, 0)
            counts[name] = index + 1
            outputs = original(*inputs, **kwargs)
            for j, output in enumerate(outputs if isinstance(outputs, tuple) else (outputs,)):
                if not isinstance(output, torch.Tensor):
                    continue
                key = f"{name}.{index}.{j}"
                if iteration == 0:
                    saved[key] = output.detach().clone()
                else:
                    print(json.dumps({"iteration": iteration, "stage": key,
                                      **metrics(output, saved[key])}), flush=True)
            return outputs
        return wrapped

    if args.trace:
        for name in ("l2norm_fwd", "chunk_local_cumsum", "chunk_scaled_dot_kkt_fwd", "solve_tril",
                     "recompute_w_u_fwd", "chunk_gated_delta_rule_fwd_h", "chunk_fwd_o"):
            originals[name] = getattr(chunk, name)
            setattr(chunk, name, trace(originals[name], name))
    inputs, kwargs = payload["args"], payload["kwargs"]
    snapshots = [value.clone() for value in inputs]
    with torch.no_grad():
        outputs = []
        for iteration in range(5):
            counts.clear()
            output = chunk.chunk_gated_delta_rule(*inputs, **kwargs)[0]
            outputs.append(output.clone())
            print(json.dumps({"iteration": iteration, "stage": "output", **metrics(output, outputs[0])}), flush=True)
            for actual, expected in zip(inputs, snapshots, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if args.reference:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_chunk_gated_delta_rule,
            )

            reference = torch_chunk_gated_delta_rule(*inputs, **kwargs)[0]
            comparison = metrics(outputs[-1], reference)
            print(json.dumps({"stage": "torch-reference", **comparison}), flush=True)
            if not torch.isfinite(reference).all() or comparison["relative_l2"] > .03:
                raise AssertionError("GDN forward oracle failed")
    for name, original in originals.items():
        setattr(chunk, name, original)
    if args.safe_runtime:
        assert all(torch.equal(output, outputs[0]) for output in outputs), "GDN forward is not repeatable"
    if args.backward:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            torch_chunk_gated_delta_rule,
        )

        torch.manual_seed(5678)
        upstream_gradient = torch.randn_like(outputs[0])
        results = []
        for name, function in (("fla", chunk.chunk_gated_delta_rule),
                               ("fla-repeat", chunk.chunk_gated_delta_rule),
                               ("torch", torch_chunk_gated_delta_rule)):
            q, k, v = [value.detach().clone().requires_grad_() for value in inputs]
            kw = dict(kwargs)
            kw["g"] = kw["g"].detach().clone().requires_grad_()
            kw["beta"] = kw["beta"].detach().clone().requires_grad_()
            output = function(q, k, v, **kw)[0]
            gradients = torch.autograd.grad(output, (q, k, v, kw["g"], kw["beta"]), upstream_gradient)
            results.append(gradients)
            for label, actual, expected in zip(("q", "k", "v", "g", "beta"), gradients, results[0], strict=True):
                metric = metrics(actual, expected)
                print(json.dumps({"stage": f"backward-{name}-{label}", **metric}), flush=True)
                if not torch.isfinite(actual).all() or metric["relative_l2"] > .03:
                    raise AssertionError("GDN gradient oracle failed")


if __name__ == "__main__":
    main()
