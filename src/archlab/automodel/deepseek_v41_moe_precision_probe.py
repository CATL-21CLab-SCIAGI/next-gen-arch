"""Isolate compiled activation rounding in the real-layer expert diagnostic.

The unchanged official activation Python body runs eagerly on the diagnostic
instance. Production execution and all upstream source files remain untouched.
"""

from functools import partial


def _inspect_stages(model, args):
    import json
    import torch
    from torch.nn import functional as F
    from nemo_automodel.components.moe.experts import _permute_tokens_for_grouped_mm
    from archlab.automodel.deepseek_v41_moe_rounding_probe import _compare

    x = args[0].reshape(-1, model.experts.gate_and_up_projs.shape[1])
    weights, indices = model.gate.weights, model.gate.indices
    expert = model.experts
    up, down = expert.gate_and_up_projs, expert.down_projs
    ids, slots, probs, counts, offs = _permute_tokens_for_grouped_mm(
        indices, weights, torch.ones(x.shape[0], dtype=torch.bool, device=x.device),
        expert.n_routed_experts, 0, return_slot_ids=True)
    grouped_up = torch._grouped_mm(x[ids], up, offs)
    grouped_act = expert.expert_activation_grouped(grouped_up, probs[:, None])
    grouped_down = torch._grouped_mm(grouped_act, down, offs)
    native_up, native_act, native_down = (torch.empty_like(v) for v in (grouped_up, grouped_act, grouped_down))
    intermediate = up.shape[-1] // 2
    first = 0
    for e, count in enumerate(counts.tolist()):
        last = first + count
        if count:
            selected = x[ids[first:last]]
            gate = F.linear(selected, up[e, :, :intermediate].T.contiguous())
            value = F.linear(selected, up[e, :, intermediate:].T.contiguous())
            native_up[first:last] = torch.cat((gate, value), -1)
            native_act[first:last] = expert.expert_activation_grouped(native_up[first:last], probs[first:last, None])
            native_down[first:last] = F.linear(native_act[first:last], down[e].T.contiguous())
        first = last
    same_act_down = torch._grouped_mm(native_act, down, offs)
    selected_x = x[ids]
    split_rows = torch.cat([torch._grouped_mm(selected_x, weight.contiguous(), offs)
                            for weight in up.chunk(2, -1)], -1)
    split_columns = torch.cat([torch._grouped_mm(
        selected_x, weight.transpose(-2, -1).contiguous().transpose(-2, -1), offs)
        for weight in up.chunk(2, -1)], -1)
    atomic = torch.zeros_like(x, dtype=torch.float32).scatter_add_(0, ids[:, None].expand_as(native_down), native_down.float())
    ordered = torch.zeros_like(atomic)
    owners = torch.zeros(8, *x.shape, device=x.device, dtype=torch.float32)
    first = 0
    for e, count in enumerate(counts.tolist()):
        last = first + count
        if count:
            ordered[ids[first:last]] += native_down[first:last].float()
            owners[e // (expert.n_routed_experts // 8), ids[first:last]] += native_down[first:last].float()
        first = last
    shared = model.shared_experts(x)
    report = {"event": "expert_precision_stages",
              "grouped_up_vs_separate_linears": _compare(grouped_up, native_up),
              "grouped_activation_vs_native_inputs": _compare(grouped_act, native_act),
              "grouped_down_vs_native": _compare(grouped_down, native_down),
              "grouped_down_with_same_input": _compare(same_act_down, native_down),
              "split_grouped_row_layout_vs_native": _compare(split_rows, native_up),
              "split_grouped_column_layout_vs_native": _compare(split_columns, native_up),
              "atomic_vs_ordered_fp32": _compare(atomic, ordered),
              "atomic_vs_ordered_final": _compare((atomic + shared.float()).bfloat16(), (ordered + shared.float()).bfloat16()),
              "ordered_vs_owner_final": _compare((ordered + shared.float()).bfloat16(), (owners.sum(0) + shared.float()).bfloat16())}
    print(json.dumps(report), flush=True)


def main():
    from archlab.automodel import deepseek_v41_official_moe as precision
    from archlab.automodel.deepseek_v41_moe_rounding_probe import main as replay

    original_install = precision.install_official_fp32_moe

    def install_eager_activation(model):
        report = original_install(model)
        for module in model.modules():
            activation = getattr(module, "expert_activation_grouped", None)
            if activation is None:
                continue
            if not isinstance(activation, partial):
                raise TypeError("expected the official clamped SwiGLU partial")
            body = getattr(activation.func, "_torchdynamo_orig_callable", None)
            if body is None or body.__name__ != "swiglu_clamped_deepep":
                raise TypeError("expected the original official clamped SwiGLU body")
            module.expert_activation_grouped = partial(body, *activation.args, **activation.keywords)
        report["diagnostic_activation"] = "unchanged-official-clamped-SwiGLU-eager"
        model.register_forward_pre_hook(_inspect_stages)
        return report

    precision.install_official_fp32_moe = install_eager_activation
    try:
        replay()
    finally:
        precision.install_official_fp32_moe = original_install


if __name__ == "__main__":
    main()
