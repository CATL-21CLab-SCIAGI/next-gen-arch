"""Distributed Flash-Next qualification checks, independent of the launch entry."""

from __future__ import annotations

import torch


def _probe_parameter_counts(model, *, data_replicas: int, expert_replicas: int):
    counts = [0, 0]
    for parameter in model.parameters():
        counts[int(not getattr(parameter, "allreduce", True))] += parameter.numel()
    totals = torch.tensor(counts, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(totals)
    dense, expert = totals.tolist()
    if dense % data_replicas or expert % expert_replicas:
        raise RuntimeError("probe parameters do not divide into complete model replicas")
    return {
        "dense_unique": dense // data_replicas,
        "expert_and_ple_unique": expert // expert_replicas,
        "total": dense // data_replicas + expert // expert_replicas,
    }


def _probe_ple_replica_equality(model, replica_group):
    """Check every trained PLE weight against its corresponding expert-DP replica."""
    dist = torch.distributed
    source = dist.get_process_group_ranks(replica_group)[0]
    checked = 0
    mismatch = torch.zeros((), dtype=torch.int64, device=torch.cuda.current_device())
    for name, parameter in model.named_parameters():
        if ".embedding.tables." not in name:
            continue
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=source, group=replica_group)
        mismatch += (~torch.isfinite(parameter)).any().to(torch.int64)
        mismatch += (parameter.detach() != reference).any().to(torch.int64)
        checked += 1
    evidence = torch.stack((mismatch, mismatch.new_tensor(checked)))
    dist.all_reduce(evidence)
    if evidence[0].item() or not evidence[1].item():
        raise RuntimeError(f"probe PLE replica equality failed: {evidence.tolist()}")
    return {"status": "passed", "checked_local_tables": evidence[1].item(), "mismatches": 0}


def _probe_restored_replica_equality(model, group):
    """Verify the new attention and residual parameters after native Muon/Adam steps."""
    checked = 0
    bad = torch.zeros((), dtype=torch.int64, device=torch.cuda.current_device())
    source = torch.distributed.get_process_group_ranks(group)[0]
    for name, parameter in model.named_parameters():
        if not any(
            part in name
            for part in (
                ".linear_qkv.",
                ".q_layernorm.",
                ".k_layernorm.",
                ".attention_residual.",
                ".mlp_residual.",
                ".final_mixer.",
                ".attention.k2.",
                ".attention.v2.",
                ".attention.k2_layernorm.",
            )
        ):
            continue
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=source, group=group)
        bad += ((parameter != reference) | ~torch.isfinite(parameter)).any().to(torch.int64)
        checked += 1
    torch.distributed.all_reduce(bad)
    if bad.item() or not checked:
        raise RuntimeError("restored attention/GR parameters differ between DP replicas")
    return {"status": "passed", "checked_parameters_per_replica": checked, "mismatches": 0}


def _effective_probe_gradient(parameter, autograd_gradient):
    """TE fused wgrad returns a placeholder after writing the native main buffer."""
    if getattr(parameter, "grad_added_to_main_grad", False):
        return parameter.main_grad
    return autograd_gradient
