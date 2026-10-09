"""GPU admission for native BF16 Adam masters, no-op batches and restoration."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main():
    import torch

    from archlab.optimizers.rl_adam import SignalFusedAdam

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(lr=3e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    parameter = torch.nn.Parameter(
        torch.tensor([0.125, -0.25], device="cuda", dtype=torch.bfloat16)
    )
    reference = torch.nn.Parameter(parameter.detach().float())
    optimizer = SignalFusedAdam([parameter], master_weights=True, **kwargs)
    oracle = torch.optim.AdamW([reference], **kwargs)
    for _ in range(10):
        parameter.grad = torch.tensor([0.1, -0.2], device="cuda", dtype=torch.bfloat16)
        reference.grad = parameter.grad.float()
        optimizer.step()
        oracle.step()
    state = optimizer.state_dict()
    master = state["state"][0]["master_param"]
    error = float((master - reference).abs().max())
    assert error < 1e-7 and not torch.equal(master, parameter.float())
    saved = copy.deepcopy(state)
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    assert optimizer.state_dict()["param_groups"][0]["step"] == saved["param_groups"][0]["step"]
    checkpoint = args.output.with_suffix(".pt")
    torch.save(dict(model=parameter.detach(), optimizer=saved), checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    second = torch.nn.Parameter(restored["model"].clone())
    resumed = SignalFusedAdam([second], master_weights=True, **kwargs)
    resumed.load_state_dict(restored["optimizer"])
    assert torch.equal(resumed.state_dict()["state"][0]["master_param"], master)
    for p in (parameter, second):
        p.grad = torch.tensor([0.2, 0.3], device="cuda", dtype=torch.bfloat16)
    optimizer.step()
    resumed.step()
    assert torch.equal(parameter, second)
    assert torch.equal(
        optimizer.state_dict()["state"][0]["master_param"],
        resumed.state_dict()["state"][0]["master_param"],
    )
    report = dict(
        passed=True,
        fp32_master_max_error=error,
        small_updates_retained=True,
        flat_batch_preserves_step=True,
        checkpoint_resume_exact=True,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
