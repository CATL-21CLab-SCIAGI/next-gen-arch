"""First-update, response-token parity between colocated serving and training."""
import json
from pathlib import Path

import torch
import torch.distributed as dist

active = None


class InitialPolicyParity:
    def __init__(self, output):
        self.output = Path(output)
        self.totals = torch.zeros(2, device="cuda", dtype=torch.float64)
        self.maximum = torch.zeros((), device="cuda", dtype=torch.float64)
        self.finished = False

    @torch.no_grad()
    def record(self, current, behavior, mask):
        from megatron.core import mpu
        if self.finished or mpu.get_tensor_model_parallel_rank() != 0:
            return
        delta = (current.detach() - behavior.detach()).abs()[mask.bool()]
        if delta.numel():
            self.totals[0] += delta.double().sum()
            self.totals[1] += delta.numel()
            self.maximum = torch.maximum(self.maximum, delta.max().double())

    @torch.no_grad()
    def finish(self):
        if self.finished:
            return
        dist.all_reduce(self.totals)
        dist.all_reduce(self.maximum, op=dist.ReduceOp.MAX)
        count = int(self.totals[1].item())
        mean, maximum = float((self.totals[0] / max(count, 1)).item()), float(self.maximum.item())
        passed = count > 0 and mean <= .05 and maximum <= .5
        receipt = dict(rank=dist.get_rank(), response_tokens=count, mean_abs_logprob_error=mean,
                       max_abs_logprob_error=maximum, mean_limit=.05, max_limit=.5,
                       policy_parity=passed, measurement="first_update_response_token_logprobs")
        (self.output / f"policy-parity-rank-{dist.get_rank():02d}.json").write_text(json.dumps(receipt, indent=2))
        if not passed:
            raise ValueError(f"initial serving/training policy parity failed: {receipt}")
        self.finished = True
