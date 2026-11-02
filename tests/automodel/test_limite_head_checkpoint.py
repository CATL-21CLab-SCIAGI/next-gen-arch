"""Head caching preserves chunk accumulation, gradients, RNG, and updates."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from archlab.automodel.limite_adapter_common import loss_sum, make_head_part


class Hidden(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embedding = nn.Embedding(19, 8, dtype=dtype)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(
            last_hidden_state=F.dropout(self.embedding(input_ids), p=0.2, training=True)
        )


class Policy(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.model = Hidden(dtype)
        self.head = nn.Linear(8, 19, bias=False, dtype=dtype)

    def _softcapped_logits(self, hidden):
        return 23.0 * torch.sigmoid((self.head(hidden).float() + 5.0) / 7.5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_head_matches_recompute_loss_gradients_rng_and_update(dtype):
    torch.manual_seed(17)
    policy = Policy(dtype)
    initial = {name: value.clone() for name, value in policy.state_dict().items()}
    ids = torch.tensor([[1, 4, 2, 7, 3], [2, 5, 8, 9, 0]])
    targets = (ids + 1) % 19
    targets[0, -1] = -100
    results = []
    for checkpoint_head in (True, False):
        policy.load_state_dict(initial)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(23)
        loss = loss_sum(
            policy,
            ids,
            targets,
            chunk=3,
            head_part=make_head_part(policy),
            checkpoint_head=checkpoint_head,
        )
        loss.backward()
        gradients = [parameter.grad.clone() for parameter in policy.parameters()]
        optimizer.step()
        results.append((loss.detach(), gradients, torch.get_rng_state(), policy.state_dict()))
        results[-1] = (*results[-1][:3], {k: v.clone() for k, v in results[-1][3].items()})
    recomputed, cached = results
    assert torch.equal(recomputed[0], cached[0])
    assert all(torch.equal(a, b) for a, b in zip(recomputed[1], cached[1], strict=True))
    assert torch.equal(recomputed[2], cached[2])
    assert all(torch.equal(recomputed[3][k], cached[3][k]) for k in recomputed[3])
