"""Small B300 oracle for the pinned V4.1 sparse-attention kernels."""

import json
import sys

import tilelang.backend.target as target
import torch

sys.modules.setdefault("tilelang.utils.target", target)
from miles_plugins.models.deepseek_v41.ops.kernel.tilelang_sparse_mla import (  # noqa: E402
    sparse_attn_tilelang,
)


def main():
    torch.manual_seed(91)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn(1, 128, 8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    s = torch.randn(8, device="cuda", requires_grad=True)
    idx = (
        torch.arange(128, device="cuda", dtype=torch.int32)[None, None, :]
        .expand(1, 128, 128)
        .clone()
    )
    mask = torch.arange(128, device="cuda")[:, None] < torch.arange(128, device="cuda")[None, :]
    idx.masked_fill_(mask[None], -1)
    y = sparse_attn_tilelang(q, k, s, idx, 512**-0.5)
    qr = q.detach().float().requires_grad_()
    kr = k.detach().float().requires_grad_()
    sr = s.detach().clone().requires_grad_()
    z = torch.einsum("bshd,btd->bsht", qr, kr) * (512**-0.5)
    z = z.masked_fill(mask[None, :, None, :], float("-inf"))
    z = torch.cat([z, sr[None, None, :, None].expand(1, 128, 8, 1)], dim=-1).softmax(-1)[..., :128]
    ref = torch.einsum("bsht,btd->bshd", z, kr)
    g = torch.randn_like(y)
    y.backward(g)
    ref.backward(g.float())
    errors = {"output": float((y.float() - ref).abs().max())}
    torch.testing.assert_close(y.float(), ref, atol=0.035, rtol=0.035)
    for name, a, b in [("q", q.grad, qr.grad), ("kv", k.grad, kr.grad), ("sink", s.grad, sr.grad)]:
        errors[name] = float((a.float() - b).abs().max())
        torch.testing.assert_close(a.float(), b, atol=0.12, rtol=0.08)
    print("V41_SPARSE_KERNEL_FORWARD_BACKWARD_PASSED", json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()
