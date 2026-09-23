import pytest
import torch
from torch.nn import functional as F

from archlab.architectures.deepseek_v41_swiglu import native_swiglu


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows,width,chunk", [(17, 32, 5), (2051, 2304, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_swiglu_preserves_clamps_outputs_and_all_gradients(
    device, rows, width, chunk, dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires the existing B300 runtime")
    torch.manual_seed(176)
    g = (torch.randn(rows, width, device=device, dtype=dtype) * 8).requires_grad_()
    v = (torch.randn_like(g) * 8).requires_grad_()
    with torch.no_grad():
        g[0, :3] = torch.tensor([-7, 0, 7], device=device, dtype=dtype)
        v[0, :3] = torch.tensor([-7, 0, 7], device=device, dtype=dtype)
    p = torch.rand(rows, 1, device=device, dtype=torch.float32, requires_grad=True)
    copies = tuple(t.detach().clone().requires_grad_() for t in (g, v, p))
    expected = (F.silu(g.float().clamp(max=7)) * v.float().clamp(-7, 7) * p).to(dtype)
    actual = native_swiglu(*copies, limit=7, chunk_rows=chunk)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for source, candidate in zip((g, v, p), copies, strict=True):
        torch.testing.assert_close(source.grad, candidate.grad, rtol=0, atol=0)
