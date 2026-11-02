import pytest
import torch

from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight, hc_split_sinkhorn


def test_mhc_gradient_and_normalization():
    torch.manual_seed(82)
    mixes = torch.randn(1, 2, 24, dtype=torch.float64, requires_grad=True)
    scale = torch.randn(3, dtype=torch.float64, requires_grad=True)
    base = torch.randn(24, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(hc_split_sinkhorn, (mixes, scale, base))
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base)
    assert pre.shape == post.shape == (1, 2, 4)
    # Twenty finite iterations give approximate, not exact, row normalization.
    torch.testing.assert_close(comb.sum(-1), torch.ones_like(pre), rtol=0, atol=1e-3)
    torch.testing.assert_close(comb.sum(-2), torch.ones_like(pre), rtol=0, atol=5e-6)


def test_fp4_codebook_nibble_order_and_scale():
    packed = torch.tensor([[i | ((15 - i) << 4) for i in range(16)]], dtype=torch.uint8).view(torch.int8)
    before = packed.clone()
    decoded = dequantize_frozen_weight(packed, torch.tensor([[2.0]]))
    codebook = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    expected = torch.tensor([[v * 2 for i in range(16) for v in (codebook[i], codebook[15 - i])]])
    torch.testing.assert_close(decoded.float(), expected, atol=0, rtol=0)
    assert torch.equal(packed, before)


def test_fp8_block_scales_and_partial_row_block():
    weight = torch.ones(33, 64).to(torch.float8_e4m3fn)
    decoded = dequantize_frozen_weight(weight, torch.tensor([[1., 2.], [3., 4.]]))
    assert (decoded[:32, :32] == 1).all() and (decoded[:32, 32:] == 2).all()
    assert (decoded[32:, :32] == 3).all() and (decoded[32:, 32:] == 4).all()
    with pytest.raises(ValueError):
        dequantize_frozen_weight(weight, torch.ones(2, 3))
