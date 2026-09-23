from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from archlab.automodel.deepseek_v41_rl_cache_shapes import (
    pad_at_position,
    replay_shape_boundaries,
)


@dataclass
class Mix:
    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class HC(torch.nn.Module):
    def forward(self, hidden):
        pre = hidden.float().sum(-1)
        return Mix(pre, pre, pre.unsqueeze(-1) * pre.unsqueeze(-2))

    @staticmethod
    def collapse(hidden, pre):
        return (hidden * pre.unsqueeze(-1)).sum(2)

    @staticmethod
    def expand(output, residual, mix):
        return output.unsqueeze(-2) * mix.post.unsqueeze(-1) + residual


@pytest.mark.parametrize("fail", [False, True])
def test_cached_overrides_preserve_values_and_restore_after_error(fail):
    model = torch.nn.Module().eval()
    model.model = SimpleNamespace(norm=torch.nn.Identity())
    layer = SimpleNamespace(attn_hc=HC(), ffn_hc=HC(),
                            attn_norm=torch.nn.Identity(), ffn_norm=torch.nn.Identity())
    modules = [model.model.norm, layer.attn_hc, layer.ffn_hc, layer.attn_norm, layer.ffn_norm]
    before = [{name: value for name, value in module.__dict__.items()
               if name in {"forward", "collapse", "expand"}} for module in modules]
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
    expected = layer.attn_hc(hidden)
    try:
        with torch.no_grad(), replay_shape_boundaries(model, [(None, layer, None)],
                                                       position=3, canvas=8):
            actual = layer.attn_hc(hidden)
            for name in ("pre", "post", "comb"):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                           rtol=0, atol=0)
            collapsed = layer.attn_hc.collapse(hidden, actual.pre)
            torch.testing.assert_close(collapsed, HC.collapse(hidden, expected.pre), rtol=0, atol=0)
            torch.testing.assert_close(layer.ffn_norm(collapsed), collapsed, rtol=0, atol=0)
            torch.testing.assert_close(layer.attn_hc.expand(collapsed, hidden, actual),
                                       HC.expand(collapsed, hidden, expected), rtol=0, atol=0)
            if fail:
                raise RuntimeError("injected decode error")
    except RuntimeError:
        assert fail
    assert before == [{name: value for name, value in module.__dict__.items()
                       if name in {"forward", "collapse", "expand"}} for module in modules]


def test_padding_preserves_position_and_rejects_invalid_canvas():
    value = torch.ones(2, 1, 4)
    padded = pad_at_position(value, 127, 256)
    assert padded.shape == (2, 256, 4) and padded.sum() == value.sum()
    torch.testing.assert_close(padded[:, 127:128], value)
    with pytest.raises(ValueError):
        pad_at_position(value, 128, 128)
