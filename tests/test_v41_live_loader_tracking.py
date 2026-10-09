import pytest
import torch

from archlab.serving.sglang_v41_live_weights import _track_loaders


class ReadOnlyLoaderParameter(torch.nn.Parameter):
    @property
    def weight_loader(self):
        return self._weight_loader


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_tracks_successful_loads_and_restores_native_loader(read_only, fail):
    cls = ReadOnlyLoaderParameter if read_only else torch.nn.Parameter
    p = cls(torch.zeros(2), requires_grad=False)

    def loader(param, value, **kwargs):
        if fail:
            raise ValueError("native load failed")
        param.data.copy_(value)

    attribute = "_weight_loader" if read_only else "weight_loader"
    setattr(p, attribute, loader)
    state = {"loaded": set(), "expert_slices": set()}
    try:
        with _track_loaders({"expert": p}, state, loader):
            p.weight_loader(p, torch.ones(2), expert_id=3, shard_id="w1")
    except ValueError:
        assert fail
    assert p.weight_loader is loader
    assert state["loaded"] == (set() if fail else {"expert"})
    assert state["expert_slices"] == (set() if fail else {("expert", 3, "w1")})
    torch.testing.assert_close(p, torch.zeros(2) if fail else torch.ones(2))


def test_default_loader_attribute_is_removed_after_failure():
    p = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    state = {"loaded": set(), "expert_slices": set()}
    with pytest.raises(RuntimeError, match="later bucket failure"):
        with _track_loaders({"plain": p}, state, lambda param, value: param.data.copy_(value)):
            p.weight_loader(p, torch.ones(2))
            raise RuntimeError("later bucket failure")
    assert not hasattr(p, "weight_loader")
    assert state["loaded"] == {"plain"}
