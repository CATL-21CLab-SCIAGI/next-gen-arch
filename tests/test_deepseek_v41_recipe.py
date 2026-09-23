import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import yaml

from archlab.automodel.deepseek_v41_recipe import V41RecipeConfig
from archlab.automodel.deepseek_v41_training import (
    learning_rate,
    restore_adapter_checkpoint,
    save_adapter_checkpoint,
)


def recipe_config():
    path = Path(__file__).parents[1] / "recipes/experiments/deepseek_v41_simplicial_math_1b.yaml"
    return V41RecipeConfig(**yaml.safe_load(path.read_text()))


def test_recipe_preserves_the_selected_experiment():
    config = recipe_config()
    assert (config.world_size, config.ep_size, config.context, config.supervised_tokens) == (32, 8, 16384, 1_000_000_000)
    for change in ({"world_size": 8}, {"ep_size": 32}, {"context": 2048},
                   {"supervised_tokens": 10_000}, {"query_chunk": 0}):
        with pytest.raises(ValueError):
            replace(config, **change)


def test_unbound_qualification_fails_before_model_construction(tmp_path, monkeypatch):
    config = recipe_config()
    path = tmp_path / "leaf.json"
    path.write_text(json.dumps({"passed": True, "implementation_sha256": {}}))
    monkeypatch.setenv("ARCHLAB_DEEPSEEK_V41_LEAF_QUALIFICATION", str(path))
    monkeypatch.setenv("ARCHLAB_DEEPSEEK_V41_QUALIFICATION", str(tmp_path))
    # This is only an admission test; it does not simulate distributed math.
    monkeypatch.setattr(dist, "get_world_size", lambda: 32)
    with pytest.raises(ValueError, match="this implementation"):
        config.build()


def test_token_schedule_warmup_and_endpoints():
    assert learning_rate(0, 0, 0) == pytest.approx(1e-7)
    assert learning_rate(99, 1000, 0) == pytest.approx(1e-5)
    assert learning_rate(100, 1000, 1000) == pytest.approx(1e-5)
    assert learning_rate(1000, 1_000_000_000, 1000) == pytest.approx(1e-6)


def test_checkpoint_reloads_next_update_and_rejects_corruption(tmp_path, monkeypatch):
    # Real one-rank Gloo for checkpoint collectives. CUDA RNG access alone is
    # replaced by CPU RNG access to keep the serialization fixture CPU-only.
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", torch.set_rng_state)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        adapter = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01)
        inputs = torch.randn(4, 3)

        def step():
            optimizer.zero_grad(set_to_none=True)
            adapter(inputs).square().sum().backward()
            optimizer.step()

        step()
        path, contract, cursor = tmp_path / "checkpoint", {"source": "fixture"}, {"step": 1}
        save_adapter_checkpoint(path, {4: adapter}, [optimizer], cursor, contract)
        step()
        expected = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        assert restore_adapter_checkpoint(path, {4: adapter}, [optimizer], contract=contract) == cursor
        step()
        for key, value in adapter.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        target = path / "adapter-state.pt"
        value = bytearray(target.read_bytes())
        value[-1] ^= 1
        target.write_bytes(value)
        with pytest.raises(ValueError, match="checksum"):
            restore_adapter_checkpoint(path, {4: adapter}, [optimizer], contract=contract)
    finally:
        dist.destroy_process_group()
