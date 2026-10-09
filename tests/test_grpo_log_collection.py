"""GRPO log transport preserves text and ordering without mutating upstream."""

import importlib.util
import json
import sys
from datetime import timedelta
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from archlab.rl.grpo_log_collection import bind_host_object_gather
from archlab.rl.rollout_rendezvous import RolloutRendezvous
from archlab.rl.rollout_rendezvous_probe import qualify_checkpoint_objects, qualify_object_logs


def gather_object(value):
    return ["default transport", *value]


def _profile(function):
    @wraps(function)
    def wrapper(self, *args, **kwargs):
        self.calls.append("profile start")
        result = function(self, *args, **kwargs)
        self.calls.append("profile end")
        return result

    return wrapper


@_profile
def _generate(self, inputs, *, suffix="done"):
    self.calls.append("generation")
    result = {key: gather_object(inputs[key]) for key in sorted(inputs)}
    self.calls.append(suffix)
    return result


class _Upstream:
    _generate_and_score_completions = _generate


def _behavior_class(monkeypatch):
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(GRPOTrainer=_Upstream))
    path = Path(__file__).parents[1] / "src/archlab/rl/limite_trainer.py"
    spec = importlib.util.spec_from_file_location("_limite_host_logs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BehaviorGRPO


def test_owned_binding_retains_upstream_code_decorators_and_globals():
    original = _generate.__wrapped__
    namespace = dict(original.__globals__)

    def collector(value):
        return ["host transport", *value]

    bound = bind_host_object_gather(_generate, collector)
    trainer = SimpleNamespace(calls=[])
    result = bound(trainer, {"completion": ["数学 — café"]}, suffix="scored")
    assert result == {"completion": ["host transport", "数学 — café"]}
    assert trainer.calls == ["profile start", "generation", "scored", "profile end"]
    assert bound.__code__ is _generate.__code__
    assert bound.__wrapped__.__code__ is original.__code__
    assert bound.__wrapped__.__globals__ is not original.__globals__
    assert original.__globals__.keys() == namespace.keys()
    assert all(original.__globals__[key] is value for key, value in namespace.items())
    trainer.calls.clear()
    assert _generate(trainer, {"completion": ["original"]}) == {
        "completion": ["default transport", "original"]
    }


def test_behavior_binding_is_instance_local_and_preserves_default_path(monkeypatch):
    behavior = _behavior_class(monkeypatch)
    host, default = behavior(), behavior()
    host.calls, default.calls = [], []
    host.archlab_rollout_rendezvous = SimpleNamespace(
        gather_object=lambda value: ["host transport", *value]
    )
    assert host._generate_and_score_completions({"completion": ["one"]}) == {
        "completion": ["host transport", "one"]
    }
    first_binding = host._archlab_host_generation
    assert host._generate_and_score_completions({"completion": ["two"]}) == {
        "completion": ["host transport", "two"]
    }
    assert host._archlab_host_generation is first_binding
    assert default._generate_and_score_completions({"completion": ["three"]}) == {
        "completion": ["default transport", "three"]
    }
    assert not hasattr(default, "_archlab_host_generation")


def test_unrecognized_upstream_dependency_is_rejected():
    def unrelated(self, inputs):
        return inputs

    with pytest.raises(RuntimeError, match="object-gather dependency"):
        bind_host_object_gather(unrelated, list)


def test_dependency_in_nested_code_is_bound_on_every_python_version():
    def nested(self, inputs):
        def collect():
            return gather_object(inputs)

        return collect()

    assert "gather_object" not in nested.__code__.co_names
    bound = bind_host_object_gather(nested, lambda value: ["host transport", *value])
    assert bound(None, ["nested completion"]) == ["host transport", "nested completion"]
    assert nested(None, ["original"]) == ["default transport", "original"]


def test_opaque_decorator_is_rejected():
    def opaque(self, inputs):
        return _generate(self, inputs)

    opaque.__wrapped__ = _generate
    with pytest.raises(RuntimeError, match="unsupported decorator"):
        bind_host_object_gather(opaque, list)


def _distributed_host_log_worker(rank, rendezvous_file, output):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        rendezvous = RolloutRendezvous(2)
        assert dist.get_backend(rendezvous.group) == "gloo"
        trainer = SimpleNamespace(calls=[])
        bound = bind_host_object_gather(_generate, rendezvous.gather_object)
        first = "数学 🧮\n" * 10000
        second = "café\n" * 12000
        third = "終わり"
        local = {
            "prompt": [f"rank {rank}"],
            "completion": [first] if rank == 0 else [second, third],
            "extra": [] if rank == 0 else [{"reason": "natural_eos"}],
        }
        rendezvous()
        result = bound(trainer, local)
        assert result == {
            "prompt": ["rank 0", "rank 1"],
            "completion": [first, second, third],
            "extra": [{"reason": "natural_eos"}],
        }
        assert trainer.calls == ["profile start", "generation", "done", "profile end"]
        # A second gather verifies the previous large/unequal payload cannot
        # leave stale object buffers or change rank order.
        assert rendezvous.gather_object([rank]) == [0, 1]
        for iteration in range(3):
            assert qualify_object_logs(rendezvous, rank, 2, iteration) > 48680
            with patch("torch.cuda.is_available", return_value=False):
                assert qualify_checkpoint_objects(rendezvous, rank, 2)["passed"]
        (Path(output) / f"rank-{rank}.json").write_text(
            json.dumps({"completed": True, "characters": len(first) + len(second) + len(third)})
        )
    finally:
        dist.destroy_process_group()


def test_real_gloo_gathers_large_unequal_unicode_completion_logs(tmp_path):
    mp.spawn(
        _distributed_host_log_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["completed"] and result["characters"] > 48680
