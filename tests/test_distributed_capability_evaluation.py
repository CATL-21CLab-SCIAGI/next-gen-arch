"""Collective scheduling oracles and an opt-in two-GPU EP/FSDP smoke test."""

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from test_capability_evaluation import CharacterTokenizer, TinyCausalModel

from archlab.automodel.evaluate import continuation_scores, math_completion
from archlab.automodel.evaluate_distributed import evaluation_rounds, restore_distributed_adapters
from archlab.benchmarks.capability import EvaluationConfig


class CollectiveModel(TinyCausalModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        # A real model's FSDP/EP layers also demand one collective per forward.
        marker = torch.tensor(1.0)
        dist.all_reduce(marker)
        self.calls += 1
        return super().forward(*args, **kwargs)


def _collective_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        tokenizer = CharacterTokenizer()
        serial, collective = TinyCausalModel(), CollectiveModel()
        # A single-token shortcut on rank zero must not skip rank one's forwards.
        choices = ["A", "B"] if rank == 0 else [" A", " BB", " CCC"]
        expected = continuation_scores(serial, tokenizer, "Q:", choices, max_context=20, device="cpu")
        actual = continuation_scores(collective, tokenizer, "Q:", choices, max_context=20,
                                     device="cpu", synchronize=True)
        assert actual == expected
        assert collective.calls == 3
        serial.table.fill_(-10)
        serial.table[:, ord("z")] = 10
        collective.table.copy_(serial.table)
        config = EvaluationConfig(max_context=20, max_new_tokens=4)
        eos = {ord("z")} if rank == 0 else {ord("x")}
        expected = math_completion(serial, tokenizer, "Q", config=config, eos_ids=eos, device="cpu")
        collective.calls = 0
        actual = math_completion(collective, tokenizer, "Q", config=config, eos_ids=eos,
                                 device="cpu", synchronize=True)
        for key in ("completion", "generated_token_ids", "stop_reason", "new_tokens", "allowed_new_tokens"):
            assert actual[key] == expected[key], key
        assert collective.calls == 4
        # A nonfinite output on one rank must fail on every rank before any new forward.
        if rank == 1:
            collective.table.fill_(float("nan"))
        try:
            continuation_scores(collective, tokenizer, "Q:", ["A", "B"], max_context=20,
                                device="cpu", synchronize=True)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("nonfinite rank did not fail collectively")
    finally:
        dist.destroy_process_group()


class DistributedEvaluationTests(unittest.TestCase):
    def test_rounds_cover_each_example_once_without_mixing_mc_and_math(self):
        rows = [{"id": f"{task}:{i}", "task": task} for task, count in
                (("mmlu", 57), ("arc_challenge", 32), ("gsm8k", 8), ("aime24", 2), ("aime25", 2))
                for i in range(count)]
        rounds = evaluation_rounds(rows, 32)
        self.assertEqual([len(r) for r in rounds], [32, 25, 32, 12])
        self.assertEqual([r["id"] for group in rounds for r in group], [r["id"] for r in rows])
        for invalid, world in ((rows, 0), (rows + rows[:1], 32), ([{"id": "x", "task": "unknown"}], 2)):
            with self.assertRaises(ValueError):
                evaluation_rounds(invalid, world)

    def test_uneven_choices_eos_and_collective_nonfinite_failures(self):
        with tempfile.TemporaryDirectory(prefix="archlab-eval-collective-") as temporary:
            mp.spawn(_collective_worker, args=(f"file://{temporary}/rendezvous",), nprocs=2, join=True)

    def test_adapter_subset_rejects_mismatched_shapes(self):
        adapters = {"3": torch.nn.Linear(5, 7)}
        with tempfile.TemporaryDirectory(prefix="archlab-eval-weights-") as temporary:
            path = Path(temporary)
            expected = {name: {key: t.clone() for key, t in module.state_dict().items()}
                        for name, module in adapters.items()}
            dcp.save({"adapters": expected, "optimizer": {"unused": torch.ones(3)}}, checkpoint_id=path / "state")
            restore_distributed_adapters(adapters, path)
            for key, tensor in adapters["3"].state_dict().items():
                torch.testing.assert_close(tensor, expected["3"][key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                restore_distributed_adapters({"3": torch.nn.Linear(5, 8)}, path)


def distributed_gpu_smoke():
    """Exercise real EP/FSDP forwards and adapter subset restore alongside training."""
    from torch.distributed.fsdp import fully_shard

    from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig
    from archlab.automodel.checkpointing import state_digest
    from archlab.automodel.evaluate import added_modules
    from archlab.automodel.execution import build_frozen_base, tiny_config
    from archlab.automodel.runtime import configure_frozen_gdn_runtime
    from archlab.automodel.simplicial import install_simplicial_modules

    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer
    try:
        configure_frozen_gdn_runtime()
        torch.manual_seed(42)
        model, mesh, precision = build_frozen_base(tiny_config(), tiny=True, checkpoint=None,
                                                  ep_size=2, activation_checkpointing=False)
        config = SimplicialAdapterConfig(hidden_size=256, query_heads=4, kv_heads=2, head_dim=64,
                                         residual_streams=4, residual_low_rank=32, short_window=4, long_window=32,
                                         output_initialization="normal")
        adapters = install_simplicial_modules(model, config, device="cuda", dtype=torch.float32)
        for adapter in adapters.values():
            fully_shard(adapter, mesh=mesh.device_mesh["dp_shard_cp"], mp_policy=precision)
        model.requires_grad_(False)
        model.eval()
        payload = {"adapters": {name: module.state_dict() for name, module in adapters.items()}}
        before = state_digest(payload["adapters"])
        path = [tempfile.mkdtemp(prefix="archlab-eval-gpu-") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(path)
        dcp.save(payload, checkpoint_id=Path(path[0]) / "state")
        assert restore_distributed_adapters(adapters, Path(path[0])) == before
        tokenizer = CharacterTokenizer()
        choices = ["A", "B"] if dist.get_rank() == 0 else [" A", " BB", " CCC"]
        for enabled in (False, True, False):
            with added_modules(model, enabled=enabled):
                result = continuation_scores(model, tokenizer, "Q:", choices, max_context=32, synchronize=True)
                assert all(torch.isfinite(torch.tensor(result["loglikelihoods"])))
                generated = math_completion(model, tokenizer, "Q:",
                    config=EvaluationConfig(max_context=32, max_new_tokens=2), eos_ids={0}, synchronize=True)
                assert generated["new_tokens"] <= 2
        assert state_digest({n: m.state_dict() for n, m in adapters.items()}) == before
        print(json.dumps({"event": "distributed_evaluation_gpu_smoke_pass", "rank": dist.get_rank(),
                          "adapter_weights_unchanged": True, "ep_size": 2}), flush=True)
    finally:
        free_buffer()
        dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" in os.environ:
        distributed_gpu_smoke()
    else:
        unittest.main()
