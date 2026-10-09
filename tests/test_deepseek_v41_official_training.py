import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from archlab.automodel.deepseek_v41_loss import frozen_head_loss
from archlab.automodel.deepseek_v41_official_training import (
    OfficialV41TrainingSession,
    _cpu_clone,
    frozen_head_unsharded,
    optimizer_step,
)
from archlab.automodel.deepseek_v41_training import (
    restore_adapter_checkpoint,
    save_adapter_checkpoint,
)


def test_frozen_head_loss_matches_dense_ce_without_a_second_label_shift():
    torch.manual_seed(23)
    head = nn.Linear(7, 19, bias=False).requires_grad_(False)
    hidden = torch.randn(1, 9, 7, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    labels = torch.randint(19, (1, 9))
    labels[0, [0, 2, 8]] = -100
    with frozen_head_unsharded(head):
        actual = frozen_head_loss(hidden, labels, head, chunk_size=2)
        actual.backward()
    expected = F.cross_entropy(head(expected_hidden).flatten(0, 1), labels.flatten(), reduction="sum")
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    assert head.weight.grad is None


@pytest.mark.parametrize("trainable,dtype", [(True, torch.float32), (False, torch.bfloat16)])
def test_head_contract_rejects_trainable_or_non_native_precision(trainable, dtype):
    head = nn.Linear(7, 19, bias=False, dtype=dtype).requires_grad_(trainable)
    with pytest.raises(ValueError, match="frozen FP32"):
        with frozen_head_unsharded(head):
            pass


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(7, 7, bias=False)

    def forward(self, hidden):
        hidden = self.base(hidden)
        if hasattr(self, "adapter"):
            hidden = hidden + self.adapter(F.dropout(hidden, .2, training=self.training))
        return hidden


class _TinyOfficialInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(19, 7)
        self.block = _TinyBlock()
        self.lm_head = nn.Linear(7, 19, bias=False)

    def forward(self, input_ids, *, return_hidden_states=False):
        assert return_hidden_states
        return SimpleNamespace(hidden_states=self.block(self.embed(input_ids)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 lifecycle requires CUDA")
def test_fsdp_head_lifetime_and_adapter_checkpoint_next_update(tmp_path, monkeypatch):
    """Real FSDP2 hooks, direct-weight CE and fresh replicated adapter masters.

    One GPU checks the sharded/unsharded lifecycle and checkpoint replay. The
    launch qualification separately checks actual multi-rank numerical behavior.
    """
    import archlab.automodel.deepseek_v41_official_training as training

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0,
                            world_size=1, device_id=torch.device("cuda", 0))
    try:
        torch.manual_seed(491)
        model = _TinyOfficialInterface().cuda().requires_grad_(False)
        # Simulate loading the base before any trainable parameter is inserted.
        loaded = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(loaded)
        fully_shard(model.embed)
        fully_shard(model.block)
        fully_shard(model.lm_head)
        fully_shard(model)
        adapter = nn.Linear(7, 7, bias=False, device="cuda", dtype=torch.float32)
        model.block.adapter = adapter
        assert isinstance(model.lm_head.weight, DTensor)
        assert not isinstance(adapter.weight, DTensor)
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(adapter.weight)}
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01)
        inputs = torch.randint(19, (1, 11), device="cuda")
        labels = torch.randint(19, (1, 11), device="cuda")
        labels[0, [0, 4, 10]] = -100
        lifecycle = []
        original_loss = training.frozen_head_loss

        def checked_loss(hidden, targets, head):
            assert not isinstance(head.weight, DTensor)
            assert head.weight.shape == (19, 7)
            lifecycle.append("loss")
            # This hook runs after the custom loss has used the head in backward.
            def after_loss_backward(gradient):
                lifecycle.append("backward")
                return gradient
            hidden.register_hook(after_loss_backward)
            return original_loss(hidden, targets, head, chunk_size=3)

        monkeypatch.setattr(training, "frozen_head_loss", checked_loss)

        def step():
            metric = optimizer_step(model, [optimizer], inputs, labels, learning_rate=.01)
            assert isinstance(model.lm_head.weight, DTensor)
            assert metric["supervised_tokens"] == 8
            assert metric["gradient_norm_before_clip"] > 0
            assert torch.isfinite(adapter.weight).all()
            return metric

        step()
        path, contract, cursor = tmp_path / "checkpoint", {"backend": "official-fixture"}, {"step": 1}
        save_adapter_checkpoint(path, {4: adapter}, [optimizer], cursor, contract)
        expected_metric = step()
        expected_adapter = _cpu_clone(adapter.state_dict())
        expected_optimizer = _cpu_clone(optimizer.state_dict())
        assert restore_adapter_checkpoint(path, {4: adapter}, [optimizer], contract=contract) == cursor
        actual_metric = step()
        assert actual_metric["loss"] == expected_metric["loss"]
        torch.testing.assert_close(_cpu_clone(adapter.state_dict()), expected_adapter, rtol=0, atol=0)
        torch.testing.assert_close(_cpu_clone(optimizer.state_dict()), expected_optimizer, rtol=0, atol=0)
        assert lifecycle == ["loss", "backward"] * 3
        for name, module in (("embed", model.embed), ("block", model.block), ("lm_head", model.lm_head)):
            module.unshard()
            try:
                for key, parameter in module.named_parameters():
                    if "adapter" not in key:
                        assert parameter.grad is None
                        torch.testing.assert_close(parameter, loaded[f"{name}.{key}"], rtol=0, atol=0)
            finally:
                module.reshard()
        # Error paths must also release the head's all-gather allocation.
        with pytest.raises(RuntimeError, match="fixture failure"):
            with frozen_head_unsharded(model.lm_head):
                raise RuntimeError("fixture failure")
        assert isinstance(model.lm_head.weight, DTensor)
    finally:
        dist.destroy_process_group()


class _TinyPilot:
    order_seed = 2234

    def __init__(self, targets):
        self.windows = [{"targets": count, "mode": "medium"} for count in targets]
        self.manifest = {"supervised_tokens": sum(targets)}
        self.accesses = []

    def __len__(self):
        return len(self.windows)

    def batch(self, index, *, device, smoke_context=None, pad_to_full=False):
        self.accesses.append((index, smoke_context, pad_to_full))
        context = smoke_context or 16
        inputs = (torch.arange(context, device=device) % 19).unsqueeze(0)
        labels = torch.full_like(inputs, -100)
        count = self.windows[index]["targets"] if index < len(self) else 0
        labels[0, :count] = inputs[0, 1:count + 1]
        return inputs, labels, count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 lifecycle requires CUDA")
def test_qualification_restores_fresh_state_and_run_consumes_exact_windows(tmp_path, monkeypatch):
    import archlab.automodel.deepseek_v41_official_training as training

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0,
                            world_size=1, device_id=torch.device("cuda", 0))
    try:
        torch.manual_seed(521)
        model = _TinyOfficialInterface().cuda().requires_grad_(False)
        fully_shard(model.embed)
        fully_shard(model.block)
        fully_shard(model.lm_head)
        fully_shard(model)
        adapter = nn.Linear(7, 7, bias=False, device="cuda", dtype=torch.float32)
        model.block.adapter = adapter
        # Only the tiny fixture substitutes its optimizer geometry. Production
        # keeps the reviewed exhaustive HeadwiseMuon/AdamW partition unchanged.
        monkeypatch.setattr(training, "adapter_optimizers", lambda model, adapters: [
            torch.optim.AdamW(adapter.parameters(), lr=1e-7)
        ])
        config = SimpleNamespace(world_size=1, context=16, supervised_tokens=9,
                                 warmup_steps=100, validation_interval_tokens=5,
                                 checkpoint_interval_tokens=5)
        train, validation = _TinyPilot([4, 5]), _TinyPilot([3])
        session = OfficialV41TrainingSession(model, {4: adapter}, train, validation,
                                             tmp_path, {"backend": "session-fixture"}, config)
        initial = _cpu_clone(adapter.state_dict())
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        report = session.qualify_mesh()
        assert report["passed"] and report["checkpoint_next_update_replay"]
        assert len(report["short_updates"]) == 2
        assert report["full_context_update"]["input_tokens"] == 16
        torch.testing.assert_close(adapter.state_dict(), {key: value.cuda() for key, value in initial.items()},
                                   rtol=0, atol=0)
        assert all(not optimizer.state for optimizer in session.optimizers)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
        accesses = len(train.accesses)
        assert session.qualify_mesh() is report
        assert len(train.accesses) == accesses
        session.run()
        assert train.accesses[accesses:] == [(0, None, False), (1, None, False)]
        metrics = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
        assert [metric["consumed_supervised_tokens"] for metric in metrics] == [4, 9]
        assert [metric["learning_rate"] for metric in metrics] == pytest.approx([1e-7, 2e-7])
        complete = json.loads((tmp_path / "TRAINING_COMPLETE.json").read_text())
        assert complete == {"steps": 2, "supervised_tokens": 9}
        marker = json.loads((tmp_path / "checkpoints/step-000002/COMPLETE.json").read_text())
        assert marker["cursor"] == {"step": 2, "supervised_tokens": 9, "warmup_tokens": 9}
        validation_metrics = [json.loads(line) for line in (tmp_path / "validation.jsonl").read_text().splitlines()]
        assert [metric["supervised_tokens"] for metric in validation_metrics] == [3, 3]
    finally:
        dist.destroy_process_group()
