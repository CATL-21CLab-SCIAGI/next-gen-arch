import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from archlab.rl.rollout import sample_rollouts


class ToyPolicy(torch.nn.Module):
    def __init__(self, *, flat=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.lm_head = torch.nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(8))
            self.lm_head.weight.zero_()
            if flat:
                # Uniform over seven tokens, with a finite but unsampleable EOS.
                self.lm_head.weight[1, :] = -1000
            else:
                # 3 -> stop immediately; 4 -> 5 -> 6 -> stop.
                for source, target in [(3, 1), (4, 5), (5, 6), (6, 1), (7, 7)]:
                    self.lm_head.weight[target, source] = 100
        self.calls = []

    def forward(self, *, input_ids, attention_mask, return_hidden_states):
        assert return_hidden_states and attention_mask.dtype == torch.bool
        assert not (attention_mask[:, 1:] & ~attention_mask[:, :-1]).any()
        assert not torch.is_grad_enabled() and not self.training
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        return SimpleNamespace(hidden_states=self.embedding(input_ids))


def sample(model, prompts, **kwargs):
    options = dict(policy_version="checkpoint-sha:123/update:0", max_new_tokens=4,
                   context_limit=16, eos_token_ids={1}, pad_token_id=2,
                   seed=23, bucket_multiple=4)
    options.update(kwargs)
    return sample_rollouts(model, prompts, **options)


def test_variable_prompts_append_without_gaps_stop_and_align_scores():
    model = ToyPolicy()
    model.embedding.eval()  # Mixed module modes must be preserved.
    batch = sample(model, [[0, 3], [0, 7, 7, 4]], prompt_group_ids=["a", "b"])
    assert batch.generated_ids == [[1], [5, 6, 1]]
    assert batch.finish_reasons == ["stop", "stop"]
    assert batch.receipt["forward_count"] == 3
    assert model.training and not model.embedding.training
    assert batch.input_ids[0, :3].tolist() == [0, 3, 1]
    assert batch.input_ids[1, :7].tolist() == [0, 7, 7, 4, 5, 6, 1]
    assert batch.labels[0, :3].tolist() == [-100, 1, -100]
    assert batch.labels[1, 3:6].tolist() == [5, 6, 1]
    assert torch.equal(batch.response_mask, batch.labels != -100)
    assert torch.count_nonzero(batch.policy_log_probs[~batch.response_mask]) == 0
    dense = model.lm_head(model.embedding(batch.input_ids)).log_softmax(-1)
    selected = dense.gather(-1, batch.labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(batch.policy_log_probs[batch.response_mask], selected[batch.response_mask])
    torch.testing.assert_close(batch.policy_log_probs, batch.behavior_log_probs)
    # On the next forward, new tokens immediately occupy old padding positions.
    assert model.calls[1][0][0, :3].tolist() == [0, 3, 1]
    assert model.calls[1][0][1, :5].tolist() == [0, 7, 7, 4, 5]


def test_length_truncation_has_no_invented_stop_and_seed_is_private():
    model = ToyPolicy(flat=True)
    torch.manual_seed(541)
    previous_rng = torch.get_rng_state().clone()
    batch = sample(model, [[0, 4], [0, 4]], max_new_tokens=8)
    again = sample(model, [[0, 4], [0, 4]], max_new_tokens=8)
    assert torch.equal(torch.get_rng_state(), previous_rng)
    assert batch.generated_ids == again.generated_ids
    assert batch.generated_ids[0] != batch.generated_ids[1]
    assert batch.finish_reasons == ["length", "length"]
    assert batch.receipt["generated_tokens"] == 16
    assert batch.receipt["cached"] is False
    assert batch.receipt["on_policy_sampling"] is True
    assert torch.allclose(batch.policy_log_probs[batch.response_mask], torch.full((16,), -torch.log(torch.tensor(7.)).item()))


def test_top_p_and_temperature_record_actual_behavior_probability():
    model = ToyPolicy(flat=True)
    batch = sample(model, [[0, 4]], temperature=.7,
                   top_p=.1, max_new_tokens=2)
    assert batch.receipt["on_policy_sampling"] is False
    torch.testing.assert_close(batch.behavior_log_probs[batch.response_mask], torch.zeros(2))
    assert (batch.policy_log_probs[batch.response_mask] < -1.9).all()


def test_invalid_inputs_and_modes_restore_on_failure():
    model = ToyPolicy()
    for overrides in [dict(max_new_tokens=0), dict(temperature=-1), dict(eos_token_ids=set()),
                      dict(context_limit=4), dict(prompt_group_ids=[]), dict(policy_version=""),
                      dict(eos_token_ids={999}), dict(pad_token_id=999), dict(temperature=0, top_p=.8)]:
        with pytest.raises(ValueError):
            sample(model, [[0, 4]], **overrides)
    with torch.no_grad():
        model.lm_head.weight.fill_(torch.nan)
    with pytest.raises(FloatingPointError):
        sample(model, [[0, 4]])
    assert model.training


def test_temperature_zero_is_greedy_with_actual_policy_scores():
    model = ToyPolicy(flat=True)
    first = sample(model, [[0, 4]], temperature=0, max_new_tokens=3)
    second = sample(model, [[0, 4]], temperature=0, max_new_tokens=3, seed=777)
    assert first.generated_ids == second.generated_ids == [[0, 0, 0]]
    assert torch.count_nonzero(first.behavior_log_probs) == 0
    assert (first.policy_log_probs[first.response_mask] < -1.9).all()
    assert first.receipt["on_policy_sampling"] is False


def test_retention_is_scoped_uses_head_callback_and_preserves_sample_math(monkeypatch):
    model = ToyPolicy(flat=True)
    reference = sample(model, [[0, 4], [0, 4]])
    events = []

    @contextmanager
    def retained(actor, minimum_free_gib):
        assert actor is model and minimum_free_gib == 16
        receipt = {"cleanup_verified": False}

        @contextmanager
        def head_context(head):
            events.append("head")
            yield head

        events.append("enter")
        try:
            yield SimpleNamespace(inference_head=head_context, receipt=receipt)
        finally:
            receipt["cleanup_verified"] = True
            events.append("exit")

    monkeypatch.setattr("archlab.rl.weight_residency.retained_fsdp_weights", retained)
    model._archlab_rl_retain_weights = True
    actual = sample(model, [[0, 4], [0, 4]])
    assert actual.generated_ids == reference.generated_ids
    torch.testing.assert_close(actual.policy_log_probs, reference.policy_log_probs, rtol=0, atol=0)
    torch.testing.assert_close(actual.behavior_log_probs, reference.behavior_log_probs, rtol=0, atol=0)
    assert events == ["enter", *["head"] * actual.receipt["forward_count"], "exit"]
    assert actual.receipt["retained_weights"] is True
    assert actual.receipt["weight_residency"]["cleanup_verified"] is True
    assert actual.receipt["backend"] == "resident-model-full-prefix-retained-weights"
    previous = list(events)
    baseline = sample(model, [[0, 4]], retain_weights=False)
    assert events == previous and baseline.receipt["retained_weights"] is False


def test_retention_cleanup_precedes_error_propagation_and_mode_restoration(monkeypatch):
    model = ToyPolicy()
    model.embedding.eval()
    model._archlab_rl_retain_weights = True
    events = []

    @contextmanager
    def retained(actor, minimum_free_gib):
        @contextmanager
        def fail_head(head):
            raise RuntimeError("injected projection failure")
            yield head

        try:
            yield SimpleNamespace(inference_head=fail_head, receipt={})
        finally:
            events.append(("cleanup", actor.training))

    monkeypatch.setattr("archlab.rl.weight_residency.retained_fsdp_weights", retained)
    with pytest.raises(RuntimeError, match="injected projection"):
        sample(model, [[0, 4]])
    assert events == [("cleanup", False)]
    assert model.training and not model.embedding.training


def _distributed_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = ToyPolicy()
        prompt = [[0, 3]] if rank == 0 else [[0, 7, 7, 4]]
        batch = sample(model, prompt)
        Path(output_dir, f"{rank}.json").write_text(json.dumps({
            "shapes": batch.receipt["forward_shapes"], "generated": batch.generated_ids,
            "receipt": batch.receipt, "mask": batch.attention_mask.tolist(),
        }))
    finally:
        dist.destroy_process_group()


def test_two_ranks_keep_forward_counts_and_buckets_after_one_finishes(tmp_path):
    # Real CPU collectives exercise unequal prompt lengths and early termination.
    import torch.multiprocessing as mp

    mp.spawn(_distributed_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2)
    first, second = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    assert first["generated"] == [[1]] and second["generated"] == [[5, 6, 1]]
    assert first["shapes"] == second["shapes"] == [[1, 4], [1, 8], [1, 8]]
    assert first["receipt"]["effective_rank_seed"] + 1 == second["receipt"]["effective_rank_seed"]
