import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist

from archlab.rl.evaluation import evaluate_policy


class NumericPolicy(torch.nn.Module):
    def __init__(self, *, consume_rng=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 9)
        self.lm_head = torch.nn.Linear(9, 9, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(9))
            self.lm_head.weight.zero_()
            for source, target in [(3, 4), (4, 1), (5, 6), (6, 1), (7, 7)]:
                self.lm_head.weight[target, source] = 100
        self.consume_rng = consume_rng
        self.inputs = []

    def forward(self, *, input_ids, attention_mask, return_hidden_states):
        assert not torch.is_grad_enabled() and not self.training
        assert return_hidden_states
        assert not (attention_mask[:, 1:] & ~attention_mask[:, :-1]).any()
        self.inputs.append(input_ids.clone())
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        return SimpleNamespace(hidden_states=self.embedding(input_ids))


class NumericTokenizer:
    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        assert 1 not in ids  # Only terminal EOS is explicitly removed before decoding.
        return "".join({4: "42", 6: "41", 7: "?"}.get(token, "!") for token in ids)


def examples():
    return [
        {"problem_id": "correct", "prompt_ids": [0, 3], "expected_answer": "42"},
        {"problem_id": "wrong", "prompt_ids": [0, 5], "expected_answer": "42"},
        {"problem_id": "invalid", "prompt_ids": [0, 7], "expected_answer": "42"},
        {"problem_id": "outside-prefix", "prompt_ids": [0, 3], "expected_answer": "41"},
    ]


def evaluate(model, rows=None, **kwargs):
    options = dict(policy_version="matched-checkpoint/update-3", max_new_tokens=3,
                   context_limit=8, eos_token_ids={1}, pad_token_id=2,
                   eval_count=3, local_batch_size=2, seed=19)
    options.update(kwargs)
    return evaluate_policy(model, NumericTokenizer(), examples() if rows is None else rows, **options)


def test_exact_prefix_dummy_rows_and_actual_generated_scores():
    model = NumericPolicy()
    summary, records = evaluate(model)
    assert summary["count"] == 3 and summary["correct"] == 1
    assert summary["pass_at_1"] == pytest.approx(1 / 3)
    assert summary["valid_answer_rate"] == pytest.approx(2 / 3)
    assert summary["truncation_rate"] == pytest.approx(1 / 3)
    assert summary["dummy_rows_excluded"] == 1
    assert summary["rollout_batches"] == 2
    assert [row["problem_id"] for row in records] == ["correct", "wrong", "invalid"]
    assert [row["completion"] for row in records] == ["42", "41", "???"]
    assert [row["generated_ids"] for row in records] == [[4, 1], [6, 1], [7, 7, 7]]
    assert all(row["policy_version"] == summary["policy_version"] for row in records)
    assert all(receipt["temperature"] == 0 and receipt["top_p"] == 1
               for receipt in summary["rollout_receipts_by_rank"][0])


def test_references_are_not_fed_to_model_and_digest_detects_gold_changes():
    first_model, second_model = NumericPolicy(), NumericPolicy()
    rows = examples()
    first, first_records = evaluate(first_model, rows)
    for row in rows:
        row["expected_answer"] = "41"
    second, second_records = evaluate(second_model, rows)
    assert [r["generated_ids"] for r in first_records] == [r["generated_ids"] for r in second_records]
    for first_ids, second_ids in zip(first_model.inputs, second_model.inputs, strict=True):
        assert torch.equal(first_ids, second_ids)
    assert first["prompt_split_digest"] == second["prompt_split_digest"]
    assert first["reference_digest"] != second["reference_digest"]
    assert first_records[0]["correct"] and not second_records[0]["correct"]
    assert not first_records[1]["correct"] and second_records[1]["correct"]


def test_rng_mixed_modes_weights_and_existing_gradients_are_preserved():
    model = NumericPolicy(consume_rng=True)
    model.embedding.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = [(parameter.clone(), parameter.grad.clone()) for parameter in model.parameters()]
    random.seed(229)
    np.random.seed(230)
    torch.manual_seed(231)
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    evaluate(model)
    assert random.getstate() == state[0]
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == state[1][0]
    assert np.array_equal(actual_numpy[1], state[1][1])
    assert actual_numpy[2:] == state[1][2:]
    assert torch.equal(torch.get_rng_state(), state[2])
    assert model.training and not model.embedding.training
    for parameter, (weight, gradient) in zip(model.parameters(), before, strict=True):
        assert torch.equal(parameter, weight) and torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize("count", [0, 5, -1])
def test_budget_failures_do_not_silently_change_count(count):
    model = NumericPolicy()
    with pytest.raises(ValueError, match="exact prefix"):
        evaluate(model, eval_count=count)
    assert not model.inputs


def test_missing_unsupported_or_duplicate_gold_is_rejected_before_generation():
    for mutation in ("missing", "unsupported", "duplicate", "over-context"):
        rows, model = examples(), NumericPolicy()
        if mutation == "missing":
            del rows[0]["expected_answer"]
        elif mutation == "unsupported":
            rows[0]["expected_answer"] = "x + y"
        elif mutation == "duplicate":
            rows[1]["problem_id"] = rows[0]["problem_id"]
        else:
            rows[0]["prompt_ids"] = [0] * 8
        with pytest.raises(ValueError):
            evaluate(model, rows)
        assert not model.inputs


def _worker(rank, init_file, directory, count):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = NumericPolicy()
        summary, records = evaluate(model, eval_count=count)
        Path(directory, f"rank{rank}.json").write_text(json.dumps({
            "summary": summary, "records": records, "forward_shapes": [list(t.shape) for t in model.inputs],
        }))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("count", [1, 3])
def test_distributed_exact_n_even_with_whole_dummy_rank(tmp_path, count):
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(str(tmp_path / "init"), str(tmp_path), count), nprocs=2)
    first, second = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    assert first["summary"] == second["summary"]
    assert first["summary"]["count"] == count
    assert first["summary"]["dummy_rows_excluded"] == 4 - count
    assert [r["problem_id"] for r in first["records"]] == [r["problem_id"] for r in examples()[:count]]
    assert len(second["records"]) == max(0, count - 2)
    assert first["forward_shapes"] == second["forward_shapes"]
