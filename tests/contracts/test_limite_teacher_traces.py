"""Teacher-trace prompt recovery, grading and assistant-only row selection."""

import json
from argparse import Namespace

import pytest

from archlab.preprocessing import limite_teacher_traces as traces

PROMPT = ("<|endoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\nFind 2+3.\n\nGive the answer in \\boxed{} format.<|im_end|>\n"
          "<|im_start|>assistant\n")


def record(**changes):
    base = dict(split="train", uuid="u", problem_sha256="p", sample=0, prompt_ids=[1, 2, 3],
                response_ids=[7, 8, traces.TEACHER_EOS], correct=True, reasoning_complete=True,
                natural_eos=True, stop_token=traces.TEACHER_EOS)
    return {**base, **changes}


def test_question_is_the_single_user_turn():
    assert traces.question(PROMPT) == "Find 2+3.\n\nGive the answer in \\boxed{} format."
    with pytest.raises(ValueError, match="one user turn"):
        traces.question(PROMPT + PROMPT)
    with pytest.raises(ValueError, match="one user turn"):
        traces.question(PROMPT.removesuffix("assistant\n"))


def test_grade_requires_a_natural_stop_token_for_eos():
    graded = traces.grade("<think>2+3</think>\nThe answer is \\boxed{5}.", "5", "stop", traces.GENERATION_EOS)
    assert graded == dict(correct=True, reasoning_complete=True, natural_eos=True,
                          stop_token=traces.GENERATION_EOS)
    assert not traces.grade("<think>2+3", "5", "length", 11)["natural_eos"]
    assert not traces.grade("<think>2+3 \\boxed{5}", "5", "stop", traces.GENERATION_EOS)["correct"]


def test_supervised_rows_mask_only_the_teacher_response_including_eos():
    row = traces.supervised(record(), max_length=6)
    assert row["input_ids"] == [1, 2, 3, 7, 8, traces.TEACHER_EOS]
    assert row["loss_mask"] == [0, 0, 0, 1, 1, 1]
    assert row["position_ids"] == list(range(6)) and row["attention_mask"] == [1] * 6
    for rejected in (dict(correct=False), dict(reasoning_complete=False), dict(natural_eos=False),
                     dict(stop_token=traces.GENERATION_EOS)):
        assert traces.supervised(record(**rejected), max_length=6) is None
    assert traces.supervised(record(), max_length=5) is None


def test_prepare_selects_verified_rows_and_rejects_changed_shards(tmp_path):
    pytest.importorskip("pyarrow")
    shards = tmp_path / "traces"
    shards.mkdir()
    for split, uuid, problem in (("train", "a", "pa"), ("heldout", "b", "pb")):
        path = shards / f"{split}-shard-000-of-001.jsonl"
        rows = [record(split=split, uuid=uuid, problem_sha256=problem, sample=i, correct=i == 0)
                for i in range(2)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        traces.atomic_write_json(path.with_suffix(".complete.json"),
                                 dict(file=path.name, file_sha256=traces.sha256_file(path)))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tokenizer / name).write_text(name)
    manifest = traces.prepare(Namespace(traces=shards, tokenizer=tokenizer,
                                        output=tmp_path / "sft", max_length=16))
    assert manifest["files"]["train"]["rows"] == manifest["files"]["validation"]["rows"] == 1
    assert manifest["files"]["train"]["supervised_tokens"] == 3
    assert manifest["tokenization"]["loss_mask_alignment"] == "unshifted-target-token"
    assert manifest["tokenization"]["generation_eos_token_id"] == traces.TEACHER_EOS
    (shards / "train-shard-000-of-001.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="changed after completion"):
        traces.prepare(Namespace(traces=shards, tokenizer=tokenizer,
                                 output=tmp_path / "sft-2", max_length=16))
