from pathlib import Path

import numpy as np
import pytest
import torch

from archlab.architectures.qwen38_flash_next import Qwen38FlashNextConfig
from archlab.megatron.qwen38_muon import (
    FROBENIUS_EPSILON,
    POLAR_EXPRESS_COEFFICIENTS,
    _canonical_optimizer_step,
    _combine_grad_norms,
    _filter_and_reorder_optimizer_groups,
    _validate_local_matrix_metadata,
    polar_express_zeroth_power,
)
from archlab.megatron.qwen38_train import (
    BinaryTokenBatches,
    _data_prefixes,
    _loss_func,
    _megatron_argv,
    _parser,
    _partition_prefixes,
)


def _part(root: Path, split: str, index: int, tokens: list[int]) -> Path:
    prefix = root / split / f"part-{index:05d}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype=np.int32).tofile(f"{prefix}.bin")
    Path(f"{prefix}.idx").write_bytes(b"idx")
    Path(f"{prefix}.json").write_text("{}")
    return prefix


def test_binary_batches_shift_labels_and_wrap(tmp_path):
    prefix = _part(tmp_path, "train", 0, list(range(10)))
    batches = BinaryTokenBatches(
        [prefix],
        batch_size=1,
        sequence_len=4,
        start_batch=2,
        device=torch.device("cpu"),
    )

    first = next(batches)
    second = next(batches)
    assert first["tokens"].tolist() == [[8, 9, 0, 1]]
    assert first["labels"].tolist() == [[9, 0, 1, 2]]
    assert second["tokens"].tolist() == [[2, 3, 4, 5]]


def test_prefix_validation_and_rank_partition(tmp_path):
    prefixes = [_part(tmp_path, "train", i, list(range(8))) for i in range(4)]
    assert _data_prefixes(tmp_path, "train") == prefixes
    assert _partition_prefixes(prefixes, 1, 2) == [prefixes[1], prefixes[3]]
    assert _partition_prefixes(prefixes, 7, 8) == [prefixes[3]]


def test_qwen38_training_defaults_to_bf16(tmp_path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.precision == "bf16"
    assert args.gdn_kernel == "fla"
    assert args.freeze_ngram_tables is False


def test_qwen38_training_accepts_frozen_ngram_control(tmp_path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--freeze-ngram-tables",
        ]
    )

    assert args.freeze_ngram_tables is True


def test_megatron_argv_uses_exact_distributed_muon_and_speedups(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    argv = _megatron_argv(args, Qwen38FlashNextConfig())

    expected_pairs = {
        "--optimizer": "muon",
        "--muon-momentum": "0.95",
        "--muon-scale-mode": "spectral",
        "--muon-extra-scale-factor": "0.2",
        "--muon-fp32-matmul-prec": "medium",
        "--muon-coefficient-type": "polar_express",
        "--muon-num-ns-steps": "8",
        "--muon-scalar-optimizer": "adam",
        "--seed": "42",
    }
    for flag, value in expected_pairs.items():
        assert argv[argv.index(flag) + 1] == value
    for flag in (
        "--muon-nesterov",
        "--muon-no-split-qkv",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--ddp-pad-buckets-for-high-nccl-busbw",
        "--optimizer-cuda-graph",
    ):
        assert flag in argv
    assert "--no-gradient-accumulation-fusion" not in argv


def test_loss_func_reports_expert_and_component_metrics():
    output = torch.tensor([1.0, 3.0], requires_grad=True)
    components = {
        "cross entropy": torch.tensor(1.5),
        "expert balance loss": torch.tensor(0.002),
    }

    loss, report = _loss_func(output, components)

    assert loss.item() == 2.0
    assert set(report) == {"lm loss", "cross entropy", "expert balance loss"}
    assert torch.allclose(report["expert balance loss"], torch.tensor([0.002, 1.0]))


def test_polar_express_kernel_is_batched_finite_and_exactly_eight_steps():
    torch.manual_seed(23)
    matrices = torch.randn(2, 4, 8)
    batched = polar_express_zeroth_power(matrices, use_bfloat16_matmul=False)
    separate = torch.cat(
        [
            polar_express_zeroth_power(matrix[None], use_bfloat16_matmul=False)
            for matrix in matrices
        ]
    )
    zeros = polar_express_zeroth_power(torch.zeros_like(matrices), use_bfloat16_matmul=False)

    assert len(POLAR_EXPRESS_COEFFICIENTS) == 8
    assert FROBENIUS_EPSILON == 1e-14
    assert torch.allclose(batched, separate)
    assert torch.isfinite(batched).all()
    assert torch.equal(zeros, torch.zeros_like(zeros))


def test_muon_accepts_transformer_engine_partition_metadata_only_at_tp1():
    parameter = torch.nn.Parameter(torch.empty(8, 4))
    parameter.partition_dim = 0

    _validate_local_matrix_metadata(parameter, tp_size=1)
    with pytest.raises(ValueError, match="requires TP=EP=1"):
        _validate_local_matrix_metadata(parameter, tp_size=2)


def test_chained_grad_norm_combination_stays_on_device():
    first = torch.tensor(3.0)
    combined = _combine_grad_norms([first, torch.tensor(4.0), 0.0])

    assert isinstance(combined, torch.Tensor)
    assert combined.device == first.device
    assert combined.item() == 5.0


def test_capturable_optimizer_steps_compare_by_value_not_tensor_identity():
    first = torch.tensor([5], dtype=torch.int32)
    second = torch.tensor([5], dtype=torch.int32)

    assert _canonical_optimizer_step([first, second]) is first
    with pytest.raises(ValueError, match="diverged"):
        _canonical_optimizer_step([first, torch.tensor([6], dtype=torch.int32)])


def test_checkpoint_group_matching_preserves_duplicate_muon_groups_one_to_one():
    keys = ("wd_mult", "lr_mult", "is_expert_parallel", "is_decoupled_lr")

    def group(parameter, split_rows):
        return {
            "params": [parameter],
            "wd_mult": 1.0,
            "lr_mult": 1.0,
            "is_expert_parallel": False,
            "is_decoupled_lr": False,
            "archlab_muon_split_rows": split_rows,
        }

    current = [group("runtime-64", 64), group("runtime-160", 160)]
    loaded = [group("checkpoint-64", 64), group("checkpoint-160", 160)]
    reordered = _filter_and_reorder_optimizer_groups(current, loaded, keys)

    assert reordered[0] is not reordered[1]
    assert [item["archlab_muon_split_rows"] for item in reordered] == [64, 160]
    assert [item["params"] for item in reordered] == [
        ["checkpoint-64"],
        ["checkpoint-160"],
    ]
