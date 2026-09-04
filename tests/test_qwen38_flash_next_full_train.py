from pathlib import Path

import pytest
import torch

from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE,
    FourStreamGatedResidual,
    GatedDeltaNet,
    Qwen38FlashNextFullConfig,
)
from archlab.megatron.qwen38_flash_next_full_train import (
    CHECKPOINT_INTERVAL_STEPS,
    EFFECTIVE_TOKENS,
    TOKENS_PER_STEP,
    TRAIN_STEPS,
    _assert_pipeline_data_rank_layout,
    _megatron_argv,
    _native_muon_contract,
    _parser,
    _tag_native_optimizer_fallbacks,
    mtp_weighted_mean,
    partition_prefixes_for_dp_rank,
    shifted_mtp_targets,
)


def test_pipeline_sample_layout_is_checked_without_a_training_step_collective():
    for data_parallel_rank in range(8):
        pipeline_ranks = tuple(data_parallel_rank + 8 * stage for stage in range(4))
        for global_rank in pipeline_ranks:
            _assert_pipeline_data_rank_layout(
                global_rank=global_rank,
                data_parallel_rank=data_parallel_rank,
                data_parallel_world_size=8,
                pipeline_global_ranks=pipeline_ranks,
            )

    with pytest.raises(RuntimeError, match="deterministic data rank"):
        _assert_pipeline_data_rank_layout(
            global_rank=0,
            data_parallel_rank=0,
            data_parallel_world_size=8,
            pipeline_global_ranks=(0, 9, 16, 24),
        )


class _OptimizerGroupingFixture(torch.nn.Module):
    def __init__(self, config: Qwen38FlashNextFullConfig):
        super().__init__()
        self.backbone = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gdn = GatedDeltaNet(config)
        self.gr = FourStreamGatedResidual(config)
        self.router = torch.nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.ple = DistributedPLE(config, owner_rank=0, owner_world_size=1)


def test_training_defaults_are_the_exact_100b_contract(tmp_path: Path):
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

    assert args.sequence_length == 2_048
    assert args.micro_batch_size == 1
    assert args.global_batch_size == 4_096
    assert args.target_train_tokens == EFFECTIVE_TOKENS == 100_000_595_968
    assert TOKENS_PER_STEP == 8_388_608
    assert TRAIN_STEPS == 11_921
    assert CHECKPOINT_INTERVAL_STEPS == 1_192
    assert args.learning_rate == 1.76e-3
    assert args.minimum_learning_rate == 1.76e-4
    assert args.warmup_fraction == 0.01
    assert args.clip_grad == 0.5


def test_megatron_argv_uses_pp4_ep8_native_muon_and_native_mtp(tmp_path: Path):
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
    argv = _megatron_argv(args, Qwen38FlashNextFullConfig())
    pairs = {
        "--train-iters": "11921",
        "--save-interval": "1192",
        "--pipeline-model-parallel-size": "4",
        "--decoder-first-pipeline-num-layers": "12",
        "--decoder-last-pipeline-num-layers": "10",
        "--expert-model-parallel-size": "8",
        "--num-experts": "512",
        "--moe-router-topk": "10",
        "--mtp-num-layers": "3",
        "--mtp-loss-scaling-factor": "0.1",
        "--muon-momentum": "0.95",
        "--muon-coefficient-type": "polar_express",
        "--muon-num-ns-steps": "8",
        "--muon-extra-scale-factor": "0.2",
        "--muon-fp32-matmul-prec": "medium",
    }
    for flag, value in pairs.items():
        assert argv[argv.index(flag) + 1] == value
    for flag in (
        "--mtp-use-repeated-layer",
        "--muon-nesterov",
        "--bf16",
        "--use-distributed-optimizer",
        "--no-use-layer-wise-param-layout",
        "--exit-signal-handler",
    ):
        assert flag in argv
    assert "--muon-no-split-qkv" not in argv


def test_native_muon_contract_records_known_deviations():
    contract = _native_muon_contract()

    assert contract["fp32_matmul_precision"] == "medium"
    assert "coarser" in contract["qkv_split"]
    assert contract["released_private_optimizer"] == "Canzona unavailable"
    assert "no adapter" in contract["integration"]


def test_optimizer_grouping_keeps_muon_maps_and_scalar_adamw_ple_adam_distinct():
    config = Qwen38FlashNextFullConfig.tiny(ngram_partitions=1)
    model = _OptimizerGroupingFixture(config)
    counts = _tag_native_optimizer_fallbacks(model)

    assert model.backbone.weight.archlab_optimizer == "muon"
    assert model.gdn.in_proj_qkv.weight.archlab_optimizer == "muon"
    assert model.gdn.in_proj_a.weight.archlab_optimizer == "adamw"
    assert model.gdn.in_proj_b.weight.archlab_optimizer == "adamw"
    assert model.gr.input_mix_weight_down.weight.archlab_optimizer == "adamw"
    assert model.router.weight.archlab_optimizer == "adamw"
    table = model.ple.embedding.tables[0]
    assert table.archlab_optimizer == "adam"
    assert table.archlab_no_weight_decay is True
    assert table.ndim == 1
    assert counts["muon"] > 0
    assert counts["adamw"] > 0
    assert counts["ple_adam_no_decay"] == table.numel()


def test_data_partition_depends_on_dp_rank_not_global_model_parallel_rank():
    prefixes = [Path(f"part-{index:02d}") for index in range(32)]

    # All 32 PP4xEP8 model-parallel ranks have DP rank zero and therefore
    # consume the same ordered shard set.
    assignments = [partition_prefixes_for_dp_rank(prefixes, 0, 1) for _ in range(32)]

    assert all(assignment == prefixes for assignment in assignments)
    assert partition_prefixes_for_dp_rank(prefixes, 1, 4) == prefixes[1::4]


def test_three_shifted_targets_mean_before_scaling_and_early_backbone_gradient():
    labels = torch.tensor([[10, 11, 12, 13, 14]])
    targets = shifted_mtp_targets(labels)

    assert [target.tolist() for target in targets] == [
        [[11, 12, 13, 14, -1]],
        [[12, 13, 14, -1, -1]],
        [[13, 14, -1, -1, -1]],
    ]
    backbone = torch.nn.Linear(4, 4, bias=False)
    shared_mtp = torch.nn.Linear(4, 4, bias=False)
    hidden = backbone(torch.randn(2, 4))
    layer_ids = []
    losses = []
    for _depth in range(3):
        layer_ids.append(id(shared_mtp.weight))
        hidden = shared_mtp(hidden)
        losses.append(hidden.square().mean())
    objective = mtp_weighted_mean(torch.stack(losses))
    objective.backward()

    assert len(set(layer_ids)) == 1
    assert torch.allclose(objective.detach(), 0.1 * torch.stack(losses).mean().detach())
    assert backbone.weight.grad is not None
    assert backbone.weight.grad.abs().sum() > 0


def test_legacy_full_launcher_is_a_narrow_flash_next_forwarder():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "run_qwen38_27b_full_dlc.sh").read_text()

    assert "100000595968" in launcher
    assert "NGA_GLOBAL_BATCH_SIZE:-}" in launcher
    assert "run_qwen38_flash_next_full_dlc.sh" in launcher
    assert "run_qwen38_27b_quarter_dlc.sh" not in launcher
