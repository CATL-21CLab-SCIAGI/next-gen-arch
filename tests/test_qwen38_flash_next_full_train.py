import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import archlab.megatron.qwen38_flash_next_full_train as flash_next_train
from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE,
    FourStreamGatedResidual,
    GatedDeltaNet,
    Qwen38FlashNextFullConfig,
)
from archlab.megatron.qwen38_flash_next_full_train import (
    BILLION_DEPTH48_NO_MTP_MODEL_VARIANT,
    CHECKPOINT_INTERVAL_STEPS,
    EFFECTIVE_TOKENS,
    LOSS_NORMALIZATION,
    QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT,
    TOKENS_PER_STEP,
    TRAIN_STEPS,
    _assert_pipeline_data_rank_layout,
    _bind_native_moe_layer_number,
    _execute_checkpoint_request_by_local_rank,
    _loss_func,
    _megatron_argv,
    _native_muon_contract,
    _parser,
    _resolve_qwen_layer_number,
    _tag_native_optimizer_fallbacks,
    mtp_weighted_mean,
    partition_prefixes_for_dp_rank,
    shifted_mtp_targets,
)


@pytest.mark.parametrize("mask_values", [[1, 0, 1, 0], [0, 0, 0, 0]])
def test_loss_callback_returns_token_sum_and_integer_count(mask_values):
    losses = torch.tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    mask = torch.tensor(mask_values, dtype=torch.float32)

    loss_sum, token_count, report = _loss_func(mask, losses)

    assert loss_sum.item() == (losses.detach() * mask).sum().item()
    assert token_count.item() == sum(mask_values)
    assert token_count.dtype == torch.int64
    assert token_count.device == losses.device
    assert report["lm loss"].tolist() == [loss_sum.item(), token_count.item()]
    assert not report["lm loss"].requires_grad
    loss_sum.backward()
    torch.testing.assert_close(losses.grad, mask)


@pytest.mark.parametrize("microbatches", [1, 2, 4])
def test_token_normalization_is_independent_of_microbatch_partition(microbatches):
    parameter = torch.tensor(0.25, requires_grad=True)
    features = torch.arange(1, 9, dtype=torch.float32)
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.float32)
    total_tokens = 0
    for feature_part, mask_part in zip(
        features.chunk(microbatches), mask.chunk(microbatches), strict=True
    ):
        loss_sum, count, _ = _loss_func(mask_part, (parameter * feature_part).square())
        loss_sum.backward()
        total_tokens += count.item()
    actual = parameter.grad / total_tokens
    reference = torch.tensor(0.25, requires_grad=True)
    objective = ((reference * features).square() * mask).sum() / mask.sum()
    objective.backward()

    torch.testing.assert_close(actual, reference.grad)


@pytest.mark.parametrize("microbatches", [1, 4])
def test_frozen_native_schedule_preserves_main_and_auxiliary_gradient_ratio(
    monkeypatch, microbatches
):
    schedules = pytest.importorskip("megatron.core.pipeline_parallel.schedules")
    moe_utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    scaler = moe_utils.MoEAuxLossAutoScaler
    monkeypatch.setattr(scaler, "main_loss_backward_scale", None)
    config = SimpleNamespace(
        calculate_per_token_loss=True,
        timers=None,
        num_moe_experts=2,
        mtp_num_layers=None,
        grad_scale_func=None,
    )
    main = torch.tensor(0.25, requires_grad=True)
    auxiliary = torch.tensor(0.5, requires_grad=True)
    features = torch.arange(1, 9, dtype=torch.float32)
    masks = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.float32)
    token_count = 0
    reports = []
    for values, mask in zip(features.chunk(microbatches), masks.chunk(microbatches), strict=True):
        # The container's router attaches a token-summed auxiliary objective;
        # its native schedule independently scales the auxiliary backward pass.
        output = scaler.apply((main * values).square(), 0.01 * auxiliary.square() * mask.sum())
        loss, count = schedules.forward_step_calc_loss(
            torch.nn.Identity(),
            output,
            partial(_loss_func, mask),
            config,
            vp_stage=None,
            collect_non_loss_data=False,
            num_microbatches=microbatches,
            forward_data_store=reports,
            cp_group_size=1,
            is_last_stage=True,
        )
        loss.backward()
        token_count += count.item()

    assert token_count == int(masks.sum()), "native schedule must receive every valid token"
    reference_main = torch.tensor(0.25, requires_grad=True)
    reference_auxiliary = torch.tensor(0.5, requires_grad=True)
    reference_loss = (
        (reference_main * features).square() * masks
    ).sum() / masks.sum() + 0.01 * reference_auxiliary.square()
    reference_loss.backward()
    torch.testing.assert_close(main.grad / token_count, reference_main.grad)
    torch.testing.assert_close(auxiliary.grad / token_count, reference_auxiliary.grad)


@pytest.mark.parametrize("legacy_contract", [False, True])
def test_run_contract_records_correct_scaling_and_rejects_legacy_resume(
    tmp_path, monkeypatch, legacy_contract
):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-variant",
            QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT,
        ]
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        flash_next_train,
        "_sha256",
        lambda path: (
            flash_next_train.TOKENIZER_SHA256
            if path.name == "tokenizer.json"
            else flash_next_train.SOURCE_CONFIG_SHA256
        ),
    )
    monkeypatch.setattr(flash_next_train, "validate_runtime", lambda **kwargs: {"provider": "test"})
    args.run_dir.mkdir()
    contract = args.run_dir / "RUN_CONTRACT.json"
    if legacy_contract:
        contract.write_text(json.dumps({"training": {"train_steps": 11921}}))
        with pytest.raises(RuntimeError, match="loss normalization changed"):
            flash_next_train._write_contract(
                args, Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
            )
        assert not (args.run_dir / "contracts").exists()
    else:
        for _ in range(2):
            flash_next_train._write_contract(
                args, Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
            )
        assert (
            json.loads(contract.read_text())["training"]["loss_normalization"] == LOSS_NORMALIZATION
        )


def test_distributed_checkpoint_host_staging_runs_only_on_its_local_turn():
    events = []

    def preload(write_buckets, non_blocking=True):
        events.append(("preload", write_buckets, non_blocking))
        return "cpu-state"

    class Request:
        async_fn_args = ("rank", None, "queue")
        async_fn_kwargs = {"suffix": "written"}
        preload_fn = partial(preload, "write-buckets", True)
        finalize_fns = [lambda: events.append("finalize")]

        @staticmethod
        def async_fn(rank, state, queue, *, suffix):
            events.append((rank, state, queue, suffix))

    _execute_checkpoint_request_by_local_rank(
        Request(),
        local_rank=2,
        local_world_size=4,
        barrier=lambda: events.append("barrier"),
    )

    assert events == [
        "barrier",
        "barrier",
        ("preload", "write-buckets", False),
        ("rank", "cpu-state", "queue", "written"),
        "barrier",
        "barrier",
        "finalize",
    ]


def test_distributed_checkpoint_host_staging_rejects_changed_preload_abi():
    class Request:
        async_fn_args = ("rank", None, "queue")
        async_fn_kwargs = {}
        preload_fn = staticmethod(lambda: None)
        finalize_fns = []

    with pytest.raises(RuntimeError, match="preload function changed its ABI"):
        _execute_checkpoint_request_by_local_rank(
            Request(),
            local_rank=0,
            local_world_size=1,
            barrier=lambda: None,
        )


@pytest.mark.parametrize("local_rank,local_world_size", [(-1, 8), (8, 8), (0, 0)])
def test_distributed_checkpoint_host_staging_rejects_invalid_topology(local_rank, local_world_size):
    with pytest.raises(RuntimeError, match="invalid local checkpoint topology"):
        _execute_checkpoint_request_by_local_rank(
            object(),
            local_rank=local_rank,
            local_world_size=local_world_size,
            barrier=lambda: None,
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


def test_native_moe_layer_number_is_propagated_to_the_router():
    class FakeRouter:
        layer_number = None

    class FakeMoE:
        layer_number = None
        router = FakeRouter()

        def set_layer_number(self, layer_number):
            self.layer_number = layer_number
            self.router.layer_number = layer_number

    moe = FakeMoE()
    _bind_native_moe_layer_number(moe, 49)
    assert moe.layer_number == moe.router.layer_number == 49


def test_repeated_mtp_keeps_the_depth_local_layer_number_for_native_metrics():
    assert _resolve_qwen_layer_number(1, is_mtp_layer=True, backbone_offset=0) == 1
    assert _resolve_qwen_layer_number(2, is_mtp_layer=False, backbone_offset=12) == 14


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
        "--dist-ckpt-workers": "8",
        "--distributed-timeout-minutes": "60",
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
        "--calculate-per-token-loss",
    ):
        assert flag in argv
    assert "--muon-no-split-qkv" not in argv


def test_quarter_depth48_argv_has_even_pipeline_and_no_mtp(tmp_path: Path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-variant",
            QUARTER_DEPTH48_NO_MTP_MODEL_VARIANT,
        ]
    )
    config = Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
    argv = _megatron_argv(args, config)

    pairs = {
        "--num-layers": "48",
        "--hidden-size": "640",
        "--decoder-first-pipeline-num-layers": "12",
        "--decoder-last-pipeline-num-layers": "12",
        "--num-experts": "128",
        "--moe-router-topk": "3",
        "--moe-ffn-hidden-size": "160",
        "--moe-aux-loss-coeff": "0.01",
        "--moe-z-loss-coeff": "0.001",
    }
    for flag, value in pairs.items():
        assert argv[argv.index(flag) + 1] == value
    assert not any(flag.startswith("--mtp-") for flag in argv)
    assert "--calculate-per-token-loss" in argv


def test_billion_argv_uses_pp1_ep8_and_larger_microbatches(tmp_path, monkeypatch):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-variant",
            BILLION_DEPTH48_NO_MTP_MODEL_VARIANT,
            "--micro-batch-size",
            "4",
        ]
    )
    config = Qwen38FlashNextFullConfig.billion_depth48_no_mtp()
    argv = _megatron_argv(args, config)
    for flag, value in {
        "--pipeline-model-parallel-size": "1",
        "--expert-model-parallel-size": "8",
        "--num-layers": "48",
        "--hidden-size": "384",
        "--num-experts": "64",
        "--moe-ffn-hidden-size": "112",
        "--micro-batch-size": "4",
        "--global-batch-size": "4096",
    }.items():
        assert argv[argv.index(flag) + 1] == value
    assert "--decoder-first-pipeline-num-layers" not in argv
    assert "--decoder-last-pipeline-num-layers" not in argv
    assert not any(flag.startswith("--mtp-") for flag in argv)
    for rank in range(32):
        _assert_pipeline_data_rank_layout(
            global_rank=rank,
            data_parallel_rank=rank,
            data_parallel_world_size=32,
            pipeline_global_ranks=(rank,),
            pipeline_world_size=1,
        )
    # Exercise the installed container CLI validation without initializing CUDA.
    native = pytest.importorskip("megatron.training.arguments")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr("sys.argv", argv)
    parsed = native.validate_args(native.parse_args())
    assert parsed.data_parallel_size == 32
    assert parsed.pipeline_model_parallel_size == 1
    assert parsed.mtp_num_layers is None
    assert parsed.group_query_attention
    native_config = native.core_transformer_config_from_args(parsed)
    assert native_config.num_query_groups == config.attention_kv_heads == 1
    assert native_config.num_attention_heads == 6


def test_dp_only_argv_enables_supported_fusions_and_preserves_depth(tmp_path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-variant",
            BILLION_DEPTH48_NO_MTP_MODEL_VARIANT,
            "--parallelism",
            "dp-only",
            "--fused-moe",
            "--fused-cross-entropy",
        ]
    )
    argv = _megatron_argv(args, Qwen38FlashNextFullConfig.billion_depth48_no_mtp())
    for flag in (
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
        "--context-parallel-size",
    ):
        assert argv[argv.index(flag) + 1] == "1"
    for flag in ("--moe-permute-fusion", "--moe-router-fusion", "--cross-entropy-loss-fusion"):
        assert flag in argv
    assert argv[argv.index("--cross-entropy-fusion-impl") + 1] == "native"
    assert argv[argv.index("--num-layers") + 1] == "48"
    assert "--sequence-parallel" not in argv
    assert not any(flag.startswith("--mtp-") for flag in argv)
    with pytest.raises(ValueError, match="single-stage"):
        _megatron_argv(args, Qwen38FlashNextFullConfig.quarter_depth48_no_mtp())


@pytest.mark.parametrize("bad_group", [None, "tp", "pp", "ep", "cp", "expt_tp", "dp", "expt_dp"])
def test_dp_only_runtime_group_guard(monkeypatch, bad_group):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 32)
    sizes = {
        name: (32 if name in ("dp", "expt_dp") else 1)
        for name in ("tp", "pp", "ep", "cp", "expt_tp", "dp", "expt_dp")
    }
    if bad_group is not None:
        sizes[bad_group] = 8
    groups = SimpleNamespace(
        **{name: SimpleNamespace(size=lambda size=size: size) for name, size in sizes.items()}
    )
    if bad_group is None:
        assert flash_next_train._assert_dp_only_groups(groups) == sizes
    else:
        with pytest.raises(RuntimeError, match="DP-only"):
            flash_next_train._assert_dp_only_groups(groups)


def test_explicit_checkpoint_source_cannot_silently_start_fresh(tmp_path):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "new"),
            "--load-dir",
            str(tmp_path / "missing"),
        ]
    )
    with pytest.raises(ValueError, match="completed checkpoint marker"):
        _megatron_argv(args, Qwen38FlashNextFullConfig())


@pytest.mark.parametrize("drift", ["model geometry", "native attention grouping"])
def test_resume_rejects_previous_model_geometry_or_attention(tmp_path, monkeypatch, drift):
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--run-dir",
            str(tmp_path / "run"),
            "--model-variant",
            BILLION_DEPTH48_NO_MTP_MODEL_VARIANT,
        ]
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        flash_next_train,
        "_sha256",
        lambda path: (
            flash_next_train.TOKENIZER_SHA256
            if path.name == "tokenizer.json"
            else flash_next_train.SOURCE_CONFIG_SHA256
        ),
    )
    monkeypatch.setattr(flash_next_train, "validate_runtime", lambda **kwargs: {})
    args.run_dir.mkdir()
    contract = args.run_dir / "RUN_CONTRACT.json"
    contract.write_text(
        json.dumps(
            {
                "training": {"loss_normalization": LOSS_NORMALIZATION},
                "model_config": (
                    Qwen38FlashNextFullConfig.quarter_depth48_no_mtp()
                    if drift == "model geometry"
                    else Qwen38FlashNextFullConfig.billion_depth48_no_mtp()
                ).to_dict(),
            }
        )
    )
    with pytest.raises(RuntimeError, match=f"{drift} changed"):
        flash_next_train._write_contract(args, Qwen38FlashNextFullConfig.billion_depth48_no_mtp())


def test_probe_resume_overrides_only_the_probe_scheduler_horizon(tmp_path: Path):
    run_dir = tmp_path / "run"
    marker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("1")
    common = [
        "--data-root",
        str(tmp_path / "data"),
        "--tokenizer",
        str(tmp_path / "tokenizer"),
        "--run-dir",
        str(run_dir),
    ]

    production_argv = _megatron_argv(_parser().parse_args(common), Qwen38FlashNextFullConfig())
    probe_argv = _megatron_argv(
        _parser().parse_args([*common, "--probe-steps", "2"]),
        Qwen38FlashNextFullConfig(),
    )

    assert "--load" in production_argv
    assert "--override-opt-param-scheduler" not in production_argv
    assert "--load" in probe_argv
    assert "--override-opt-param-scheduler" in probe_argv


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
    supported_launcher = (root / "scripts" / "run_qwen38_flash_next_full_dlc.sh").read_text()

    assert "100000595968" in launcher
    assert "NGA_GLOBAL_BATCH_SIZE:-}" in launcher
    assert "run_qwen38_flash_next_full_dlc.sh" in launcher
    assert "run_qwen38_27b_quarter_dlc.sh" not in launcher
    assert "export NGA_EXPECTED_NODES NGA_GPUS_PER_NODE" in supported_launcher


def test_resident_controller_launcher_dispatches_only_the_flash_next_handle():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "run_qwen38_27b_quarter_dlc.sh").read_text()

    assert "/compat-qwen38-flash-next-*" in launcher
    assert "run_qwen38_27b_full_dlc.sh" in launcher
    assert "/mnt/nas/evergreen/compat-qwen38-flash-next-*" in launcher


def test_compatibility_launcher_selects_depth48_quarter_without_mtp():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts" / "run_qwen38_27b_full_dlc.sh").read_text()
    supported_launcher = (root / "scripts" / "run_qwen38_flash_next_full_dlc.sh").read_text()

    assert "qwen38-flash-next-quarter-depth48-nomtp-*" in launcher
    assert "NGA_FLASH_NEXT_MODEL_VARIANT=quarter-depth48-no-mtp" in launcher
    assert "NGA_PROBE_SAVE_INTERVAL=1" in launcher
    assert '--model-variant "$NGA_FLASH_NEXT_MODEL_VARIANT"' in supported_launcher
