import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter
from archlab.architectures.deepseek_v41_scratch import (
    CHANNEL_FIELDS,
    SCRATCH_WIDTHS,
    scaled_scratch_config,
    scaling_experts_per_token,
    scratch_adapter_head_dim,
)
from archlab.automodel.deepseek_v41_scaling_campaign import validate_qualification_pair


def base_config():
    return {
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "moe_intermediate_size": 2304,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "index_head_dim": 128,
            "engram_head_dim": 256,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "engram_layer_ids": [1, 14],
            "candidate_source_layer_id": 20,
            "n_shared_experts": 1,
            "n_routed_experts": 384,
            "num_experts_per_tok": 6,
            "num_attention_heads": 64,
            "engram_num_embeddings": [384006168, 384016682],
        }
    }


@pytest.mark.parametrize(
    "width,channels,adapter_dim",
    [
        (128, [128, 128, 16, 2, 32, 32, 4, 8], 16),
        (384, [384, 128, 64, 6, 96, 80, 16, 24], 16),
        (640, [640, 128, 64, 8, 160, 128, 16, 32], 16),
        (1280, [1280, 128, 128, 16, 320, 256, 32, 64], 32),
    ],
)
def test_declared_width_geometry_and_unchanged_categorical_controls(width, channels, adapter_dim):
    base = base_config()
    original = copy.deepcopy(base)
    text = scaled_scratch_config(base, width=width, scaling_study=True)["text_config"]
    assert [text[key] for key in CHANNEL_FIELDS] == channels
    assert scratch_adapter_head_dim(width) == adapter_dim
    assert text["num_hidden_layers"] == 20
    for key in (
        "n_routed_experts",
        "n_shared_experts",
        "num_attention_heads",
        "engram_num_embeddings",
    ):
        assert text[key] == base["text_config"][key]
    assert text["qk_rope_head_dim"] % 2 == 0
    assert text["qk_rope_head_dim"] <= min(text["head_dim"], text["index_head_dim"])
    assert Fraction(
        (text["num_experts_per_tok"] + text["n_shared_experts"]) * text["moe_intermediate_size"],
        width,
    ) == Fraction(3, 1)
    assert text["num_experts_per_tok"] == scaling_experts_per_token(width)
    assert base == original


@pytest.mark.parametrize("width", [0, True, 128.0, 256, 5120])
def test_reject_undeclared_width(width):
    with pytest.raises(ValueError, match="scratch width"):
        scaled_scratch_config(base_config(), width=width)


@pytest.mark.parametrize("width", SCRATCH_WIDTHS)
def test_paired_branch_shared_initialization_and_parameter_difference(width):
    config = V41AdapterConfig(width=width, head_dim=scratch_adapter_head_dim(width))
    baseline = V41NormalAttentionAdapter(config, backend="reference")
    simplicial = V41SimplicialAdapter(config, backend="reference")
    normal_state, simplex_state = baseline.state_dict(), simplicial.state_dict()
    for key, value in normal_state.items():
        mapped = {
            "k.weight": "k2.weight",
            "v.weight": "v2.weight",
            "k_norm.weight": "k2_norm.weight",
        }.get(key, key)
        assert torch.equal(value, simplex_state[mapped]), key
    difference = sum(p.numel() for p in simplicial.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    assert difference == 2 * width * config.kv_heads * config.head_dim + config.head_dim


def test_recipe_budget_is_eight_independent_ten_billion_token_runs():
    import yaml

    recipe = yaml.safe_load(
        (
            Path(__file__).parents[1] / "recipes/deepseek_v41/scratch_scaling.yaml"
        ).read_text()
    )
    assert len(recipe["widths"]) * len(recipe["variants"]) == 8
    assert recipe["training"]["supervised_tokens_per_run"] == 10_000_000_000
    assert recipe["training"]["total_supervised_token_allowance"] == 80_000_000_000
    assert json.loads(json.dumps(recipe))["widths"] == list(SCRATCH_WIDTHS)
    assert recipe["geometry"]["active_width_ratio"] == 3.0
    assert "engram_anchor_width" not in recipe["geometry"]
    assert recipe["geometry"]["engram_bucket_scale"] == "width / 5120"
    assert recipe["training"]["performance_contract"].endswith("/performance-v1.json")
    for width in SCRATCH_WIDTHS:
        assert width % 128 == 0
        assert recipe["geometry"]["routed_top_k"][width] == scaling_experts_per_token(width)


def qualified_pair(tmp_path):
    pair = []
    for variant in ("normal", "simplicial"):
        root = tmp_path / variant
        root.mkdir()
        pair.append({"variant": variant, "width": 640, "qualification": str(root)})
        contract = {
            "variant": variant,
            "target_supervised_tokens": 10_000_000_000,
            "runtime": {
                "variant": variant,
                "geometry": scaled_scratch_config(base_config(), scaling_study=True),
                "parameters": 1_000_000 + (327_808 if variant == "simplicial" else 0),
            },
            "data_contract": {"source_sha256": "identical"},
        }
        (root / "QUALIFIED.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "world_size": 16,
                    "context": 2048,
                    "contract": contract,
                }
            )
        )
        for rank in range(16):
            (root / f"rank-{rank:02d}-qualified.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "exact_checkpoint_restore": True,
                        "identical_next_update_loss": True,
                    }
                )
            )
            (root / f"rank-{rank:02d}-initial.json").write_text(
                json.dumps({"common_sha256": str(rank)})
            )
    return pair


def test_pair_admission_checks_all_rank_initializations(tmp_path):
    pair = qualified_pair(tmp_path)
    assert validate_qualification_pair(*pair)["passed"]
    path = Path(pair[1]["qualification"]) / "rank-15-initial.json"
    path.write_text(json.dumps({"common_sha256": "changed"}))
    with pytest.raises(ValueError, match="initialization differs"):
        validate_qualification_pair(*pair)


@pytest.mark.parametrize("mutation", ["data", "budget", "parameters"])
def test_pair_admission_rejects_confounded_controls(tmp_path, mutation):
    pair = qualified_pair(tmp_path)
    path = Path(pair[1]["qualification"]) / "QUALIFIED.json"
    value = json.loads(path.read_text())
    if mutation == "data":
        value["contract"]["data_contract"]["source_sha256"] = "other-data"
    elif mutation == "budget":
        value["contract"]["target_supervised_tokens"] = 1_000_000_000
    else:
        value["contract"]["runtime"]["parameters"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_qualification_pair(*pair)


def test_two_node_batch_preserves_every_global_window_once():
    from archlab.automodel.deepseek_v41_scratch_training import build_batches

    class Data:
        def batch(self, indices, device):
            return list(indices)

    windows = [
        i
        for rank in range(16)
        for batch in build_batches(Data(), 640, rank, microbatch=4, world_size=16)
        for i in batch
    ]
    assert sorted(windows) == list(range(640, 704))
    assert len(set(windows)) == 64


def campaign_plan(tmp_path):
    nodes = {name: {"hostname": name} for name in ("a", "b", "c", "d")}
    runs = []
    for variant, owners in (("normal", ["a", "b"]), ("simplicial", ["c", "d"])):
        for index, width in enumerate((640, 128, 384, 1280)):
            runs.append(
                {
                    "variant": variant,
                    "width": width,
                    "nodes": owners,
                    "queue_index": index,
                    "master_port": 28410 + 2 * index,
                    "output": str(tmp_path / f"production-{width}-{variant}"),
                    "qualification": str(tmp_path / f"qualification-{width}-{variant}"),
                }
            )
    return {
        "root": str(tmp_path),
        "nodes": nodes,
        "runs": runs,
        "token_budget_per_run": 10_000_000_000,
        "environment": {},
        "source_commit": "test",
    }


def test_campaign_rejects_gpu_overlap_between_variants(tmp_path):
    from archlab.automodel.deepseek_v41_scaling_campaign import validate_plan

    plan = campaign_plan(tmp_path)
    validate_plan(plan)
    for run in plan["runs"]:
        if run["variant"] == "simplicial":
            run["nodes"] = ["b", "c"]
    with pytest.raises(ValueError, match="partition"):
        validate_plan(plan)


def test_campaign_rejects_reusing_a_previous_phase_rendezvous_port(tmp_path):
    from archlab.automodel.deepseek_v41_scaling_campaign import validate_plan

    plan = campaign_plan(tmp_path)
    plan["runs"][1]["master_port"] = plan["runs"][0]["master_port"] + 1
    with pytest.raises(ValueError, match="distinct rendezvous ports"):
        validate_plan(plan)


def test_faster_variant_starts_every_next_width_without_peer_completion(tmp_path, monkeypatch):
    from archlab.automodel import deepseek_v41_scaling_campaign as campaign

    plan = campaign_plan(tmp_path)
    (tmp_path / "RL_RETIRED.json").write_text('{"all_gpus_free":true}')
    for run in plan["runs"]:
        if run["variant"] == "simplicial":
            qualification = Path(run["qualification"])
            qualification.mkdir()
            (qualification / "QUALIFIED.json").write_text("{}")
    phases = []

    def run_phase(plan, run, node, mode, environment, state_path):
        phases.append((mode, run["width"]))
        if mode == "qualify":
            return
        root = Path(run["output"])
        root.mkdir()
        (root / "COMPLETE.json").write_text('{"passed":true,"supervised_tokens":10000000000}')
        checkpoints = []
        for i in range(1, 6):
            target = root / f"oss-{i}"
            target.mkdir()
            link = root / f"checkpoint-{i}"
            link.symlink_to(target, target_is_directory=True)
            checkpoints.append(
                {"path": str(link), "oss_path": str(target), "milestone_tokens": i * 2_000_000_000}
            )
        (root / "EVAL_CHECKPOINTS.json").write_text(json.dumps({"checkpoints": checkpoints}))

    monkeypatch.setattr(campaign, "run_phase", run_phase)
    monkeypatch.setattr(campaign, "validate_qualification_pair", lambda *args: {"passed": True})
    monkeypatch.setattr(campaign.socket, "gethostname", lambda: "a")
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *args, **kwargs: "")
    campaign.worker(plan, "a")
    assert [width for mode, width in phases if mode == "train"] == [640, 128, 384, 1280]
    assert all(
        not Path(run["output"]).exists() for run in plan["runs"] if run["variant"] == "simplicial"
    )


def test_admission_resume_reuses_successful_preflight(tmp_path, monkeypatch):
    from archlab.automodel import deepseek_v41_scaling_campaign as campaign

    plan = campaign_plan(tmp_path)
    preflight = dict(plan["runs"][0], qualification=str(tmp_path / "preflight"))
    plan["preflight_runs"] = [preflight]
    receipt = Path(preflight["qualification"])
    receipt.mkdir()
    (receipt / "QUALIFIED.json").write_text('{"passed":true}')
    (tmp_path / "RL_RETIRED.json").write_text('{"all_gpus_free":true}')
    (tmp_path / "STOP_REQUEST").touch()
    monkeypatch.setattr(campaign.socket, "gethostname", lambda: "a")
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(campaign, "run_phase", lambda *a, **k: pytest.fail("repeated preflight"))
    campaign.worker(plan, "a", resume=True)
    (receipt / "QUALIFIED.json").write_text('{"passed":false}')
    with pytest.raises(ValueError, match="failed preflight"):
        campaign.worker(plan, "a", resume=True)


def test_admission_resume_refuses_existing_production(tmp_path, monkeypatch):
    from archlab.automodel import deepseek_v41_scaling_campaign as campaign

    plan = campaign_plan(tmp_path)
    output = Path(plan["runs"][0]["output"])
    output.mkdir()
    (output / "launcher-a.log").touch()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(
        campaign.sys, "argv", ["campaign", "--plan", str(path), "--node", "a", "--resume-admission"]
    )
    with pytest.raises(ValueError, match="cannot restart production"):
        campaign.main()


def test_fixed_engram_geometry_keeps_budget_through_width_and_depth_sweeps():
    base = base_config()
    base["text_config"].update(engram_vocab_size=16_000_000, engram_max_ngram_size=4,
                               engram_n_heads=8)
    anchor = None
    for width in (128, 256, 384):
        for depth in (10, 20, 45):
            text = scaled_scratch_config(base, width=width, depth=depth, scaling_study=True,
                                         sweep_geometry=True, scale_engram=True,
                                         engram_anchor_width=128)["text_config"]
            geometry = tuple(text[key] for key in ("engram_head_dim", "engram_vocab_size",
                                                    "engram_num_embeddings"))
            if anchor is None:
                anchor = geometry
            assert geometry == anchor
            assert len(text["engram_layer_ids"]) == len(text["engram_num_embeddings"]) == 2
            assert (text["num_experts_per_tok"] + 1) * 128 == 3 * width
    with pytest.raises(ValueError, match="requires scale_engram"):
        scaled_scratch_config(base, engram_anchor_width=128)
