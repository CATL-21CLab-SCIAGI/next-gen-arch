"""Scientific manifest controls and admission gates, independent of GPU runtime."""

import copy
import json
from pathlib import Path

import pytest

from archlab.artifacts import sha256_file
from archlab.automodel.triadic_scratch_qualification import (
    COMMON_MIXER_TENSORS,
    DELTA_SHARED_TENSORS,
    LAYERS,
    VARIANTS,
    assemble_matched_report,
    build_manifest,
    mixer_specification,
    read_matched_contract,
    state_accounting,
    validate_matched_contract,
    verify_admission,
)

RECIPE = Path(__file__).parents[1] / "recipes/triadic/matched_scratch_w640_d20.yaml"
REVISION = "a" * 40


@pytest.fixture
def contract():
    return read_matched_contract(RECIPE)


def test_registered_geometry_is_a_new_comparison(contract):
    assert contract["geometry"]["head_dim"] == 128
    assert contract["geometry"]["query_heads"] == contract["geometry"]["kv_heads"] == 2
    assert contract["training"]["micro_batch_size"] * contract["training"]["accumulation"] * 8 == 64
    assert contract["training"]["checkpoint_tokens"][-1] == 10_000_000_000
    assert contract["matching"]["compute_match"] is False
    assert contract["matching"]["state_match"] is False


@pytest.mark.parametrize("variant,features", [("linear", None), ("linsimp", None), ("gdn", 1), ("triadic", 4)])
def test_variant_geometry_and_axes(contract, variant, features):
    spec = mixer_specification(contract, variant)
    assert spec["feature_dim"] == features
    assert spec["feature_rank"] == 512
    assert spec["adapter_config"]["head_dim"] == 128
    spec["compensation"]["alignment"] = 1
    assert contract["mixer_contract"]["compensation"]["alignment"] == 16


@pytest.mark.parametrize("section,key,value", [
    ("geometry", "head_dim", 16),
    ("geometry", "query_heads", 8),
    ("geometry", "width", 384),
    ("geometry", "branch_layers", [2, 4]),
    ("mixer_contract", "feature_rank", 128),
    ("mixer_contract", "triadic_features", 8),
    ("training", "micro_batch_size", 8),
    ("training", "supervised_tokens", 1_000_000_000),
    ("training", "checkpoint_tokens", [10_000_000_000]),
    ("training", "cross_document_packing", True),
    ("training", "execution", dict(optimized_synchronization=False, trim_alignment=128)),
    ("training", "execution", dict(optimized_synchronization=True, trim_alignment=64)),
    ("matching", "compute_match", True),
    ("matching", "state_match", True),
    ("qualification", "full_model_world_size", 2),
    ("initialization", "delta_named_rng_streams", []),
    ("initialization", "require_constructor_hash_receipts", False),
])
def test_reject_silent_scientific_contract_changes(contract, section, key, value):
    contract[section][key] = value
    with pytest.raises(ValueError):
        validate_matched_contract(contract)


def test_state_accounting_reports_residuals(contract):
    sizes = state_accounting(contract)
    assert sizes["linear"]["global_matrix"] == sizes["triadic"]["global_matrix"] == 65536
    assert sizes["linear"]["total"] == 66048
    assert sizes["linsimp"]["total"] == 74240
    assert sizes["gdn"]["total"] == 16384


@pytest.fixture
def manifest(tmp_path, contract):
    source = tmp_path / "source"
    (source / "src/archlab").mkdir(parents=True)
    (source / "SOURCE_REVISION").write_text(REVISION + "\n")
    (source / "src/archlab/example.py").write_text("PINNED = True\n")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_bytes(RECIPE.read_bytes())
    data = tmp_path / "DATA.json"
    data.write_text(json.dumps(dict(training_targets=10_000_000_000, validation_targets=1_000_000,
                                   sequence=2048, order_sha256="b" * 64)))
    return build_manifest(contract, recipe=recipe, source=source, revision=REVISION,
                          output_root=tmp_path / "runs", data_contract=data, container_image="pinned-container")


def test_manifest_reuses_existing_trainer_without_launching(manifest):
    assert manifest["status"] == "prepared-not-qualified"
    assert manifest["launches_performed"] is False
    assert len(manifest["stages"]) == 12
    assert all(not s["name"].endswith("-train") for s in manifest["stages"][:8])
    assert all(s["name"].endswith("-train") for s in manifest["stages"][8:])
    for variant in VARIANTS:
        stages = [s for s in manifest["stages"] if s["name"].startswith(variant + "-")]
        assert [s["world"] for s in stages] == [2, 8, 8]
        assert stages[0]["scope"] == "adapter-only"
        assert stages[1]["module"] == stages[2]["module"] == "archlab.automodel.deepseek_v41_scratch_training"
        assert "--matched-mixer-contract" in stages[2]["args"]
        assert len(stages[2]["requires"]) == 3
    with pytest.raises(FileNotFoundError):
        verify_admission(manifest)


def write_receipts(manifest):
    data = json.loads(Path(manifest["data_contract"]).read_text())
    for stage in manifest["stages"]:
        if "required_receipt" not in stage:
            continue
        variant = stage["name"].removesuffix("-adapter2").removesuffix("-full8")
        path = Path(stage["required_receipt"])
        path.parent.mkdir(parents=True, exist_ok=True)
        report = dict(passed=True, world_size=stage["world"], variant=variant, source_revision=REVISION)
        initialization = dict(format="archlab-matched-mixer-initialization-v2", seed=42,
            common_core_sha256={name:"c" * 64 for name in COMMON_MIXER_TENSORS},
            delta_shared_sha256={name:"d" * 64 for name in DELTA_SHARED_TENSORS}
                if variant in ("gdn", "triadic") else {},
            delta_rng_streams=["beta", "qkv-conv"] if variant in ("gdn", "triadic") else [])
        report["initialization_contract"] = initialization
        if stage["scope"] == "full-backbone":
            report["contract"] = dict(project_commit=REVISION, variant=variant,
                                      matched_study=manifest["study"], data_contract=data)
            report["ranks"] = [f"rank-{i:02d}-qualified.json" for i in range(8)]
            from archlab.automodel.deepseek_v41_scratch_training import summarize_training_timings

            timing = summarize_training_timings([
                dict(wall_seconds=2, supervised_tokens=1000, window_cursor=64 * (i + 1))
                for i in range(5)
            ], warmup_updates=2)
            for rank, name in enumerate(report["ranks"]):
                (path.parent / name).write_text(json.dumps(dict(passed=True, exact_checkpoint_restore=True,
                    identical_next_update_state=True, common_initial_sha256=f"same-backbone-rank{rank}",
                    real_data_timing=timing)))
            layer = dict(variant=variant, total_parameters=375_000, target_parameters=375_000,
                         relative_parameter_error=0)
            report["contract"]["runtime"] = dict(branch_parameters=3_000_000,
                nonembedding_parameters=4_000_000_000, parameters=29_000_000_000,
                matched_parameter_contracts={str(i):copy.deepcopy(layer) for i in LAYERS},
                matched_initialization_contracts={str(i):copy.deepcopy(initialization) for i in LAYERS})
            (path.parent / "RUN_CONTRACT.json").write_text(json.dumps(report["contract"]))
        path.write_text(json.dumps(report))
    match_path = Path(next(s for s in manifest["stages"] if "requires" in s)["requires"][-1])
    match_path.write_text(json.dumps(assemble_matched_report(manifest)))
    return match_path


def test_admission_requires_all_actual_receipts_and_rank_aligned_backbone(manifest):
    match = write_receipts(manifest)
    report = verify_admission(manifest)
    assert report["passed"] is True
    assert report["receipt_sha256"][str(match)] == sha256_file(match)


def test_matched_report_derives_actual_constructor_counts(manifest):
    write_receipts(manifest)
    report = assemble_matched_report(manifest)
    assert report["passed"] is True
    assert report["variants"]["triadic"]["branch_parameters"] == 3_000_000
    assert report["variants"]["triadic"]["nonembedding_parameters"] == 4_000_000_000
    assert report["compute_matched"] is False and report["state_matched"] is False
    assert len(report["evidence_sha256"]) == 44
    assert report["mixer_initialization"]["gdn"] == report["mixer_initialization"]["triadic"]


def test_matched_report_rejects_constructor_counts_without_consistent_evidence(manifest):
    write_receipts(manifest)
    full = Path(next(s for s in manifest["stages"] if s["name"] == "triadic-full8")["required_receipt"])
    run = full.parent / "RUN_CONTRACT.json"
    value = json.loads(run.read_text())
    value["runtime"]["branch_parameters"] += 1
    run.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="contract differs"):
        assemble_matched_report(manifest)


@pytest.mark.parametrize("field,tensor", [
    ("common_core_sha256", "q"),
    ("delta_shared_sha256", "beta_projection"),
    ("delta_shared_sha256", "qkv_causal_conv"),
])
@pytest.mark.parametrize("scope", ["adapter2", "full8"])
def test_admission_and_report_reject_unmatched_mixer_initialization(manifest, field, tensor, scope):
    write_receipts(manifest)
    path = Path(next(s for s in manifest["stages"] if s["name"] == "triadic-" + scope)["required_receipt"])
    receipt = json.loads(path.read_text())
    if scope == "adapter2":
        receipt["initialization_contract"][field][tensor] = "f" * 64
    else:
        receipt["contract"]["runtime"]["matched_initialization_contracts"][str(LAYERS[0])][field][tensor] = "f" * 64
        (path.parent / "RUN_CONTRACT.json").write_text(json.dumps(receipt["contract"]))
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="initialization"):
        verify_admission(manifest)
    with pytest.raises(ValueError, match="initialization"):
        assemble_matched_report(manifest)


def test_admission_rejects_old_receipts_without_named_initialization_evidence(manifest):
    write_receipts(manifest)
    path = Path(next(s for s in manifest["stages"] if s["name"] == "gdn-adapter2")["required_receipt"])
    receipt = json.loads(path.read_text())
    del receipt["initialization_contract"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="initialization"):
        verify_admission(manifest)


@pytest.mark.parametrize("change", ["source", "data", "recipe", "world", "variant", "backbone", "optimizer", "rankset"])
def test_admission_rejects_mutated_or_mismatched_evidence(manifest, change):
    write_receipts(manifest)
    full = Path(next(s for s in manifest["stages"] if s["name"] == "triadic-full8")["required_receipt"])
    if change == "source":
        (Path(manifest["source"]) / "src/archlab/example.py").write_text("MUTATED = True\n")
    elif change == "data":
        Path(manifest["data_contract"]).write_text("{}")
    elif change == "recipe":
        Path(manifest["recipe"]).write_text("{}")
    elif change in ("world", "variant", "rankset"):
        report = json.loads(full.read_text())
        if change == "world":
            report["world_size"] = 2
        elif change == "variant":
            report["contract"]["variant"] = "linear"
        else:
            report["ranks"] = report["ranks"][:2]
        full.write_text(json.dumps(report))
    else:
        rank = full.parent / "rank-00-qualified.json"
        report = json.loads(rank.read_text())
        if change == "backbone":
            report["common_initial_sha256"] = "different-initial-backbone"
        else:
            report["identical_next_update_state"] = False
        rank.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_admission(manifest)


@pytest.mark.parametrize("bad_error", [0.02, -0.02, float("nan"), float("inf")])
def test_admission_rejects_unmatched_parameter_budgets(manifest, bad_error):
    match = write_receipts(manifest)
    report = json.loads(match.read_text())
    report["variants"]["gdn"]["branch_relative_parameter_error"] = bad_error
    match.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="parameter audit"):
        verify_admission(manifest)


def test_build_manifest_rejects_wrong_data_order_contract(tmp_path, manifest):
    data = Path(manifest["data_contract"])
    wrong = json.loads(data.read_text())
    wrong["training_targets"] = 1_000_000_000
    data.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="sealed corpus"):
        build_manifest(manifest["study"], recipe=manifest["recipe"], source=manifest["source"],
                       revision=REVISION, output_root=tmp_path, data_contract=data, container_image="same")


def test_validate_does_not_mutate_callers(contract):
    original = copy.deepcopy(contract)
    result = validate_matched_contract(contract)
    result["geometry"]["width"] = 1
    assert contract == original
