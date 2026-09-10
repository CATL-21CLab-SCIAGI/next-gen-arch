"""Static proposal consistency only; these do not qualify a V4.1 training backend."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def proposal():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "recipes/proposals/deepseek_v41_global_simplicial_math.yaml").read_text()
    )


def test_added_parameter_budget(proposal):
    adapter = proposal["adapter_proposal"]
    width, dim = adapter["input_output_width"], adapter["head_dim"]
    q = adapter["query_heads"] * dim
    kv = adapter["kv_heads"] * dim
    streams = proposal["model"]["backbone_residual_streams"]
    matrices = width * (3 * q + 4 * kv)
    auxiliary = width + 3 * dim + 2 * streams
    assert matrices + auxiliary == adapter["parameters_per_adapter"]
    assert (matrices + auxiliary) * len(adapter["layers_1based"]) == adapter["total_trainable_parameters"]


def test_window_complexity_and_boundary_convention(proposal):
    adapter = proposal["adapter_proposal"]
    short, long = adapter["key_axis_windows"]
    assert (short, long) == (32, 512)
    assert adapter["window_counts_include_current_token"]
    assert adapter["include_diagonal_pairs"]
    assert not adapter["global_pair_coverage"]
    assert not adapter["whole_model_linear_complexity_claim"]
    assert sum(min(t, short) * min(t, long) for t in range(1, 16385)) == 264243888


def test_optimizer_partition_and_inherited_scaling(proposal):
    adapter = proposal["adapter_proposal"]
    train = proposal["training_proposal"]
    muon = train["muon"]
    assert set(muon["headwise"]) == {"Q", "K1", "K2"}
    assert set(muon["full_matrix"]) == {"V1", "V2", "output_projection", "output_gate"}
    assert muon["head_matrix_shape"] == [adapter["head_dim"], adapter["input_output_width"]]
    assert muon["heads_per_parameter"] == {
        "Q": adapter["query_heads"], "K1": adapter["kv_heads"], "K2": adapter["kv_heads"],
    }
    assert muon["nesterov"] and muon["momentum"] == 0.95
    assert muon["update_rms"] == 0.18
    assert not muon["force_measured_rms_renormalization"]
    ns = muon["orthogonalization"]
    assert ns["fast_steps"] == 8 and ns["stabilization_steps"] == 2
    assert ns["fast_coefficients"] == [3.4445, -4.775, 2.0315]
    assert ns["stabilization_coefficients"] == [2.0, -1.5, 0.5]
    assert train["adamw"]["norm_weight_decay"] == muon["weight_decay"] == 0.1
    assert train["adamw"]["scalar_weight_decay"] == 0.0
    assert train["adamw"]["eps"] == 1e-20


def test_identity_initialization_and_frozen_state_are_explicit(proposal):
    model = proposal["model"]
    init = proposal["adapter_proposal"]["initialization"]
    assert model["freeze_existing_parameters"]
    assert model["freeze_existing_mutable_training_state"]
    assert model["router_bias_updates"] == "disabled"
    assert init["output_projection_zero"] and init["output_projection_std"] == 0
    assert init["qkv_and_output_gate_std"] > 0
    assert init["rmsnorm_effective_scale"] == 1
    assert init["effective_stream_write_coefficients"] == 1


def test_not_a_launchable_or_data_complete_contract(proposal):
    assert not proposal["adapter_proposal"]["implemented"]
    assert not proposal["training_proposal"]["launch_now"]
    assert not proposal["qualification"]["exact_V41_training_backend_found"]
    assert proposal["data_preparation"]["current_status"].startswith("failed-")
    assert proposal["runtime"]["package_installations"] == "forbidden"
    assert proposal["runtime"]["subagents"] == "forbidden"
    assert proposal["parallelism_proposal"]["world_size"] == 32
    assert proposal["parallelism_proposal"]["expert_parallel_size"] == 8
