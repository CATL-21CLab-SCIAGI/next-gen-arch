from fractions import Fraction

import torch
from test_deepseek_v41_scaling import base_config

from archlab.architectures.deepseek_v41_scratch import scaled_scratch_config
from archlab.architectures.prelude_loop_coda import balanced_layout
from archlab.automodel.loop_regularization import cell_name, cells, validate_cell
from archlab.optimizers.sharded_adafactor import ShardedAdafactor


def test_registered_grid_has_all_paper_cells_and_requested_d128_anchor_first():
    grid = cells()
    assert len(grid) == len({cell_name(c) for c in grid}) == 222
    assert (
        grid[0]["hidden_size"],
        grid[0]["stored_layers"],
        grid[0]["recursions"],
        grid[0]["weight_decay"],
    ) == (128, 20, 2, 0.8)
    for cell in grid:
        validate_cell(cell)
        assert cell["unique_targets"] * cell["epochs"] == 1_000_000_000


def test_all_size_controls_keep_exact_moe_ratio_and_valid_native_ownership():
    from pathlib import Path

    import yaml

    recipe = yaml.safe_load(
        (Path(__file__).parents[1] / "recipes/deepseek_v41/loop_regularization.yaml").read_text()
    )
    for cell in cells()[::6] + cells()[:6]:
        text = scaled_scratch_config(
            base_config(),
            width=cell["hidden_size"],
            depth=cell["stored_layers"],
            scaling_study=True,
            sweep_geometry=True,
        )["text_config"]
        assert Fraction((text["num_experts_per_tok"] + 1) * 128, text["hidden_size"]) == Fraction(
            3, 1
        )
        assert text["num_experts_per_tok"] > 0
        declared = recipe["grid"]["geometry_by_reference_depth"][cell["reference_depth"]]
        assert (text["hidden_size"], text["num_hidden_layers"], text["num_experts_per_tok"]) == (
            declared["hidden_size"], declared["stored_layers"], declared["routed_topk"]
        )
        assert len(text["compress_ratios"]) == cell["stored_layers"]
        assert set(text["kv_source_layer_ids"]).issubset(text["index_source_layer_ids"])
        assert all(text["compress_ratios"][i] > 0 for i in text["index_source_layer_ids"])
        assert text["compress_ratios"][text["candidate_source_layer_id"]] == 1
        layout = balanced_layout(cell["stored_layers"])
        assert layout.stored_depth == cell["stored_layers"]
        assert (
            len(layout.execution(cell["recursions"]))
            == layout.prelude + cell["recursions"] * layout.core + layout.coda
        )


def test_matrix_decay_matches_decoupled_equation_and_excludes_vectors():
    torch.manual_seed(7)
    for shape in ((7,), (9, 7), (3, 9, 7)):
        initial = torch.randn(shape)
        gradient = torch.randn(shape)
        zero, decayed = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
        for parameter, wd in ((zero, 0), (decayed, 0.8)):
            optimizer = ShardedAdafactor(
                [parameter], lr=0.01, weight_decay=wd, chunk_elements=14, stochastic_rounding=False
            )
            parameter.grad = gradient.clone()
            optimizer.step()
        expected = zero.detach() - initial * (0.008 if len(shape) > 1 else 0)
        torch.testing.assert_close(decayed, expected, rtol=1e-6, atol=1e-7)
