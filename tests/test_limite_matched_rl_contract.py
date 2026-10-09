from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from archlab.rl.limite_contract import check_matched_recipe


def contract():
    spec = yaml.safe_load((Path(__file__).parents[1] / "recipes/limite/full_math_rl_native_context.yaml").read_text())
    effective = dict(learning_rate=1e-5, seed=42, data_seed=42, num_generations=4,
                     generation_batch_size=64, max_completion_length=131072, max_steps=400,
                     gradient_accumulation_steps=8, loss_type="dapo", scale_rewards="group",
                     beta=0., num_iterations=1, temperature=1., top_p=1., top_k=0,
                     mask_truncated_completions=False)
    options = dict(variant="normal", world_size=8, phase_start=0, checkpoint_steps=10,
                   split_sha256=spec["data"]["split_sha256"], heldout_count=128)
    return spec, effective, options


def test_matched_recipe_binds_effective_training_settings_and_allows_small_probe():
    spec, effective, options = contract()
    report = check_matched_recipe(spec, effective, **options)
    assert report["effective_training"]["max_completion_length"] == 131072
    assert not report["correctness_fixture"]
    probe = effective | dict(max_steps=3, max_completion_length=128, gradient_accumulation_steps=32)
    check_matched_recipe(spec, probe, **(options | dict(world_size=2, checkpoint_steps=1)),
                         correctness_fixture=True)
    assert spec["rollout"]["max_tokens"] == 131072


@pytest.mark.parametrize("change", [dict(learning_rate=1e-6), dict(seed=43), dict(num_generations=8),
                                    dict(generation_batch_size=32), dict(max_steps=300),
                                    dict(max_completion_length=16384), dict(beta=.1),
                                    dict(loss_type="grpo"), dict(mask_truncated_completions=True)])
def test_ignored_recipe_knobs_or_changed_objective_are_rejected(change):
    spec, effective, options = contract()
    with pytest.raises(ValueError, match="differs from the recipe"):
        check_matched_recipe(spec, effective | change, **options)


@pytest.mark.parametrize("change", [dict(world_size=4), dict(checkpoint_steps=20),
                                    dict(phase_start=1), dict(split_sha256="other"), dict(heldout_count=127)])
def test_unmatched_topology_clock_or_dataset_is_rejected(change):
    spec, effective, options = contract()
    with pytest.raises(ValueError, match="differs from the recipe"):
        check_matched_recipe(spec, effective, **(options | change))


def test_invalid_accumulation_or_async_contract_is_rejected():
    spec, effective, options = contract()
    with pytest.raises(ValueError, match="accumulation"):
        check_matched_recipe(spec, effective | dict(gradient_accumulation_steps=4), **options)
    changed = deepcopy(spec)
    changed["execution"]["max_policy_lag"] = 2
    with pytest.raises(ValueError, match="policy lag"):
        check_matched_recipe(changed, effective, **options)
