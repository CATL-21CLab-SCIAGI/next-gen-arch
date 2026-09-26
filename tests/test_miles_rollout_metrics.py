import ast
from types import SimpleNamespace

from archlab.rl.miles_rollout_metrics import (
    generated_tokens_at_version,
    log_eval,
    normalize_reward_scalars,
    reward_group_metrics,
)


def sample(group, reward, length=10, removed=False):
    result = SimpleNamespace(group_index=group, reward=reward, effective_response_length=length,
                             remove_sample=removed)
    result.get_reward_value = lambda args: result.reward
    return result


def test_constant_groups_and_signal_tokens_are_counted_separately():
    rows = [sample(0, 0), sample(0, 0), sample(1, 1), sample(1, 1),
            sample(2, 0, 3), sample(2, 1, 7), sample(3, 0.5), sample(3, 0.5)]
    metrics = reward_group_metrics(None, rows)
    assert metrics['groups'] == 4
    assert metrics['constant_groups'] == 3
    assert metrics['all_zero_groups'] == metrics['all_one_groups'] == 1
    assert metrics['contrast_groups'] == 1
    assert metrics['reward_contrast_tokens'] == 10
    assert metrics['response_tokens'] == 70
    assert metrics['group_contrast_fraction'] == 0.25
    assert metrics['token_contrast_fraction'] == 1 / 7


def test_reward_scalar_normalization_fixes_native_string_lookup_without_changing_rewards():
    rows = [sample(0, 0), sample(0, 0), sample(1, 1), sample(1, 1)]
    before = [row.reward for row in rows]
    normalize_reward_scalars(rows)
    assert [row.reward for row in rows] == before
    # Reproduce the pinned logger's key construction and percentage lookup.
    keys = [str(round(rows[i].reward, 1)) for i in (0, 2)]
    assert keys.count('0.0') / 2 == keys.count('1.0') / 2 == 0.5


def test_invalid_rewards_are_not_counted_as_useful_gradient_signal():
    metrics = reward_group_metrics(None, [sample(0, None), sample(0, 1),
                                         sample(1, float('nan')), sample(1, 0)])
    assert metrics['invalid_groups'] == 2
    assert metrics['contrast_groups'] == 0


def test_removed_samples_do_not_inflate_signal_token_counts():
    metrics = reward_group_metrics(None, [sample(0, 0, removed=True), sample(0, 1, length=3)])
    assert metrics['reward_contrast_tokens'] == metrics['response_tokens'] == 3


def test_eval_rows_recover_prompt_groups_without_changing_rewards():
    rows = [sample(None, reward) for reward in (0, 0, 1, 1)]
    for index, row in enumerate(rows):
        row.index = index
    args = SimpleNamespace(eval_datasets=[SimpleNamespace(name='heldout', n_samples_per_eval_prompt=2)])
    assert log_eval(0, args, {'heldout': {'samples': rows}}, None) is False
    metrics = reward_group_metrics(args, rows)
    assert metrics['groups'] == 2
    assert metrics['all_zero_groups'] == metrics['all_one_groups'] == 1
    assert [row.reward for row in rows] == [0, 0, 1, 1]


def test_generation_accounting_excludes_reused_prefixes():
    row = SimpleNamespace(all_weight_version_spans=[
        SimpleNamespace(version='1', abs_start=100, abs_end=150),
        SimpleNamespace(version='2', abs_start=150, abs_end=170),
    ])
    assert generated_tokens_at_version([row], 2) == 20
    assert generated_tokens_at_version([row], 3) == 0


def test_partial_age_keeps_boundary_and_drops_only_stale_groups():
    # This pure function is shared with the GPU-only rollout class. Extract it
    # without importing Miles' runtime dependencies into the portable CPU suite.
    from pathlib import Path

    source = Path(__file__).parents[1] / 'src/archlab/rl/miles_qualified_rollout.py'
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'split_stale_groups')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    groups = [[SimpleNamespace(oldest_weight_version=version)] for version in (None, 2, 3, 5)]
    kept, stale = namespace['split_stale_groups'](groups, current_version=5, max_age=2)
    assert kept == [groups[0], groups[2], groups[3]]
    assert stale == [groups[1]]
