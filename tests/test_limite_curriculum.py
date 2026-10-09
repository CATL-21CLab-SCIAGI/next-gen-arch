import pytest

from archlab.rl.limite_curriculum import MathCurriculumSampler


def make_sampler(start, **kwargs):
    return MathCurriculumSampler(
        [{"problem_sha256": str(i)} for i in range(20)], {"problem_sha256": ["0", "1", "2", "3"]},
        max_steps=8, phase_start=start, prompts_per_batch=4, num_generations=4, repeat_count=4,
        anneal_steps=3, **kwargs,
    )


def test_phase_alignment_preserves_groups_reuses_and_shared_data_order():
    left, right = list(make_sampler(2)), list(make_sampler(4))
    batch = 4 * 4 * 4
    assert left[2 * batch:6 * batch] == right[4 * batch:8 * batch]
    for offset in range(0, len(left), batch):
        block = left[offset:offset + batch]
        assert block[:16] == block[16:32] == block[32:48] == block[48:64]
        assert all(len(set(block[i:i + 4])) == 1 for i in range(0, 16, 4))
        assert len(set(block[:16])) == 4


def test_curriculum_rejects_heldout_or_unknown_hashes():
    with pytest.raises(ValueError, match="training-only"):
        MathCurriculumSampler([{"problem_sha256": "train"}], {"problem_sha256": ["heldout"]}, max_steps=1, phase_start=0, prompts_per_batch=1, num_generations=4, repeat_count=4)


def test_curriculum_does_not_remove_full_distribution_and_anneals():
    left, baseline = list(make_sampler(0)), list(make_sampler(0, initial_fraction=0))
    assert left[3 * 64:] == baseline[3 * 64:]
