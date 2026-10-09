from types import SimpleNamespace

import pytest
import torch

from archlab.rl.limite_rollout import SamplingLogprobs


def test_streamed_probabilities_match_full_scores_including_eos_padding():
    torch.manual_seed(7)
    scores = [torch.randn(4, 41, dtype=torch.float32) for _ in range(9)]
    # Includes an early EOS followed by padding, as native generation does.
    sampled = torch.tensor([[3, 0, 0, 0, 0, 0, 0, 0, 0], [4, 7, 3, 0, 0, 0, 0, 0, 0],
                            [1, 2, 5, 7, 8, 9, 10, 11, 3], [8, 9, 8, 9, 8, 9, 8, 9, 8]])
    prefix = torch.ones(4, 6, dtype=torch.long)
    recorder = SamplingLogprobs(SimpleNamespace())
    rng = torch.get_rng_state().clone()
    for index, score in enumerate(scores):
        original = score.clone()
        assert recorder(torch.cat([prefix, sampled[:, :index]], dim=1), score) is score
        assert torch.equal(score, original)
    actual = recorder.finish(torch.cat([prefix, sampled], dim=1))
    expected = torch.stack([score.log_softmax(-1).gather(-1, sampled[:, i:i + 1]).squeeze(-1)
                            for i, score in enumerate(scores)], dim=1)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    assert recorder.previous is None and not recorder.selected
    assert recorder.max_distribution_bytes == 4 * 41 * 4


@pytest.mark.parametrize("setting", [dict(temperature=0.7), dict(top_k=50), dict(top_p=0.9),
                                    dict(min_p=0.1), dict(renormalize_logits=True), dict(num_beams=2)])
def test_streaming_rejects_post_processor_distribution_changes(setting):
    with pytest.raises(ValueError, match="probabilities|sampling"):
        SamplingLogprobs(SimpleNamespace(**setting))


def test_streaming_accepts_transformers_unset_identity_processors():
    # Transformers 5 leaves neutral processors unset in GenerationConfig.
    SamplingLogprobs(SimpleNamespace(
        temperature=1.0, top_p=1.0, top_k=0, num_beams=None, typical_p=None,
        epsilon_cutoff=None, eta_cutoff=None, min_p=None, top_h=None,
    ))
