from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from archlab.rl.limite_actor import _decode_capacities
from archlab.rl.limite_phase import discard_prefetched_rollouts, protocol_transition
from archlab.rl.limite_protocol import MathRolloutProtocol, response_budget, score_math_rollout
from archlab.rl.limite_scoring import (
    chunked_policy_scores,
    enable_chunked_policy_scores,
    native_replay_scores,
)


def new_spec():
    return yaml.safe_load((Path(__file__).parents[1] / "recipes/limite/violetto_math_rl_native_context.yaml").read_text())


def test_native_budget_uses_actual_prompt_and_has_one_precaptured_capacity():
    config = SimpleNamespace(max_new_tokens=131072, archlab_budget_mode="native_context")
    for prompt in (1, 99, 2048):
        assert response_budget(config, prompt, 131072) == 131072 - prompt
    assert _decode_capacities(131072, 2048, 1024, 131072, budget_mode="native_context") == (131072,)
    with pytest.raises(ValueError, match="prompt"):
        response_budget(config, 131072, 131072)
    with pytest.raises(ValueError, match="match"):
        response_budget(config, 100, 65536)
    assert response_budget(SimpleNamespace(max_new_tokens=16384), 100, 131072) == 16384


def test_capped_context_is_a_verification_failure_without_artificial_length_penalty():
    spec = new_spec()
    protocol = MathRolloutProtocol(**spec["rollout"])
    tokens = [42] * (131072 - 2048)
    capped = score_math_rollout(r"<think>\boxed{42}", "42", tokens, "length", protocol,
                               training=True, completion_budget=len(tokens))
    assert capped["accuracy"] == capped["reward"] == 0
    assert capped["native_context_exhausted"]
    assert capped["overlong_penalty"] == capped["unfinished_penalty"] == 0
    complete = score_math_rollout(r"</think>\boxed{42}", "42", tokens[:-1] + [151645], "eos", protocol,
                                 training=True, completion_budget=len(tokens))
    assert complete["reward"] == 1
    with pytest.raises(ValueError, match="actual per-prompt"):
        score_math_rollout("42", "42", tokens, "length", protocol, training=True)
    with pytest.raises(ValueError, match="exceeds"):
        score_math_rollout("42", "42", tokens, "length", protocol, training=True,
                           completion_budget=len(tokens) - 1)
    with pytest.raises(ValueError, match="fewer tokens"):
        score_math_rollout("42", "42", [42] * 16384, "length", protocol, training=True,
                           completion_budget=len(tokens))


def test_historical_contract_is_identical_and_phase_migration_is_explicit():
    spec = new_spec()
    assert MathRolloutProtocol().contract() == spec["protocol_transition"]["previous_protocol"]
    protocol = MathRolloutProtocol(**spec["rollout"])
    receipt = dict(step=142, phase_start=0, math_protocol=MathRolloutProtocol().contract(), curriculum_sha256="same")
    kwargs = dict(phase_start=142, curriculum_sha256="same", new_tracking_phase=True)
    result = protocol_transition(receipt, protocol, spec, **kwargs)
    assert result["optimizer_scheduler_retained"] and not result["same_experiment_phase"]
    for change in (dict(phase_start=141), dict(new_tracking_phase=False), dict(curriculum_sha256="other")):
        with pytest.raises(ValueError, match="explicit phase"):
            protocol_transition(receipt, protocol, spec, **(kwargs | change))
    with pytest.raises(ValueError, match="explicit phase"):
        protocol_transition(receipt, protocol, {}, **kwargs)


def test_discard_old_protocol_prefetch_rewinds_actor_but_retains_learner_and_checkpoint():
    before, after = torch.tensor([1, 2]), torch.tensor([3, 4])
    state = dict(world_size=2, scheduler={"last_epoch": 142}, evidence=dict(applied_updates=141, flat_batches=7),
                 rank_rng=[dict(cuda=after.clone(), rollout=dict(generator=after.clone(),
                                pending=dict(rng_before=before.clone(), completion_ids=[[42] * 16384])))
                           for _ in range(2)])
    result = discard_prefetched_rollouts(state)
    assert result["scheduler"] == state["scheduler"]
    assert result["evidence"]["applied_updates"] == 141 and result["evidence"]["flat_batches"] == 0
    assert result["evidence"]["discarded_old_protocol_prefetched_batches"] == 2
    for old, new in zip(state["rank_rng"], result["rank_rng"], strict=True):
        assert old["rollout"]["pending"] is not None
        assert new["rollout"]["pending"] is None
        assert torch.equal(new["cuda"], after)
        assert torch.equal(new["rollout"]["generator"], before)


@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_chunked_softcapped_scores_match_full_values_entropy_and_parameter_gradients(chunk):
    torch.manual_seed(1)
    x = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 7))
    advantages = torch.randn(2, 7, dtype=torch.float64)
    def head(hidden):
        return 30 * ((hidden @ weight.T + 2) / 10).sigmoid()
    logits = head(x)
    expected = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    entropy = -(logits.log_softmax(-1) * logits.softmax(-1)).sum(-1)
    grad = torch.autograd.grad((expected * advantages).sum(), (x, weight))
    actual, actual_entropy = chunked_policy_scores(head, x, labels, chunk_size=chunk, compute_entropy=True)
    actual_grad = torch.autograd.grad((actual * advantages).sum(), (x, weight))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_entropy, entropy)
    assert not actual_entropy.requires_grad
    for got, want in zip(actual_grad, grad, strict=True):
        torch.testing.assert_close(got, want)


def test_owned_forward_preserves_normal_forward_and_causal_shift_without_state_changes():
    class Backbone(torch.nn.Embedding):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=super().forward(input_ids))

    class Model(torch.nn.Module):
        archlab_native_checkpoint = True
        def __init__(self):
            super().__init__()
            self.model = Backbone(11, 5)
            self.lm_head = torch.nn.Linear(5, 11)
        def _softcapped_logits(self, x):
            return 30 * (self.lm_head(x) / 10).sigmoid()
        def forward(self, input_ids, **kwargs):
            return self._softcapped_logits(self.model(input_ids).last_hidden_state)
    model = Model()
    before = deepcopy(model.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    full = model(ids)
    enable_chunked_policy_scores(model, chunk_size=2)
    torch.testing.assert_close(model(ids), full)
    scores, entropy = model(input_ids=ids, use_cache=False, archlab_replay=(3, 1.0, False))
    torch.testing.assert_close(scores, full[:, 1:-1].log_softmax(-1).gather(-1, ids[:, -3:, None]).squeeze(-1))
    assert entropy is None and before.keys() == model.state_dict().keys()
    for name in before:
        torch.testing.assert_close(before[name], model.state_dict()[name])


def test_native_replay_strips_only_padding_and_restores_original_grpo_geometry():
    calls = []
    def model(input_ids, attention_mask, use_cache, archlab_replay):
        keep, _, entropy = archlab_replay
        assert attention_mask is None and not use_cache
        calls.append(input_ids.tolist())
        scores = input_ids[:, -keep:].float()
        return scores, scores + 1 if entropy else None
    ids = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 5, 6]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    scores, entropy = native_replay_scores(model, ids, mask, 3, temperature=1., compute_entropy=True)
    assert calls == [[[1, 2, 3, 4]], [[1, 2, 3, 4, 5, 6]]]
    assert scores.tolist() == [[3, 4, 0], [4, 5, 6]]
    assert entropy.tolist() == [[4, 5, 0], [5, 6, 7]]
    mask[1, 3] = 0
    with pytest.raises(ValueError, match="contiguous"):
        native_replay_scores(model, ids, mask, 3, temperature=1., compute_entropy=False)


def test_zero_loss_rank_retains_a_zero_gradient_graph_without_unmasking_tokens():
    weight = torch.nn.Parameter(torch.tensor(1.))
    def model(input_ids, attention_mask, use_cache, archlab_replay):
        assert input_ids.tolist() == [[1, 2, 3]]
        assert archlab_replay[0] == 1
        return weight * input_ids[:, -1:].float(), None
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 1, 1, 0, 0, 0]])
    old_mask = mask.clone()
    scores, _ = native_replay_scores(model, ids, mask, 3, temperature=1., compute_entropy=False)
    (scores * mask[:, -3:]).sum().backward()
    assert scores.shape == (1, 3) and weight.grad.item() == 0
    assert torch.equal(old_mask, mask)
