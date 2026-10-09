"""Memory-bounded native Limite log-probabilities for GRPO replay.

Retain the publisher backbone and softcapped head, checkpointing each head
chunk so backward never retains a sequence-by-vocabulary tensor. The owned
forward binding goes through DDP's public forward boundary.
"""

from functools import wraps
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def chunked_policy_scores(head, hidden, targets, *, chunk_size=512, temperature=1.0, compute_entropy=False):
    if chunk_size < 1 or temperature <= 0 or hidden.shape[:-1] != targets.shape:
        raise ValueError("invalid policy score chunk geometry")
    shape = targets.shape
    flat, labels = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)

    def part(x, y):
        logits = head(x) / temperature
        logps = logits.gather(-1, y[:, None]).squeeze(-1) - logits.logsumexp(-1)
        with torch.no_grad():
            entropy = (logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)
                       if compute_entropy else logits.new_empty(0))
        return logps, entropy

    values, entropies = [], []
    for start in range(0, len(flat), chunk_size):
        x, y = flat[start:start + chunk_size], labels[start:start + chunk_size]
        value, entropy = (checkpoint(part, x, y, use_reentrant=False) if torch.is_grad_enabled() else part(x, y))
        values.append(value)
        entropies.append(entropy)
    return torch.cat(values).reshape(shape), torch.cat(entropies).reshape(shape) if compute_entropy else None


def enable_chunked_policy_scores(model, *, chunk_size=512):
    # The inserted-attention comparisons keep the same publisher LM head and
    # backbone output interface. Admit their verified wrapper too; otherwise
    # an uncapped adapter rollout falls back to a sequence-by-vocabulary tensor.
    from archlab.architectures.limite_adapter import PreludeBackbone

    verified_adapter = (
        isinstance(getattr(model, "model", None), PreludeBackbone)
        and bool(getattr(model, "archlab_base_snapshot_sha256", None))
    )
    if not (getattr(model, "archlab_native_checkpoint", False) or verified_adapter) or chunk_size < 1:
        raise ValueError("chunked native replay requires a verified Limite checkpoint")
    if getattr(model, "archlab_chunked_policy_scores", False):
        raise ValueError("native policy scoring already configured")
    original = model.forward

    @wraps(original)
    def forward(self, *args, archlab_replay=None, **kwargs):
        if archlab_replay is None:
            return original(*args, **kwargs)
        if args or kwargs.get("use_cache") is not False:
            raise ValueError("native replay requires keyword inputs and no decode cache")
        keep, temperature, entropy = archlab_replay
        ids = kwargs["input_ids"]
        if not 0 < keep < ids.shape[1]:
            raise ValueError("replay requires a nonempty prompt and completion")
        hidden = self.model(**kwargs, return_dict=True).last_hidden_state[:, -(keep + 1):-1]
        return chunked_policy_scores(self._softcapped_logits, hidden, ids[:, -keep:],
                                     chunk_size=chunk_size, temperature=temperature, compute_entropy=entropy)

    model.forward = MethodType(forward, model)
    model.archlab_chunked_policy_scores = True


def native_replay_scores(model, input_ids, attention_mask, logits_to_keep, *, temperature, compute_entropy):
    """Replay each real sequence; restore padded score geometry expected by TRL."""
    if attention_mask.shape != input_ids.shape or not 0 < logits_to_keep < input_ids.shape[1]:
        raise ValueError("invalid native replay mask or completion geometry")
    scores, entropies = [], []
    boundary = input_ids.shape[1] - logits_to_keep
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        valid = mask.nonzero().flatten()
        if not len(valid):
            raise ValueError("native replay cannot score an empty sequence")
        start, end = int(valid[0]), int(valid[-1]) + 1
        keep = end - boundary
        if not bool(mask[start:end].all()) or keep > logits_to_keep or start >= boundary:
            raise ValueError("native replay requires contiguous prompt and completion tokens")
        if keep <= 0:
            if end != boundary:
                raise ValueError("native replay requires a complete prompt")
            # A rank can have zero loss tokens while peers have signal. Retain
            # one existing completion in its zero-weight graph so DDP sees the
            # same parameters; do not invent a loss token or change its mask.
            keep, end = 1, boundary + 1
        logps, entropy = model(input_ids=ids[None, start:end], attention_mask=None,
                              use_cache=False, archlab_replay=(keep, temperature, compute_entropy))
        scores.append(torch.nn.functional.pad(logps, (0, logits_to_keep - keep)))
        if compute_entropy:
            entropies.append(torch.nn.functional.pad(entropy, (0, logits_to_keep - keep)))
    return torch.cat(scores), torch.cat(entropies) if compute_entropy else None
