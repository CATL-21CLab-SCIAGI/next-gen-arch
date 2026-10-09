"""Masked attention-teacher KL for the discrete CSA2 indexer's score model."""
import torch


def attention_indexer_kl(scores, teacher_queries, teacher_keys, allowed, *, scale):
    """Mean KL over valid sampled queries; teacher values carry no derivatives.

    Scores are [B,Q,K], queries [B,Q,H,D], keys [B,K,D]. Each teacher head
    normalizes over visible compressed keys; the target averages these heads.
    """
    valid = allowed.any(-1)
    safe_allowed = allowed.clone()
    safe_allowed[..., 0] |= ~valid
    with torch.no_grad():
        target = torch.zeros_like(scores, dtype=torch.float32)
        for head in range(teacher_queries.shape[-2]):
            logits = teacher_queries[:, :, head].float() @ teacher_keys.float().transpose(-1, -2)
            logits.mul_(scale).masked_fill_(~safe_allowed, -torch.inf)
            target.add_(logits.softmax(-1))
        target.div_(teacher_queries.shape[-2])
    log_prob = scores.float().masked_fill(~safe_allowed, -torch.inf).log_softmax(-1)
    # Zero-probability keys must not form 0 * infinity.
    terms = target * (target.clamp_min(1e-30).log() - log_prob.masked_fill(~safe_allowed, 0))
    return (terms.sum(-1) * valid).sum() / valid.sum().clamp_min(1)
