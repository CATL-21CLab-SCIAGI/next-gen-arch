"""Container-owned Adam with a no-signal no-op, preserving native checkpoints."""

from transformer_engine.pytorch.optimizers import FusedAdam


class SignalFusedAdam(FusedAdam):
    def step(self, closure=None, **kwargs):
        # TE has a group-wide step counter, so grad=None alone is insufficient
        # to keep a flat-reward batch from advancing the Adam bias correction.
        if not any(p.grad is not None for group in self.param_groups for p in group["params"]):
            return None
        return super().step(closure=closure, **kwargs)
