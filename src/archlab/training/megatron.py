"""Package import boundary for native Limite under verl's Megatron engine."""
from archlab.training.runtime import flash_attn_namespace

# Transformer Engine imports the container's optional FA4 extension.
flash_attn_namespace()

from archlab.megatron.limite import NativeLimiteBridge  # noqa: E402, F401
from archlab.training.metrics import register_flops  # noqa: E402

register_flops()
