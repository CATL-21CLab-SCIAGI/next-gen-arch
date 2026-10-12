"""Package import boundary for native Limite under verl's Megatron engine."""
from importlib.metadata import PackageNotFoundError, version

try:
    version("flash-attn-4")
except PackageNotFoundError:
    pass
else:
    # FA4 and the container's FA2 share a namespace. Reuse its verified loader
    # before Transformer Engine imports the installed optional FA4 extension.
    from archlab.architectures.fa4_attention import fa4_runtime_contract

    fa4_runtime_contract()

from archlab.megatron.limite import NativeLimiteBridge  # noqa: E402, F401
from archlab.training.metrics import register_flops  # noqa: E402

register_flops()
