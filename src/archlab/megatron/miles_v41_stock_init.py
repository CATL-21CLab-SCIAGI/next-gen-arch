"""Checkpoint compatibility for the unmodified upstream Miles training loop."""


def initialize(args):
    from miles.backends.megatron_utils import checkpoint

    from archlab.megatron.miles_v41_checkpoint import load_parent
    from archlab.megatron.miles_v41_sync import install
    from archlab.megatron.miles_v41_weight_session import install as install_session

    checkpoint._load_hf_weights_with_mbridge = load_parent
    install()
    install_session()
