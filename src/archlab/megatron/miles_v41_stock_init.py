"""Checkpoint compatibility for the unmodified upstream Miles training loop."""


def initialize(args):
    import os

    from miles.backends.megatron_utils import checkpoint

    from archlab.megatron.miles_v41_checkpoint import load_parent
    from archlab.megatron.miles_v41_checkpoint_kind import install as install_checkpoint_kind
    from archlab.megatron.miles_v41_checkpoint_writer import install as install_checkpoint_writer
    from archlab.megatron.miles_v41_sync import install
    from archlab.megatron.miles_v41_weight_session import install as install_session

    checkpoint._load_hf_weights_with_mbridge = load_parent
    install_checkpoint_kind()
    install_checkpoint_writer()
    install()
    install_session()
    if os.environ.get("ARCHLAB_MILES_RESIDENT_POLICY") == "1":
        from miles.backends.megatron_utils.actor import MegatronTrainRayActor

        from archlab.megatron.miles_v41_resident_policy import install as install_resident

        if args.offload_train or not args.colocate or args.keep_old_actor:
            raise ValueError("resident policy requires colocated, non-offloaded single-actor training")
        install_resident(MegatronTrainRayActor)
