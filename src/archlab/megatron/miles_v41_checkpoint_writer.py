"""Use native synchronous DCP writing without whole-policy CPU staging."""

from functools import wraps


def make_strategy():
    import torch.distributed.checkpoint as dcp
    from megatron.core.dist_checkpointing.strategies.torch import (
        MCoreSavePlanner,
        TorchDistSaveShardedStrategy,
        _replace_state_dict_keys_with_sharded_keys,
        mcore_to_pyt_state_dict,
    )

    class SerialTorchDistSave(TorchDistSaveShardedStrategy):
        def save(self, sharded_state_dict, checkpoint_dir):
            translated, _, _ = _replace_state_dict_keys_with_sharded_keys(
                sharded_state_dict, self.keep_only_main_replica
            )
            dcp.save(
                state_dict=mcore_to_pyt_state_dict(translated, False),
                storage_writer=dcp.FileSystemWriter(
                    checkpoint_dir,
                    thread_count=1,
                    per_thread_copy_ahead=0,
                    # This pinned PyTorch retains tensor_dict until each file
                    # closes, including for torch serialization. One tensor per
                    # file therefore also bounds retained CPU copies.
                    single_file_per_rank=False,
                ),
                planner=MCoreSavePlanner(
                    dedup_replicated_tensors=False,
                    flatten_state_dict=False,
                    flatten_sharded_tensors=False,
                ),
            )

    return SerialTorchDistSave()


def install():
    from miles.backends.megatron_utils import model

    original = model.save_checkpoint
    if getattr(original, "_archlab_serial_writer", False):
        return

    @wraps(original)
    def save_checkpoint(*args, **kwargs):
        config = model.get_args()
        if config.async_save or config.ckpt_format != "torch_dist" or config.ckpt_assume_constant_structure:
            raise ValueError("serial checkpoint writer requires synchronous torch_dist with sharding validation")
        context = dict(kwargs.get("checkpointing_context") or {})
        context["save_strategy"] = make_strategy()
        kwargs["checkpointing_context"] = context
        return original(*args, **kwargs)

    save_checkpoint._archlab_serial_writer = True
    model.save_checkpoint = save_checkpoint
