"""Synchronous MCore-compatible checkpoints with bounded host staging."""
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)
from torch.distributed import checkpoint
from torch.distributed.checkpoint import FileSystemWriter


class StreamingSaveStrategy(TorchDistSaveShardedStrategy):
    def save(self, sharded_state_dict, checkpoint_dir):
        sharded, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica)
        state = mcore_to_pyt_state_dict(sharded, False)
        # MCore's default synchronous wrapper stages every tensor on CPU before
        # writing. The public DCP writer stages one tensor plus bounded lookahead.
        checkpoint.save(state, storage_writer=FileSystemWriter(
            checkpoint_dir, thread_count=1, per_thread_copy_ahead=16 * 2**20),
            planner=MCoreSavePlanner(dedup_replicated_tensors=not self.keep_only_main_replica,
                                    flatten_state_dict=False, flatten_sharded_tensors=False))


def install(args):
    """Use MCore's save-strategy extension through the Miles adapter boundary."""
    from miles.backends.megatron_utils import model
    if args.async_save:
        raise ValueError("resident qualification requires synchronous checkpoints")
    if getattr(model.save_checkpoint, "_archlab_streaming", False):
        return dict(installed=True, already_installed=True)
    original = model.save_checkpoint
    runtime_args = args

    def save_checkpoint(*args, **kwargs):
        context = dict(kwargs.get("checkpointing_context") or {})
        context["save_strategy"] = StreamingSaveStrategy()
        kwargs["checkpointing_context"] = context
        result = original(*args, **kwargs)
        import json
        from pathlib import Path
        launch = json.loads((Path(runtime_args.save).parent / "LAUNCH_PROCESS.json").read_text())
        if launch["action"] == "pilot":
            from archlab.megatron.miles_v41_checkpoint_verify import restore_and_verify
            restore_and_verify(runtime_args, *args[:4])
        return result

    save_checkpoint._archlab_streaming = True
    model.save_checkpoint = save_checkpoint
    return dict(installed=True, host_copy_ahead_bytes=16 * 2**20)
