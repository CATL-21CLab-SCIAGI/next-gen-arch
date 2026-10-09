"""Explicit extension registration, leaving the installed runtime source intact."""


def initialize(args):
    import os

    from miles.backends.megatron_utils import checkpoint, model

    from archlab.megatron.miles_muown_streaming import build_optimizer
    from archlab.megatron.miles_v41_checkpoint import load_parent
    from archlab.megatron.miles_v41_sync import install

    resident = os.environ.get("ARCHLAB_RL_OFFLOAD_POLICY") == "forbidden"
    if resident:
        if args.offload_train or os.environ.get("ARCHLAB_RL_FREEZE_ENGRAM") != "1":
            raise ValueError("resident RL requires frozen Engram and explicit no-offload flags")
        from archlab.megatron.miles_v41_resident import build_optimizer, install_live_policy
        install_live_policy()
        from archlab.megatron.miles_v41_streaming_checkpoint import install as install_checkpoint
        install_checkpoint(args)
    model.get_megatron_muon_optimizer = build_optimizer
    checkpoint._load_hf_weights_with_mbridge = load_parent
    install()
    if os.environ.get("EVERGREENTREE_WEIGHT_CACHE_DIR") and not resident:
        from archlab.megatron.miles_v41_storage import install_backuper
        install_backuper()
    print("EvergreenTree optimizer=Muown fresh_state=True ratio_mask=[0.2,5.0]", flush=True)


def prepare_weight_updater(args):
    """Wait for asymmetric checkpoint/backup IO before starting collectives."""
    import json
    import os
    import time
    from datetime import timedelta
    from pathlib import Path

    import torch.distributed as dist
    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    root = Path(args.save).parent
    started = json.loads((root / "LAUNCH_PROCESS.json").read_text())["started_at_epoch"]
    ready = root / f"weight-backup-ready-rank-{dist.get_rank():02d}.json"
    ready.write_text(json.dumps({"launch_epoch": started, "rank": dist.get_rank(), "pid": os.getpid()}))
    print(f"EvergreenTree rank {dist.get_rank()} weight backup ready; waiting for all ranks", flush=True)
    deadline = time.monotonic() + 7200
    while True:
        complete = 0
        for rank in range(dist.get_world_size()):
            path = root / f"weight-backup-ready-rank-{rank:02d}.json"
            try:
                complete += json.loads(path.read_text())["launch_epoch"] == started
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        if complete == dist.get_world_size():
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Only {complete} ranks finished parent load and weight backup")
        time.sleep(5)
    # The first launch used the default ten-minute timeout. Preserve those loaded
    # models while giving the first disk-backed transfer enough time on NAS.
    timeout = timedelta(minutes=120)
    for group in ReloadableProcessGroup.GROUPS.get(os.getpid(), []):
        group.inner_kwargs["timeout"] = timeout
        if group.group is not None and group.group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.distributed_c10d._set_pg_timeout(timeout, group.group)
    print("EvergreenTree all ranks loaded and backed up; weight transfer admission ready", flush=True)
