"""Write immutable evaluation checkpoints through NAS links to a verified OSS mount."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from archlab.artifacts import atomic_write_json
from archlab.storage.checkpoint_retention import validate


def require_oss_mount(target):
    existing = target
    while not existing.exists():
        existing = existing.parent
    info = json.loads(
        subprocess.check_output(
            ["findmnt", "--json", "--target", str(existing), "--output", "TARGET,FSTYPE"], text=True
        )
    )["filesystems"][0]
    if not info["fstype"].startswith("fuse.ossfs"):
        raise ValueError(f"checkpoint destination is not mounted OSS: {target}")
    return info


def prepare_link(local, target):
    local, target = Path(local), Path(target)
    if not target.is_absolute() or target.resolve() != target:
        raise ValueError("OSS destination must be an absolute path without symlinks")
    if local.exists() or local.is_symlink() or target.exists():
        raise FileExistsError("checkpoint names are immutable; refusing to overwrite")
    require_oss_mount(target)
    target.mkdir(parents=True, exist_ok=False)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.symlink_to(target, target_is_directory=True)
    return local


def on_rank_zero(function):
    import torch.distributed as dist

    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {"value": function()}
        except Exception as error:
            result[0] = {"error": repr(error)}
    dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def prepare_distributed_checkpoint(local, target):
    on_rank_zero(lambda: str(prepare_link(local, target)))
    return Path(local)


def record_evaluation_checkpoint(output, checkpoint, *, milestone, budget=10_000_000_000):
    def record():
        if budget not in (1_000_000_000, 10_000_000_000) or milestone not in range(
            budget // 5, budget + 1, budget // 5
        ):
            raise ValueError("evaluation checkpoint is outside the five declared milestones")
        if not checkpoint.is_symlink():
            raise ValueError("evaluation checkpoint must link to OSS")
        target = checkpoint.resolve(strict=True)
        require_oss_mount(target)
        receipt = validate(target)
        marker = json.loads((target / "COMPLETE.json").read_text())
        if marker.get("contract", {}).get("target_supervised_tokens", budget) != budget:
            raise ValueError("checkpoint training budget differs from its milestone schedule")
        if marker["cursor"]["supervised_tokens"] < milestone:
            raise ValueError("checkpoint predates its token milestone")
        path = output / "EVAL_CHECKPOINTS.json"
        catalog = json.loads(path.read_text()) if path.exists() else {"checkpoints": []}
        if any(row["milestone_tokens"] == milestone for row in catalog["checkpoints"]):
            raise ValueError("duplicate evaluation checkpoint milestone")
        catalog["checkpoints"].append(
            {
                "milestone_tokens": milestone,
                "path": str(checkpoint),
                "oss_path": str(target),
                "cursor": marker["cursor"],
                "verification": receipt,
            }
        )
        atomic_write_json(path, catalog)
        return receipt

    return on_rank_zero(record)
