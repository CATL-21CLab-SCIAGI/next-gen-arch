"""Bounded FP32-master Muown updates for the 16-GPU BF16 V4.1 actors.

Masters and momentum live on persistent storage. Each matrix update is atomic;
Adam embeddings are elementwise and can be split into bounded chunks. Model
parameters and accumulated gradients remain on their original Megatron shards.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.optimizer.optimizer import FP32Optimizer
from megatron.training import get_args

from archlab.optimizers.muown import Muown


def _cpu_state(value):
    return {key: item.detach().cpu() if isinstance(item, torch.Tensor) else item
            for key, item in value.items()}


class DiskMuown(torch.optim.Optimizer):
    def __init__(self, named, root, lr, pg_collection):
        self.named = list(named)
        super().__init__([p for _, p in self.named], dict(lr=lr, weight_decay=0.0,
                                                       lr_mult=1.0, wd_mult=1.0,
                                                       is_decoupled_lr=False))
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise ValueError("fresh RL launch cannot overwrite existing optimizer state")
        self.pg_collection = pg_collection
        self.completed_steps = 0
        self.files = set()

    @torch.no_grad()
    def _update(self, name, p, grad, *, adam):
        chunk = 16 * 2**20 if adam else p.numel()
        for first in range(0, p.numel(), chunk):
            count = min(chunk, p.numel() - first)
            key = hashlib.sha256(f"{name}:{first}".encode()).hexdigest() + ".pt"
            path = self.root / key
            selected = p.reshape(-1)[first:first + count] if adam else p
            gradient = grad.reshape(-1)[first:first + count] if adam else grad
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                master = torch.nn.Parameter(payload["master"].to(p.device))
                state = {k: v.to(p.device) if isinstance(v, torch.Tensor) else v
                         for k, v in payload["state"].items()}
            else:
                master = torch.nn.Parameter(selected.float().clone())
                state = {}
            master.grad = gradient.float()
            lr = self.param_groups[0]["lr"]
            if adam:
                state = self._adam(master, state, lr)
            else:
                for attr in ("tensor_model_parallel", "partition_dim", "partition_stride", "expert_tp"):
                    if hasattr(p, attr):
                        setattr(master, attr, getattr(p, attr))
                opt = Muown([master], lr=lr, pg_collection=self.pg_collection)
                opt.state[master] = state
                opt.step()
                state = opt.state[master]
            selected.copy_(master)
            temporary = path.with_suffix(".tmp")
            torch.save({"name": name, "offset": first, "master": master.detach().cpu(),
                        "state": _cpu_state(state)}, temporary)
            os.replace(temporary, path)
            self.files.add(key)

    @staticmethod
    def _adam(master, state, lr):
        if not state:
            state.update(step=0, exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master))
        state["step"] += 1
        state["exp_avg"].lerp_(master.grad, 0.05)
        state["exp_avg_sq"].lerp_(master.grad.square(), 0.05)
        correction = 1 - 0.95 ** state["step"]
        denominator = (state["exp_avg_sq"] / correction).sqrt().add_(1e-8)
        master.addcdiv_(state["exp_avg"], denominator, value=-lr / correction)
        return state

    @torch.no_grad()
    def step(self, closure=None):
        started = time.monotonic()
        for index, (name, p) in enumerate(self.named):
            grad = getattr(p, "main_grad", p.grad)
            if grad is None:
                raise ValueError(f"missing gradient for trainable parameter: {name}")
            adam = p.ndim != 2 or getattr(p, "is_embedding_or_output_parameter", False)
            self._update(name, p, grad, adam=adam)
            if index % 100 == 0:
                print(json.dumps({"event": "muown_update_progress", "rank": dist.get_rank(),
                                  "parameter": index, "total": len(self.named), "name": name}), flush=True)
        self.completed_steps += 1
        print(json.dumps({"event": "muown_optimizer_step_complete", "rank": dist.get_rank(),
                          "step": self.completed_steps, "seconds": time.monotonic() - started}), flush=True)

    def snapshot(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in sorted(self.files):
            target = destination / name
            if not target.exists():
                os.link(self.root / name, target)
        metadata = {"format": "archlab-muown-fp32-stream-v1", "step": self.completed_steps,
                    "files": sorted(self.files), "root": str(destination)}
        (destination / "COMPLETE.json").write_text(json.dumps(metadata, indent=2))
        return metadata


class StreamingOptimizer(FP32Optimizer):
    @torch.no_grad()
    def step(self):
        admission = Path(get_args().save).parents[1] / "numerical-admission-v1"
        if not all((admission / f"rank-{rank}" / "PASSED.json").exists() for rank in range(2)):
            raise RuntimeError("distributed Muown/Engram admission has not passed")
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        tp_rank = mpu.get_tensor_model_parallel_rank()
        for name, p in self.optimizer.named:
            grad = p.main_grad
            expert = ".experts." in name and ".shared_experts." not in name
            counted = expert or getattr(p, "tensor_model_parallel", False) or tp_rank == 0
            if counted:
                for chunk in grad.reshape(-1).split(16 * 2**20):
                    total.add_(chunk.float().square().sum(dtype=torch.float64))
        dist.all_reduce(total)
        norm = total.sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite global policy gradient")
        factor = min(1.0, self.config.clip_grad / (norm.item() + 1e-6))
        for _, p in self.optimizer.named:
            p.main_grad.mul_(factor)
        self.optimizer.step()
        self.grad_norms_by_group = {}
        return True, norm.item(), None

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        root = Path(get_args().save) / "optimizer-snapshots" / f"step-{self.optimizer.completed_steps:06d}" / f"rank-{dist.get_rank():02d}"
        data = {} if is_loading else self.optimizer.snapshot(root)
        return {"archlab_streaming_state": ShardedObject(
            "optimizer.archlab_streaming_state", data,
            global_shape=(dist.get_world_size(),), global_offset=(dist.get_rank(),), replica_id=0)}

    def load_state_dict(self, state_dict):
        state = state_dict["archlab_streaming_state"]
        if state["format"] != "archlab-muown-fp32-stream-v1":
            raise ValueError("unsupported optimizer state")
        for name in state["files"]:
            if Path(name).name != name:
                raise ValueError("invalid optimizer state filename")
            os.link(Path(state["root"]) / name, self.optimizer.root / name)
        self.optimizer.files = set(state["files"])
        self.optimizer.completed_steps = state["step"]


def build_optimizer(config, model_chunks, **kwargs):
    from megatron.core.process_groups_config import ProcessGroupCollection
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers

    args = get_args()
    if (mpu.get_data_parallel_world_size() != 1 or mpu.get_expert_tensor_parallel_world_size() != 1):
        raise ValueError("streaming Muown admission requires dense DP1 and expert TP1")
    named = [(name, p) for name, p in named_params_and_buffers(args, model_chunks) if p.requires_grad]
    for name, param in named:
        if ".experts." in name and ".shared_experts." not in name:
            param.expert_tp = True
    root = Path(args.save).parent / "optimizer-live" / f"rank-{dist.get_rank():02d}"
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "expt_tp"])
    inner = DiskMuown(named, root, config.lr, groups)
    return StreamingOptimizer(inner, config, lambda *_: None)
