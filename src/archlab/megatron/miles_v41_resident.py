"""GPU-only mixed-precision Muown/Sinkhorn for frozen-Engram RL.

No live tensor is copied to CPU or disk. Checkpoint serialization is the sole
exception and is explicitly called by Megatron at a checkpoint boundary.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.optimizer.optimizer import FP32Optimizer

from archlab.optimizers.muown import Muown
from archlab.optimizers.sinkhorn import sinkhorn_step


class ResidentMuown(torch.optim.Optimizer):
    def __init__(self, named, lr, pg_collection, momentum_dtype=torch.bfloat16,
                 orthogonalization_dtype=torch.float32):
        self.named = list(named)
        super().__init__([p for _, p in self.named], dict(lr=lr, lr_mult=1., wd_mult=1.,
                         weight_decay=0., is_decoupled_lr=False))
        self.pg_collection = pg_collection
        self.completed_steps = 0
        self.entries = {}
        self.momentum_dtype = momentum_dtype
        self.orthogonalization_dtype = orthogonalization_dtype

    @torch.no_grad()
    def initialize_state(self):
        for name, p in self.named:
            if name in self.entries:
                continue
            if p.device.type != "cuda" or not p.requires_grad or name.endswith("v41_engram.table"):
                raise ValueError(f"invalid resident optimizer parameter: {name}")
            master = torch.nn.Parameter(p.detach().float().clone())
            for key in ("tensor_model_parallel", "partition_dim", "partition_stride", "expert_tp"):
                if hasattr(p, key):
                    setattr(master, key, getattr(p, key))
            embedding = getattr(p, "is_embedding_or_output_parameter", False)
            if embedding:
                if p.ndim != 2 or getattr(p, "partition_dim", 0) != 0:
                    raise ValueError("resident Sinkhorn requires row-sharded embeddings")
                entry = dict(master=master, kind="sinkhorn", momentum=torch.zeros_like(p, dtype=self.momentum_dtype))
                if self.momentum_dtype == torch.float16:
                    entry["momentum_scale"] = torch.ones((), device=p.device, dtype=torch.float32)
            elif p.ndim == 2:
                opt = Muown([master], lr=self.param_groups[0]["lr"], pg_collection=self.pg_collection,
                            momentum_dtype=self.momentum_dtype, orthogonalization_dtype=self.orthogonalization_dtype)
                opt._init_group(opt.param_groups[0], skip_non_grad_params=False)
                entry = dict(master=master, kind="muown", optimizer=opt)
            else:
                entry = dict(master=master, kind="adam", step=0,
                             exp_avg=torch.zeros_like(master), exp_avg_sq=torch.zeros_like(master))
            self.entries[name] = entry

    @torch.no_grad()
    def step(self, closure=None):
        self.initialize_state()
        lr = self.param_groups[0]["lr"]
        for name, p in self.named:
            entry = self.entries[name]
            master = entry["master"]
            grad = getattr(p, "main_grad", p.grad)
            if grad is None:
                raise ValueError(f"missing gradient: {name}")
            if entry["kind"] == "muown":
                master.grad = grad.float()
                opt = entry["optimizer"]
                opt.param_groups[0]["lr"] = lr
                opt.step()
                master.grad = None
            elif entry["kind"] == "sinkhorn":
                group = self.pg_collection.tp if getattr(p, "tensor_model_parallel", False) else None
                sinkhorn_step(master, grad, entry["momentum"], lr=lr, group=group,
                              momentum_scale=entry.get("momentum_scale"))
            else:
                entry["step"] += 1
                gradient = grad.float()
                entry["exp_avg"].lerp_(gradient, .05)
                entry["exp_avg_sq"].lerp_(gradient.square(), .05)
                correction = 1 - .95 ** entry["step"]
                denom = (entry["exp_avg_sq"] / correction).sqrt().add_(1e-8)
                master.addcdiv_(entry["exp_avg"], denom, value=-lr / correction)
            p.copy_(master)
        self.completed_steps += 1

    def resident_state(self):
        data = {}
        for name, entry in self.entries.items():
            data[name] = {k: (v.state_dict() if k == "optimizer" else v) for k, v in entry.items()}
        return dict(format="archlab-resident-momentum-v2", momentum_dtype=str(self.momentum_dtype),
                    orthogonalization_dtype=str(self.orthogonalization_dtype),
                    step=self.completed_steps, entries=data)

    def restore(self, payload):
        if (payload["format"] != "archlab-resident-momentum-v2"
                or payload["orthogonalization_dtype"] != str(self.orthogonalization_dtype)
                or payload["momentum_dtype"] != str(self.momentum_dtype)):
            raise ValueError("incompatible optimizer state")
        self.initialize_state()
        if self.entries.keys() != payload["entries"].keys():
            raise ValueError("optimizer parameter coverage mismatch")
        with torch.no_grad():
            for name, saved in payload["entries"].items():
                entry = self.entries[name]
                if saved["kind"] != entry["kind"]:
                    raise ValueError("optimizer kind changed")
                for key, value in saved.items():
                    if key == "optimizer":
                        opt = entry[key]
                        from archlab.megatron.miles_v41_inplace_restore import load_muown_in_place
                        load_muown_in_place(opt, value)
                    elif isinstance(value, torch.Tensor):
                        entry[key].copy_(value)
                    else:
                        entry[key] = value
        self.completed_steps = payload["step"]


class ResidentOptimizer(FP32Optimizer):
    @torch.no_grad()
    def step(self):
        from archlab.megatron.miles_v41_policy_parity import active
        if active is not None:
            active.finish()
        total = torch.zeros((), device="cuda", dtype=torch.float64)
        for name, p in self.optimizer.named:
            grad = p.main_grad
            expert = ".experts." in name and ".shared_experts." not in name
            if expert or getattr(p, "tensor_model_parallel", False) or mpu.get_tensor_model_parallel_rank() == 0:
                for chunk in grad.reshape(-1).split(16 * 2**20):
                    total += chunk.float().square().sum(dtype=torch.float64)
        dist.all_reduce(total)
        norm = total.sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        if norm.item() == 0:
            return True, 0., None
        scale = min(1., self.config.clip_grad / (norm.item() + 1e-6))
        for _, p in self.optimizer.named:
            p.main_grad.mul_(scale)
        self.optimizer.step()
        self.grad_norms_by_group = {}
        print(f"ARCHLAB_RESIDENT_UPDATE step={self.optimizer.completed_steps} norm={norm.item()}", flush=True)
        return True, norm.item(), None

    def state_dict(self):
        return {"archlab_resident": self.optimizer.resident_state()}

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
        from archlab.megatron.miles_v41_resident_checkpoint import pack_state
        self.optimizer.initialize_state()
        return pack_state(self.optimizer.resident_state())

    def load_state_dict(self, state_dict):
        from archlab.megatron.miles_v41_resident_checkpoint import unpack_state
        payload = (state_dict["archlab_resident"] if "archlab_resident" in state_dict
                   else unpack_state(state_dict))
        self.optimizer.restore(payload)


def build_optimizer(config, model_chunks, **kwargs):
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.training import get_args
    from miles.backends.megatron_utils.named_weights import named_params_and_buffers

    from archlab.megatron import miles_v41_policy_parity
    torch.backends.cuda.matmul.allow_tf32 = False
    miles_v41_policy_parity.active = miles_v41_policy_parity.InitialPolicyParity(Path(get_args().save).parent)

    if mpu.get_data_parallel_world_size() != 1 or mpu.get_expert_tensor_parallel_world_size() != 1:
        raise ValueError("resident optimizer requires DP1 and expert TP1")
    named = [(n, p) for n, p in named_params_and_buffers(get_args(), model_chunks) if p.requires_grad]
    for name, p in named:
        if name.endswith("v41_engram.table"):
            raise ValueError("Engram tables must be frozen before DDP construction")
        if ".experts." in name and ".shared_experts." not in name:
            p.expert_tp = True
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "expt_tp"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[os.environ["ARCHLAB_RL_MOMENTUM_DTYPE"]]
    return ResidentOptimizer(ResidentMuown(named, config.lr, groups, dtype), config, lambda *_: None)


class LivePolicy:
    """Alias the live GPU policy; never create a backup or restore a snapshot."""
    def __init__(self, getter):
        self.getter = getter
        self.actor = getter.__self__
        self.gradients_paused = False
        self.audit_written = False
        for model in self.actor.model:
            model.register_forward_pre_hook(self._before_forward)
            original_zero = model.zero_grad_buffer

            def zero_grad_buffer(*args, _original=original_zero, **kwargs):
                self._before_forward(None, None)
                return _original(*args, **kwargs)

            model.zero_grad_buffer = zero_grad_buffer

    def _before_forward(self, module, inputs):
        if self.gradients_paused:
            from torch_memory_saver import torch_memory_saver
            torch_memory_saver.resume(tag="grad_buffer")
            for _, p in self.actor.optimizer.optimizer.named:
                p.main_grad.zero_()
            self.gradients_paused = False

    @property
    def backup_tags(self):
        return ["actor"]

    def get(self, tag):
        if tag != "actor":
            raise ValueError("only one policy is permitted")
        weights = dict(self.getter())
        if any(t.device.type != "cuda" for t in weights.values()):
            raise ValueError("policy offloading is forbidden")
        return weights

    def backup(self, tag):
        self.get(tag)
        if not self.gradients_paused:
            from torch_memory_saver import torch_memory_saver
            torch.cuda.synchronize()
            torch_memory_saver.pause(tag="grad_buffer")
            self.gradients_paused = True
        # Loading has completed. Masters must never be initialized from the
        # uninitialized model that exists before checkpoint import.
        self.actor.optimizer.optimizer.initialize_state()
        if not self.audit_written:
            from archlab.megatron.miles_v41_runtime_audit import audit_actor
            audit_actor(self.actor)
            self.audit_written = True
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        optimizer = self.actor.optimizer.optimizer
        receipt = dict(rank=dist.get_rank(), world_size=dist.get_world_size(),
                       completed_updates=optimizer.completed_steps,
                       momentum_dtype=str(optimizer.momentum_dtype),
                       trainable_parameters=sum(p.numel() for _, p in optimizer.named),
                       optimizer_entries=len(optimizer.entries),
                       gradients_discarded=self.gradients_paused,
                       free_hbm_bytes=free, total_hbm_bytes=total,
                       establishes_admission=False)
        path = Path(self.actor.args.save).parent / f"resident-rank-{dist.get_rank():02d}.json"
        path.write_text(json.dumps(receipt, indent=2))

    def restore(self, tag):
        self.get(tag)


def install_live_policy():
    from miles.utils.tensor_backper import TensorBackuper

    def create(source_getter, main_cast_ctx=None):
        actor = source_getter.__self__
        if main_cast_ctx is not None or actor.with_ref or actor.with_opd_teacher or actor.args.keep_old_actor:
            raise ValueError("offload-free run requires one live actor policy")
        if actor.args.offload_train:
            raise ValueError("offloading is forbidden")
        return LivePolicy(source_getter)

    TensorBackuper.create = staticmethod(create)
