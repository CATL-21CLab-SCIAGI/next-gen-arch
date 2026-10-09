"""EvergreenTree's single local weight copy for colocated Miles actors.

NAS holds durable checkpoints. A packed node-local file is the sole hot weight
copy, used both by policy transfer and by GPU rematerialization after offload.
"""

import hashlib
import json
import os
from pathlib import Path

import torch


def cache_root(rank):
    return Path(os.environ["EVERGREENTREE_WEIGHT_CACHE_DIR"]) / f"rank-{rank:02d}"


class LocalWeightCache:
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "COMPLETE.json").read_text())
        self.path = self.root / "weights.bin"
        if self.path.stat().st_size != self.metadata["bytes"]:
            raise ValueError("local weight cache size differs from its manifest")
        self.storage = torch.from_file(str(self.path), shared=True,
                                       size=self.metadata["bytes"], dtype=torch.uint8)
        self.entries = {entry["name"]: entry for entry in self.metadata["entries"]}
        self.weights = {}
        for name, entry in self.entries.items():
            raw = self.storage[entry["offset"]:entry["offset"] + entry["nbytes"]]
            dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
            self.weights[name] = raw.view(dtype).view(entry["shape"])

    def validate(self, named, parent_sha256):
        if parent_sha256 != self.metadata["parent"]:
            raise ValueError("local cache belongs to a different SFT parent")
        expected = dict(named)
        if expected.keys() != self.weights.keys():
            raise ValueError("local cache parameter inventory differs from model")
        for name, tensor in expected.items():
            value = self.weights[name]
            if tensor.shape != value.shape or tensor.dtype != value.dtype:
                raise ValueError(f"local cache tensor contract differs: {name}")
            digest = hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy()).hexdigest()
            if digest != self.entries[name]["sha256"]:
                raise ValueError(f"local cache checksum mismatch: {name}")

    @torch.no_grad()
    def restore(self, named):
        cuda = False
        for name, tensor in named:
            tensor.copy_(self.weights[name])
            cuda |= tensor.is_cuda
        if cuda:
            torch.cuda.synchronize()


class LocalTensorBackuper:
    """One local cache, with small derived model buffers kept in host memory."""

    def __init__(self, source_getter):
        self.source_getter = source_getter
        self.actor = source_getter.__self__
        self.cache = LocalWeightCache(cache_root(torch.distributed.get_rank()))
        self.first_backup = True
        self.buffers = []
        for model in self.actor.model:
            for _, buffer in model.named_buffers():
                if buffer.numel() * buffer.element_size() > 256 * 2**20:
                    raise ValueError("unexpected large derived model buffer")
                self.buffers.append((buffer, buffer.detach().cpu().clone()))

    @property
    def backup_tags(self):
        return ["actor"]

    def get(self, tag):
        if tag != "actor":
            raise ValueError("single-cache policy does not support reference/teacher copies")
        return self.cache.weights

    @torch.no_grad()
    def backup(self, tag):
        self.get(tag)
        # The loader just validated and restored this very cache. Rewriting it
        # at startup would create the redundant I/O this backend avoids.
        if self.first_backup:
            self.first_backup = False
            return
        for name, tensor in self.source_getter():
            self.cache.weights[name].copy_(tensor.detach())
        torch.cuda.synchronize()
        fd = os.open(self.cache.path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        for buffer, saved in self.buffers:
            saved.copy_(buffer)

    def restore(self, tag):
        self.get(tag)
        self.cache.restore(self.source_getter())

    @torch.no_grad()
    def on_resume(self):
        self.restore("actor")
        for buffer, saved in self.buffers:
            buffer.copy_(saved)
        torch.cuda.synchronize()


def install_backuper():
    from miles.utils.tensor_backper import TensorBackuper

    def create(source_getter, main_cast_ctx=None):
        if main_cast_ctx is not None:
            raise ValueError("local policy cache cannot combine with master rematerialization")
        actor = source_getter.__self__
        if actor.with_ref or actor.with_opd_teacher or actor.args.keep_old_actor:
            raise ValueError("local policy cache requires one actor policy")
        return LocalTensorBackuper(source_getter)

    TensorBackuper.create = staticmethod(create)
