"""Layerwise teacher-input/output matching for the Qwen3.5 mechanism switches.

OPAL stage 1 motivates independent squared reconstruction losses. This experiment
fits mixer outputs and all parameters of the replacement mixers, including their
inherited projections. The original teacher and unchanged backbone remain frozen.
Each layer's raw MSE averages windows/tokens/channels; the objective sums layers.
Normalized error is a diagnostic only. Evaluation composes all switches and measures
next-token KL/CE.

Run: torchrun --nproc_per_node N -m archlab.training.latent_match run.json
Data: archlab-latent-documents-v1 manifest with tokenizer_files and checksum-pinned
files named tokens (1-D integer .npy), offsets (document boundaries .npy), and groups
(JSON string list). Stable document/task group IDs select disjoint train/eval sets;
windows never cross document boundaries. No pretokenized corpus is accepted without
exact tokenizer provenance. Outputs are new directories, or explicit same-run resumes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from archlab.artifacts import sha256_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")


@dataclass(frozen=True)
class SourceConfig:
    manifest: str
    weight: float = 1.0

    def __post_init__(self):
        if not self.manifest or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("each source needs a manifest and a positive finite weight")


@dataclass(frozen=True)
class RunConfig:
    teacher: str
    output: str
    variant: Literal["simplicial", "triadic"]
    sources: tuple[SourceConfig, ...]
    sequence_length: int
    batch_tokens: int
    tokens: int
    eval_windows: int
    lr: float
    seed: int = 42
    init_std: float = 0.02
    feature_dim: int = 4
    backend: Literal["kernel", "reference"] = "kernel"
    loss: Literal["mse"] = "mse"
    microbatch_windows: int = 1
    heldout_fraction: float = 0.1
    split_seed: int = 42
    warmup_fraction: float = 0.03
    final_lr_fraction: float = 0.1
    clip_norm: float = 1.0
    eval_every: int = 50
    log_every: int = 1
    eval_chunk_tokens: int = 128
    expected_layers: int | None = None

    def __post_init__(self):
        if self.variant not in ("simplicial", "triadic") or self.backend not in ("kernel", "reference"):
            raise ValueError("unknown switch variant/backend")
        if self.loss != "mse":
            raise ValueError("stage 1 uses raw MSE; target normalization is diagnostic only")
        for name in ("sequence_length", "batch_tokens", "tokens", "eval_windows", "microbatch_windows",
                     "eval_every", "log_every", "eval_chunk_tokens", "feature_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length < 2 or self.batch_tokens % self.sequence_length or self.tokens % self.batch_tokens:
            raise ValueError("budgets must contain whole sequences and whole optimizer updates")
        if not self.sources or not self.teacher or not self.output:
            raise ValueError("teacher, output and sources are required")
        if not 0 < self.heldout_fraction < 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("invalid heldout/warmup fraction")
        if not 0 <= self.final_lr_fraction <= 1:
            raise ValueError("invalid final LR fraction")
        for name in ("lr", "clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.init_std) or self.init_std < 0:
            raise ValueError("init_std must be nonnegative and finite")
        if type(self.seed) is not int or self.seed < 0 or type(self.split_seed) is not int:
            raise ValueError("seed must be a nonnegative integer and split_seed an integer")
        if self.expected_layers is not None and self.expected_layers <= 0:
            raise ValueError("expected_layers must be positive")

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        values["sources"] = tuple(SourceConfig(**source) for source in values["sources"])
        return cls(**values)

    def windows_per_rank(self, world):
        if self.batch_tokens % (self.sequence_length * world):
            raise ValueError("batch_tokens must divide exactly over ranks and sequence_length")
        return self.batch_tokens // (self.sequence_length * world)


def tokenizer_identity(teacher):
    root = Path(teacher)
    if not all((root / name).is_file() for name in TOKENIZER_FILES[:2]):
        raise ValueError("teacher needs local tokenizer.json and tokenizer_config.json")
    return {name: sha256_file(root / name) for name in TOKENIZER_FILES if (root / name).is_file()}


def teacher_identity(teacher):
    """Pin the local safetensors checkpoint, including weights rather than config alone."""
    root = Path(teacher)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        names.append(index.name)
    else:
        names = ["model.safetensors"]
    names.append("config.json")
    if any(Path(name).name != name for name in names):
        raise ValueError("teacher checkpoint files must be directly inside the model directory")
    return {name: sha256_file(root / name) for name in names}


def heldout_group(group: str, seed: int, fraction: float) -> bool:
    value = int.from_bytes(hashlib.sha256(f"latent-match:{seed}:{group}".encode()).digest()[:8], "big")
    return value < fraction * 2**64


class MixtureWindows:
    """Checksum-verified documents, sampled by mixture weight then eligible document/window.

    The same RNG seed, world size and data contract produce identical streams for both
    architectural arms. Repeated windows are allowed and explicitly counted as processed
    tokens, not unique corpus tokens. Short documents are excluded, never padded or joined.
    """

    def __init__(self, sources, length, *, tokenizer_files, vocab_size, heldout_fraction=0.1, split_seed=42):
        if length < 2 or not 0 < heldout_fraction < 1 or vocab_size < 1:
            raise ValueError("invalid document-window contract")
        self.length, self.parts, self.receipts = length, [], []
        self.weights = np.asarray([source.weight for source in sources], dtype=np.float64)
        if not len(self.weights):
            raise ValueError("empty corpus")
        self.weights /= self.weights.sum()
        for source in sources:
            path = Path(source.manifest).resolve(strict=True)
            manifest = json.loads(path.read_text())
            if manifest.get("format") != "archlab-latent-documents-v1":
                raise ValueError("unsupported corpus format; raw Megatron token IDs are not accepted")
            if manifest.get("tokenizer_files") != tokenizer_files:
                raise ValueError(f"tokenizer identity differs from teacher: {path}")
            files = {}
            for key in ("tokens", "offsets", "groups"):
                record = manifest["files"][key]
                asset = (path.parent / record["path"]).resolve(strict=True)
                if not asset.is_relative_to(path.parent) or sha256_file(asset) != record["sha256"]:
                    raise ValueError(f"corpus asset escapes source or checksum differs: {key}")
                files[key] = asset
            tokens = np.load(files["tokens"], mmap_mode="r", allow_pickle=False)
            offsets = np.load(files["offsets"], mmap_mode="r", allow_pickle=False)
            groups = json.loads(files["groups"].read_text())
            if tokens.ndim != 1 or tokens.dtype not in (np.dtype("int32"), np.dtype("int64")):
                raise ValueError("tokens must be a one-dimensional int32/int64 array")
            if offsets.ndim != 1 or offsets.dtype not in (np.dtype("int32"), np.dtype("int64")):
                raise ValueError("offsets must be a one-dimensional integer array")
            if (len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(tokens)
                    or np.any(np.diff(offsets) <= 0)):
                raise ValueError("offsets must partition every token into nonempty documents")
            if not isinstance(groups, list) or len(groups) != len(offsets) - 1:
                raise ValueError("one stable document/task group is required per document")
            if any(not isinstance(group, str) or not group for group in groups):
                raise ValueError("group IDs must be nonempty strings")
            for start in range(0, len(tokens), 1 << 20):
                chunk = tokens[start:start + (1 << 20)]
                if np.any(chunk < 0) or np.any(chunk >= vocab_size):
                    raise ValueError("corpus token ID is outside the teacher vocabulary")
            selections = {False: [], True: []}
            for index, group in enumerate(groups):
                if offsets[index + 1] - offsets[index] >= length:
                    selections[heldout_group(group, split_seed, heldout_fraction)].append(index)
            if not selections[False] or not selections[True]:
                raise ValueError("each source needs nonempty disjoint train and heldout eligible documents")
            self.parts.append((tokens, offsets, selections))
            self.receipts.append({"manifest": str(path), "sha256": sha256_file(path), "weight": source.weight,
                                  "train_documents": len(selections[False]), "heldout_documents": len(selections[True]),
                                  "short_documents": len(groups) - sum(map(len, selections.values()))})

    def sample(self, rng, count, *, heldout=False):
        if count < 1:
            raise ValueError("sample count must be positive")
        windows = []
        for source in rng.choice(len(self.parts), size=count, p=self.weights):
            tokens, offsets, selections = self.parts[source]
            documents = selections[heldout]
            index = documents[int(rng.integers(len(documents)))]
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            start = int(rng.integers(lo, hi - self.length + 1))
            windows.append(np.asarray(tokens[start:start + self.length], dtype=np.int64))
        return torch.from_numpy(np.stack(windows))


@contextmanager
def capture(mixers):
    """Detached teacher mixer inputs, rotary embeddings and outputs; hooks always removed."""
    store, handles = {}, []
    try:
        for index, module in mixers.items():
            def hook(module, args, kwargs, output, index=index):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor) or not isinstance(out, torch.Tensor):
                    raise TypeError("mixer capture requires tensor hidden states and tensor output")
                rotary = kwargs.get("position_embeddings")
                store[index] = (hidden.detach(), tuple(v.detach() for v in rotary) if rotary is not None else None,
                                out.detach())
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def reconstruction_loss(output, target, kind="mse"):
    if output.shape != target.shape or output.ndim != 3:
        raise ValueError("mixer outputs must have matching [batch, tokens, hidden] shape")
    errors = (output.float() - target.detach().float()).square().mean(dim=(1, 2))
    if kind == "normalized_mse":
        errors = errors / target.detach().float().square().mean(dim=(1, 2)).clamp_min(1e-12)
    elif kind != "mse":
        raise ValueError("unknown reconstruction loss")
    return errors.mean()


def layer_losses(switches, store, kind="mse"):
    if set(switches) != set(store):
        raise ValueError("teacher did not visit exactly the switched layers")
    for index, switch in switches.items():
        hidden, rotary, target = store[index]
        kwargs = {"position_embeddings": rotary} if rotary is not None else {}
        with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.device.type == "cuda"):
            output = switch(hidden, **kwargs)
        yield index, reconstruction_loss(output, target, kind)


def text_model(model):
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


class Patched:
    """Temporarily compose all switches, preserving teacher module registration and forward methods."""

    def __init__(self, mixers, switches):
        self.mixers, self.switches, self.saved = mixers, switches, {}

    def __enter__(self):
        try:
            for index, module in self.mixers.items():
                self.saved[index] = module.forward
                student = self.switches[index]
                is_attention = hasattr(module, "q_proj")

                def forward(*args, _student=student, _attention=is_attention, **kwargs):
                    hidden = kwargs.pop("hidden_states", args[0] if args else None)
                    # evaluate supplies only unpadded input_ids and use_cache=False.
                    # HF may still synthesize a causal mask; both switches are causal intrinsically.
                    kwargs.pop("attention_mask", None)
                    with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.device.type == "cuda"):
                        output = _student(hidden, **kwargs)
                    return (output, None) if _attention else output

                module.forward = forward
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        for index, forward in self.saved.items():
            self.mixers[index].forward = forward
        self.saved.clear()


def train_step(model, mixers, switches, optimizer, ids, *, microbatch_windows=1, loss_kind="mse", clip_norm=1.0):
    """One global update; teacher forward once per microbatch, independent switch backward passes."""
    params = [p for module in switches.values() for p in module.parameters() if p.requires_grad]
    if not params or not len(ids) or microbatch_windows < 1:
        raise ValueError("training needs parameters, examples and a positive microbatch")
    if any(p.requires_grad for p in model.parameters()) or {id(p) for p in params} & {id(p) for p in model.parameters()}:
        raise ValueError("teacher must be frozen and switch parameters must be independent")
    optimizer.zero_grad(set_to_none=True)
    total = ids.new_zeros((), dtype=torch.float32)
    for start in range(0, len(ids), microbatch_windows):
        batch = ids[start:start + microbatch_windows]
        with capture(mixers) as store:
            with torch.no_grad():
                text_model(model)(input_ids=batch, use_cache=False)
        scale = len(batch) / len(ids)
        for _, loss in layer_losses(switches, store, loss_kind):
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite layer reconstruction loss")
            (loss * scale).backward()
            total += loss.detach() * scale
    if any(p.grad is None for p in params):
        raise RuntimeError("a trainable switch parameter received no gradient")
    if dist.is_initialized():
        for parameter in params:
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())
        dist.all_reduce(total)
        total.div_(dist.get_world_size())
    norm = torch.nn.utils.clip_grad_norm_(params, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError("optimizer produced nonfinite switch parameters")
    return {"train_loss": float(total), "train_layer_error_mean": float(total) / len(switches),
            "grad_norm": float(norm)}


@torch.no_grad()
def evaluate(model, mixers, switches, windows, device, *, loss_kind="mse", chunk_tokens=128):
    """Full composition, with LM-head KL/CE chunks instead of full sequence-vocabulary logits."""
    if not len(windows) or chunk_tokens < 1:
        raise ValueError("evaluation requires windows and a positive logits chunk")
    sums = {index: 0.0 for index in switches}
    normalized_sums = {index: 0.0 for index in switches}
    kl = ce_teacher = ce_student = 0.0
    tokens = 0
    head = model.get_output_embeddings()
    for window in windows:
        ids = window[None].to(device)
        with capture(mixers) as store:
            teacher_hidden = text_model(model)(input_ids=ids, use_cache=False).last_hidden_state
        for index, loss in layer_losses(switches, store, loss_kind):
            sums[index] += float(loss)
            energy = store[index][2].float().square().mean().clamp_min(1e-12)
            normalized_sums[index] += float(loss / energy) if loss_kind == "mse" else float(loss)
        del store
        with Patched(mixers, switches):
            student_hidden = text_model(model)(input_ids=ids, use_cache=False).last_hidden_state
        for start in range(0, ids.shape[1] - 1, chunk_tokens):
            end = min(start + chunk_tokens, ids.shape[1] - 1)
            t_logp = head(teacher_hidden[:, start:end]).float().log_softmax(-1)
            s_logp = head(student_hidden[:, start:end]).float().log_softmax(-1)
            targets = ids[:, start + 1:end + 1].reshape(-1)
            kl += float((t_logp.exp() * (t_logp - s_logp)).sum())
            ce_teacher += float(F.nll_loss(t_logp.flatten(0, 1), targets, reduction="sum"))
            ce_student += float(F.nll_loss(s_logp.flatten(0, 1), targets, reduction="sum"))
            tokens += targets.numel()
    per_layer = {str(index): total / len(windows) for index, total in sums.items()}
    normalized = {str(index): total / len(windows) for index, total in normalized_sums.items()}
    result = {"layer_error": per_layer, "layer_error_mean": sum(per_layer.values()) / len(per_layer),
              "layer_normalized_error": normalized, "layer_normalized_error_mean": sum(normalized.values()) / len(normalized),
              "kl": kl / tokens, "ce_teacher": ce_teacher / tokens, "ce_student": ce_student / tokens,
              "eval_prediction_tokens": tokens}
    if not all(math.isfinite(v) for v in (*per_layer.values(), *normalized.values(),
                                        result["kl"], result["ce_teacher"], result["ce_student"])):
        raise FloatingPointError("nonfinite evaluation metrics")
    return result


def lr_at(step, total, peak, warmup_fraction, final_fraction):
    warmup = int(total * warmup_fraction)
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rng_state(rng, device):
    return {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}


def save_checkpoint(output, step, switches, optimizer, rngs, fingerprint):
    destination = Path(output) / "checkpoints" / f"step-{step:08d}.pt"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.partial")
    state = {"format": "archlab-latent-checkpoint-v1", "contract_sha256": fingerprint, "step": step,
             "switches": {str(i): m.state_dict() for i, m in switches.items()},
             "optimizer": optimizer.state_dict(), "rngs": rngs}
    try:
        with temporary.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(path, switches, optimizer, rng, fingerprint, *, rank=0, world=1, device=None):
    device = torch.device("cpu") if device is None else device
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "archlab-latent-checkpoint-v1" or state.get("contract_sha256") != fingerprint:
        raise ValueError("checkpoint belongs to another experiment contract")
    if len(state["rngs"]) != world or set(state["switches"]) != {str(i) for i in switches}:
        raise ValueError("checkpoint rank count or switched layers differ")
    for index, module in switches.items():
        module.load_state_dict(state["switches"][str(index)], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    saved = state["rngs"][rank]
    rng.bit_generator.state = saved["numpy"]
    torch.set_rng_state(saved["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"], device)
    return int(state["step"])


@contextmanager
def run_directory(output, contract, resume=None):
    """Hold an exclusive local/NAS lock; never mix different contracts or overwrite completed steps."""
    output = Path(output)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("resume needs its existing run directory")
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = output / "RUN.json"
        if resume is None:
            receipt.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        else:
            if json.loads(receipt.read_text()) != contract:
                raise ValueError("resume contract differs from RUN.json")
            checkpoints = sorted((output / "checkpoints").glob("step-*.pt"))
            if not checkpoints or Path(resume).resolve() != checkpoints[-1].resolve():
                raise ValueError("resume must select this run's latest complete checkpoint")
        yield output


def main(argv=None):
    from contextlib import ExitStack

    from transformers import AutoConfig, AutoModelForCausalLM

    from archlab.architectures import qwen35_switch as switch_lib
    from archlab.architectures.triadic_attention import official_dependency_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--resume", type=Path, help="explicit latest checkpoint in the same output directory")
    parser.add_argument("--stop-after", type=int,
                        help="stop and checkpoint at this total optimizer-step count; preserve the full-budget LR schedule")
    args = parser.parse_args(argv)
    run = RunConfig.read(args.run)
    distributed = "RANK" in os.environ
    if distributed:
        # Full causal simplex evaluation can take longer than the default process-group timeout.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
        count = run.windows_per_rank(world)
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        torch.cuda.set_device(device)
        torch.manual_seed(run.seed)
        identity = tokenizer_identity(run.teacher)
        teacher_files = [teacher_identity(run.teacher) if rank == 0 else None]
        if distributed:
            dist.broadcast_object_list(teacher_files, src=0)
        hf_config = AutoConfig.from_pretrained(run.teacher, local_files_only=True, trust_remote_code=False)
        config = hf_config.get_text_config()
        data = MixtureWindows(run.sources, run.sequence_length, tokenizer_files=identity, vocab_size=config.vocab_size,
                              heldout_fraction=run.heldout_fraction, split_seed=run.split_seed)
        rng = np.random.default_rng(np.random.SeedSequence([run.seed, rank]))
        heldout = data.sample(np.random.default_rng(run.seed + 7), run.eval_windows, heldout=True)
        model = AutoModelForCausalLM.from_pretrained(run.teacher, dtype=torch.bfloat16, device_map={"": device},
                                                   local_files_only=True, trust_remote_code=False)
        model.eval().requires_grad_(False)
        model.config.use_cache = False
        layers = text_model(model).layers
        switches = switch_lib.build_switches(layers, config.layer_types, run.variant,
                    switch_lib.SwitchConfig(init_std=run.init_std, feature_dim=run.feature_dim, backend=run.backend), seed=run.seed)
        expected = run.expected_layers if run.expected_layers is not None else {"simplicial": 8, "triadic": 24}[run.variant]
        if len(switches) != expected:
            raise ValueError(f"expected {expected} switched layers, got {len(switches)}")
        for module in switches.values():
            module.to(device=device, dtype=torch.float32).eval()
        mixers = {i: switch_lib.mixer(layers[i])[1] for i in switches}
        params = [p for module in switches.values() for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=run.lr, betas=(0.9, 0.95), weight_decay=0.0)
        architecture_dir = Path(switch_lib.__file__).parent
        source_paths = (Path(__file__), *(architecture_dir / name for name in (
            "qwen35_switch.py", "simplicial_attention.py", "simplicial_kernels.py", "triadic_attention.py")))
        versions = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "transformers": metadata.version("transformers"), "numpy": np.__version__,
                    "triton": metadata.version("triton")}
        contract = {"format": "archlab-latent-run-v1", "config": json.loads(json.dumps(asdict(run))), "world": world,
                    "switched_layers": sorted(switches), "trainable_parameters": sum(p.numel() for p in params),
                    "teacher_files_sha256": teacher_files[0], "tokenizer_files": identity, "versions": versions,
                    "data": data.receipts, "source_sha256": {p.name: sha256_file(p) for p in source_paths},
                    "eval_windows_sha256": hashlib.sha256(heldout.numpy().tobytes()).hexdigest(),
                    "train_scope": "all_replacement_mixer_parameters", "target": "teacher_mixer_output",
                    "layer_reduction": "sum_of_per_layer_mean_squared_errors",
                    "window_sampling": "with_replacement_within_documents"}
        if run.variant == "triadic" and run.backend == "kernel":
            contract["triadic_dependency"] = official_dependency_contract()
        fingerprint = contract_hash(contract)
        with ExitStack() as stack:
            output = Path(run.output)
            if rank == 0:
                stack.enter_context(run_directory(output, contract, args.resume))
            if distributed:
                dist.barrier()
            start_step = load_checkpoint(args.resume, switches, optimizer, rng, fingerprint, rank=rank, world=world,
                                         device=device) if args.resume else 0
            steps = run.tokens // run.batch_tokens
            if not 0 <= start_step <= steps:
                raise ValueError("checkpoint step exceeds this run's budget")
            stop_step = steps if args.stop_after is None else args.stop_after
            if not 0 < stop_step <= steps or stop_step < start_step:
                raise ValueError("stop-after must be within the run budget and not before the resumed step")
            metrics = None
            if rank == 0:
                attempts = output / "attempts"
                attempts.mkdir(exist_ok=True)
                metrics = stack.enter_context((attempts / f"{uuid.uuid4().hex}.jsonl").open("x"))

            def log(record):
                if metrics:
                    metrics.write(json.dumps({"at": time.time(), **record}, allow_nan=False) + "\n")
                    metrics.flush()

            if rank == 0:
                log({"step": start_step, "tokens": start_step * run.batch_tokens, "resumed": bool(args.resume),
                     **evaluate(model, mixers, switches, heldout, device, loss_kind=run.loss, chunk_tokens=run.eval_chunk_tokens)})
            if distributed:
                dist.barrier()
            started = time.monotonic()
            for step in range(start_step, stop_step):
                lr = lr_at(step, steps, run.lr, run.warmup_fraction, run.final_lr_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                ids = data.sample(rng, count).to(device)
                result = train_step(model, mixers, switches, optimizer, ids, microbatch_windows=run.microbatch_windows,
                                    loss_kind=run.loss, clip_norm=run.clip_norm)
                done = (step + 1) * run.batch_tokens
                if (step + 1) % run.log_every == 0:
                    log({"step": step + 1, "tokens": done, "lr": lr, **result,
                         "tokens_per_second": (step + 1 - start_step) * run.batch_tokens / (time.monotonic() - started)})
                if (step + 1) % run.eval_every == 0 or step + 1 == stop_step:
                    if rank == 0:
                        log({"step": step + 1, "tokens": done,
                             **evaluate(model, mixers, switches, heldout, device, loss_kind=run.loss, chunk_tokens=run.eval_chunk_tokens)})
                    local_rng = rng_state(rng, device)
                    states = [None] * world if rank == 0 else None
                    if distributed:
                        dist.gather_object(local_rng, states, dst=0)
                    else:
                        states = [local_rng]
                    if rank == 0:
                        save_checkpoint(output, step + 1, switches, optimizer, states, fingerprint)
                    if distributed:
                        dist.barrier()
            elapsed = time.monotonic() - started
            log({"event": "stopped" if stop_step < steps else "complete", "step": stop_step,
                 "tokens": stop_step * run.batch_tokens, "attempt_seconds": elapsed,
                 "attempt_tokens_per_second": (stop_step - start_step) * run.batch_tokens / max(elapsed, 1e-9)})
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
