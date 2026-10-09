"""Adapter-only training through the pinned official V4.1 forward/backward.

The output head is frozen FP32 and FSDP-sharded. Chunked cross entropy reads
its weight directly, so its explicit unshard lifetime includes loss backward.
MathPilot already shifts labels and assigns unique windows to global ranks.
"""

from __future__ import annotations

import hashlib
import math
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from archlab.artifacts import atomic_write_json
from archlab.automodel.deepseek_v41_execution import adapter_optimizers
from archlab.automodel.deepseek_v41_loss import frozen_head_loss
from archlab.automodel.deepseek_v41_training import (
    append_metric,
    emit,
    global_numbers,
    globally_reduce_gradients,
    learning_rate,
    restore_adapter_checkpoint,
    save_adapter_checkpoint,
)


@contextmanager
def frozen_head_unsharded(head):
    """Keep a native FP32 head resident until the caller finishes backward.

Calling ``head.forward`` would materialize the full [sequence, vocabulary]
logits. Calling our custom loss on a DTensor would instead bypass FSDP hooks.
The public FSDP2 lifecycle lets the memory-bounded loss use the full weight.
"""
    sharded = isinstance(head, FSDPModule)
    if sharded:
        head.unshard()
    try:
        if isinstance(head.weight, DTensor):
            raise TypeError("the output head must be unsharded before chunked cross entropy")
        if head.weight.requires_grad or head.weight.dtype != torch.float32:
            raise ValueError("official V4.1 requires a frozen FP32 output head")
        yield head
    finally:
        if sharded:
            head.reshard()


def _hidden(model, inputs):
    output = model(input_ids=inputs, return_hidden_states=True)
    hidden = output.hidden_states
    if not isinstance(hidden, torch.Tensor) or hidden.shape[:2] != inputs.shape:
        raise TypeError("official V4.1 must return its final hidden Tensor")
    return hidden


def optimizer_step(model, optimizers, inputs, labels, *, learning_rate):
    """Sum supervised CE, reduce across all unique data ranks, then clip at 1."""
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    start = time.perf_counter()
    hidden = _hidden(model, inputs)
    with frozen_head_unsharded(model.lm_head) as head:
        loss = frozen_head_loss(hidden, labels, head)
        loss_sum, target_count, input_count = global_numbers(
            float(loss.detach()), int((labels != -100).sum()), inputs.numel()
        )
        if not math.isfinite(loss_sum) or target_count < 1:
            raise FloatingPointError("nonfinite loss or empty global training batch")
        loss.backward()
    parameters = [p for optimizer in optimizers for group in optimizer.param_groups
                  for p in group["params"]]
    grad_norm = globally_reduce_gradients(parameters, target_count)
    for optimizer in optimizers:
        optimizer.step()
    torch.cuda.synchronize()
    maxima = torch.tensor([time.perf_counter() - start,
                           torch.cuda.max_memory_allocated() / 2**30],
                          device=inputs.device, dtype=torch.float64)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    return {"loss": loss_sum / target_count, "supervised_tokens": int(target_count),
            "input_tokens": int(input_count), "seconds": maxima[0].item(),
            "learning_rate": learning_rate, "gradient_norm_before_clip": grad_norm,
            "max_memory_allocated_gib": maxima[1].item()}


@torch.no_grad()
def evaluate(model, data, *, step):
    """Evaluate the sealed target masks once, including empty trailing ranks."""
    was_training = model.training
    model.eval()
    local_sum, local_count = 0., 0
    by_mode = {mode: [0., 0.] for mode in ("low", "medium", "high")}
    try:
        for cursor in range(math.ceil(len(data) / dist.get_world_size())):
            index = cursor * dist.get_world_size() + dist.get_rank()
            inputs, labels, count = data.batch(index, device="cuda")
            hidden = _hidden(model, inputs)
            with frozen_head_unsharded(model.lm_head) as head:
                value = float(frozen_head_loss(hidden, labels, head))
            local_sum += value
            local_count += count
            if index < len(data):
                mode = data.windows[index]["mode"]
                by_mode[mode][0] += value
                by_mode[mode][1] += count
        loss_sum, count = global_numbers(local_sum, local_count)
        if not math.isfinite(loss_sum) or count <= 0:
            raise FloatingPointError("nonfinite validation loss or no validation targets")
        if count != data.manifest["supervised_tokens"]:
            raise RuntimeError("validation targets disagree with the sealed pilot")
        report = {"step": step, "loss": loss_sum / count,
                  "supervised_tokens": int(count), "by_mode": {}}
        for mode, values in by_mode.items():
            total, tokens = global_numbers(*values)
            report["by_mode"][mode] = {"loss": total / tokens if tokens else None,
                                       "targets": int(tokens)}
        return report
    finally:
        model.train(was_training)


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


# Exact checkpoint restoration is a separate contract from recomputing through
# native sparse-attention / EP kernels, which do not promise bitwise backward.
REPLAY_TOLERANCES = {"loss": {"rtol": 1e-5, "atol": 1e-7},
                     "updated_state": {"rtol": 1e-4, "atol": 1e-6}}
REPLAY_PROTOCOL = {"version": "native-update-baseline-v1", "min_in_memory_repeats": 2,
                   "max_update_relative_l2": 1e-2, "baseline_multiplier": 2.0,
                   "baseline_numerical_floor": 1e-5,
                   "loss_tolerance": {"rtol": 1e-5, "atol": 1e-7}}


def capture_adapter_training_state(adapters, optimizers):
    """Own independent CPU copies of every resumable adapter state component."""
    return {"adapters": _cpu_clone({i: adapter.state_dict() for i, adapter in adapters.items()}),
            "optimizers": _cpu_clone([optimizer.state_dict() for optimizer in optimizers]),
            "rng": {"cpu": torch.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state().cpu().clone()}}


def training_state_sha256(state):
    """Include tensor bytes, dtype/shape, structure and scalar optimizer options."""
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode() + b"\0")
            for chunk in tensor.reshape(-1).view(torch.uint8).split(8 * 1024 * 1024):
                digest.update(chunk.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        else:
            digest.update(repr(value).encode() + b"\0")

    visit(state)
    return digest.hexdigest()


def assert_exact_training_state(expected, actual):
    """Run before recomputation so serialization errors cannot hide in tolerance."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_sha, actual_sha = training_state_sha256(expected), training_state_sha256(actual)
    if expected_sha != actual_sha:
        raise AssertionError("checkpoint state differs in exact bytes or metadata")
    return {"exact": True, "sha256": actual_sha,
            "components": ["adapters", "optimizers", "cpu_rng", "cuda_rng"]}


def restore_in_memory_training_state(state, adapters, optimizers):
    """Reset the same state without any file serialization for a control repeat."""
    for index, adapter in adapters.items():
        adapter.load_state_dict(state["adapters"][index], strict=True)
    for optimizer, saved in zip(optimizers, state["optimizers"], strict=True):
        # load_state_dict may retain CPU scalar step tensors. Never let the next
        # optimizer update mutate the snapshot that defines the control repeat.
        optimizer.load_state_dict(_cpu_clone(saved))
    torch.set_rng_state(state["rng"]["cpu"])
    torch.cuda.set_rng_state(state["rng"]["cuda"])


def state_difference_statistics(expected, actual):
    """Report bounded-memory absolute and relative-L2 tensor differences."""
    max_abs, error_squared, reference_squared = 0.0, 0.0, 0.0
    elements, differing, tensors, worst_tensor = 0, 0, 0, None

    def visit(left, right, path):
        nonlocal max_abs, error_squared, reference_squared, elements, differing, tensors, worst_tensor
        if isinstance(left, torch.Tensor):
            tensors += 1
            left_chunks = left.detach().cpu().reshape(-1).split(1024 * 1024)
            right_chunks = right.detach().cpu().reshape(-1).split(1024 * 1024)
            for a, b in zip(left_chunks, right_chunks, strict=True):
                a, b = a.double(), b.double()
                delta = b - a
                peak = float(delta.abs().max()) if delta.numel() else 0.0
                if peak > max_abs:
                    max_abs, worst_tensor = peak, path
                error_squared += float(delta.square().sum())
                reference_squared += float(a.square().sum())
                elements += delta.numel()
                differing += int(delta.count_nonzero())
        elif isinstance(left, dict):
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, f"{path}.{index}")

    visit(expected, actual, "state")
    return {"max_abs": max_abs, "relative_l2": math.sqrt(error_squared / max(reference_squared, 1e-300)),
            "differing_elements": differing, "elements": elements,
            "tensor_count": tensors, "worst_tensor": worst_tensor}


def compare_replayed_update(expected_metric, expected_state, actual_metric, actual_state):
    """Use fixed native-compute tolerances only after an exact state restoration."""
    torch.testing.assert_close(actual_metric["loss"], expected_metric["loss"], **REPLAY_TOLERANCES["loss"])
    for component in ("adapters", "optimizers"):
        torch.testing.assert_close(actual_state[component], expected_state[component],
                                   **REPLAY_TOLERANCES["updated_state"])
    return {"passed": True, "replay_tolerance": _cpu_clone(REPLAY_TOLERANCES["updated_state"]),
            "loss_tolerance": _cpu_clone(REPLAY_TOLERANCES["loss"]),
            "loss_abs_difference": abs(actual_metric["loss"] - expected_metric["loss"]),
            **{component: state_difference_statistics(expected_state[component], actual_state[component])
               for component in ("adapters", "optimizers")}}


def _optimizer_moments_and_metadata(optimizers):
    """Remove floating moment tensors from metadata; scalar counters stay exact."""
    moments = {}

    def visit(value, path):
        if isinstance(value, torch.Tensor):
            in_parameter_state = len(path) >= 4 and path[1] == "state"
            if in_parameter_state and path[-1] != "step" and value.is_floating_point():
                moments[path] = value
                return ("moment_tensor", tuple(value.shape), str(value.dtype))
            return value
        if isinstance(value, dict):
            return {key: visit(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, (*path, i)) for i, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(visit(item, (*path, i)) for i, item in enumerate(value))
        return value

    return moments, visit(optimizers, ())


def _assert_finite_state(state):
    if isinstance(state, torch.Tensor):
        if not bool(state.isfinite().all()):
            raise AssertionError("nonfinite replay state tensor")
    elif isinstance(state, dict):
        for value in state.values():
            _assert_finite_state(value)
    elif isinstance(state, (list, tuple)):
        for value in state:
            _assert_finite_state(value)
    elif isinstance(state, float) and not math.isfinite(state):
        raise AssertionError("nonfinite replay metadata")


def _update_vector_statistics(before, expected, actual):
    """Normalize numerical error by the reference update, not stored weights."""
    error_squared = reference_squared = update_squared = max_abs = 0.0
    elements = differing = 0
    worst_tensor = None

    def visit(start, target, value, path):
        nonlocal error_squared, reference_squared, update_squared, max_abs, elements, differing, worst_tensor
        if isinstance(target, torch.Tensor):
            if not isinstance(start, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise AssertionError(f"replay tensor structure differs at {path}")
            if not (start.shape == target.shape == value.shape and start.dtype == target.dtype == value.dtype):
                raise AssertionError(f"replay tensor shape/dtype differs at {path}")
            chunks = [tensor.detach().cpu().reshape(-1).split(1024 * 1024)
                      for tensor in (start, target, value)]
            for a, b, c in zip(*chunks, strict=True):
                if not all(bool(part.isfinite().all()) for part in (a, b, c)):
                    raise AssertionError(f"nonfinite replay state at {path}")
                a, b, c = a.double(), b.double(), c.double()
                error = c - b
                peak = float(error.abs().max()) if error.numel() else 0.0
                if peak > max_abs:
                    max_abs, worst_tensor = peak, path
                error_squared += float(error.square().sum())
                reference_squared += float(b.square().sum())
                update_squared += float((b - a).square().sum())
                elements += error.numel()
                differing += int(error.count_nonzero())
        elif isinstance(target, dict):
            if not isinstance(start, dict) or not isinstance(value, dict) or not (start.keys() == target.keys() == value.keys()):
                raise AssertionError(f"replay state keys differ at {path}")
            for key in target:
                visit(start[key], target[key], value[key], f"{path}.{key}")
        elif isinstance(target, (list, tuple)):
            if type(start) is not type(target) or type(value) is not type(target) or not (len(start) == len(target) == len(value)):
                raise AssertionError(f"replay state sequence differs at {path}")
            for index, (a, b, c) in enumerate(zip(start, target, value, strict=True)):
                visit(a, b, c, f"{path}.{index}")
        else:
            torch.testing.assert_close(value, target, rtol=0, atol=0)

    visit(before, expected, actual, "state")
    error_l2, update_l2, reference_l2 = map(math.sqrt, (error_squared, update_squared, reference_squared))
    return {"max_abs": max_abs, "global_relative_l2": error_l2 / max(reference_l2, 1e-150),
            "update_relative_l2": error_l2 / max(update_l2, 1e-150),
            "error_l2": error_l2, "update_l2": update_l2, "reference_l2": reference_l2,
            "elements": elements, "differing_elements": differing, "worst_tensor": worst_tensor}


def assess_replay_protocol(saved_state, expected_metric, expected_state,
                           checkpoint_metric, checkpoint_state, memory_replays):
    """Assess already-collected continuations after exact restoration checks.

    Every continuation must stay within the fixed update-vector health ceiling.
    The serialized continuation must also fit the measured in-memory baseline.
    Scalar optimizer counters and all parameter-group metadata remain exact.
    """
    if len(memory_replays) < REPLAY_PROTOCOL["min_in_memory_repeats"]:
        raise AssertionError("replay protocol requires at least two in-memory continuations")
    if not math.isfinite(expected_metric["loss"]):
        raise AssertionError("nonfinite reference continuation loss")
    _assert_finite_state(saved_state)
    _assert_finite_state(expected_state)
    saved_moments, _ = _optimizer_moments_and_metadata(saved_state["optimizers"])
    expected_moments, expected_metadata = _optimizer_moments_and_metadata(expected_state["optimizers"])
    measurements = []
    for metric, state in [(checkpoint_metric, checkpoint_state), *memory_replays]:
        _assert_finite_state(state)
        if not math.isfinite(metric["loss"]):
            raise AssertionError("nonfinite replay continuation loss")
        torch.testing.assert_close(metric["loss"], expected_metric["loss"], **REPLAY_PROTOCOL["loss_tolerance"])
        moments, metadata = _optimizer_moments_and_metadata(state["optimizers"])
        if training_state_sha256(metadata) != training_state_sha256(expected_metadata):
            raise AssertionError("optimizer counters or parameter-group metadata changed")
        measurements.append({
            "adapters": _update_vector_statistics(saved_state["adapters"], expected_state["adapters"], state["adapters"]),
            "optimizer_moments": _update_vector_statistics(saved_moments, expected_moments, moments),
            "loss_abs_difference": abs(metric["loss"] - expected_metric["loss"]),
        })
    components = {}
    for name in ("adapters", "optimizer_moments"):
        checkpoint = measurements[0][name]
        memories = [measurement[name] for measurement in measurements[1:]]
        memory_max = max(value["update_relative_l2"] for value in memories)
        baseline_bound = max(REPLAY_PROTOCOL["baseline_multiplier"] * memory_max,
                             REPLAY_PROTOCOL["baseline_numerical_floor"])
        ceiling = REPLAY_PROTOCOL["max_update_relative_l2"]
        if any(value["update_relative_l2"] > ceiling for value in (checkpoint, *memories)):
            raise AssertionError(f"{name} continuation exceeds update-relative L2 health ceiling {ceiling}: "
                                 f"checkpoint={checkpoint['update_relative_l2']}, in_memory={memory_max}")
        if checkpoint["update_relative_l2"] > baseline_bound:
            raise AssertionError(f"{name} checkpoint error exceeds measured in-memory bound {baseline_bound}: "
                                 f"{checkpoint['update_relative_l2']}")
        components[name] = {"checkpoint": checkpoint, "in_memory": memories,
                            "max_in_memory_update_relative_l2": memory_max,
                            "baseline_bound": baseline_bound, "fixed_health_ceiling": ceiling}
    return {"passed": True, "protocol": REPLAY_PROTOCOL["version"],
            "protocol_config": _cpu_clone(REPLAY_PROTOCOL), "components": components,
            "optimizer_counters_and_groups_exact": True,
            "loss_tolerance": dict(REPLAY_PROTOCOL["loss_tolerance"]),
            "loss_abs_differences": [measurement["loss_abs_difference"] for measurement in measurements]}


def _replay_collective_check(errors):
    failures = [errors]
    if dist.is_initialized():
        failures = [None] * dist.get_world_size()
        dist.all_gather_object(failures, errors)
    grouped = {}
    for rank, messages in enumerate(failures):
        for message in messages:
            grouped.setdefault(message, []).append(rank)
    if grouped:
        raise AssertionError(f"replay qualification failed: "
                             f"{[{'ranks': ranks, 'error': error} for error, ranks in grouped.items()]}")


def _replay_world_agreement(adapters, state):
    errors, gradients = [], {}
    try:
        _assert_finite_state(state)
    except AssertionError as error:
        errors.append(str(error))
    for index, adapter in adapters.items():
        for name, parameter in adapter.named_parameters():
            key = f"{index}.{name}"
            if isinstance(parameter, DTensor) or parameter.dtype != torch.float32:
                errors.append(f"{key} is not a replicated FP32 adapter master")
            if parameter.grad is None or not bool(parameter.grad.isfinite().all()):
                errors.append(f"{key} has missing/nonfinite gradients")
            else:
                gradients[key] = parameter.grad.detach().cpu()
    _replay_collective_check(errors)
    fingerprint = {"state_sha256": training_state_sha256({key: state[key] for key in ("adapters", "optimizers")}),
                   "gradient_sha256": training_state_sha256(gradients)}
    copies = [fingerprint]
    if dist.is_initialized():
        copies = [None] * dist.get_world_size()
        dist.all_gather_object(copies, fingerprint)
    _replay_collective_check([] if all(copy == copies[0] for copy in copies) else ["world state/gradient disagreement"])
    return {**fingerprint, "world_state_and_gradient_agreement": True}


def qualify_checkpoint_replay(model, adapters, optimizers, inputs, labels, *, path, cursor, contract,
                              learning_rate=1e-5, memory_repeats=2):
    """Collect every native baseline continuation before making a health decision."""
    if memory_repeats < REPLAY_PROTOCOL["min_in_memory_repeats"]:
        raise ValueError("checkpoint replay needs at least two in-memory repeats")
    saved = capture_adapter_training_state(adapters, optimizers)
    initial_agreement = _replay_world_agreement(adapters, saved)
    saved_sha = training_state_sha256(saved)
    save_adapter_checkpoint(path, adapters, optimizers, cursor, contract)
    gradient_snapshots = []

    def continuation():
        metric = optimizer_step(model, optimizers, inputs, labels, learning_rate=learning_rate)
        state = capture_adapter_training_state(adapters, optimizers)
        agreement = _replay_world_agreement(adapters, state)
        if not dist.is_initialized() or dist.get_rank() == 0:
            gradient_snapshots.append({i: {name: p.grad.detach().cpu().clone()
                for name, p in adapter.named_parameters()} for i, adapter in adapters.items()})
        return metric, state, agreement

    expected_metric, expected, expected_agreement = continuation()
    restored_cursor = restore_adapter_checkpoint(path, adapters, optimizers, contract=contract)
    errors = [] if restored_cursor == cursor else ["checkpoint changed the data cursor"]
    try:
        exact_checkpoint = assert_exact_training_state(saved, capture_adapter_training_state(adapters, optimizers))
    except AssertionError as error:
        errors.append(f"checkpoint exact restoration: {error}")
    _replay_collective_check(errors)
    checkpoint_metric, checkpoint, checkpoint_agreement = continuation()
    memories, memory_receipts = [], []
    for repeat in range(memory_repeats):
        restore_in_memory_training_state(saved, adapters, optimizers)
        errors = []
        try:
            exact_memory = assert_exact_training_state(saved, capture_adapter_training_state(adapters, optimizers))
        except AssertionError as error:
            errors.append(f"in-memory exact restoration: {error}")
        _replay_collective_check(errors)
        metric, state, agreement = continuation()
        memories.append((metric, state))
        memory_receipts.append({"repeat": repeat + 1, "exact_restoration": exact_memory,
                                "metric": metric, **agreement})
    errors = []
    try:
        assessment = assess_replay_protocol(saved, expected_metric, expected,
                                            checkpoint_metric, checkpoint, memories)
    except AssertionError as error:
        errors.append(str(error))
        if not dist.is_initialized() or dist.get_rank() == 0:
            diagnostic = {"passed": False, "reason": str(error), "expected_metric": expected_metric,
                          "continuations": []}
            for replay_index, (label, (metric, state)) in enumerate(zip(
                ["checkpoint", *[f"memory-{i + 1}" for i in range(len(memories))]],
                [(checkpoint_metric, checkpoint), *memories], strict=True,
            ), start=1):
                diagnostic["continuations"].append({
                    "kind": label, "metric": metric,
                    "adapters": _update_vector_statistics(saved["adapters"], expected["adapters"], state["adapters"]),
                    "per_adapter": {str(i): _update_vector_statistics(saved["adapters"][i],
                        expected["adapters"][i], state["adapters"][i]) for i in saved["adapters"]},
                    "gradients": state_difference_statistics(gradient_snapshots[0], gradient_snapshots[replay_index]),
                    "per_adapter_gradients": {str(i): state_difference_statistics(gradient_snapshots[0][i],
                        gradient_snapshots[replay_index][i]) for i in saved["adapters"]},
                })
            atomic_write_json(path.parent / f"{path.name}-FAILED_REPLAY.json", diagnostic, allow_nan=False)
    _replay_collective_check(errors)
    return {**assessment, "exact_state_restoration": True, "saved_state_sha256": saved_sha,
            "exact_restoration": exact_checkpoint, "initial_agreement": initial_agreement,
            "expected": {"metric": expected_metric, **expected_agreement},
            "checkpoint": {"metric": checkpoint_metric, **checkpoint_agreement},
            "in_memory_repeats": memory_receipts}


class OfficialV41TrainingSession:
    def __init__(self, model, adapters, train, validation, output, contract, config):
        self.model, self.adapters = model, adapters
        self.train, self.validation = train, validation
        self.output, self.contract, self.config = Path(output), contract, config
        if dist.get_world_size() != config.world_size:
            raise ValueError("training mesh does not match the experiment contract")
        if train.order_seed != 2234 or validation.order_seed != 2234:
            raise ValueError("the existing MathPilot order seed must remain 2234")
        self.optimizers = adapter_optimizers(model, adapters)
        self.stop_requested = False
        self.qualified = False
        self.qualification_report = None
        signal.signal(signal.SIGUSR1, self._request_stop)

    def _request_stop(self, signum, frame):
        self.stop_requested = True

    def qualify_mesh(self):
        """Two short updates, next-update replay, full-context update, then reset."""
        if self.qualified:
            return self.qualification_report
        initial = {i: _cpu_clone(adapter.state_dict()) for i, adapter in self.adapters.items()}
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        was_training = self.model.training
        self.model.train()
        report = {"short_updates": [], "context": self.config.context, "passed": False}
        frozen = [(p, p._version) for p in self.model.parameters() if not p.requires_grad]
        try:
            inputs, labels, _ = self.train.batch(dist.get_rank(), device="cuda", smoke_context=128)
            for step in range(2):
                metric = optimizer_step(self.model, self.optimizers, inputs, labels, learning_rate=1e-5)
                report["short_updates"].append(metric)
                emit("production_mesh_qualification", step=step, **metric)
                if step == 1:
                    missing = any(p.grad is None or not bool(p.grad.count_nonzero())
                                  for adapter in self.adapters.values() for p in adapter.parameters())
                    bad = torch.tensor(int(missing), device="cuda", dtype=torch.int32)
                    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                    if bad.item():
                        raise ValueError("missing second-step adapter gradients on the production mesh")
            path = self.output / "qualification-checkpoint"
            cursor = {"step": 2, "supervised_tokens": 0, "qualification_only": True}
            report["checkpoint_replay"] = qualify_checkpoint_replay(
                self.model, self.adapters, self.optimizers, inputs, labels,
                path=path, cursor=cursor, contract=self.contract)
            report["checkpoint_next_update_replay"] = True
            long_inputs, long_labels, _ = self.train.batch(
                dist.get_rank(), device="cuda", smoke_context=self.config.context, pad_to_full=True
            )
            torch.cuda.reset_peak_memory_stats()
            metric = optimizer_step(self.model, self.optimizers, long_inputs, long_labels, learning_rate=1e-7)
            report["full_context_update"] = metric
            emit("production_mesh_context_qualification", context=self.config.context, **metric)
            del long_inputs, long_labels
            if any(p._version != version or p.grad is not None for p, version in frozen):
                raise ValueError("the frozen base changed during production-mesh qualification")
            report["passed"] = True
        finally:
            for i, adapter in self.adapters.items():
                adapter.load_state_dict(initial[i], strict=True)
            self.optimizers = adapter_optimizers(self.model, self.adapters)
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
            torch.cuda.empty_cache()
        self.qualified = True
        self.qualification_report = report
        emit("production_mesh_qualified_fresh_state_restored")
        return report

    def run(self, *, resume_from: Path | None = None):
        self.qualify_mesh()
        self.model.train()
        start_step, consumed, warmup_tokens, last_saved = 0, 0, 0, -1
        total_steps = math.ceil(len(self.train) / self.config.world_size)
        if resume_from is not None:
            cursor = restore_adapter_checkpoint(resume_from, self.adapters, self.optimizers,
                                                contract=self.contract)
            start_step, consumed, warmup_tokens = (cursor["step"], cursor["supervised_tokens"],
                                                    cursor["warmup_tokens"])
            expected = sum(window["targets"] for window in self.train.windows[:start_step * self.config.world_size])
            if not 0 <= start_step <= total_steps or consumed != expected or not 0 <= warmup_tokens <= consumed:
                raise ValueError("checkpoint cursor disagrees with the sealed data order")
        append_metric(self.output / "validation.jsonl", evaluate(self.model, self.validation, step=start_step))
        next_eval = (consumed // self.config.validation_interval_tokens + 1) * self.config.validation_interval_tokens
        next_save = (consumed // self.config.checkpoint_interval_tokens + 1) * self.config.checkpoint_interval_tokens
        for step in range(start_step, total_steps):
            first = step * self.config.world_size
            inputs, labels, _ = self.train.batch(first + dist.get_rank(), device="cuda")
            rate = learning_rate(step, consumed, warmup_tokens, budget=self.config.supervised_tokens,
                                 warmup_steps=self.config.warmup_steps)
            metric = optimizer_step(self.model, self.optimizers, inputs, labels, learning_rate=rate)
            expected = sum(window["targets"] for window in self.train.windows[first:first + self.config.world_size])
            if metric["supervised_tokens"] != expected:
                raise RuntimeError("training update did not consume its unique sealed windows exactly once")
            consumed += metric["supervised_tokens"]
            if step < self.config.warmup_steps:
                warmup_tokens = consumed
            if consumed > self.config.supervised_tokens:
                raise RuntimeError("supervised token budget exceeded")
            record = {"step": step + 1, "consumed_supervised_tokens": consumed, **metric}
            append_metric(self.output / "train.jsonl", record)
            emit("train_step", **record)
            requested = self.stop_requested or (self.output / "STOP_REQUEST").exists()
            stop = torch.tensor(int(requested), device="cuda", dtype=torch.int32)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if consumed >= next_eval:
                append_metric(self.output / "validation.jsonl",
                              evaluate(self.model, self.validation, step=step + 1))
                next_eval = (consumed // self.config.validation_interval_tokens + 1) * self.config.validation_interval_tokens
            if consumed >= next_save or stop.item() or step + 1 == total_steps:
                cursor = {"step": step + 1, "supervised_tokens": consumed, "warmup_tokens": warmup_tokens}
                save_adapter_checkpoint(self.output / "checkpoints" / f"step-{step + 1:06d}",
                                        self.adapters, self.optimizers, cursor, self.contract)
                last_saved = step + 1
                next_save = (consumed // self.config.checkpoint_interval_tokens + 1) * self.config.checkpoint_interval_tokens
            if stop.item():
                emit("training_paused", step=step + 1, supervised_tokens=consumed, checkpoint_step=last_saved)
                return
        if consumed != self.config.supervised_tokens:
            raise RuntimeError("the sealed data did not reach exactly the requested budget")
        if dist.get_rank() == 0:
            atomic_write_json(self.output / "TRAINING_COMPLETE.json",
                              {"steps": total_steps, "supervised_tokens": consumed})
