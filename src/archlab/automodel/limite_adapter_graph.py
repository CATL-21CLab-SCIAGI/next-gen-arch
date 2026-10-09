"""Opt-in replay of unchanged native Limite forward/backward kernels."""

from __future__ import annotations

import time

import torch


def graph_contract(mode="none", *, static_gradient_sync=False):
    if mode not in ("none", "manual"):
        raise ValueError("invalid Limite CUDA graph mode")
    if mode == "none":
        if static_gradient_sync:
            raise ValueError("static gradient synchronization requires manual graphs")
        return {"mode": mode}
    result = {
        "mode": mode,
        "scope": "zero persistent gradients and native forward/backward only",
        "optimizer_clip_communication": "outside capture; unchanged ordering",
        "normalization": "native loss sum multiplied by world then divided by remaining tokens",
        "partial_token_step": "unchanged uncaptured forward/backward",
        "preparation": "three forward/backward warmups; no optimizer update or data advance; RNG restored",
        "stream": "one dedicated nondefault stream for construction, training and capture",
    }
    if static_gradient_sync:
        result["gradient_presence"] = "qualified_all_used_manual_graph"
    return result


class ManualTrainingGraph:
    """One fixed-shape graph with persistent gradients and dynamic token inputs.

    Callers retain the original optimizer, clipping and FP32 distributed mean.
    Preparation never updates parameters or optimizer state. Partial final steps
    use the original objective and denominator with the same attached gradients.
    """

    def __init__(
        self, objective, optimizer, *, world_size, global_batch, context, static_gradient_sync=False
    ):
        if world_size < 1 or global_batch < 1 or global_batch % world_size or context < 1:
            raise ValueError("invalid graph batch/context contract")
        model = objective.policy
        backbone = model.model
        if (
            backbone.adapter_config["variant"] != "normal"
            or backbone.adapter_config["attention_backend"] != "tilelang"
            or backbone.trainable_mode != "full"
            or backbone.normal_kernel != "gqa"
            or objective.compilation["mode"] != "strict-blocks"
            or objective.chunk != 512
            or objective.checkpoint_head
        ):
            raise ValueError("manual graphs require the qualified full-weight normal configuration")
        attentions = tuple(layer.self_attn for layer in backbone.base.layers) + tuple(
            adapter.native for adapter in backbone.adapters
        )
        if any(attention.attention_dropout for attention in attentions):
            raise ValueError("manual graphs require native zero dropout")
        self.objective, self.optimizer = objective, optimizer
        self.world_size = world_size
        self.shape = (global_batch // world_size, context)
        if self.shape != (8, 2048):
            raise ValueError(
                "manual graph execution is qualified only for local batch 8/context 2048"
            )
        self.full_tokens = global_batch * context
        self.parameters = tuple(model.parameters())
        if static_gradient_sync and (
            world_size != 16 or global_batch != 128 or len(self.parameters) != 903
        ):
            raise ValueError("static synchronization requires the qualified 16-rank/903-tensor graph")
        self._static_gradient_sync = static_gradient_sync
        self.gradient_groups = None
        self.parameter_identities = tuple(id(parameter) for parameter in self.parameters)
        self.geometry = self._geometry()
        self.stream = torch.cuda.current_stream()
        if not self.stream.cuda_stream:
            raise ValueError("set a dedicated nondefault stream before constructing the model")
        backbone.enable_static_training_context()
        self.graph = None
        self.ids = self.labels = self.loss = None
        self.input_strides = None
        self.native_context = None
        self.gradient_storage = None
        self.parameter_storage = None
        self.capture_seconds = None
        self.replays = self.partial_fallbacks = 0

    @property
    def static_gradient_sync(self):
        return self._static_gradient_sync

    def _storage(self, gradients=False):
        return tuple(
            None
            if gradients and parameter.grad is None
            else (parameter.grad if gradients else parameter).data_ptr()
            for parameter in self.parameters
        )

    def _geometry(self):
        config = self.objective.policy.config
        return (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.sliding_window,
            tuple(config.global_layers),
            tuple(config.layer_types),
            config._attn_implementation,
            getattr(self.objective.policy.model, "normal_backward", "tilelang"),
            self.static_gradient_sync,
        )

    def _validate(self, ids, labels):
        backbone = self.objective.policy.model
        if not backbone.training or not backbone.base.training:
            raise RuntimeError("training graph replay requires full-weight training mode")
        if backbone.normal_kernel != "gqa":
            raise RuntimeError("training graph kernel changed after construction")
        if (
            self._geometry() != self.geometry
            or tuple(id(parameter) for parameter in self.objective.policy.parameters())
            != self.parameter_identities
        ):
            raise RuntimeError("training graph model geometry or parameter identities changed")
        if tuple(ids.shape) != self.shape or labels.shape != ids.shape:
            raise ValueError("training graph inputs changed shape")
        if ids.dtype != torch.int64 or labels.dtype != torch.int64:
            raise ValueError("training graph inputs require native integer token IDs and targets")
        if ids.device != self.parameters[0].device or labels.device != ids.device:
            raise ValueError("training graph inputs changed device")
        if self.input_strides is not None and (ids.stride(), labels.stride()) != self.input_strides:
            raise ValueError("training graph input layouts changed")
        if torch.cuda.current_stream().cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("training graph replay changed CUDA stream")
        if self.graph is not None and (
            self._storage() != self.parameter_storage
            or self._storage(True) != self.gradient_storage
        ):
            raise RuntimeError("training graph parameter or gradient storage changed")

    def _backward(self, ids, labels, remaining):
        self.optimizer.zero_grad(set_to_none=False)
        # Keep both native eager scalar operations, in their original order.
        loss = self.objective(ids, labels) * self.world_size / remaining
        loss.backward()
        return loss

    def _prepare_gradient_groups(self):
        if self.static_gradient_sync:
            from archlab.automodel.limite_adapter_communication import StaticGradientGroups

            if self.gradient_groups is None:
                self.gradient_groups = StaticGradientGroups(self.parameters)
            else:
                self.gradient_groups.validate()

    def _capture(self):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.ids.device)
        started = time.perf_counter()
        try:
            # These tensors are allocated before capture, outside its private
            # pool. Keep them alive when evaluation invalidates the model cache.
            self.native_context = self.objective.policy.model.prepare_static_training_context(
                self.ids
            )
            for _ in range(3):
                self._backward(self.ids, self.labels, self.full_tokens)
            torch.cuda.synchronize(self.ids.device)
            self.parameter_storage = self._storage()
            self.gradient_storage = self._storage(True)
            if any(pointer is None for pointer in self.gradient_storage):
                raise RuntimeError("training graph warmup left a trainable parameter unused")
            self._prepare_gradient_groups()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                self.loss = self._backward(self.ids, self.labels, self.full_tokens)
            if (
                self._storage() != self.parameter_storage
                or self._storage(True) != self.gradient_storage
            ):
                raise RuntimeError("capture changed parameter or gradient storage")
            self.graph = graph
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.ids.device)
        self.capture_seconds = time.perf_counter() - started

    def synchronize_gradients(self):
        """Reduce the qualified persistent gradients through the unchanged FP32 mean."""
        if not self.static_gradient_sync or self.gradient_groups is None:
            raise RuntimeError("static synchronization requires a prepared qualified graph")
        if torch.cuda.current_stream().cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("training graph synchronization changed CUDA stream")
        self.gradient_groups.synchronize(world=self.world_size)

    def backward(self, ids, labels, *, remaining_tokens):
        self._validate(ids, labels)
        if not 0 < remaining_tokens <= self.full_tokens:
            raise ValueError("invalid remaining-token denominator")
        if remaining_tokens != self.full_tokens:
            self.partial_fallbacks += 1
            self.objective.policy.model.prepare_static_training_context(ids)
            loss = self._backward(ids, labels, remaining_tokens)
            # A resume may contain only the final partial step. Qualify its
            # ordinary backward too, without requiring a prior graph capture.
            self._prepare_gradient_groups()
            return loss
        if self.ids is None:
            self.input_strides = (ids.stride(), labels.stride())
            self.ids = torch.empty_strided(
                self.shape, ids.stride(), dtype=ids.dtype, device=ids.device
            )
            self.labels = torch.empty_strided(
                self.shape, labels.stride(), dtype=labels.dtype, device=labels.device
            )
        self.ids.copy_(ids)
        self.labels.copy_(labels)
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self.replays += 1
        return self.loss

    def runtime(self):
        return dict(
            **graph_contract("manual", static_gradient_sync=self.static_gradient_sync),
            captured=self.graph is not None,
            shape=list(self.shape),
            input_strides=self.input_strides,
            qualified_production_shape=[8, 2048],
            capture_seconds=self.capture_seconds,
            replays=self.replays,
            partial_fallbacks=self.partial_fallbacks,
            parameter_tensors=len(self.parameters),
            persistent_gradient_tensors=(
                sum(pointer is not None for pointer in self.gradient_storage)
                if self.gradient_storage is not None
                else 0
            ),
            normal_backward=getattr(self.objective.policy.model, "normal_backward", "tilelang"),
        )
