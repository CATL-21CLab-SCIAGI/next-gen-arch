"""CUDA graph decode over preallocated native Limite K/V and short-axis caches.

Prefill uses the publisher's unpadded DynamicCache. Only one-token decoding uses
fixed storage: global keys occupy a prefix, local keys retain chronological
order in an exact 1025-slot window, and unused slots are masked. No framework
or publisher source is patched.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch
from transformers import DynamicCache

from archlab.architectures.limite_decode_state import InferenceBuffers


class DecodeCache(DynamicCache):
    def __init__(self, source, config, capacity, position):
        super().__init__(config=config)
        self.position = position
        self.capacity = capacity
        # LimiteConfig already converts the serialized history width to its
        # inclusive key span. Adding one here admits an extra key/zero slot.
        self.local_capacity = int(config.sliding_window)
        self.prefill_length = source.get_seq_length()
        self._limite_unpadded = False
        self.archlab_no_padding = True
        self.archlab_short = {}
        device = source.layers[0].keys.device
        self.global_indices = torch.arange(capacity, device=device)
        self.local_indices = torch.arange(self.local_capacity, device=device)
        self.global_lengths = torch.empty(3, device=device, dtype=torch.int32)
        self.local_lengths = torch.empty_like(self.global_lengths)
        self.buffers = []
        self.global_layers = set(config.global_layers)
        for i, layer in enumerate(source.layers):
            size = capacity if i in config.global_layers else self.local_capacity
            # Keep the publisher's [B,H,L,D] cache interface while owning
            # [B,L,H,D] storage. Both TileLang and FlashAttention consume the
            # latter: transposing these views avoids copying the entire global
            # context on every generated token, including unused cache slots.
            shape = (layer.keys.shape[0], size, layer.keys.shape[1], layer.keys.shape[-1])
            self.buffers.append([
                layer.keys.new_zeros(shape).transpose(1, 2),
                layer.values.new_zeros(shape).transpose(1, 2),
            ])
        self.load(source)

    def load(self, source):
        self.prefill_length = source.get_seq_length()
        if self.prefill_length > self.capacity:
            raise ValueError("prefill exceeds decode capacity")
        for layer_idx, (buffers, layer) in enumerate(zip(self.buffers, source.layers, strict=True)):
            for dest, data in zip(buffers, (layer.keys, layer.values), strict=True):
                dest.zero_()
                if layer_idx in self.global_layers:
                    dest[:, :, :data.shape[2]].copy_(data)
                else:
                    dest[:, :, -data.shape[2]:].copy_(data)
        for index, values in getattr(source, "archlab_short", {}).items():
            stored = self.archlab_short.get(index)
            if stored is None:
                stored = [data.new_zeros((data.shape[0], 16, *data.shape[2:])) for data in values]
            for dest, data in zip(stored, values, strict=True):
                dest.zero_()
                dest[:, -data.shape[1]:].copy_(data)
            self.archlab_short[index] = stored

    def get_seq_length(self, layer_idx=0):
        return self.prefill_length

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        buffers = self.buffers[layer_idx]
        for dest, data in zip(buffers, (key_states, value_states), strict=True):
            if layer_idx in self.global_layers:
                dest.index_copy_(2, self.position.reshape(1), data)
            else:
                dest.copy_(torch.cat((dest[:, :, 1:], data), dim=2))
        return buffers

    def update_short(self, key, value, layer_idx):
        stored = self.archlab_short[layer_idx]
        for dest, data in zip(stored, (key, value), strict=True):
            # Sixteen entries; fixed shapes/pointers survive graph replay.
            dest.copy_(torch.cat((dest[:, 1:], data), dim=1))
        return stored

    def decode_masks(self):
        length = self.position[0] + 1
        self.global_lengths[0].copy_(length)
        self.global_lengths[1].copy_(length.clamp(max=16))
        self.global_lengths[2].zero_()
        self.local_lengths[0].copy_(length.clamp(max=self.local_capacity))
        self.local_lengths[1].copy_(length.clamp(max=16))
        self.local_lengths[2].copy_((self.local_capacity - length).clamp(min=0))
        return dict(
            full_attention=(self.global_indices <= self.position[0])[None, None, None, :],
            sliding_attention=(self.local_indices >= self.local_lengths[2])[None, None, None, :],
        )

    def decode_lengths(self, is_global):
        return self.global_lengths if is_global else self.local_lengths


class GraphDecoder:
    """A reusable native decode graph with owned, fixed-address caches."""

    def __init__(self, model, source_cache, first_token, capacity, *, capture=True):
        if model.training or torch.is_grad_enabled():
            raise ValueError("graph decoding requires eval mode under no_grad")
        self.model = model
        self.input_ids = first_token.clone()
        self.position = torch.tensor([source_cache.get_seq_length()], device=first_token.device)
        self.cache = DecodeCache(source_cache, model.config, capacity, self.position)
        if hasattr(source_cache, "archlab_native_preludes"):
            self.cache.archlab_native_preludes = DecodeCache(
                source_cache.archlab_native_preludes, model.config, capacity, self.position
            )
        self.graph = None
        if capture:
            self.graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(device=first_token.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Initialize TileLang/SDPA/compiler workspaces before capture.
                for _ in range(2):
                    self.forward()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self.logits = self.forward()
        # CUDA graphs keep raw addresses, not Python references to external
        # eval constants. Model.train/eval can replace these buffers later.
        self.inference_buffers = tuple(
            value for module in model.modules() for name, value in module._buffers.items()
            if name.startswith("_inference_") and value is not None
        )
        self.reset(source_cache, first_token)

    def reset(self, source_cache, first_token):
        if first_token.shape != self.input_ids.shape:
            raise ValueError("decode graph batch changed")
        self.position.fill_(source_cache.get_seq_length())
        self.input_ids.copy_(first_token)
        self.cache.load(source_cache)
        if hasattr(self.cache, "archlab_native_preludes"):
            self.cache.archlab_native_preludes.load(source_cache.archlab_native_preludes)

    def forward(self):
        native_masks = ({} if hasattr(self.cache, "archlab_native_preludes") else
                        dict(attention_mask=self.cache.decode_masks()))
        return self.model(
            input_ids=self.input_ids, position_ids=self.position[None, :].expand(self.input_ids.shape[0], -1),
            past_key_values=self.cache, use_cache=True, logits_to_keep=1,
            **native_masks,
        ).logits[:, -1]

    def __call__(self, token, position):
        if not 0 <= position < self.cache.capacity:
            raise ValueError("decode position exceeds the preallocated context")
        self.input_ids.copy_(token)
        self.position.fill_(position)
        if self.graph is None:
            self.logits = self.forward()
        else:
            self.graph.replay()
        return self.logits


class GraphDecoderPool:
    """Bound graph residency and refresh captured native constants per policy."""

    def __init__(self, model, max_entries=8, capacity_quantum=1024):
        if max_entries < 1 or capacity_quantum < 1:
            raise ValueError("decode pool limits must be positive")
        self.model = model
        self.max_entries = max_entries
        self.capacity_quantum = capacity_quantum
        self.buffers = InferenceBuffers()
        self.entries = OrderedDict()
        self.hits = self.misses = 0
        self.capture_seconds = 0.0
        self.capture_allowed = True
        self.eager_misses = 0
        self._startup_keys = set()

    def freeze_capture(self):
        """Reuse startup graphs; unexpected keys keep identical fixed caches.

        PyTorch forbids concurrent uncaptured work during capture. In addition,
        its graph context synchronizes the entire device, so actor pool misses
        must never capture while the learner can enter a collective.
        """
        self.capture_allowed = False
        self._startup_keys.update(key for key, decoder in self.entries.items() if decoder.graph is not None)

    def synchronize(self):
        self.buffers.synchronize(self.model)

    def get(self, source, token, capacity):
        rounded = min(
            ((capacity + self.capacity_quantum - 1) // self.capacity_quantum) * self.capacity_quantum,
            self.model.config.max_position_embeddings,
        )
        if rounded < capacity:
            raise ValueError("requested decode context exceeds the model limit")
        key = (token.shape[0], rounded)
        decoder = self.entries.pop(key, None)
        if decoder is None:
            started = time.perf_counter()
            decoder = GraphDecoder(self.model, source, token, rounded, capture=self.capture_allowed)
            if self.capture_allowed:
                self.capture_seconds += time.perf_counter() - started
            else:
                self.eager_misses += 1
            self.misses += 1
        else:
            decoder.reset(source, token)
            self.hits += 1
        self.entries[key] = decoder
        while len(self.entries) > self.max_entries:
            evicted = next(key for key in self.entries if key not in self._startup_keys)
            del self.entries[evicted]
        return decoder
