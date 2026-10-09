"""Execute native V4.1 blocks as a tied prelude/core/coda without copying them."""

from types import MethodType

from torch import nn

from archlab.architectures.prelude_loop_coda import LoopLayout, boundary


class _BoundaryStep:
    def __init__(self, layer, apply):
        self.layer = layer
        self.apply = apply

    @property
    def engram(self):
        return self.layer.engram

    def __call__(self, hidden, pre_mix, state, **kwargs):
        hidden, state = self.apply(hidden, state)
        return self.layer(hidden, pre_mix, state, **kwargs)


class LoopedLayers(nn.ModuleDict):
    """Unique module registration, with a separate per-forward execution iterator.

    Boundary operations sit outside activation-checkpointed native blocks. Each
    iterator owns its anchor; backward recomputation cannot overwrite it. Native
    inspection/initialization still sees every physical block exactly once.
    """

    def __init__(self, layers, layout):
        super().__init__(layers.items())
        if list(self.keys()) != [str(i) for i in range(layout.stored_depth)]:
            raise ValueError("loop layout does not match the native block registry")
        self.layout = layout
        self.repetitions = 2
        self.executing = False

    def values(self):
        return self._execution_values() if self.executing else super().values()

    def _execution_values(self):
        anchor = state_at_prelude = None

        def capture(hidden, state):
            nonlocal anchor, state_at_prelude
            anchor, state_at_prelude = hidden, state
            return hidden, state

        def recur(hidden, state):
            # CSA2 ownership follows physical layer IDs. Each core pass starts
            # from prelude-owned KV/index state, while residual streams recur.
            return boundary(hidden, anchor), state_at_prelude

        def finish(hidden, state):
            return boundary(hidden, anchor), state

        start, end = self.layout.prelude, self.layout.prelude + self.layout.core
        for i in range(start):
            yield self[str(i)]
        for repetition in range(self.repetitions):
            yield _BoundaryStep(self[str(start)], capture if repetition == 0 else recur)
            for i in range(start + 1, end):
                yield self[str(i)]
        yield _BoundaryStep(self[str(end)], finish)
        for i in range(end + 1, self.layout.stored_depth):
            yield self[str(i)]


def _forward(self, *args, **kwargs):
    if self.layers.executing:
        raise RuntimeError("nested loop decoder forward is unsupported")
    self.layers.executing = True
    try:
        return self._archlab_loop_original_forward(*args, **kwargs)
    finally:
        self.layers.executing = False


def install_loop(model, *, layout=None):
    decoder = model.model
    if hasattr(decoder, "_archlab_loop_original_forward"):
        raise ValueError("loop execution is already installed")
    layout = layout or LoopLayout()
    before = {name: id(p) for name, p in model.named_parameters()}
    decoder.layers = LoopedLayers(decoder.layers, layout)
    decoder._archlab_loop_original_forward = decoder.forward
    decoder.forward = MethodType(_forward, decoder)
    if before != {name: id(p) for name, p in model.named_parameters()}:
        raise AssertionError("loop installation changed parameter registration")
    set_repetitions(model, 2)
    return {
        "reference_commit": "9139c396957a57b6de9a1e7efe7e35a4a863f1a6",
        "prelude": layout.prelude,
        "core": layout.core,
        "coda": layout.coda,
        "stored_depth": layout.stored_depth,
        "boundary": "FP32 feature RMSNorm + raw prelude / sqrt(2), after every core pass",
        "first_core_input": "raw prelude, following released code",
        "shared_parameters": True,
        "backpropagation": "all passes",
        "csa2_state": "prelude state at each core entry; final core state enters coda",
        "mhc": "feature normalization per stream; carried FP32 pre-mix retained",
        "auxiliary_normalization": "mean over executed indexers; router coefficient scaled by stored/executed blocks",
    }


def set_repetitions(model, repetitions):
    layers = model.model.layers
    execution = layers.layout.execution(repetitions)
    layers.repetitions = repetitions
    for key, layer in layers.items():
        layer.ffn.gate.aux_loss_coeff = 0.01 * layers.layout.stored_depth / len(execution)
        if layer.attn.indexer is not None:
            layer.attn.indexer._archlab_loop_visits = execution.count(int(key))
    return len(execution)
