"""Preserve replay reduction shapes around one-token resident decoding.

FP32 mixing and RMS reductions can round differently for a single row. Their
small operations keep the replay canvas; backbone attention and experts remain
incremental. All instance overrides are restored before leaving the context.
"""

from contextlib import contextmanager


def pad_at_position(value, position, canvas):
    if value.shape[1] != 1 or not 0 <= position < canvas:
        raise ValueError("expected one token inside its replay canvas")
    padded = value.new_zeros(value.shape[0], canvas, *value.shape[2:])
    padded[:, position : position + 1] = value
    return padded


def padded_collapse(original, hidden, pre, *, position, canvas):
    return original(
        pad_at_position(hidden, position, canvas), pad_at_position(pre, position, canvas)
    )[:, position : position + 1].contiguous()


@contextmanager
def replay_shape_boundaries(model, layers, *, position, canvas):
    import torch

    if torch.is_grad_enabled() or model.training:
        raise ValueError("cached replay shapes are inference-only")
    missing = object()
    originals = []

    def bind(module, name, function):
        originals.append((module, name, module.__dict__.get(name, missing)))
        setattr(module, name, function)

    try:
        norms = [model.model.norm]
        for _, inner, _ in layers:
            norms.extend((inner.attn_norm, inner.ffn_norm))
            for hc in (inner.attn_hc, inner.ffn_hc):
                original = hc.forward

                def forward(hidden, *, original=original):
                    mix = original(pad_at_position(hidden, position, canvas))
                    return type(mix)(
                        *(getattr(mix, name)[:, position : position + 1].contiguous()
                          for name in ("pre", "post", "comb"))
                    )

                bind(hc, "forward", forward)
                original = hc.collapse

                def collapse(hidden, pre, *, original=original):
                    return padded_collapse(original, hidden, pre, position=position, canvas=canvas)

                bind(hc, "collapse", collapse)
                name = "_archlab_original_expand" if hasattr(hc, "_archlab_original_expand") else "expand"
                original = getattr(hc, name)

                def expand(output, residual, mix, *, original=original):
                    padded_mix = type(mix)(
                        *(pad_at_position(getattr(mix, key), position, canvas)
                          for key in ("pre", "post", "comb"))
                    )
                    return original(
                        pad_at_position(output, position, canvas),
                        pad_at_position(residual, position, canvas), padded_mix,
                    )[:, position : position + 1].contiguous()

                bind(hc, name, expand)
        for norm in norms:
            original = norm.forward

            def forward(hidden, *, original=original):
                return original(pad_at_position(hidden, position, canvas))[
                    :, position : position + 1
                ].contiguous()

            bind(norm, "forward", forward)
        yield
    finally:
        for module, name, original in reversed(originals):
            if original is missing:
                module.__dict__.pop(name, None)
            else:
                setattr(module, name, original)
