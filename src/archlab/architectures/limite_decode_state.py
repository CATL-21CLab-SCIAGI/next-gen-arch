"""Native inference constants and exact row views for reusable decode graphs."""

from types import SimpleNamespace


class InferenceBuffers:
    """Keep captured pointers stable while refreshing every native eval constant."""

    def __init__(self):
        self.buffers = {}

    def synchronize(self, model):
        if model.training:
            raise ValueError("inference buffer refresh requires eval mode")
        current = {}
        for module_name, module in model.named_modules():
            for name, value in module._buffers.items():
                if name.startswith("_inference_") and value is not None:
                    current[(module_name, name)] = (module, value)
        if self.buffers and set(current) != set(self.buffers):
            raise ValueError("native inference buffer topology changed")
        for key, (module, value) in current.items():
            stable = self.buffers.get(key)
            if stable is None:
                # A publisher may return a detached view of a Parameter here.
                # Own the storage; never copy folded values into trainable weights.
                stable = value.detach().clone()
                self.buffers[key] = stable
            elif stable.shape != value.shape or stable.dtype != value.dtype or stable.device != value.device:
                raise ValueError("native inference buffer geometry changed")
            elif stable is not value:
                stable.copy_(value)
            setattr(module, key[1], stable)


def cache_rows(source, rows, *, length=None):
    """Select complete native histories, including prelude and short-axis state."""
    length = source.get_seq_length() if length is None else length
    if length < 1:
        raise ValueError("decode cache selection requires a nonempty prefix")
    if hasattr(source, "buffers"):
        layers = []
        for index, (key, value) in enumerate(source.buffers):
            extent = min(length, key.shape[2])
            sl = slice(0, extent) if index in source.global_layers else slice(-extent, None)
            layers.append(SimpleNamespace(
                keys=key[:, :, sl].index_select(0, rows),
                values=value[:, :, sl].index_select(0, rows),
            ))
    else:
        layers = [SimpleNamespace(keys=x.keys.index_select(0, rows), values=x.values.index_select(0, rows))
                  for x in source.layers]
    result = SimpleNamespace(
        layers=layers, get_seq_length=lambda: length,
        archlab_short={index: [value[:, -min(length, value.shape[1]):].index_select(0, rows)
                              for value in pair]
                       for index, pair in getattr(source, "archlab_short", {}).items()},
    )
    if hasattr(source, "archlab_native_preludes"):
        result.archlab_native_preludes = cache_rows(source.archlab_native_preludes, rows, length=length)
    return result
