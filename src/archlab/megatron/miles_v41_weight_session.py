"""Miles session framing for the pinned BF16 V4.1 in-stream transaction.

This runtime has pause/flush/update/version/resume RPCs, but no quantized-base
or LoRA begin/end RPCs. Archlab's iterator and model loader instead exchange
begin/end tensors inside the acknowledged weight stream. The loader rejects
partial parameter coverage before Miles resumes generation.
"""


def begin(engines, selector="all", *, sync_base=True):
    if selector != "all" or not sync_base:
        raise ValueError("V4.1 in-stream transactions require complete base weights")


def end(engines, *, expected_lora_checksums=None):
    if expected_lora_checksums is not None:
        raise ValueError("V4.1 in-stream transactions do not support LoRA")


def install():
    import os

    from miles.backends.training_utils.weight_update import updater
    if os.environ.get("ARCHLAB_MILES_LIVE_WEIGHTS") != "1":
        raise ValueError("in-stream session requires the V4.1 live-weight loader")
    # Leave pause, flush, transfer acknowledgments, versioning and resume intact.
    updater.begin_weight_update = begin
    updater.end_weight_update = end
