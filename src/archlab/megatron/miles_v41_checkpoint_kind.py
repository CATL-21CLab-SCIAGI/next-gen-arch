"""Keep architectural attention adapters out of Miles's LoRA save path."""


def is_lora_model(model):
    for chunk in model:
        if hasattr(chunk.module, "peft_config"):
            return True
        for name, _ in chunk.named_parameters():
            if "lora_" in name or ("adapter" in name and ".archlab_adapter." not in name):
                return True
    return False


def install():
    from miles.backends.megatron_utils import checkpoint, hf_export, model

    for module in (checkpoint, hf_export, model):
        module.is_lora_model = is_lora_model
