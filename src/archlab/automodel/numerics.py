"""Small-model EP/FSDP numerical reference, never used during finetuning."""

from __future__ import annotations

import torch
import torch.distributed as dist

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration

from archlab.automodel.simplicial import ADAPTER_MARKER, install_simplicial_modules


def compare_combined_batch_reference(model, config, adapter_config, tokens, targets) -> dict:
    """Compare distributed local losses/averaged gradients with one global batch.

    All ranks participate in gathers. The reference is unsharded and uses the
    same upstream grouped experts, but the ordinary torch token dispatcher.
    Only call for the tiny qualification model, never for full-size weights.
    """
    if config.text_config.hidden_size > 256 or config.text_config.num_experts > 64:
        raise ValueError("the unsharded numerical reference is tiny-model-only")
    full_state = {name: (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().clone()
                  for name, value in model.state_dict().items()}
    reference = Qwen3_8_FlashNextForConditionalGeneration(
        config,
        backend=BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32", experts="torch_mm",
                              dispatcher="torch", rope_fusion=False, gate_precision="float32",
                              enable_hf_state_dict_adapter=False),
        moe_overrides={"aux_loss_coeff": 0.0},
    )
    reference.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    reference.to(device=tokens.device)
    # FSDP computes the added FP32 master weights in BF16. Match that compute
    # representation here, not an FP32 reference with different norm weights.
    install_simplicial_modules(reference, adapter_config, device=tokens.device, dtype=torch.bfloat16)
    reference.load_state_dict(full_state, strict=True)
    del full_state
    gathered_tokens = [torch.empty_like(tokens) for _ in range(dist.get_world_size())]
    gathered_targets = [torch.empty_like(targets) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tokens, tokens)
    dist.all_gather(gathered_targets, targets)
    global_tokens, global_targets = torch.cat(gathered_tokens), torch.cat(gathered_targets)
    model.zero_grad(set_to_none=True)
    local_logits = model(input_ids=tokens).logits
    loss_fn = MaskedCrossEntropy(reduction="mean")
    loss = loss_fn(local_logits, targets)
    loss.backward()
    reference_logits = reference(input_ids=global_tokens).logits
    reference_loss = loss_fn(reference_logits, global_targets)
    reference_loss.backward()
    expected_logits = reference_logits.chunk(dist.get_world_size())[dist.get_rank()]
    torch.testing.assert_close(local_logits, expected_logits, rtol=.02, atol=.02)
    mean_loss = loss.detach().clone()
    dist.all_reduce(mean_loss)
    mean_loss.div_(dist.get_world_size())
    torch.testing.assert_close(mean_loss, reference_loss, rtol=1e-3, atol=1e-3)
    reference_parameters = dict(reference.named_parameters())
    squared_difference, squared_reference = 0., 0.
    max_absolute = 0.
    for name, parameter in model.named_parameters():
        if ADAPTER_MARKER not in name:
            continue
        actual = parameter.grad.full_tensor().float()
        # named_parameters exposes checkpoint wrapper internals; state_dict
        # deliberately strips that wrapper. Compare the same logical name.
        logical_name = name.replace("._checkpoint_wrapped_module.", ".")
        expected = reference_parameters[logical_name].grad.float()
        difference = actual - expected
        squared_difference += difference.square().sum().item()
        squared_reference += expected.square().sum().item()
        max_absolute = max(max_absolute, difference.abs().max().item())
    relative_l2 = (squared_difference / max(squared_reference, 1e-30)) ** .5
    if relative_l2 > .03:
        raise AssertionError(f"EP/FSDP mean-gradient reference mismatch: relative L2 {relative_l2}")
    model.zero_grad(set_to_none=True)
    return {"logits_max_abs": (local_logits - expected_logits).abs().max().item(),
            "mean_loss": mean_loss.item(), "reference_loss": reference_loss.item(),
            "adapter_gradient_relative_l2": relative_l2, "adapter_gradient_max_abs": max_absolute}
