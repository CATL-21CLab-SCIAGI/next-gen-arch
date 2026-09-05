"""Sample a native Flash-Next checkpoint as one complete DP-only replica.

Full-prefix recomputation is intentional: GDN and PLE do not implement a shared
incremental inference-cache contract. This is a qualitative check, not a serving
benchmark, and never loads or changes optimizer state.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from archlab.architectures.qwen38_flash_next_full import Qwen38FlashNextFullConfig
from archlab.megatron.qwen38_flash_next_full_train import (
    _assert_dp_only_groups,
    _atomic_json,
    _megatron_argv,
    build_model,
)
from archlab.megatron.backend import validate_runtime
from archlab.megatron.qwen38_flash_next_full_train import (
    _parser as trainer_parser,
)
from archlab.prompts import load_prompts


def select_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise RuntimeError("sampling logits contain nonfinite values")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, indices = probabilities.sort(descending=True, dim=-1)
    excluded = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(excluded, 0)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return indices.gather(-1, sampled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).parents[1] / "prompts" / "backbone_validation.yaml",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite samples: {args.output}")
    if args.max_new_tokens < 1 or args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid generation controls")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("sampling uses exactly one complete model replica")
    if not (args.checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
        raise ValueError("sampling requires a completed checkpoint marker")

    config = Qwen38FlashNextFullConfig.billion_depth48_no_mtp()
    trainer = trainer_parser().parse_args(
        [
            "--data-root",
            str(args.tokenizer),
            "--tokenizer",
            str(args.tokenizer),
            "--run-dir",
            str(args.output.parent / "sampling-runtime"),
            "--load-dir",
            str(args.checkpoint_root),
            "--model-variant",
            "1b-depth48-no-mtp",
            "--parallelism",
            "dp-only",
            "--global-batch-size",
            "1",
            "--micro-batch-size",
            "1",
            "--probe-steps",
            "1",
        ]
    )
    sys.argv = _megatron_argv(trainer, config) + ["--no-load-optim", "--no-load-rng"]
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import Float16Module
    from megatron.training.arguments import (
        core_transformer_config_from_args,
        parse_args,
        validate_args,
    )
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import initialize_megatron
    from tokenizers import Tokenizer

    native = validate_args(parse_args())
    native.tensorboard_dir = None
    set_global_variables(native)
    initialize_megatron()
    groups = ProcessGroupCollection.use_mpu_process_groups()
    topology = _assert_dp_only_groups(groups)
    native_config = core_transformer_config_from_args(native)
    model = Float16Module(native_config, build_model(config, native_config, groups).cuda())
    iteration, _ = load_checkpoint([model], None, None, strict=True)
    model.eval()
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    records = []
    evidence = {
        "checkpoint_root": str(args.checkpoint_root),
        "iteration": iteration,
        "parallelism": topology,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "full-prefix recomputation; no chat template",
        "samples": records,
        "runtime": validate_runtime(require_pretrain=False),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with torch.inference_mode():
        for prompt in load_prompts(args.prompts):
            initial = tokenizer.encode(prompt.text, add_special_tokens=False).ids
            tokens = torch.tensor([initial], device="cuda", dtype=torch.long)
            generated = []
            for _ in range(args.max_new_tokens):
                positions = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
                logits = model(tokens, positions, None)
                token = select_token(
                    logits[:, -1, :], temperature=args.temperature, top_p=args.top_p
                )
                token_id = token.item()
                generated.append(token_id)
                if token_id == config.eos_token_id:
                    break
                tokens = torch.cat((tokens, token), dim=-1)
            record = {
                "id": prompt.id,
                "prompt": prompt.text,
                "token_ids": generated,
                "continuation": tokenizer.decode(generated, skip_special_tokens=True),
            }
            records.append(record)
            _atomic_json(args.output, evidence)
            print(record, flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
