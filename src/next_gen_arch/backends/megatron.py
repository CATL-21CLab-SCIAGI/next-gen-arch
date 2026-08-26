"""Megatron-LM command renderer with a strict read-only submodule boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from next_gen_arch.architectures import require_backend_support
from next_gen_arch.backends.base import LaunchPlan
from next_gen_arch.spec import ExperimentSpec, SpecError

MEGATRON_URL = "https://github.com/NVIDIA/Megatron-LM.git"
MEGATRON_COMMIT = "55ac7082517c3878ae653c07c09c534b8aed49f6"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = REPOSITORY_ROOT / "third_party" / "Megatron-LM"


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={MEGATRON_ROOT}", *args],
        cwd=MEGATRON_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def validate_submodule() -> dict[str, str]:
    """Verify that the submodule exists, is pinned, and has no local patches."""
    if not (MEGATRON_ROOT / "megatron" / "core").is_dir():
        raise RuntimeError(
            "Megatron-LM submodule is not initialized; run "
            "`git submodule update --init --recursive`"
        )
    actual = _git_output("rev-parse", "HEAD")
    if actual != MEGATRON_COMMIT:
        raise RuntimeError(f"Megatron-LM commit {actual} != locked {MEGATRON_COMMIT}")
    dirty = _git_output("status", "--porcelain")
    if dirty:
        raise RuntimeError(f"read-only Megatron-LM submodule has local changes:\n{dirty}")
    return {"url": MEGATRON_URL, "commit": actual, "path": str(MEGATRON_ROOT)}


def _required(mapping: dict[str, Any], *keys: str, section: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise SpecError(f"{section} is missing: {', '.join(missing)}")


class MegatronBackend:
    name = "megatron"

    def doctor(self) -> dict[str, str]:
        return validate_submodule()

    def render(
        self,
        spec: ExperimentSpec,
        *,
        path_overrides: dict[str, str] | None = None,
    ) -> LaunchPlan:
        require_backend_support(spec.variant, self.name)
        submodule = validate_submodule()
        paths = spec.resolve_paths(path_overrides)
        _required(paths, "train_data", "valid_data", "tokenizer", "output_dir", section="paths")

        model = spec.model
        training = spec.training
        parallel = spec.parallelism
        _required(
            model,
            "num_layers",
            "hidden_size",
            "ffn_hidden_size",
            "num_attention_heads",
            "sequence_length",
            section="model",
        )
        _required(
            training,
            "micro_batch_size",
            "global_batch_size",
            "target_train_tokens",
            "learning_rate",
            "minimum_learning_rate",
            section="training",
        )

        nodes = int(parallel.get("nodes", 1))
        gpus = int(parallel.get("gpus_per_node", 8))
        tensor_parallel = int(parallel.get("tensor", 1))
        pipeline_parallel = int(parallel.get("pipeline", 1))
        context_parallel = int(parallel.get("context", 1))
        expert_parallel = int(parallel.get("expert", 1))
        sequence_parallel = bool(parallel.get("sequence_parallel", tensor_parallel > 1))
        world_size = nodes * gpus
        model_parallel = tensor_parallel * pipeline_parallel * context_parallel
        if min(
            nodes,
            gpus,
            tensor_parallel,
            pipeline_parallel,
            context_parallel,
            expert_parallel,
        ) < 1:
            raise SpecError("all parallelism values must be positive")
        if world_size % model_parallel:
            raise SpecError(
                f"world size {world_size} is not divisible by model parallel size {model_parallel}"
            )
        data_parallel = world_size // model_parallel
        if data_parallel % expert_parallel:
            raise SpecError(
                f"data parallel size {data_parallel} is not divisible by expert parallel "
                f"size {expert_parallel}"
            )

        sequence_length = int(model["sequence_length"])
        num_layers = int(model["num_layers"])
        hidden_size = int(model["hidden_size"])
        ffn_hidden_size = int(model["ffn_hidden_size"])
        moe_ffn_hidden_size = int(model.get("moe_ffn_hidden_size", ffn_hidden_size))
        num_attention_heads = int(model["num_attention_heads"])
        if (
            hidden_size % tensor_parallel
            or ffn_hidden_size % tensor_parallel
            or moe_ffn_hidden_size % tensor_parallel
        ):
            raise SpecError("hidden and FFN sizes must be divisible by tensor parallel size")
        if num_attention_heads % tensor_parallel:
            raise SpecError("attention heads must be divisible by tensor parallel size")
        if num_layers % pipeline_parallel:
            raise SpecError("layers must be divisible by pipeline parallel size")
        if context_parallel > 1 and sequence_length % (2 * context_parallel):
            raise SpecError("sequence length must be divisible by 2 * context parallel size")

        num_experts = model.get("num_experts")
        if expert_parallel > 1 and num_experts is None:
            raise SpecError("expert parallelism requires model.num_experts")
        if num_experts is not None:
            num_experts = int(num_experts)
            if num_experts < 1 or num_experts % expert_parallel:
                raise SpecError("num_experts must be positive and divisible by expert parallel size")
        if expert_parallel > 1 and tensor_parallel > 1 and not sequence_parallel:
            raise SpecError("combined tensor and expert parallelism requires sequence parallelism")

        micro_batch = int(training["micro_batch_size"])
        global_batch = int(training["global_batch_size"])
        distributed_micro_batch = micro_batch * data_parallel
        if global_batch % distributed_micro_batch:
            raise SpecError(
                f"global batch {global_batch} is not divisible by micro batch {micro_batch} "
                f"times data parallel size {data_parallel}"
            )
        num_microbatches = global_batch // distributed_micro_batch
        if pipeline_parallel > 1 and num_microbatches < pipeline_parallel:
            raise SpecError("pipeline parallelism requires at least one microbatch per stage")
        tokens_per_iteration = sequence_length * global_batch
        target_tokens = int(training["target_train_tokens"])
        train_iters = target_tokens // tokens_per_iteration
        if train_iters < 1:
            raise SpecError("target_train_tokens is smaller than one global batch")
        effective_tokens = train_iters * tokens_per_iteration

        if nodes == 1:
            launcher = ["torchrun", "--standalone", f"--nproc-per-node={gpus}"]
        else:
            launcher = [
                "torchrun",
                f"--nnodes={nodes}",
                f"--nproc-per-node={gpus}",
                "--node-rank",
                "env:NODE_RANK",
                "--master-addr",
                "env:MASTER_ADDR",
                "--master-port",
                "env:MASTER_PORT",
            ]

        output = Path(paths["output_dir"])
        argv = [
            *launcher,
            str(MEGATRON_ROOT / "pretrain_gpt.py"),
            "--use-mcore-models",
            "--num-layers",
            str(num_layers),
            "--hidden-size",
            str(hidden_size),
            "--ffn-hidden-size",
            str(ffn_hidden_size),
            "--num-attention-heads",
            str(num_attention_heads),
            "--seq-length",
            str(sequence_length),
            "--max-position-embeddings",
            str(sequence_length),
            "--position-embedding-type",
            "rope",
            "--normalization",
            "RMSNorm",
            "--swiglu",
            "--disable-bias-linear",
            "--untie-embeddings-and-output-weights",
            "--micro-batch-size",
            str(micro_batch),
            "--global-batch-size",
            str(global_batch),
            "--train-iters",
            str(train_iters),
            "--tensor-model-parallel-size",
            str(tensor_parallel),
            "--pipeline-model-parallel-size",
            str(pipeline_parallel),
            "--context-parallel-size",
            str(context_parallel),
            "--expert-model-parallel-size",
            str(expert_parallel),
            "--distributed-backend",
            "nccl",
            "--optimizer",
            str(training.get("optimizer", "adam")),
            "--lr",
            str(training["learning_rate"]),
            "--min-lr",
            str(training["minimum_learning_rate"]),
            "--lr-decay-style",
            str(training.get("lr_decay_style", "cosine")),
            "--lr-warmup-fraction",
            str(training.get("lr_warmup_fraction", 0.01)),
            "--weight-decay",
            str(training.get("weight_decay", 0.1)),
            "--clip-grad",
            str(training.get("clip_grad", 1.0)),
            "--bf16",
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            paths["tokenizer"],
            "--train-data-path",
            paths["train_data"],
            "--valid-data-path",
            paths["valid_data"],
            "--seed",
            str(spec.seed),
            "--eval-interval",
            str(training.get("eval_interval", 250)),
            "--eval-iters",
            str(training.get("eval_iters", 20)),
            "--log-interval",
            str(training.get("log_interval", 1)),
        ]
        if training.get("save", True):
            argv.extend(
                (
                    "--save",
                    str(output / "checkpoints"),
                    "--save-interval",
                    str(training.get("save_interval", 1000)),
                )
            )
        if training.get("tensorboard", True):
            argv.extend(("--tensorboard-dir", str(output / "tensorboard")))
        if model.get("transformer_impl"):
            argv.extend(("--transformer-impl", str(model["transformer_impl"])))
        if "attention_dropout" in model:
            argv.extend(("--attention-dropout", str(model["attention_dropout"])))
        if "hidden_dropout" in model:
            argv.extend(("--hidden-dropout", str(model["hidden_dropout"])))
        if "init_method_std" in model:
            argv.extend(("--init-method-std", str(model["init_method_std"])))
        if sequence_parallel:
            argv.append("--sequence-parallel")
        if context_parallel > 1:
            argv.extend(("--cp-comm-type", str(parallel.get("context_comm_type", "p2p"))))
        if num_experts is not None:
            argv.extend(
                (
                    "--num-experts",
                    str(num_experts),
                    "--moe-ffn-hidden-size",
                    str(moe_ffn_hidden_size),
                    "--moe-router-topk",
                    str(model.get("moe_router_topk", 1)),
                    "--moe-router-load-balancing-type",
                    str(model.get("moe_router_load_balancing_type", "aux_loss")),
                    "--moe-aux-loss-coeff",
                    str(model.get("moe_aux_loss_coeff", 0.0)),
                    "--moe-token-dispatcher-type",
                    str(model.get("moe_token_dispatcher_type", "alltoall")),
                )
            )
            if model.get("moe_router_dtype"):
                argv.extend(("--moe-router-dtype", str(model["moe_router_dtype"])))
            if model.get("moe_grouped_gemm", False):
                argv.append("--moe-grouped-gemm")
            if model.get("moe_permute_fusion", False):
                argv.append("--moe-permute-fusion")
        if paths.get("data_cache"):
            argv.extend(("--data-cache-path", paths["data_cache"]))
        if training.get("optimizer") == "muon":
            argv.extend(("--muon-momentum", str(training.get("muon_momentum", 0.95))))
        if "adam_beta1" in training:
            argv.extend(("--adam-beta1", str(training["adam_beta1"])))
        if "adam_beta2" in training:
            argv.extend(("--adam-beta2", str(training["adam_beta2"])))
        if "adam_epsilon" in training:
            argv.extend(("--adam-eps", str(training["adam_epsilon"])))
        if training.get("calculate_per_token_loss", False):
            argv.append("--calculate-per-token-loss")
        if training.get("log_throughput", False):
            argv.append("--log-throughput")
        if training.get("deterministic", False):
            argv.extend(("--deterministic-mode", "--no-gradient-accumulation-fusion"))
        if training.get("use_distributed_optimizer", False):
            argv.append("--use-distributed-optimizer")

        launch_env = {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}
        if training.get("deterministic", False):
            launch_env.update(
                {
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "NCCL_ALGO": "Ring",
                    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
                }
            )

        return LaunchPlan(
            backend=self.name,
            argv=tuple(argv),
            env=launch_env,
            metadata={
                "variant": spec.variant,
                "world_size": world_size,
                "model_parallel_size": model_parallel,
                "data_parallel_size": data_parallel,
                "expert_parallel_size": expert_parallel,
                "expert_data_parallel_size": data_parallel // expert_parallel,
                "num_microbatches": num_microbatches,
                "target_training_tokens": target_tokens,
                "effective_training_tokens": effective_tokens,
                "prompt_file": spec.resolve_reference(spec.prompts),
                "submodule": submodule,
                "semantic_equivalence": (
                    "scaling backend only; speedrun optimizer/kernel equivalence is not claimed"
                ),
            },
        )
