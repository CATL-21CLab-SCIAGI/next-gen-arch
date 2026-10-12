"""Compose a native Limite bootstrap contract and delegate SFT to pinned verl.

Launch ``train`` through torchrun. This entry composes configuration only; its
existence does not qualify an architecture, checkpoint or dataset for production.
The prepared dataset owns native tokenization and assistant target masks, while
verl owns the CE objective, distributed updates, checkpointing and monitoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import MISSING, DictConfig, OmegaConf

from archlab.artifacts import sha256_file


class BootstrapParentStage(str, Enum):
    original_base = "original_base"
    package_sft = "package_sft"


@dataclass
class BootstrapExperiment:
    name: str = MISSING
    parent_checkpoint: str = MISSING
    data_manifest: str = MISSING
    parent_stage: BootstrapParentStage = BootstrapParentStage.original_base
    variant: str = "simplicial"
    objective: str = "assistant_ce"
    native_context: int = 131072
    settings: dict[str, Any] = field(default_factory=dict)


def read_bootstrap(path: Path, overrides: list[str] = ()) -> BootstrapExperiment:
    config = OmegaConf.merge(OmegaConf.structured(BootstrapExperiment), OmegaConf.load(path))
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator or not key:
            raise ValueError("overrides must use key=value")
        value = OmegaConf.from_dotlist([f"value={raw}"]).value
        if key.startswith("settings."):
            config.settings[key.removeprefix("settings.")] = value
        else:
            OmegaConf.update(config, key, value, merge=False)
    experiment = OmegaConf.to_object(config)
    if experiment.objective != "assistant_ce":
        raise ValueError("bootstrap requires assistant-only cross entropy")
    if experiment.variant not in {"normal", "simplicial"}:
        raise ValueError("bootstrap requires a native normal or simplicial export")
    if experiment.native_context <= 0:
        raise ValueError("native_context must be positive")
    return experiment


def compose_sft(experiment: BootstrapExperiment, upstream: Path) -> DictConfig:
    with initialize_config_dir(
        config_dir=str((upstream / "verl/trainer/config").resolve()), version_base=None
    ):
        config = compose(
            config_name="sft_trainer_engine", overrides=["engine=megatron", "optim=megatron"]
        )
    OmegaConf.set_struct(config, True)
    for key, value in experiment.settings.items():
        OmegaConf.update(config, key, value, merge=False)
    config.trainer.experiment_name = experiment.name
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    validate_sft(experiment, config)
    return config


def validate_sft(experiment: BootstrapExperiment, config: DictConfig) -> None:
    model, engine, data = config.model, config.engine, config.data
    if model.path != experiment.parent_checkpoint:
        raise ValueError("bootstrap must start from its explicit parent export")
    if model.external_lib != "archlab.training.megatron" or model.trust_remote_code:
        raise ValueError("bootstrap requires the registered native Megatron provider")
    if (model.override_config or model.lora_rank or model.lora.rank
            or model.lora_adapter_path or model.lora.adapter_path):
        raise ValueError("bootstrap requires full weights and unchanged exported geometry")
    if (model.use_remove_padding or model.use_fused_kernels or model.use_liger
            or model.mtp.enable):
        raise ValueError("bootstrap requires the native unfused model path")
    if engine._target_ != "verl.workers.config.McoreEngineConfig":
        raise ValueError("bootstrap requires upstream Megatron execution")
    if (engine.dtype != "bfloat16" or engine.patch_recomputation_backward
            or not engine.use_distributed_optimizer or engine.use_megatron_fsdp
            or config.optim.use_precision_aware_optimizer):
        raise ValueError("bootstrap requires native mixed precision and distributed Adam")
    if any(getattr(engine, key) != 1 for key in (
        "tensor_model_parallel_size", "pipeline_model_parallel_size",
        "context_parallel_size", "expert_model_parallel_size",
    )) or engine.expert_tensor_parallel_size not in (None, 1):
        raise ValueError("bootstrap supports data parallelism only")
    if (engine.sequence_parallel or engine.virtual_pipeline_model_parallel_size is not None
            or engine.use_remove_padding or engine.dynamic_context_parallel
            or engine.pad_bshd_to_minibatch_max):
        raise ValueError("bootstrap requires one unpacked native sequence per microbatch")
    if (config.trainer.nnodes != 1 or config.trainer.n_gpus_per_node != 8
            or data.micro_batch_size_per_gpu != 1 or data.use_dynamic_bsz
            or data.train_batch_size < 8 or data.train_batch_size % 8):
        raise ValueError("bootstrap requires DP8 and a microbatch of one sequence")
    if (data.pad_mode != "no_padding" or data.truncation != "error"
            or not 0 < data.max_length <= experiment.native_context):
        raise ValueError("bootstrap requires jagged dataset tensors without truncation")
    if (data.custom_cls.path != "pkg://archlab.data.agentic_sft"
            or data.custom_cls.name != "PreparedAgenticSFTDataset"):
        raise ValueError("bootstrap requires prepared native assistant masks")
    if config.trainer.resume_mode != "disable":
        raise ValueError("bootstrap starts a fresh full-weight CE experiment")


def validate_assets(experiment: BootstrapExperiment, config: DictConfig) -> dict:
    """Check this conversion, original-base lineage and prepared native masks."""
    root = Path(experiment.parent_checkpoint)
    exported = json.loads((root / "config.json").read_text())
    converted = bool(exported.get("conversion"))
    if converted != (experiment.variant == "simplicial"):
        raise ValueError("export conversion variant differs from bootstrap contract")
    if exported.get("trainable_mode") != "full" or (converted and exported.get("adapter")):
        raise ValueError("bootstrap requires a full-weight native export")
    generation = json.loads((root / "generation_config.json").read_text())
    if (exported.get("eos_token_id") != 151645 or generation.get("eos_token_id") != 151645
            or generation.get("pad_token_id") != 151643):
        raise ValueError("exported native generation EOS/PAD contract differs")
    publisher = Path(exported["publisher_snapshot"])
    source_path = publisher / "DOWNLOAD_VERIFIED.json"
    source = json.loads(source_path.read_text())
    if source.get("repo") != "paradigma-inc/limite-1b-base" or not source.get("revision"):
        raise ValueError("bootstrap lineage must be the pinned original Limite base")
    for item in source["files"]:
        name = item["path"]
        if Path(name).name != name or sha256_file(publisher / name) != item["sha256"]:
            raise ValueError("publisher source checksum mismatch: " + name)

    manifest_path = Path(experiment.data_manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "archlab.prepared-agentic-sft.v1":
        raise ValueError("bootstrap requires a prepared agentic SFT manifest")
    tokenization = manifest["tokenization"]
    if (tokenization.get("generation_eos_token_id") != 151645
            or tokenization.get("tokenizer_eos_token_id") != 151643
            or tokenization.get("loss_mask_alignment") != "unshifted-target-token"
            or tokenization.get("max_length") != config.data.max_length):
        raise ValueError("prepared native EOS, mask alignment or context differs")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if tokenization["files"].get(name) != sha256_file(root / name):
            raise ValueError("prepared tokenizer/template checksum differs: " + name)
    for name, expected in tokenization["files"].items():
        if Path(name).name != name or sha256_file(root / name) != expected:
            raise ValueError("prepared native asset checksum differs: " + name)
    template_hash = sha256_file(root / "chat_template.jinja")
    if (tokenization.get("chat_template_sha256") != template_hash
            or tokenization.get("loaded_chat_template_sha256") != template_hash):
        raise ValueError("prepared chat template differs from the exported model")
    for split in ("train", "validation", "index"):
        item = manifest["files"][split]
        name = item["name"]
        if Path(name).name != name or sha256_file(manifest_path.parent / name) != item["sha256"]:
            raise ValueError("prepared data checksum mismatch: " + split)
    for split, files in (("train", config.data.train_files), ("validation", config.data.val_files)):
        expected = manifest_path.parent / manifest["files"][split]["name"]
        if [str(Path(path).resolve()) for path in files] != [str(expected.resolve())]:
            raise ValueError("upstream SFT files differ from the prepared manifest: " + split)
    if manifest["files"]["train"]["rows"] < config.data.train_batch_size:
        raise ValueError("bootstrap data cannot provide one complete DP8 training batch")

    # Verify export files independently of the model runtime.
    from archlab.training.limite_artifacts import verify_export

    if experiment.parent_stage == BootstrapParentStage.package_sft:
        from archlab.training.sft_export import verify_package_sft_export

        receipt = verify_package_sft_export(root)
        training = receipt["training"]
        if (not converted or exported.get("num_hidden_layers") != 48
                or exported.get("publisher_config", {}).get("num_hidden_layers") != 48
                or exported.get("max_position_embeddings") != experiment.native_context
                or exported.get("conversion") != {"short_window": 16, "backend": "tilelang"}
                or training.get("format") != "archlab-limite-agentic-sft-v1"
                or training.get("objective") != experiment.objective
                or training.get("initialization") != "inherited_trained_weights"
                or training.get("original_publisher") != {
                    "repo": source["repo"], "revision": source["revision"],
                    "receipt_sha256": sha256_file(source_path)}
                or type(training.get("global_step")) is not int or training["global_step"] <= 0
                or training.get("token_budget", {}).get("completed_samples", 0) <= 0
                or not receipt.get("state_sha256")
                or receipt["state_sha256"] == training.get("parent_export", {}).get("state_sha256")):
            raise ValueError("package SFT parent does not establish trained original-base replacement weights")
    else:
        receipt = verify_export(root)
    if converted:
        conversion = receipt.get("conversion", {})
        if (conversion.get("format") != "archlab-limite-conversion-v1"
                or conversion.get("intervention") != "replace_attention"
                or conversion.get("mechanism") != "joint-softmax-2-simplicial"
                or conversion.get("config") != exported["conversion"]
                or conversion.get("layers") != list(range(exported["num_hidden_layers"]))):
            raise ValueError("conversion receipt does not establish the declared replacement")
        if experiment.parent_stage == BootstrapParentStage.original_base:
            if conversion.get("initialization") != "zero_short_deltas;inherited_native_attention_parameters":
                raise ValueError("conversion receipt does not establish the replacement initialization")
        elif (conversion.get("weights_state") != "trained_by_assistant_ce"
              or conversion.get("initialization")):
            raise ValueError("package SFT parent lacks explicit trained conversion metadata")
    result = {
        "export_receipt_sha256": sha256_file(root / "EXPORT.json"),
        "publisher_receipt_sha256": sha256_file(source_path),
        "publisher_repo": source["repo"], "publisher_revision": source["revision"],
        "data_manifest_sha256": sha256_file(manifest_path),
        "conversion": receipt.get("conversion"),
        "parent_stage": experiment.parent_stage.value,
    }
    if experiment.parent_stage == BootstrapParentStage.package_sft:
        result["parent_training"] = receipt["training"]
    if "supervision" in manifest:
        if not isinstance(manifest["supervision"], dict):
            raise ValueError("prepared supervision must be an explicit manifest object")
        result["supervision"] = manifest["supervision"]
    return result


def source_snapshot(upstream: Path, recipe: Path) -> dict:
    """Freeze the local Python package plus the upstream SFT execution boundary."""
    package = Path(__file__).resolve().parents[1]
    paths = set(package.rglob("*.py")) | {recipe.resolve()}
    paths.update((upstream / "verl/trainer/config").rglob("*.yaml"))
    paths.update(upstream / name for name in (
        "verl/trainer/sft_trainer.py", "verl/workers/engine_workers.py",
        "verl/workers/utils/losses.py", "verl/workers/engine/megatron/transformer_impl.py",
        "verl/models/mcore/model_forward.py", "verl/utils/dataset/dataset_utils.py",
        "verl/utils/import_utils.py",
    ))
    files = {str(path.resolve()): sha256_file(path) for path in sorted(paths)}
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": files}


def verify_source(snapshot: dict) -> None:
    for name, expected in snapshot["files"].items():
        if sha256_file(name) != expected:
            raise ValueError("bootstrap source changed after freeze: " + name)


def _write_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text() != value:
            raise ValueError("immutable bootstrap receipt already differs: " + str(path))
        return
    with path.open("x") as stream:
        stream.write(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resolve", "train"))
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--verl", type=Path, default=Path("third_party/verl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_intermixed_args(argv)
    experiment = read_bootstrap(args.recipe, args.overrides)
    config = compose_sft(experiment, args.verl)
    if args.action == "train" and int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise ValueError("bootstrap train must be launched with torchrun --nproc_per_node=8")
    launch = None
    if args.action == "train":
        launch = {"format": "archlab-bootstrap-launch-v1", "assets": validate_assets(experiment, config),
                  "source": source_snapshot(args.verl, args.recipe)}
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_once(args.output / "resolved.yaml", OmegaConf.to_yaml(config, resolve=True, sort_keys=True))
        _write_once(args.output / "experiment.json", json.dumps(asdict(experiment), indent=2, sort_keys=True) + "\n")
        if launch:
            launch["resolved_config_sha256"] = sha256_file(args.output / "resolved.yaml")
            _write_once(args.output / "bootstrap-launch.json", json.dumps(launch, indent=2, sort_keys=True) + "\n")
    if args.action == "train":
        # Register the native provider before importing the engine registry.
        # Reversing this order re-enters ModelOpt's optional Megatron plugin
        # during its first import in the pinned container.
        import_module(config.model.external_lib)
        from verl.trainer.sft_trainer import run_sft

        verify_source(launch["source"])
        run_sft(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
