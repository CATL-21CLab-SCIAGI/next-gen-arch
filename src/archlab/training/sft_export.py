"""Finalize package-written native SFT weights without constructing a model.

verl/Megatron-Bridge owns checkpoint serialization. This boundary restores the
pinned native assets and publishes the checksum and trained-state lineage needed
by the existing Limite loader. Optimizer and resume payloads are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.training.limite_artifacts import verify_export

_ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "chat_template.jinja", "generation_config.json")
_GEOMETRY = ("model_type", "publisher_snapshot", "publisher_config", "adapter", "conversion",
             "parameter_dtypes", "normal_kernel", "normal_backward", "trainable_mode",
             "backbone_attention", "vocab_size", "hidden_size", "num_hidden_layers",
             "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
             "max_position_embeddings", "tie_word_embeddings")


def _json(path):
    return json.loads(Path(path).read_text())


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Package content path escapes its checkpoint")
    return path


def _inventory(path):
    with safe_open(path, framework="pt", device="cpu") as source:
        return {name: (source.get_slice(name).get_shape(), source.get_slice(name).get_dtype())
                for name in source.keys()}


def native_state_digest(weights, *, parent_weights):
    """Match Limite's policy state digest, including its validated tied alias."""
    with safe_open(parent_weights, framework="pt", device="cpu") as parent:
        keys = set(parent.keys())
        aliases = {name: value for name, value in (parent.metadata() or {}).items()
                   if value in keys}
    if aliases not in ({"policy.model.embed_tokens.weight": "policy.lm_head.weight"},
                       {"policy.model.base.embed_tokens.weight": "policy.lm_head.weight"}):
        raise ValueError("Unknown native tied-weight alias inventory")
    with safe_open(weights, framework="pt", device="cpu") as source:
        lookup = {name.removeprefix("policy."): name for name in source.keys()}
        lookup.update({name.removeprefix("policy."): value for name, value in aliases.items()})
        digest = hashlib.sha256()
        for name, key in sorted(lookup.items()):
            tensor = source.get_tensor(key).contiguous()
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def _lineage(parent, checkpoint, launch_path, hf_root):
    parent_receipt = verify_export(parent)
    launch = _json(launch_path)
    config_path, experiment_path = launch_path.parent / "resolved.yaml", launch_path.parent / "experiment.json"
    config, experiment = yaml.safe_load(config_path.read_text()), _json(experiment_path)
    if (launch.get("format") != "archlab-bootstrap-launch-v1"
            or sha256_file(config_path) != launch["resolved_config_sha256"]
            or experiment.get("objective") != "assistant_ce"
            or Path(experiment["parent_checkpoint"]).resolve() != parent
            or Path(config["model"]["path"]).resolve() != parent):
        raise ValueError("SFT launch does not establish this parent and assistant-CE objective")
    if launch["assets"]["export_receipt_sha256"] != sha256_file(parent / "EXPORT.json"):
        raise ValueError("SFT parent receipt differs from the launched parent")
    manifest_path = Path(experiment["data_manifest"]).resolve()
    manifest = _json(manifest_path)
    if (manifest.get("format") != "archlab.prepared-agentic-sft.v1"
            or launch["assets"]["data_manifest_sha256"] != sha256_file(manifest_path)):
        raise ValueError("SFT prepared-data lineage differs from the launch")
    tokenization = manifest["tokenization"]
    if (tokenization["generation_eos_token_id"] != 151645
            or tokenization["tokenizer_eos_token_id"] != 151643
            or tokenization["loss_mask_alignment"] != "unshifted-target-token"
            or tokenization["max_length"] != config["data"]["max_length"]):
        raise ValueError("SFT native tokenization contract differs")
    parent_config = _json(parent / "config.json")
    if (parent_config.get("model_type") != "archlab_limite"
            or parent_config.get("trainable_mode") != "full"
            or parent_config.get("eos_token_id") != 151645):
        raise ValueError("SFT finalization requires a full-weight native parent")
    publisher = Path(parent_config["publisher_snapshot"])
    publisher_path = publisher / "DOWNLOAD_VERIFIED.json"
    publisher_receipt = _json(publisher_path)
    if (publisher_receipt.get("repo") != "paradigma-inc/limite-1b-base"
            or not publisher_receipt.get("revision")
            or launch["assets"]["publisher_receipt_sha256"] != sha256_file(publisher_path)):
        raise ValueError("SFT original publisher lineage differs")
    publisher_files = {item["path"]: item["sha256"] for item in publisher_receipt["files"]}
    for name in _ASSETS:
        expected = tokenization["files"].get(name)
        if expected != publisher_files.get(name) or expected != sha256_file(parent / name):
            raise ValueError("SFT pinned native asset differs: " + name)
    template_hash = sha256_file(parent / "chat_template.jinja")
    if (tokenization.get("chat_template_sha256") != template_hash
            or tokenization.get("loaded_chat_template_sha256") != template_hash):
        raise ValueError("SFT native template differs")
    for split in ("train", "validation", "index"):
        item = manifest["files"][split]
        if sha256_file(_inside(manifest_path.parent, item["name"])) != item["sha256"]:
            raise ValueError("SFT data checksum differs: " + split)
    for split, key in (("train", "train_files"), ("validation", "val_files")):
        expected = _inside(manifest_path.parent, manifest["files"][split]["name"])
        if [Path(path).resolve() for path in config["data"][key]] != [expected]:
            raise ValueError("SFT resolved dataset paths differ")
    package = _json(checkpoint / "ckpt_contents.json")
    contents, step = package["contents"], package["global_step"]
    world = config["trainer"]["nnodes"] * config["trainer"]["n_gpus_per_node"]
    if (package.get("schema_version") != 2 or package.get("framework") != "megatron"
            or checkpoint.parent != Path(config["trainer"]["default_local_dir"]).resolve()
            or package.get("world_size") != world or type(step) is not int or step <= 0
            or checkpoint.name != f"global_step_{step}"
            or not {"model", "optimizer", "extra", "hf_model"} <= set(package["save_contents"])
            or _inside(checkpoint, contents["hf_model"]["path"]) != hf_root
            or contents["hf_model"]["format"] != "huggingface"):
        raise ValueError("Incomplete or mismatched package SFT checkpoint")
    for key in ("optimizer", "lr_scheduler", "rng_state"):
        directory = _inside(checkpoint, contents[key]["path"])
        if not directory.is_dir() or not any(path.is_file() and path.stat().st_size
                                             for path in directory.rglob("*")):
            raise ValueError("Missing package resume payload: " + key)
    for rank in range(world):
        path = checkpoint / f"data_{rank}.pt"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("Missing SFT dataloader resume state")
    batch = config["data"]["train_batch_size"]
    total_steps = config["trainer"].get("total_training_steps")
    if total_steps is None:
        total_steps = manifest["files"]["train"]["rows"] // batch * config["trainer"]["total_epochs"]
    if step > total_steps:
        raise ValueError("Package step exceeds the launched SFT token budget")
    index = [json.loads(line) for line in _inside(manifest_path.parent, manifest["files"]["index"]["name"]).read_text().splitlines()]
    train = [row for row in index if row["split"] == "train"]
    lineage = {
        "format": "archlab-limite-agentic-sft-v1", "objective": "assistant_ce",
        "initialization": "inherited_trained_weights", "global_step": step,
        "parent_export": {"path": str(parent), "receipt_sha256": sha256_file(parent / "EXPORT.json"),
                          "state_sha256": parent_receipt["state_sha256"]},
        "launch": {"path": str(launch_path), "sha256": sha256_file(launch_path),
                   "resolved_config_sha256": sha256_file(config_path),
                   "experiment_sha256": sha256_file(experiment_path),
                   "source_sha256": launch["source"]["sha256"]},
        "original_publisher": {"repo": publisher_receipt["repo"], "revision": publisher_receipt["revision"],
                               "receipt_sha256": sha256_file(publisher_path)},
        "data": {"manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                 "source": manifest["source"], "tokenization": tokenization, "counts": manifest["counts"],
                 "train_input_tokens": sum(row["input_tokens"] for row in train),
                 "train_assistant_tokens": sum(row["assistant_tokens"] for row in train)},
        "token_budget": {"max_length": tokenization["max_length"], "train_batch_size": batch,
                         "completed_samples": step * batch,
                         "maximum_input_tokens": step * batch * tokenization["max_length"],
                         "actual_processed_input_tokens": None, "actual_processed_assistant_tokens": None},
    }
    source_checkpoint = {"path": str(checkpoint), "receipt_sha256": sha256_file(checkpoint / "ckpt_contents.json")}
    return parent_receipt, lineage, source_checkpoint


def _validate_weights(root, parent):
    if list(root.glob("model*.safetensors")) != [root / "model.safetensors"] or (root / "model.safetensors.index.json").exists():
        raise ValueError("Native SFT export requires the parent's single safetensors layout")
    if _inventory(root / "model.safetensors") != _inventory(parent / "model.safetensors"):
        raise ValueError("Package SFT tensor inventory, shapes or mixed dtypes differ")
    candidate, original = _json(root / "config.json"), _json(parent / "config.json")
    if any(candidate.get(key) != original.get(key) for key in _GEOMETRY):
        raise ValueError("Package SFT config changed native geometry")
    # HFModelConfig normalizes these fields from tokenizer metadata. They do
    # not change the CE model geometry; restore publisher generation values.
    if (candidate.get("eos_token_id") not in {151643, 151645}
            or candidate.get("bos_token_id") not in {None, 151643}
            or candidate.get("pad_token_id") not in {None, 151643}):
        raise ValueError("Unknown package native token metadata normalization")
    return native_state_digest(root / "model.safetensors", parent_weights=parent / "model.safetensors")


def _conversion(parent_receipt):
    conversion = dict(parent_receipt.get("conversion") or {})
    if conversion:
        conversion.pop("initialization", None)
        conversion["cached_decode_qualified"] = False
        conversion["weights_state"] = "trained_by_assistant_ce"
    return conversion


def verify_package_sft_export(hf_root):
    """Return the verified trained receipt; retain separate SFT resume lineage."""
    root = Path(hf_root).resolve()
    receipt = verify_export(root)
    training = receipt.get("training", {})
    if training.get("format") != "archlab-limite-agentic-sft-v1" or receipt.get("optimizer_migrated") is not False:
        raise ValueError("Export lacks explicit package-SFT trained-state lineage")
    parent = Path(training["parent_export"]["path"]).resolve()
    parent_receipt, expected, checkpoint = _lineage(
        parent, Path(receipt["source_checkpoint"]["path"]).resolve(),
        Path(training["launch"]["path"]).resolve(), root,
    )
    if training != expected or receipt["source_checkpoint"] != checkpoint or receipt.get("conversion", {}) != _conversion(parent_receipt):
        raise ValueError("SFT export lineage changed after finalization")
    if receipt["state_sha256"] != _validate_weights(root, parent) or receipt["state_sha256"] == parent_receipt["state_sha256"]:
        raise ValueError("SFT export does not establish changed trained weights")
    for name in _ASSETS + ("config.json",):
        if sha256_file(root / name) != sha256_file(parent / name):
            raise ValueError("Finalized native asset differs from parent: " + name)
    return receipt


def finalize_package_hf_export(hf_root, *, parent_export, package_checkpoint, training_receipt):
    """Publish EXPORT.json last around existing, complete package-written weights.

    training_receipt is the bootstrap-launch.json emitted beside resolved.yaml
    and experiment.json. Call after upstream SFT returns or after an immutable
    completed checkpoint is available. No scheduler or model instance is needed.
    """
    root, parent = Path(hf_root).resolve(), Path(parent_export).resolve()
    checkpoint, launch = Path(package_checkpoint).resolve(), Path(training_receipt).resolve()
    if root == parent:
        raise ValueError("Cannot finalize over the immutable SFT parent")
    parent_receipt, training, source = _lineage(parent, checkpoint, launch, root)
    if (root / "EXPORT.json").exists():
        receipt = verify_package_sft_export(root)
        if receipt["training"] != training or receipt["source_checkpoint"] != source:
            raise ValueError("Existing finalized export belongs to another SFT run")
        return receipt
    state = _validate_weights(root, parent)
    if state == parent_receipt["state_sha256"]:
        raise ValueError("Package SFT checkpoint has unchanged parent weights")
    for name in _ASSETS + ("config.json",):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=f".{name}.", delete=False) as stream:
                temporary = Path(stream.name)
                with (parent / name).open("rb") as original:
                    shutil.copyfileobj(original, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / name)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    receipt = {"format": "archlab-limite-hf-v1", "source_checkpoint": source,
               "optimizer_migrated": False, "state_sha256": state, "training": training,
               "files": {path.name: sha256_file(path) for path in root.iterdir() if path.is_file()}}
    conversion = _conversion(parent_receipt)
    if conversion:
        receipt["conversion"] = conversion
    atomic_write_json(root / "EXPORT.json", receipt)
    return verify_package_sft_export(root)
