"""Finalize real safetensors payloads; never construct or initialize a model."""

import hashlib
import json
import shutil

import pytest
import torch
import yaml
from safetensors.torch import save_file

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.training.sft_export import (
    finalize_package_hf_export,
    native_state_digest,
    verify_package_sft_export,
)


def write_json(path, value):
    atomic_write_json(path, value)


@pytest.fixture
def run(tmp_path, monkeypatch):
    from archlab.training.limite import LimiteForCausalLM

    def forbidden(*args, **kwargs):
        raise AssertionError("Finalization must not construct a model")

    monkeypatch.setattr(LimiteForCausalLM, "__init__", forbidden)
    publisher, parent = tmp_path / "publisher", tmp_path / "parent"
    publisher.mkdir()
    parent.mkdir()
    config = {"model_type": "archlab_limite", "publisher_snapshot": str(publisher),
              "publisher_config": {"vocab_size": 4, "hidden_size": 3}, "trainable_mode": "full",
              "conversion": {"short_window": 16, "backend": "tilelang"},
              "vocab_size": 4, "hidden_size": 3, "tie_word_embeddings": True,
              "eos_token_id": 151645, "bos_token_id": 151643, "pad_token_id": 151643,
              "parameter_dtypes": {"model.embed_tokens.weight": "bfloat16", "model.scale": "float32"}}
    write_json(parent / "config.json", config)
    assets = {"tokenizer.json": '{"version":"1.0"}\n',
              "tokenizer_config.json": '{"eos_token":"<|endoftext|>"}\n',
              "special_tokens_map.json": '{}\n', "chat_template.jinja": 'native_template\n',
              "generation_config.json": '{"eos_token_id":151645,"pad_token_id":151643}\n'}
    for name, content in assets.items():
        (parent / name).write_text(content)
        shutil.copyfile(parent / name, publisher / name)
    publisher_receipt = {"repo": "paradigma-inc/limite-1b-base", "revision": "cc612baf",
                         "files": [{"path": name, "sha256": sha256_file(parent / name)} for name in assets]}
    write_json(publisher / "DOWNLOAD_VERIFIED.json", publisher_receipt)
    weights = {"policy.lm_head.weight": torch.arange(12).reshape(4, 3).to(torch.bfloat16),
               "policy.model.scale": torch.tensor([1., 2., 3.], dtype=torch.float32)}
    alias = {"policy.model.embed_tokens.weight": "policy.lm_head.weight"}
    save_file(weights, parent / "model.safetensors", metadata=alias)
    parent_state = native_state_digest(parent / "model.safetensors", parent_weights=parent / "model.safetensors")
    conversion = {"format": "archlab-limite-conversion-v1", "initialization": "zero_short_deltas;inherited_native_attention_parameters",
                  "cached_decode_qualified": False, "config": config["conversion"]}
    write_json(parent / "EXPORT.json", {"format": "archlab-limite-hf-v1", "state_sha256": parent_state,
               "optimizer_migrated": False, "conversion": conversion,
               "files": {path.name: sha256_file(path) for path in parent.iterdir()}})
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.parquet", "validation.parquet"):
        (data / name).write_bytes(b"Pinned public prepared data")
    index = [{"split": "train", "input_tokens": 10 + i, "assistant_tokens": 3} for i in range(4)]
    (data / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
    template = sha256_file(parent / "chat_template.jinja")
    manifest = {"format": "archlab.prepared-agentic-sft.v1", "source": {"repo_id": "public/source", "revision": "pin"},
                "counts": {"train": 4, "validation": 1}, "tokenization": {
                    "generation_eos_token_id": 151645, "tokenizer_eos_token_id": 151643,
                    "loss_mask_alignment": "unshifted-target-token", "max_length": 16,
                    "chat_template_sha256": template, "loaded_chat_template_sha256": template,
                    "files": {name: sha256_file(parent / name) for name in assets}},
                "files": {split: {"name": name, "sha256": sha256_file(data / name), "rows": rows}
                          for split, name, rows in (("train", "train.parquet", 4),
                                                     ("validation", "validation.parquet", 1),
                                                     ("index", "index.jsonl", 4))}}
    write_json(data / "manifest.json", manifest)
    launch_root = tmp_path / "run"
    launch_root.mkdir()
    config = {"model": {"path": str(parent)}, "data": {"max_length": 16, "train_batch_size": 2,
              "train_files": [str(data / "train.parquet")], "val_files": [str(data / "validation.parquet")]},
              "trainer": {"nnodes": 1, "n_gpus_per_node": 2, "total_epochs": 2,
                          "default_local_dir": str(launch_root / "checkpoints")}}
    (launch_root / "resolved.yaml").write_text(yaml.safe_dump(config))
    write_json(launch_root / "experiment.json", {"objective": "assistant_ce", "parent_checkpoint": str(parent),
                                                "data_manifest": str(data / "manifest.json")})
    launch = {"format": "archlab-bootstrap-launch-v1", "source": {"sha256": "frozen-source"},
              "resolved_config_sha256": sha256_file(launch_root / "resolved.yaml"), "assets": {
                  "export_receipt_sha256": sha256_file(parent / "EXPORT.json"),
                  "publisher_receipt_sha256": sha256_file(publisher / "DOWNLOAD_VERIFIED.json"),
                  "data_manifest_sha256": sha256_file(data / "manifest.json")}}
    write_json(launch_root / "bootstrap-launch.json", launch)
    checkpoint = launch_root / "checkpoints/global_step_2"
    hf = checkpoint / "model/huggingface"
    hf.mkdir(parents=True)
    for path in parent.iterdir():
        if path.name != "EXPORT.json":
            shutil.copyfile(path, hf / path.name)
    weights = {name: value + 0.5 for name, value in weights.items()}
    save_file(weights, hf / "model.safetensors")  # Actual package writer omits alias metadata.
    package_config = json.loads((hf / "config.json").read_text())
    package_config.update(bos_token_id=None, eos_token_id=151643)
    write_json(hf / "config.json", package_config)
    (hf / "tokenizer_config.json").write_text('{"normalized":"by package save_pretrained"}\n')
    for subdir in ("optimizer/dist_ckpt", "extra/dist_ckpt"):
        path = checkpoint / subdir
        path.mkdir(parents=True)
        (path / "common.pt").write_bytes(b"Untouched package resume payload")
    for rank in range(2):
        (checkpoint / f"data_{rank}.pt").write_bytes(b"Dataloader state")
    contents = {"model": {"path": "model/huggingface", "format": "huggingface"},
                "hf_model": {"path": "model/huggingface", "format": "huggingface"},
                "optimizer": {"path": "optimizer/dist_ckpt"}, "lr_scheduler": {"path": "optimizer/dist_ckpt"},
                "rng_state": {"path": "extra/dist_ckpt"}}
    write_json(checkpoint / "ckpt_contents.json", {"schema_version": 2, "framework": "megatron", "world_size": 2,
        "global_step": 2, "save_contents": ["model", "optimizer", "extra", "hf_model"], "contents": contents})
    return {"hf_root": hf, "parent_export": parent, "package_checkpoint": checkpoint,
            "training_receipt": launch_root / "bootstrap-launch.json"}


def test_finalizes_actual_weights_without_model_or_optimizer_rewrite(run):
    hf, parent, checkpoint = run["hf_root"], run["parent_export"], run["package_checkpoint"]
    weights_before = sha256_file(hf / "model.safetensors")
    resume_before = {str(path): sha256_file(path) for path in checkpoint.rglob("*.pt")}
    receipt = finalize_package_hf_export(**run)
    assert receipt == verify_package_sft_export(hf) == finalize_package_hf_export(**run)
    assert sha256_file(hf / "model.safetensors") == weights_before
    assert resume_before == {str(path): sha256_file(path) for path in checkpoint.rglob("*.pt")}
    assert (hf / "tokenizer_config.json").read_bytes() == (parent / "tokenizer_config.json").read_bytes()
    assert json.loads((hf / "config.json").read_text())["eos_token_id"] == 151645
    assert receipt["optimizer_migrated"] is False
    assert receipt["source_checkpoint"]["path"] == str(checkpoint)
    assert receipt["training"]["initialization"] == "inherited_trained_weights"
    assert "initialization" not in receipt["conversion"]
    assert receipt["conversion"]["weights_state"] == "trained_by_assistant_ce"
    assert receipt["training"]["data"]["train_input_tokens"] == 46
    assert receipt["training"]["data"]["train_assistant_tokens"] == 12
    assert receipt["training"]["token_budget"]["maximum_input_tokens"] == 64
    assert receipt["training"]["token_budget"]["actual_processed_input_tokens"] is None


@pytest.mark.parametrize("mutation,match", [
    ("no_complete_marker", "ckpt_contents"),
    ("no_rng", "resume payload"),
    ("no_dataloader", "dataloader"),
    ("launch_changed", "parent and assistant-CE"),
    ("parent_asset_changed", "checksum mismatch"),
    ("unknown_tensor", "inventory"),
    ("wrong_dtype", "inventory"),
    ("wrong_shape", "inventory"),
    ("changed_geometry", "geometry"),
    ("unknown_token_metadata", "token metadata"),
    ("unchanged_weights", "unchanged parent"),
    ("escaped_hf_path", "escapes"),
])
def test_rejects_incomplete_or_incompatible_exports_before_receipt(run, mutation, match):
    hf, parent, checkpoint = run["hf_root"], run["parent_export"], run["package_checkpoint"]
    if mutation == "no_complete_marker":
        (checkpoint / "ckpt_contents.json").unlink()
    elif mutation == "no_rng":
        shutil.rmtree(checkpoint / "extra/dist_ckpt")
    elif mutation == "no_dataloader":
        (checkpoint / "data_1.pt").unlink()
    elif mutation == "launch_changed":
        (run["training_receipt"].parent / "resolved.yaml").write_text("changed")
    elif mutation == "parent_asset_changed":
        (parent / "chat_template.jinja").write_text("changed")
    elif mutation in {"unknown_tensor", "wrong_dtype", "wrong_shape"}:
        from safetensors.torch import load_file

        state = load_file(hf / "model.safetensors")
        if mutation == "unknown_tensor":
            state["policy.unmapped"] = torch.ones(1)
        elif mutation == "wrong_dtype":
            state["policy.model.scale"] = state["policy.model.scale"].bfloat16()
        else:
            state["policy.model.scale"] = torch.ones(2)
        save_file(state, hf / "model.safetensors")
    elif mutation in {"changed_geometry", "unknown_token_metadata"}:
        path = hf / "config.json"
        config = json.loads(path.read_text())
        if mutation == "changed_geometry":
            config["conversion"]["short_window"] = 32
        else:
            config["eos_token_id"] = 999
        write_json(path, config)
    elif mutation == "unchanged_weights":
        shutil.copyfile(parent / "model.safetensors", hf / "model.safetensors")
    else:
        path = checkpoint / "ckpt_contents.json"
        package = json.loads(path.read_text())
        package["contents"]["hf_model"]["path"] = "../../outside"
        write_json(path, package)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        finalize_package_hf_export(**run)
    assert not (hf / "EXPORT.json").exists()


def test_verifier_rejects_changed_weights_or_lineage(run):
    finalize_package_hf_export(**run)
    path = run["hf_root"] / "EXPORT.json"
    receipt = json.loads(path.read_text())
    receipt["training"]["initialization"] = "zero_short_deltas"
    write_json(path, receipt)
    with pytest.raises(ValueError, match="lineage changed"):
        verify_package_sft_export(run["hf_root"])
    receipt["training"]["initialization"] = "inherited_trained_weights"
    write_json(path, receipt)
    with (run["hf_root"] / "model.safetensors").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_package_sft_export(run["hf_root"])


def test_streaming_digest_matches_the_native_policy_digest(run):
    from safetensors.torch import load_file

    state = load_file(run["hf_root"] / "model.safetensors")
    native = {name.removeprefix("policy."): tensor for name, tensor in state.items()}
    native["model.embed_tokens.weight"] = state["policy.lm_head.weight"]
    expected = hashlib.sha256()
    for name, tensor in sorted(native.items()):
        expected.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        expected.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    assert native_state_digest(run["hf_root"] / "model.safetensors",
                               parent_weights=run["parent_export"] / "model.safetensors") == expected.hexdigest()
