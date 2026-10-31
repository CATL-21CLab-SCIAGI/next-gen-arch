import json
import sys
import types
from pathlib import Path

import pytest
from omegaconf.errors import (
    ConfigAttributeError,
    ConfigKeyError,
    MissingMandatoryValue,
    ValidationError,
)

from archlab.artifacts import sha256_file
from archlab.training import bootstrap
from archlab.training.bootstrap import compose_sft, main, read_bootstrap

RECIPE = Path("recipes/sft/limite_agentic_bootstrap.yaml")
UPSTREAM = Path("third_party/verl")


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    for name in ("MODEL", "DATA", "OUTPUT"):
        monkeypatch.setenv("ARCHLAB_BOOTSTRAP_" + name, str(tmp_path / name.lower()))
    return tmp_path


def test_bootstrap_rejects_unknown_contract_and_missing_lineage(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("name: bootstrap\nparent_checkpoint: parent\ndata_manifest: data\nengien: dp\n")
    with pytest.raises(ConfigKeyError):
        read_bootstrap(recipe)
    recipe.write_text("name: bootstrap\ndata_manifest: data\n")
    with pytest.raises(MissingMandatoryValue):
        read_bootstrap(recipe)


def test_bootstrap_composes_pinned_upstream_schema(deployment):
    experiment = read_bootstrap(RECIPE)
    config = compose_sft(experiment, UPSTREAM)
    assert experiment.parent_checkpoint == str(deployment / "model")
    assert experiment.parent_stage == bootstrap.BootstrapParentStage.original_base
    assert experiment.data_manifest == str(deployment / "data/manifest.json")
    assert config.model.path == experiment.parent_checkpoint
    assert config.model.external_lib == "archlab.training.megatron"
    assert config.engine._target_ == "verl.workers.config.McoreEngineConfig"
    assert config.optim._target_ == "verl.workers.config.McoreOptimizerConfig"
    assert dict(config.optim.override_optimizer_config) == {
        "adam_beta1": .9, "adam_beta2": .95}
    assert config.optim.use_checkpoint_opt_param_scheduler
    assert config.data.custom_cls.name == "PreparedAgenticSFTDataset"
    assert config.data.pad_mode == "no_padding" and not config.engine.use_remove_padding
    assert config.data.micro_batch_size_per_gpu == 1 and config.data.train_batch_size == 8
    assert config.trainer.seed == 42 and config.trainer.n_gpus_per_node == 8
    assert config.trainer.default_local_dir == str(deployment / "output/checkpoints")
    assert not config.model.lora_rank and not config.model.lora.rank


@pytest.mark.parametrize("override", [
    "settings.model.lora_rank=4",
    "settings.model.external_lib=other.provider",
    "settings.engine.dtype=float16",
    "settings.engine.patch_recomputation_backward=true",
    "settings.engine.tensor_model_parallel_size=2",
    "settings.engine.use_remove_padding=true",
    "settings.data.pad_mode=right",
    "settings.data.truncation=left",
    "settings.data.micro_batch_size_per_gpu=2",
    "settings.data.custom_cls.name=MultiTurnSFTDataset",
    "settings.trainer.resume_mode=auto",
])
def test_bootstrap_rejects_incompatible_execution(deployment, override):
    with pytest.raises(ValueError):
        compose_sft(read_bootstrap(RECIPE, [override]), UPSTREAM)


def test_bootstrap_upstream_schema_rejects_unknown_setting(deployment):
    with pytest.raises(ConfigAttributeError):
        compose_sft(read_bootstrap(RECIPE, ["settings.engine.unknown_setting=true"]), UPSTREAM)


def test_bootstrap_resolve_and_train_delegate_without_replacing_trainer(deployment, monkeypatch):
    calls = []
    imports = []
    monkeypatch.setattr(bootstrap, "import_module", imports.append)
    module = types.ModuleType("verl.trainer.sft_trainer")
    module.run_sft = calls.append
    monkeypatch.setitem(sys.modules, "verl.trainer.sft_trainer", module)
    monkeypatch.setattr(bootstrap, "validate_assets", lambda *args: {"verified": True})
    output = deployment / "receipt"
    args = [str(RECIPE), "--output", str(output)]
    assert main(["resolve", *args]) == 0
    assert not calls
    assert not imports
    assert json.loads((output / "experiment.json").read_text())["objective"] == "assistant_ce"
    with pytest.raises(ValueError, match="torchrun"):
        main(["train", *args])
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert main(["train", *args]) == 0
    assert len(calls) == 1 and calls[0].model.path == str(deployment / "model")
    assert imports == ["archlab.training.megatron"]
    launch = json.loads((output / "bootstrap-launch.json").read_text())
    assert launch["resolved_config_sha256"] == sha256_file(output / "resolved.yaml")
    assert launch["source"]["files"][str(RECIPE.resolve())] == sha256_file(RECIPE)


def test_bootstrap_receipts_and_frozen_source_reject_changes(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original\n")
    snapshot = {"files": {str(path): sha256_file(path)}}
    bootstrap.verify_source(snapshot)
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="changed after freeze"):
        bootstrap.verify_source(snapshot)
    receipt = tmp_path / "receipt.json"
    bootstrap._write_once(receipt, "original\n")
    bootstrap._write_once(receipt, "original\n")
    with pytest.raises(ValueError, match="immutable"):
        bootstrap._write_once(receipt, "changed\n")


@pytest.fixture
def prepared_assets(deployment, monkeypatch):
    model, data, publisher = (deployment / name for name in ("model", "data", "publisher"))
    for path in (model, data, publisher):
        path.mkdir()
    (publisher / "model.safetensors").write_bytes(b"publisher")
    source = {"repo": "paradigma-inc/limite-1b-base", "revision": "pinned-base",
              "files": [{"path": "model.safetensors",
                         "sha256": sha256_file(publisher / "model.safetensors")}]}
    (publisher / "DOWNLOAD_VERIFIED.json").write_text(json.dumps(source))
    export_config = {"conversion": {"short_window": 16, "backend": "tilelang"},
                     "adapter": {}, "trainable_mode": "full", "num_hidden_layers": 2,
                     "publisher_snapshot": str(publisher), "eos_token_id": 151645}
    (model / "config.json").write_text(json.dumps(export_config))
    (model / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": 151645, "pad_token_id": 151643}))
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (model / name).write_text(name)
        files[name] = sha256_file(model / name)
    prepared = {}
    for split, name in (("train", "train.parquet"), ("validation", "validation.parquet"),
                        ("index", "index.jsonl")):
        (data / name).write_text(split)
        prepared[split] = {"name": name, "sha256": sha256_file(data / name), "rows": 8}
    manifest = {"format": "archlab.prepared-agentic-sft.v1", "files": prepared,
                "tokenization": {"files": files, "generation_eos_token_id": 151645,
                                 "tokenizer_eos_token_id": 151643, "max_length": 16384,
                                 "loss_mask_alignment": "unshifted-target-token",
                                 "chat_template_sha256": files["chat_template.jinja"],
                                 "loaded_chat_template_sha256": files["chat_template.jinja"]}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    receipt = {"conversion": {"format": "archlab-limite-conversion-v1",
                              "intervention": "replace_attention",
                              "mechanism": "joint-softmax-2-simplicial",
                              "config": export_config["conversion"], "layers": [0, 1],
                              "initialization": "zero_short_deltas;inherited_native_attention_parameters"}}
    (model / "EXPORT.json").write_text(json.dumps(receipt))
    module = types.ModuleType("archlab.training.limite_artifacts")
    module.verify_export = lambda root: json.loads((root / "EXPORT.json").read_text())
    monkeypatch.setitem(sys.modules, "archlab.training.limite_artifacts", module)
    return model, data, publisher


def test_bootstrap_admits_original_base_conversion_and_prepared_native_assets(prepared_assets):
    experiment = read_bootstrap(RECIPE)
    result = bootstrap.validate_assets(experiment, compose_sft(experiment, UPSTREAM))
    assert result["publisher_repo"] == "paradigma-inc/limite-1b-base"
    assert result["publisher_revision"] == "pinned-base"
    assert result["conversion"]["layers"] == [0, 1]


@pytest.mark.parametrize("changed", ["variant", "publisher", "data", "tokenizer", "eos", "conversion"])
def test_bootstrap_admission_rejects_mismatched_assets(prepared_assets, changed):
    model, data, publisher = prepared_assets
    experiment = read_bootstrap(RECIPE)
    config = compose_sft(experiment, UPSTREAM)
    if changed == "variant":
        experiment.variant = "normal"
    elif changed == "publisher":
        (publisher / "model.safetensors").write_bytes(b"different weights")
    elif changed == "data":
        (data / "train.parquet").write_bytes(b"different data")
    elif changed == "tokenizer":
        (model / "tokenizer.json").write_text("different tokenizer")
    elif changed == "eos":
        path = data / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["tokenization"]["generation_eos_token_id"] = 151643
        path.write_text(json.dumps(manifest))
    else:
        path = model / "EXPORT.json"
        receipt = json.loads(path.read_text())
        receipt["conversion"]["layers"] = [0]
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        bootstrap.validate_assets(experiment, config)


INSERTION = ["variant=normal", "trainable_mode=adapter", "insertion={variant: gdn, heads: 5}",
             "generation_eos_token_id=151643"]


@pytest.fixture
def inserted_assets(prepared_assets):
    model, data, publisher = prepared_assets
    adapter = {"variant": "gdn", "heads": 5, "feature_dim": 1, "compensation": 184,
               "attention_backend": "official", "native_shape": {"width": 1280}}
    config = json.loads((model / "config.json").read_text())
    config.update(conversion={}, adapter=adapter, trainable_mode="adapter")
    (model / "config.json").write_text(json.dumps(config))
    receipt = {"insertion": {"format": "archlab-limite-insertion-v1",
                             "intervention": "insert_residual_prelude",
                             "config": {k: v for k, v in adapter.items() if k != "native_shape"},
                             "layers": [0, 1], "trainable_mode": "adapter", "output_state": "zero"},
               "chat_template": {"repo": "paradigma-inc/limite-1b-violetto", "revision": "pinned"}}
    (model / "EXPORT.json").write_text(json.dumps(receipt))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["tokenization"]["generation_eos_token_id"] = 151643
    (data / "manifest.json").write_text(json.dumps(manifest))
    return model, data


def test_bootstrap_admits_fresh_insertion_with_declared_supervised_eos(inserted_assets):
    experiment = read_bootstrap(RECIPE, INSERTION)
    config = compose_sft(experiment, UPSTREAM)
    assert config.data.generation_eos_token_id == 151643
    result = bootstrap.validate_assets(experiment, config)
    assert result["insertion"]["config"]["heads"] == 5 and result["conversion"] is None
    assert result["chat_template"]["repo"] == "paradigma-inc/limite-1b-violetto"


@pytest.mark.parametrize("defect", ["geometry", "scope", "trained", "eos", "layers"])
def test_bootstrap_insertion_rejects_mismatched_parent(inserted_assets, defect):
    model, data = inserted_assets
    overrides = list(INSERTION)
    receipt = json.loads((model / "EXPORT.json").read_text())
    if defect == "geometry":
        overrides[2] = "insertion={variant: gdn, heads: 4}"
    elif defect == "scope":
        overrides[1] = "trainable_mode=full"
    elif defect == "trained":
        receipt["insertion"]["output_state"] = "trained"
    elif defect == "eos":
        overrides[3] = "generation_eos_token_id=151645"
    else:
        receipt["insertion"]["layers"] = [0]
    (model / "EXPORT.json").write_text(json.dumps(receipt))
    experiment = read_bootstrap(RECIPE, overrides)
    with pytest.raises(ValueError):
        bootstrap.validate_assets(experiment, compose_sft(experiment, UPSTREAM))


@pytest.mark.parametrize("overrides", [
    ["trainable_mode=adapter"],
    ["insertion={variant: gdn}"],
    ["variant=normal", "trainable_mode=partial", "insertion={variant: gdn}"],
    ["generation_eos_token_id=42"],
])
def test_bootstrap_rejects_inconsistent_insertion_contract(deployment, overrides):
    with pytest.raises(ValueError):
        read_bootstrap(RECIPE, overrides)


def _upstream_step_targets(rows, *, batch_size, data_parallel, epochs):
    """Assistant targets per global step, drawn exactly as verl's SFT trainer draws rows."""
    import torch
    from torch.utils.data import DataLoader, DistributedSampler

    totals = []
    for epoch in range(epochs):
        loaders = []
        for rank in range(data_parallel):
            sampler = DistributedSampler(rows, shuffle=True, num_replicas=data_parallel, rank=rank,
                                         drop_last=True)
            sampler.set_epoch(epoch)
            loaders.append(DataLoader(rows, batch_size=batch_size // data_parallel,
                                      sampler=sampler, drop_last=True,
                                      collate_fn=lambda batch: sum(sum(mask) for mask in batch)))
        totals.extend(sum(int(torch.tensor(step).sum()) for step in steps)
                      for steps in zip(*loaders, strict=True))
    return totals


def test_token_budget_follows_upstream_sampler_across_epochs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    masks = [[0] * (index % 3) + [1] * (1 + (7 * index) % 23) for index in range(37)]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"loss_mask": mask} for mask in masks]), path)
    upstream = _upstream_step_targets(masks, batch_size=8, data_parallel=4, epochs=3)
    budget = sum(upstream[:6]) + 1
    schedule = bootstrap.supervised_schedule(path, batch_size=8, data_parallel=4,
                                             budget=budget, evaluations=3)
    assert schedule["steps_per_epoch"] == 4 and schedule["epochs"] == 2
    assert schedule["steps"] == 7 and schedule["frequency"] == 2
    assert schedule["supervised_tokens"] == sum(upstream[:7]) >= budget
    assert schedule["evaluations"] == {step: sum(upstream[:step]) for step in (2, 4, 6, 7)}
    with pytest.raises(ValueError, match="epoch limit"):
        bootstrap.supervised_schedule(path, batch_size=8, data_parallel=4,
                                      budget=sum(upstream) * 10, evaluations=1, max_epochs=3)


def test_bootstrap_token_budget_sets_upstream_steps(deployment):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (deployment / "data").mkdir()
    pq.write_table(pa.Table.from_pylist([{"loss_mask": [0] + [1] * 10}] * 64),
                   deployment / "data/train.parquet")
    experiment = read_bootstrap(RECIPE, ["supervised_tokens=1000", "evaluations=4"])
    config = compose_sft(experiment, UPSTREAM)
    assert config.trainer.total_training_steps == 13 and config.trainer.total_epochs == 2
    assert config.trainer.test_freq == config.trainer.save_freq == 3
    with pytest.raises(ValueError, match="evaluations"):
        read_bootstrap(RECIPE, ["supervised_tokens=1000"])


def test_bootstrap_rejects_unknown_parent_stage(deployment):
    with pytest.raises(ValidationError):
        read_bootstrap(RECIPE, ["parent_stage=rl_resume"])


@pytest.fixture
def trained_assets(prepared_assets, monkeypatch):
    model, data, publisher = prepared_assets
    config = json.loads((model / "config.json").read_text())
    config.update(num_hidden_layers=48, publisher_config={"num_hidden_layers": 48},
                  max_position_embeddings=131072)
    (model / "config.json").write_text(json.dumps(config))
    receipt = json.loads((model / "EXPORT.json").read_text())
    receipt.update(state_sha256="trained-state", training={
        "format": "archlab-limite-agentic-sft-v1", "objective": "assistant_ce",
        "initialization": "inherited_trained_weights", "global_step": 32,
        "token_budget": {"completed_samples": 256}, "parent_export": {"state_sha256": "identity-state"},
        "original_publisher": {"repo": "paradigma-inc/limite-1b-base", "revision": "pinned-base",
                               "receipt_sha256": sha256_file(publisher / "DOWNLOAD_VERIFIED.json")}})
    receipt["conversion"].pop("initialization")
    receipt["conversion"].update(layers=list(range(48)), weights_state="trained_by_assistant_ce")
    (model / "EXPORT.json").write_text(json.dumps(receipt))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["supervision"] = {"selection": "assistant_protocol_tokens_only",
                              "token_ids": [151645, 151657, 151658],
                              "input_ids_positions_attention_unchanged": True,
                              "observations_have_loss": False}
    (data / "manifest.json").write_text(json.dumps(manifest))
    calls = []
    module = types.ModuleType("archlab.training.sft_export")

    def verify(root):
        calls.append(root)
        return json.loads((root / "EXPORT.json").read_text())

    module.verify_package_sft_export = verify
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return model, data, publisher, calls


def test_bootstrap_admits_explicit_trained_parent_and_records_subset_supervision(trained_assets):
    model, data, _, calls = trained_assets
    experiment = read_bootstrap(RECIPE, ["parent_stage=package_sft"])
    config = compose_sft(experiment, UPSTREAM)
    result = bootstrap.validate_assets(experiment, config)
    assert calls == [model]
    assert result["parent_stage"] == "package_sft"
    assert result["parent_training"]["global_step"] == 32
    assert result["supervision"] == json.loads((data / "manifest.json").read_text())["supervision"]
    assert experiment.objective == "assistant_ce" and config.trainer.resume_mode == "disable"


def test_bootstrap_original_parent_does_not_admit_trained_conversion(trained_assets):
    experiment = read_bootstrap(RECIPE)
    with pytest.raises(ValueError, match="initialization"):
        bootstrap.validate_assets(experiment, compose_sft(experiment, UPSTREAM))


def test_bootstrap_cannot_bypass_strict_trained_export_verifier(trained_assets, monkeypatch):
    def reject(root):
        raise ValueError("trained export checksum differs")

    monkeypatch.setattr(sys.modules["archlab.training.sft_export"], "verify_package_sft_export", reject)
    experiment = read_bootstrap(RECIPE, ["parent_stage=package_sft"])
    with pytest.raises(ValueError, match="checksum differs"):
        bootstrap.validate_assets(experiment, compose_sft(experiment, UPSTREAM))


@pytest.mark.parametrize("defect", [
    "layers", "geometry", "backend", "publisher", "objective", "initialization",
    "untrained_state", "zero_steps", "zero_samples", "weights_state", "zero_claim", "eos", "supervision",
])
def test_bootstrap_trained_parent_rejects_wrong_lineage_or_geometry(trained_assets, defect):
    model, data, _, _ = trained_assets
    config = json.loads((model / "config.json").read_text())
    receipt = json.loads((model / "EXPORT.json").read_text())
    if defect == "layers":
        receipt["conversion"]["layers"].pop()
    elif defect == "geometry":
        config["num_hidden_layers"] = 47
    elif defect == "backend":
        config["conversion"]["backend"] = "other"
    elif defect == "publisher":
        receipt["training"]["original_publisher"]["repo"] = "paradigma-inc/limite-1b-violetto"
    elif defect in {"objective", "initialization"}:
        receipt["training"][defect] = "other"
    elif defect == "untrained_state":
        receipt["state_sha256"] = "identity-state"
    elif defect == "zero_steps":
        receipt["training"]["global_step"] = 0
    elif defect == "zero_samples":
        receipt["training"]["token_budget"]["completed_samples"] = 0
    elif defect == "weights_state":
        receipt["conversion"]["weights_state"] = "zero"
    elif defect == "zero_claim":
        receipt["conversion"]["initialization"] = "zero_short_deltas;inherited_native_attention_parameters"
    elif defect == "eos":
        config["eos_token_id"] = 151643
    else:
        manifest = json.loads((data / "manifest.json").read_text())
        manifest["supervision"] = "unrecorded mask"
        (data / "manifest.json").write_text(json.dumps(manifest))
    (model / "config.json").write_text(json.dumps(config))
    (model / "EXPORT.json").write_text(json.dumps(receipt))
    experiment = read_bootstrap(RECIPE, ["parent_stage=package_sft"])
    with pytest.raises(ValueError):
        bootstrap.validate_assets(experiment, compose_sft(experiment, UPSTREAM))
