"""Native DCP model-only export preserves mixed model files and rejects resume."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from archlab.megatron.miles_eval_checkpoint import check_eval_load
from archlab.storage.miles_eval_checkpoint import _common, _metadata, export_checkpoint


def checkpoint(tmp_path, *, mixed=True):
    root = tmp_path / "checkpoints"
    source = root / "iter_0000000"
    source.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("19")
    common = {"iteration": 0, "args": Namespace(no_save_optim=False, no_save_rng=False),
              "optimizer": {"private": "nvme_opt_state"}, "opt_param_scheduler": {"step": 1},
              "rng_state": {"private": torch.arange(3)}, "model": {"metadata": "keep"}}
    state = {"decoder.layers.0.input_layernorm.weight": torch.arange(16).bfloat16(),
             "decoder.layers.1.archlab_adapter.weight": torch.arange(32).reshape(4, 8).float(),
             "decoder.layers.1.scale": torch.tensor(0.25),
             "decoder.layers.1.linear._extra_state/shard_0_1": [b"TE metadata"],
             "rng_state/shard_0": torch.arange(64), "common_state/shard_0_1": [common]}
    dcp.save(state, storage_writer=dcp.FileSystemWriter(source, single_file_per_rank=mixed),
             planner=dcp.DefaultSavePlanner(flatten_state_dict=False), no_dist=True)
    (source / "metadata.json").write_text('{"sharded_backend":"torch_dist","sharded_backend_version":1}')
    optimizer = source / "nvme_opt_state" / "rank00000" / "opt0_0"
    optimizer.mkdir(parents=True)
    (optimizer / "bucket00000.bin").write_bytes(b"old training state" * 17)
    (optimizer / "manifest.json").write_text('{"buckets":[]}')
    return source, state


@pytest.mark.parametrize("mixed", [False, True])
def test_native_export_and_retirement(tmp_path, mixed):
    source, expected = checkpoint(tmp_path, mixed=mixed)
    original = _metadata(source)
    model_file = next(info.relative_path for idx, info in original.storage_data.items()
                      if idx.fqn.startswith("decoder."))
    inode = (source / model_file).stat().st_ino
    receipt = export_checkpoint(source, tmp_path / "audit", retire=True)
    assert receipt["retired"] and receipt["can_resume"] is False
    assert receipt["validation"]["all_model_keys"] == 4
    assert receipt["validation"]["native_byte_state_readback"]
    assert (source / model_file).stat().st_ino == inode
    assert not (source / "nvme_opt_state").exists()
    assert (source.parent / "latest_checkpointed_iteration.txt").read_text() == "19"
    metadata = _metadata(source)
    assert not any(key.startswith("rng_state/") for key in metadata.state_dict_metadata)
    common = _common(source, "common_state/shard_0_1")
    assert not {"optimizer", "opt_param_scheduler", "rng_state"}.intersection(common)
    assert common["model"] == {"metadata": "keep"}
    assert common["args"].no_save_optim and common["args"].no_save_rng
    target = {key: torch.zeros_like(value) for key, value in expected.items()
              if key.startswith("decoder.") and isinstance(value, torch.Tensor)}
    dcp.load(target, storage_reader=dcp.FileSystemReader(source),
             planner=dcp.DefaultLoadPlanner(flatten_state_dict=False), no_dist=True)
    assert all(torch.equal(value, expected[key]) for key, value in target.items())
    assert Path(receipt["audit"], "model-files.jsonl").is_file()
    evaluation = Path(receipt["native_eval_root"])
    assert (evaluation / "latest_checkpointed_iteration.txt").read_text().strip() == "0"
    assert (evaluation / source.name).resolve() == source
    check_eval_load(Namespace(load=str(evaluation), ckpt_step=0, no_load_optim=True,
                              no_load_rng=True, load_main_params_from_ckpt=False))


@pytest.mark.parametrize("failure", ["source_rename", "retired_deletion"])
def test_interrupted_retirement_recovers_public_path_and_deletion(tmp_path, monkeypatch, failure):
    source, _ = checkpoint(tmp_path)
    import archlab.storage.miles_eval_checkpoint as export

    if failure == "source_rename":
        original = export.os.rename

        def fail(left, right):
            original(left, right)
            if Path(left) == source:
                raise RuntimeError("interrupted after source rename")

        monkeypatch.setattr(export.os, "rename", fail)
    else:
        original = export.shutil.rmtree

        def fail(path, *args, **kwargs):
            if Path(path).name.startswith(".eval-retired-"):
                (Path(path) / "nvme_opt_state" / "rank00000" / "opt0_0" / "bucket00000.bin").unlink()
                raise RuntimeError("interrupted during old-state deletion")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(export.shutil, "rmtree", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        export_checkpoint(source, tmp_path / "audit", retire=True)
    monkeypatch.undo()
    receipt = export_checkpoint(source, tmp_path / "audit", retire=True)
    assert receipt["phase"] == "complete" and source.is_dir()
    assert not Path(receipt["retirement_directory"]).exists()
    assert (source / "EVAL_ONLY.json").is_file()


def test_pin_and_explicit_protected_iteration_block_conversion(tmp_path):
    source, _ = checkpoint(tmp_path)
    (source / "PINNED.json").write_text("{}")
    with pytest.raises(ValueError, match="pin marker"):
        export_checkpoint(source, tmp_path / "audit", retire=True)
    (source / "PINNED.json").unlink()
    protected = source.with_name("iter_0000002")
    source.rename(protected)
    with pytest.raises(ValueError, match="explicitly protected"):
        export_checkpoint(protected, tmp_path / "audit", retire=True, protected_iterations=(2, 33))


def test_tracker_change_during_validation_preserves_original(tmp_path, monkeypatch):
    source, _ = checkpoint(tmp_path)
    import archlab.storage.miles_eval_checkpoint as export

    original = export.validate_export

    def change_tracker(*args, **kwargs):
        proof = original(*args, **kwargs)
        (source.parent / "latest_checkpointed_iteration.txt").write_text("33")
        return proof

    monkeypatch.setattr(export, "validate_export", change_tracker)
    with pytest.raises(ValueError, match="tracker changed"):
        export_checkpoint(source, tmp_path / "audit", retire=True)
    assert (source / "nvme_opt_state").exists()


def test_new_protection_is_enforced_on_validated_journal_replay(tmp_path):
    source, _ = checkpoint(tmp_path)
    receipt = export_checkpoint(source, tmp_path / "audit", retire=False)
    assert receipt["phase"] == "validated"
    with pytest.raises(ValueError, match="explicitly protected"):
        export_checkpoint(source, tmp_path / "audit", retire=True, protected_iterations=(0, 2, 33))
    assert (source / "nvme_opt_state").exists()


def test_reader_appearing_after_validation_blocks_publication(tmp_path, monkeypatch):
    source, _ = checkpoint(tmp_path)
    import archlab.storage.miles_eval_checkpoint as export

    original = export.validate_export

    def reader_starts(*args, **kwargs):
        proof = original(*args, **kwargs)
        (source / "READERS.json").write_text('{"reader":"evaluation"}')
        return proof

    monkeypatch.setattr(export, "validate_export", reader_starts)
    with pytest.raises(ValueError, match="reader or pin"):
        export_checkpoint(source, tmp_path / "audit", retire=True)
    assert (source / "nvme_opt_state").exists()


def test_source_is_preserved_when_native_readback_fails(tmp_path, monkeypatch):
    source, _ = checkpoint(tmp_path)
    original = (source / ".metadata").read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("readback failed")

    monkeypatch.setattr("archlab.storage.miles_eval_checkpoint.validate_export", fail)
    with pytest.raises(ValueError, match="readback failed"):
        export_checkpoint(source, tmp_path / "audit", retire=True)
    assert (source / ".metadata").read_bytes() == original
    assert (source / "nvme_opt_state").is_dir()


def test_latest_save_is_protected(tmp_path):
    source, _ = checkpoint(tmp_path)
    (source.parent / "latest_checkpointed_iteration.txt").write_text("0")
    with pytest.raises(ValueError, match="latest"):
        export_checkpoint(source, tmp_path / "audit", retire=True)


def test_eval_guard_targets_selected_save_and_requires_no_training_state(tmp_path):
    source, _ = checkpoint(tmp_path)
    (source / "EVAL_ONLY.json").write_text('{"can_resume":false}')
    (source.parent / "latest_checkpointed_iteration.txt").write_text("0")
    args = Namespace(load=str(source.parent), ckpt_step=0, no_load_optim=False,
                     no_load_rng=True, load_main_params_from_ckpt=False)
    with pytest.raises(ValueError, match="cannot resume"):
        check_eval_load(args)
    args.no_load_optim = True
    check_eval_load(args)
    args.load_main_params_from_ckpt = True
    with pytest.raises(ValueError, match="cannot resume"):
        check_eval_load(args)
    args.ckpt_step = 19  # The untouched latest save is not affected.
    check_eval_load(args)
    args.ckpt_step = 0
    (source.parent / "latest_checkpointed_iteration.txt").write_text("19")
    check_eval_load(args)  # Native step-zero fallback selects the tracker, too.
    check_eval_load(Namespace(load=None))  # HF-only initialization has no native checkpoint.
