import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from archlab.megatron.indexed_data import validated_data_prefixes
from archlab.megatron.lifecycle import invoke_pretrain
from archlab.megatron.simplicial_pilot import StridedTokenBatches
from archlab.megatron.token_batches import BinaryTokenBatches, DPRankTokenBatches


@pytest.mark.parametrize("value", [None, [], 1, "bad", {"train_parts": []},
                                       {"train_parts": [""]}, {"train_parts": [1]}])
def test_all_trainers_reject_malformed_manifests_consistently(tmp_path, value):
    from archlab.megatron.qwen38_27b_train import _validated_data_prefixes as dense
    from archlab.megatron.qwen38_flash_next_full_train import _validated_data_prefixes as full
    from archlab.megatron.qwen38_train import _validated_data_prefixes as small

    assert dense is full is small is validated_data_prefixes
    (tmp_path / "DATA_READY.json").write_text(json.dumps(value))
    for validator in (dense, full, small):
        with pytest.raises(RuntimeError, match="manifest"):
            validator(tmp_path)


@pytest.mark.parametrize("cfg_container", [False, True])
def test_pretrain_always_receives_explicit_callback(monkeypatch, cfg_container):
    sentinel = object()
    observed = []
    if cfg_container:
        monkeypatch.setitem(sys.modules, "megatron.training.arguments", SimpleNamespace(
            parse_and_validate_args=lambda **kw: kw))
        monkeypatch.setitem(sys.modules, "megatron.training.argument_utils", SimpleNamespace(
            pretrain_cfg_container_from_args=lambda parsed: {"parsed": parsed}))

        def pretrain(cfg_container, data, model, kind, forward):
            observed.append((data, model, kind, forward))
    else:
        def pretrain(data, model, kind, forward, *, args_defaults):
            assert args_defaults == {"tokenizer_type": "NullTokenizer"}
            observed.append((data, model, kind, forward))
    invoke_pretrain(SimpleNamespace(pretrain=pretrain), "data", "model", "type", forward_step=sentinel)
    assert observed == [("data", "model", "type", sentinel)]


@pytest.mark.parametrize("start,repeat", [(0, None), (5, None), (3, 2)])
def test_consolidated_batches_preserve_multifile_wrap_and_resume(tmp_path, start, repeat):
    prefixes = [tmp_path / "a", tmp_path / "b"]
    arrays = [np.arange(11, dtype=np.int32), np.arange(19, dtype=np.int32) + 100]
    for prefix, array in zip(prefixes, arrays, strict=True):
        array.tofile(str(prefix) + ".bin")
    assert DPRankTokenBatches is BinaryTokenBatches
    reader = BinaryTokenBatches(prefixes, batch_size=3, sequence_len=4, start_batch=start,
                                repeat_window_batches=repeat, device=torch.device("cpu"))
    for step in range(8):
        source = start + (step % repeat if repeat else step)
        array = arrays[source % 2]
        offset = source // 2 * 12
        expected = array[(np.arange(13) + offset) % len(array)]
        actual = next(reader)
        assert actual["tokens"].flatten().tolist() == expected[:-1].tolist()
        assert actual["labels"].flatten().tolist() == expected[1:].tolist()


def test_strided_pilot_extension_point_retains_global_stream(tmp_path):
    prefix = tmp_path / "tokens"
    np.arange(101, dtype=np.int32).tofile(str(prefix) + ".bin")
    common = dict(prefixes=[prefix], batch_size=1, sequence_len=4, start_batch=2, device=torch.device("cpu"))
    readers = [StridedTokenBatches(**common, rank=rank, world_size=3) for rank in range(3)]
    for step in range(3):
        for rank, reader in enumerate(readers):
            start = ((step + 2) * 3 + rank) * 4
            assert next(reader)["tokens"].flatten().tolist() == list(range(start, start + 4))


def test_neutral_factory_preserves_frozen_runtime_binding():
    import archlab.model_factory as neutral
    import archlab.speedrun.models as frozen
    from archlab.campaigns import campaign_model_config_kwargs, get_campaign_variant

    kwargs = campaign_model_config_kwargs("10m", get_campaign_variant("10m", "baseline"))
    old = frozen.build_model_config(**kwargs)
    new = neutral.build_model_config(**kwargs, runtime=frozen.training_architecture_runtime())
    assert neutral.model_config_to_dict(old) == neutral.model_config_to_dict(new)
    assert old.runtime == new.runtime
    assert neutral.build_model_config(**kwargs).runtime.compute_dtype == torch.float32
