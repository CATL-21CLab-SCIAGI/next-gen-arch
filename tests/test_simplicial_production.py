"""Production variant contracts must not perturb the frozen baseline recipe."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from archlab.architectures.qwen38_flash_next_full import Qwen38FlashNextFullConfig
from archlab.megatron.qwen38_flash_next_full_train import _megatron_argv, _parser
from archlab.megatron.simplicial_production import (
    EXTRA_PARAMETERS,
    SIMPLICIAL_ATTENTION,
    attention_variant_contract,
    validate_attention_resume,
)


def arguments(tmp_path, *extra):
    return _parser().parse_args([
        "--data-root", str(tmp_path / "data"), "--tokenizer", str(tmp_path / "tokenizer"),
        "--run-dir", str(tmp_path / "run"), "--model-variant", "w320-e32-depth48-no-mtp",
        "--parallelism", "dp-only", "--micro-batch-size", "4", "--no-resume", *extra,
    ])


def test_only_attention_changes_not_native_training_arguments(tmp_path):
    config = Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()
    baseline = arguments(tmp_path)
    variant = arguments(tmp_path, "--attention-variant", SIMPLICIAL_ATTENTION)
    assert _megatron_argv(baseline, config) == _megatron_argv(variant, config)
    assert attention_variant_contract(baseline) is None
    contract = attention_variant_contract(variant)
    assert contract["changed_layers_1based"] == [8, 16, 24, 32, 40, 48]
    assert contract["extra_parameters"] == EXTRA_PARAMETERS == 245952
    assert 387680960 + EXTRA_PARAMETERS == 387926912


@pytest.mark.parametrize("model,parallelism", [("full", "dp-only"),
    ("1b-depth48-no-mtp", "dp-only"), ("w320-e32-depth48-no-mtp", "legacy")])
def test_reject_incompatible_shape_or_parallelism(model, parallelism):
    with pytest.raises(ValueError, match="W320/E32 DP-only"):
        attention_variant_contract(SimpleNamespace(attention_variant=SIMPLICIAL_ATTENTION,
                                                   model_variant=model, parallelism=parallelism))


def test_resume_cannot_silently_change_attention():
    variant = {"attention_variant": attention_variant_contract(
        arguments(Path("/tmp"), "--attention-variant", SIMPLICIAL_ATTENTION))}
    validate_attention_resume({}, {})
    validate_attention_resume(variant, variant)
    for old, new in (({}, variant), (variant, {})):
        with pytest.raises(RuntimeError, match="attention variant changed"):
            validate_attention_resume(old, new)
