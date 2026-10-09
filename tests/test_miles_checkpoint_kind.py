from types import SimpleNamespace

import pytest

from archlab.megatron.miles_v41_checkpoint_kind import is_lora_model


def chunk(*names, peft=False):
    module = SimpleNamespace(peft_config={}) if peft else SimpleNamespace()
    return SimpleNamespace(module=module, named_parameters=lambda: [(n, None) for n in names])


def test_architectural_adapter_requires_full_model_checkpoint():
    assert not is_lora_model([chunk("module.decoder.layers.4.archlab_adapter.weight", "module.mlp.weight")])


@pytest.mark.parametrize("name", ["module.linear.lora_A", "module.linear.adapter.linear_in.weight"])
def test_real_peft_parameters_keep_adapter_checkpoint(name):
    assert is_lora_model([chunk("module.layer.archlab_adapter.weight", name)])


def test_peft_config_still_selects_lora():
    assert is_lora_model([chunk(peft=True)])
