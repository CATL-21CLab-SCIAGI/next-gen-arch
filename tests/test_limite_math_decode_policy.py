"""Changing benchmark shard ownership must not select a new numerical path."""

import pytest

pytest.importorskip("transformers")

from archlab.automodel.limite_math_benchmark import prepare_decode


def test_frozen_eager_does_not_touch_the_model_tokenizer_or_cuda(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("eager scheduling must not capture or probe")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    model = tokenizer = object()
    for question in ["first old shard question", "a different reassigned question"]:
        pool, admission = prepare_decode(model, tokenizer, question, dict(decode_mode="native_eager"))
        assert pool is None and admission["skipped"] and not admission["passed"]


def test_unknown_decode_policy_fails_before_sampling():
    with pytest.raises(ValueError, match="decode mode"):
        prepare_decode(object(), object(), "question", dict(decode_mode="misspelled"))
