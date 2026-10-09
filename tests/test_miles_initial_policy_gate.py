"""Exercise the serving gate without importing the GPU-owned SGLang runtime."""

import ast
import itertools
import os
import sys
import types
from pathlib import Path

import pytest


def test_initial_policy_transaction_keeps_generation_closed(monkeypatch):
    source = Path(__file__).parents[1] / "src/archlab/serving/sglang/deepseek_v41.py"
    module = ast.parse(source.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"load_weights", "forward"}]

    class Native:
        def load_weights(self, weights):
            raise AssertionError("initial transaction incorrectly used checkpoint loader")

        def forward(self):
            return "generated"

    received = []
    extension = types.ModuleType("archlab.serving.sglang_v41_live_weights")
    extension.update = lambda model, weights, native: received.extend(weights)
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    monkeypatch.setenv("ARCHLAB_MILES_LIVE_WEIGHTS", "1")
    namespace = {"NativeV41": Native, "os": os, "itertools": itertools}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    model = namespace[cls.name]()
    model._archlab_loaded = False
    packet = [("archlab_update_begin", 1), ("head.weight", object())]
    model.load_weights(iter(packet))
    assert received == packet
    with pytest.raises(RuntimeError, match="before complete"):
        model.forward()
    model._archlab_loaded = True
    assert model.forward() == "generated"
