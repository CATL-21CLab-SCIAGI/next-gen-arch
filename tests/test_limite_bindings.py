"""Private compiler namespaces preserve native bytecode and existing bindings."""

import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from archlab.architectures.limite_bindings import bind_native_forward


class OriginalInterface:
    def get_interface(self, name, fallback):
        return fallback


class ReplacementInterface:
    def __init__(self, reduction):
        self.reduction = reduction

    def get_interface(self, name, fallback):
        return self.reduction


INTERFACE = OriginalInterface()


def native_forward(value, scale=2, *, offset=3):
    return INTERFACE.get_interface("native", lambda x: x + 1)(value) * scale + offset


def test_default_binding_remains_unregistered_and_original_globals_unchanged():
    replacement = ReplacementInterface(lambda value: value - 1)
    modules = set(sys.modules)
    before = dict(native_forward.__globals__)
    bound = bind_native_forward(native_forward, (("INTERFACE", replacement),), "default")
    assert bound(4) == 9 and native_forward(4) == 13
    assert bound.__module__ == native_forward.__module__
    assert bound.__globals__ is not native_forward.__globals__
    assert set(sys.modules) == modules
    assert before.keys() == native_forward.__globals__.keys()
    assert all(native_forward.__globals__[n] is value for n, value in before.items())
    assert bind_native_forward(native_forward, (("INTERFACE", replacement),), "default") is bound
    assert (
        bind_native_forward(
            native_forward, (("INTERFACE", replacement),), "default", owned_namespace=False
        )
        is bound
    )


def test_owned_namespace_preserves_bytecode_defaults_closure_and_cached_identity():
    shift = SimpleNamespace(value=11)

    def original(value=4, *, multiplier=2):
        return (value + shift.value) * multiplier

    modules = dict(sys.modules)
    before = dict(original.__globals__)
    bound = bind_native_forward(original, (), "closure", owned_namespace=True)
    assert bound() == original() == 30
    assert bound.__code__ is not original.__code__
    for field in ("co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars"):
        assert getattr(bound.__code__, field) == getattr(original.__code__, field)
    assert bound.__defaults__ is original.__defaults__
    assert bound.__kwdefaults__ is original.__kwdefaults__
    assert bound.__closure__ is original.__closure__
    assert bound.__module__.startswith("_archlab_native_binding_")
    assert sys.modules[bound.__module__].__dict__ is bound.__globals__
    assert sys.modules[bound.__module__]._archlab_bound_forward is bound
    assert all(sys.modules[name] is module for name, module in modules.items())
    assert before.keys() == original.__globals__.keys()
    assert all(original.__globals__[n] is value for n, value in before.items())
    assert bind_native_forward(original, (), "closure", owned_namespace=True) is bound


def test_aot_compiler_guards_resolve_replacement_in_owned_module():
    torch = pytest.importorskip("torch")
    replacement = ReplacementInterface(torch.sin)
    original_interface = INTERFACE
    bound = bind_native_forward(
        native_forward, (("INTERFACE", replacement),), "aot", owned_namespace=True
    )
    reference = torch.linspace(-1, 1, 17, requires_grad=True)
    actual = reference.detach().clone().requires_grad_()
    expected = torch.sin(reference) * 2 + 3
    compiled = torch.compile(bound, backend="aot_eager", fullgraph=True)
    output = compiled(actual)
    expected.sum().backward()
    output.sum().backward()
    assert torch.equal(output, expected) and torch.equal(actual.grad, reference.grad)
    exported = torch._dynamo.export(bound)(actual.detach())
    assert any("INTERFACE" in guard.name and "reduction" in guard.name for guard in exported.guards)
    assert sys.modules[bound.__module__].INTERFACE is replacement
    assert INTERFACE is original_interface and not hasattr(INTERFACE, "reduction")


@pytest.mark.parametrize("value", ["true", 1, 0, None])
def test_owned_namespace_rejects_non_boolean_option_after_cache_population(value):
    bind_native_forward(native_forward, (), "invalid", owned_namespace=True)
    bind_native_forward(native_forward, (), "invalid", owned_namespace=False)
    with pytest.raises(ValueError, match="boolean"):
        bind_native_forward(native_forward, (), "invalid", owned_namespace=value)


def test_cache_clear_keeps_private_lookup_and_allocates_a_distinct_owner():
    bound = bind_native_forward(native_forward, (), "clear", owned_namespace=True)
    owner = sys.modules[bound.__module__]
    bind_native_forward.cache_clear()
    assert owner._archlab_bound_forward is bound
    assert owner.__dict__ is bound.__globals__
    again = bind_native_forward(native_forward, (), "clear", owned_namespace=True)
    assert again is not bound and again.__module__ != bound.__module__
    assert sys.modules[bound.__module__] is owner
    assert again(4) == bound(4) == native_forward(4)


def test_concurrent_first_calls_share_one_owned_function_and_namespace():
    replacement = ReplacementInterface(lambda value: value - 1)

    def bind(_):
        return bind_native_forward(
            native_forward, (("INTERFACE", replacement),), "threads", owned_namespace=True
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        bindings = list(executor.map(bind, range(32)))
    assert all(bound is bindings[0] for bound in bindings)
    assert sys.modules[bindings[0].__module__]._archlab_bound_forward is bindings[0]


def test_owned_namespace_refuses_to_overwrite_publisher_reserved_name():
    with pytest.raises(ValueError, match="reserved binding name"):
        bind_native_forward(
            native_forward,
            (("_archlab_bound_forward", object()),),
            "reserved",
            owned_namespace=True,
        )
