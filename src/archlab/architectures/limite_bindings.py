"""Local native-function bindings with shared compiler code and globals."""

import sys
from functools import cache
from threading import RLock
from types import FunctionType, ModuleType

_binding_lock = RLock()


def bind_native_forward(forward, replacements, tag, *, owned_namespace=False):
    """Copy a publisher namespace while retaining unchanged function bytecode.

    ``replacements`` is an immutable tuple of name/value pairs. Each binding
    shares one code object and globals dictionary across its model instances;
    distinct bindings get distinct code identities for Dynamo's generated names.
    The publisher function and its globals remain untouched.

    Opt-in ``owned_namespace`` registers a new private module for the copied
    globals. Compiler guards can then import replacement objects from their own
    namespace instead of resolving the publisher's unchanged module globals.
    The default preserves the existing unregistered namespace behavior.
    """
    if not isinstance(owned_namespace, bool):
        raise ValueError("owned_namespace must be a boolean")
    with _binding_lock:
        return _cached_bind_native_forward(forward, replacements, tag, owned_namespace)


@cache
def _cached_bind_native_forward(forward, replacements, tag, owned_namespace):
    namespace = dict(forward.__globals__)
    namespace.update(replacements)
    suffix = f"__archlab_{tag}"
    code = forward.__code__.replace(
        co_name=forward.__name__ + suffix,
        co_qualname=forward.__qualname__ + suffix,
    )
    owner = None
    if owned_namespace:
        name = f"_archlab_native_binding_{id(code):x}"
        if name in sys.modules:
            raise RuntimeError("private native binding namespace already exists")
        if "_archlab_bound_forward" in namespace:
            raise ValueError("publisher namespace contains reserved binding name")
        owner = ModuleType(name)
        owner.__dict__.update(namespace)
        owner.__dict__.update(__name__=name, __package__="", __spec__=None, __loader__=None)
        namespace = owner.__dict__
    bound = FunctionType(
        code, namespace, forward.__name__, forward.__defaults__, forward.__closure__
    )
    bound.__kwdefaults__ = forward.__kwdefaults__
    if owner is not None:
        bound.__module__ = owner.__name__
        owner.__dict__["_archlab_bound_forward"] = bound
        sys.modules[owner.__name__] = owner
    return bound


def _clear_binding_cache():
    with _binding_lock:
        _cached_bind_native_forward.cache_clear()


bind_native_forward.cache_clear = _clear_binding_cache
bind_native_forward.cache_info = _cached_bind_native_forward.cache_info
