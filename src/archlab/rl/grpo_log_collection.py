"""Instance-local binding of upstream GRPO object logs to a host collective."""

import dis
from functools import update_wrapper
from types import CellType, CodeType, FunctionType


def _uses_object_gather(code):
    # Python 3.11 gives comprehensions their own code object; Python 3.12
    # usually inlines them. Nested functions still share the owned globals.
    return any(
        instruction.opname == "LOAD_GLOBAL" and instruction.argval == "gather_object"
        for instruction in dis.get_instructions(code)
    ) or any(
        _uses_object_gather(value) for value in code.co_consts if isinstance(value, CodeType)
    )


def bind_host_object_gather(function, gather_object):
    """Retain upstream code and decorators without changing vendor globals.

    TRL imports ``gather_object`` into its generation method's globals instead
    of providing an instance hook. Bind that dependency in an owned namespace;
    preserve decorators by replacing only their reference to the wrapped
    function. Reject an unfamiliar wrapper rather than silently retaining NCCL.
    """
    if not isinstance(function, FunctionType):
        raise TypeError("GRPO log binding requires a Python function")
    wrapped = getattr(function, "__wrapped__", None)
    namespace = dict(function.__globals__)
    closure = function.__closure__
    if wrapped is None:
        if not _uses_object_gather(function.__code__) or not callable(
            namespace.get("gather_object")
        ):
            raise RuntimeError("upstream GRPO generation has no supported object-gather dependency")
        namespace["gather_object"] = gather_object
    else:
        bound_wrapped = bind_host_object_gather(wrapped, gather_object)
        if closure is None or not any(cell.cell_contents is wrapped for cell in closure):
            raise RuntimeError("upstream GRPO generation has an unsupported decorator")
        closure = tuple(
            CellType(bound_wrapped) if cell.cell_contents is wrapped else cell for cell in closure
        )
    bound = FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, closure
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    update_wrapper(bound, function)
    if wrapped is not None:
        bound.__wrapped__ = bound_wrapped
    return bound
