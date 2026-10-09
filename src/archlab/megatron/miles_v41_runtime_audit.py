"""Record loaded extension bytecode separately from files currently on disk."""

import base64
import hashlib
import inspect
import json
import marshal
import sys
import types
from pathlib import Path


def _code_identity(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return _code_identity(value)
        if isinstance(value, tuple):
            return [constant(x) for x in value]
        if isinstance(value, frozenset):
            return {"frozenset": sorted((constant(x) for x in value), key=repr)}
        return repr(value)

    return dict(bytecode=code.co_code.hex(), constants=[constant(x) for x in code.co_consts],
                names=code.co_names, variables=code.co_varnames, free=code.co_freevars,
                cells=code.co_cellvars, args=code.co_argcount,
                positional_only=code.co_posonlyargcount, keyword_only=code.co_kwonlyargcount,
                flags=code.co_flags)


def _code_hash(code):
    return hashlib.sha256(json.dumps(_code_identity(code), sort_keys=True).encode()).hexdigest()


def audit_actor(actor):
    import datetime

    import torch.distributed as dist
    from miles.backends.megatron_utils.actor import MegatronTrainRayActor

    compiled, sources, functions = {}, {}, {}

    def record(name, function):
        if not inspect.isfunction(function):
            return
        code = function.__code__
        path = Path(code.co_filename)
        if path.is_file() and str(path) not in compiled:
            source = path.read_bytes()
            sources[str(path)] = hashlib.sha256(source).hexdigest()
            codes = {}

            def walk(item):
                codes[item.co_qualname] = _code_hash(item)
                for child in item.co_consts:
                    if isinstance(child, types.CodeType):
                        walk(child)

            walk(compile(source, str(path), "exec", dont_inherit=True))
            compiled[str(path)] = codes
        identity = _code_hash(code)
        functions[name] = dict(
            code_identity_sha256=identity,
            matches_current_source=compiled.get(str(path), {}).get(code.co_qualname) == identity,
            source_file=str(path), qualified_name=code.co_qualname,
            marshal_base64=base64.b64encode(marshal.dumps(code)).decode())

    for name, module in list(sys.modules.items()):
        if not (name.startswith("archlab.megatron.miles_")
                or name in {"archlab.optimizers.muown", "archlab.rl.miles_mimo"}):
            continue
        if module is None:
            continue
        for member_name, member in list(vars(module).items()):
            if getattr(member, "__module__", None) != name:
                continue
            if inspect.isfunction(member):
                record(f"{name}.{member_name}", member)
            elif inspect.isclass(member):
                for method_name, method in vars(member).items():
                    if isinstance(method, (classmethod, staticmethod)):
                        method = method.__func__
                    elif isinstance(method, property):
                        method = method.fget
                    record(f"{name}.{member_name}.{method_name}", method)
    record("live.MegatronTrainRayActor._switch_model", MegatronTrainRayActor._switch_model)
    rank = dist.get_rank()
    result = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  rank=rank, python=sys.version, source_file_sha256=sources,
                  initialized=hasattr(actor, "weight_updater"), functions=functions)
    path = Path(actor.args.save).parent / f"loaded-code-rank-{rank:02d}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return dict(rank=rank, artifact=str(path), functions=len(functions),
                mismatches=[name for name, item in functions.items()
                            if not item["matches_current_source"]])
