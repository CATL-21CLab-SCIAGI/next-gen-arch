"""Execution backends for portable architecture experiments."""

from next_gen_arch.backends.base import LaunchPlan


def get_backend(name: str):
    if name == "speedrun":
        from next_gen_arch.backends.speedrun import SpeedrunBackend

        return SpeedrunBackend()
    if name == "megatron":
        from next_gen_arch.backends.megatron import MegatronBackend

        return MegatronBackend()
    raise ValueError(f"unknown backend: {name}")


__all__ = ["LaunchPlan", "get_backend"]
