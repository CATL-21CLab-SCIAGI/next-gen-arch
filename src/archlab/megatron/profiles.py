"""Explicit Megatron execution profiles for controlled architecture comparisons."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronBackendProfile:
    """One explicit, auditable set of Megatron wrapper optimizations."""

    name: str
    compile_architecture: bool
    use_mcore_bf16_master: bool
    finite_checks: bool = True
    compile_mode: str | None = None
    overlap_grad_reduce: bool = False
    ddp_num_buckets: int | None = None
    ddp_average_in_collective: bool = False
    compile_mode_overrides: tuple[tuple[str, str | None], ...] = ()

    def resolved_compile_mode(self, variant: str) -> str | None:
        overrides = dict(self.compile_mode_overrides)
        return overrides[variant] if variant in overrides else self.compile_mode


MEGATRON_BACKEND_PROFILES = {
    profile.name: profile
    for profile in (
        MegatronBackendProfile(
            name="legacy",
            compile_architecture=False,
            use_mcore_bf16_master=True,
        ),
        MegatronBackendProfile(
            name="compile",
            compile_architecture=True,
            use_mcore_bf16_master=True,
        ),
        MegatronBackendProfile(
            name="compile-reduce-overhead",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            compile_mode="reduce-overhead",
        ),
        MegatronBackendProfile(
            name="compile-max-autotune",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            compile_mode="max-autotune",
        ),
        MegatronBackendProfile(
            name="compile-safe-autotune",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            compile_mode="max-autotune",
            compile_mode_overrides=(
                ("kda", None),
                ("kimi-k3-kda-update", None),
                ("qwen-gdn", None),
            ),
        ),
        MegatronBackendProfile(
            name="compile-dp-overlap",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            overlap_grad_reduce=True,
            ddp_num_buckets=4,
        ),
        MegatronBackendProfile(
            name="compile-dp-overlap-average",
            compile_architecture=True,
            use_mcore_bf16_master=True,
            overlap_grad_reduce=True,
            ddp_num_buckets=4,
            ddp_average_in_collective=True,
        ),
        MegatronBackendProfile(
            name="native-master",
            compile_architecture=False,
            use_mcore_bf16_master=False,
        ),
        MegatronBackendProfile(
            name="speedrun",
            compile_architecture=True,
            use_mcore_bf16_master=False,
        ),
    )
}


def get_megatron_backend_profile(name: str) -> MegatronBackendProfile:
    try:
        return MEGATRON_BACKEND_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(MEGATRON_BACKEND_PROFILES)
        raise ValueError(
            f"unknown Megatron backend profile {name!r}; choose one of: {choices}"
        ) from error


