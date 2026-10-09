"""The original modded-nanogpt/nanochat-derived speedrun backend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from archlab.capabilities import require_backend_support
from archlab.launch import LaunchPlan
from archlab.registry import find_run
from archlab.spec import ExperimentSpec, SpecError

SPEEDRUN_UPSTREAM = "https://github.com/KellerJordan/modded-nanogpt.git"
SPEEDRUN_PROVENANCE_COMMIT = "f411b3d346aa52d3504324ca93c230fd84c6c07f"


class SpeedrunBackend:
    name = "speedrun"

    def render(
        self,
        spec: ExperimentSpec,
        *,
        path_overrides: dict[str, str] | None = None,
    ) -> LaunchPlan:
        require_backend_support(spec.variant, self.name)
        selection = spec.config.get("selection", {})
        size = selection.get("size")
        if not isinstance(size, str):
            raise SpecError("speedrun experiments require selection.size")
        paths = spec.resolve_paths(path_overrides)
        if "data_root" not in paths:
            raise SpecError("speedrun experiments require paths.data_root")

        run = find_run(size, spec.variant, spec.seed)
        command = run.command(run_name=spec.name)
        prompt_file = spec.resolve_reference(spec.prompts)
        command.append(f"--prompt-file={prompt_file}")
        comparison = spec.comparison
        artifacts = spec.artifacts
        if artifacts is not None and artifacts.mode == "research":
            if "output_dir" not in paths or "data_manifest" not in paths:
                raise SpecError(
                    "research speedrun specs require paths.output_dir and paths.data_manifest"
                )
            command = [argument for argument in command if argument != "--no-save-final-checkpoint"]
            command.extend(
                (
                    "--save-final-checkpoint",
                    f"--metrics-path={Path(paths['output_dir']) / 'metrics.jsonl'}",
                    f"--checkpoint-dir={Path(paths['output_dir']) / 'checkpoints'}",
                    f"--data-manifest={paths['data_manifest']}",
                    "--artifact-policy=research",
                    "--initialization-hash=shared",
                )
            )

        parallelism = spec.parallelism
        nodes = int(parallelism.get("nodes", 1))
        gpus = int(parallelism.get("gpus_per_node", 1))
        if nodes < 1 or gpus < 1:
            raise SpecError("nodes and gpus_per_node must be positive")
        if nodes > 1:
            command = [
                "torchrun",
                f"--nnodes={nodes}",
                f"--nproc-per-node={gpus}",
                "--node-rank=env:NODE_RANK",
                "--master-addr=env:MASTER_ADDR",
                "--master-port=env:MASTER_PORT",
                *command[1:],
            ]
        elif gpus > 1:
            command = ["torchrun", "--standalone", f"--nproc-per-node={gpus}", *command[1:]]

        return LaunchPlan(
            backend=self.name,
            argv=tuple(command),
            env={"NANOCHAT_BASE_DIR": paths["data_root"]},
            metadata={
                "run_id": run.run_id,
                "parameter_count": run.parameter_count,
                "training_tokens": run.tokens,
                "prompt_file": prompt_file,
                "upstream": SPEEDRUN_UPSTREAM,
                "provenance_commit": SPEEDRUN_PROVENANCE_COMMIT,
                "semantic_equivalence": "frozen campaign command with only package-path changes",
                "comparison": asdict(comparison) if comparison is not None else None,
                "artifacts": asdict(artifacts) if artifacts is not None else None,
                "data_manifest": paths.get("data_manifest"),
            },
        )
