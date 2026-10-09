"""Run the existing V4.1 simplicial experiment on official AutoModel.

The same process validates the bare backbone, adds the zero-initialized
adapters, qualifies distributed updates/reload, then starts the sealed pilot.
No failed numerical receipt can admit production training.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import json
import math
import os
import signal
import subprocess
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.deepseek_v41_official_execution import (
    MODEL_REVISION,
    UPSTREAM_COMMIT,
    official_core_hashes,
)


@dataclass(frozen=True)
class OfficialV41Config:
    assets: str
    weights: str
    train_data: str
    validation_data: str
    mesh_qualification: str
    output: str
    container_kernel_packages: str
    backend: dict
    world_size: int = 32
    ep_size: int = 8
    context: int = 16384
    supervised_tokens: int = 1_000_000_000
    validation_tokens: int = 1_000_000
    validation_interval_tokens: int = 10_000_000
    checkpoint_interval_tokens: int = 50_000_000
    warmup_steps: int = 100
    query_chunk: int = 32
    adapter_variant: str = "simplicial"

    def __post_init__(self):
        if self.adapter_variant not in ("simplicial", "normal"):
            raise ValueError("unknown adapter variant")
        if (self.world_size, self.ep_size, self.context, self.supervised_tokens) != (
            32,
            8,
            16384,
            1_000_000_000,
        ):
            raise ValueError("preserve the selected world32/EP8/16K/1B experiment")
        if self.backend != {
            "implementation": "official-automodel-deepseek-v41",
            "upstream_commit": UPSTREAM_COMMIT,
            "dense_fsdp": 32,
            "expert_fsdp": 4,
            "engram_owners": 32,
            "attention": "tilelang",
            "sinkhorn": "native-tilelang",
            "experts": "torch_mm",
            "dispatcher": "torch",
            "linear_compute": "bfloat16",
            "moe_combine": "ordered-float32-through-shared-add-single-final-cast",
            "expert_gate_up": "native-separate-bfloat16-linear",
            "expert_activation": "eager-float32-clamped-swiglu",
            "expert_input_gradient": "float32-local-and-expert-parallel",
            "sparse_backward": "private-query-deterministic-fp32-kv",
            "deterministic_pytorch": False,
            "adapter_core": (
                "deterministic-triton-private-window-gradients"
                if self.adapter_variant == "simplicial"
                else "flash-attn-deterministic-bf16-local512"
            ),
            "expert_gather_backward": "private-clone-before-allreduce",
            "kv_index_quantization": "retained",
            "parity_reference": "released-structure-matching-bf16-compute",
        }:
            raise ValueError("official backend does not match the reviewed migration contract")
        for name in (
            "world_size",
            "ep_size",
            "context",
            "supervised_tokens",
            "validation_tokens",
            "validation_interval_tokens",
            "checkpoint_interval_tokens",
            "warmup_steps",
            "query_chunk",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def resolve_path(value):
    if not value.startswith("env:") or not os.environ.get(value[4:]):
        raise ValueError(f"populate the recipe path variable: {value}")
    return Path(os.environ[value[4:]]).resolve()


def admit_mesh(path, hashes, *, runtime=None, adapter_variant="simplicial"):
    if adapter_variant not in ("simplicial", "normal"):
        raise ValueError("unknown mesh adapter variant")
    expected_backend = (
        "deterministic" if adapter_variant == "simplicial" else "flash-attn-deterministic"
    )
    from archlab.automodel.deepseek_v41_official_training import REPLAY_PROTOCOL

    marker = json.loads((path / "COMPLETE.json").read_text())
    if (
        not marker.get("passed")
        or marker.get("world_size") != 32
        or marker.get("rank_receipts") != [f"rank{rank}.json" for rank in range(32)]
    ):
        raise ValueError("the actual 32-rank tiny mesh must qualify before loading production")
    receipts = []
    runtime_keys = ("container_image", "packages", "cuda", "nccl", "reproducibility")
    expected_runtime = None if runtime is None else {key: runtime[key] for key in runtime_keys}
    for rank in range(32):
        receipt_path = path / f"rank{rank}.json"
        receipt = json.loads(receipt_path.read_text())
        if (
            not receipt.get("passed")
            or receipt["rank"] != rank
            or receipt.get("automodel_commit") != UPSTREAM_COMMIT
            or receipt.get("implementation_sha256") != hashes
        ):
            raise ValueError(f"missing, failed, or stale official mesh receipt: {receipt_path}")
        expected_topology = {
            "world_size": 32,
            "ep_size": 8,
            "expert_fsdp_size": 4,
            "engram_owners": 32,
        }
        if any(receipt.get(key) != value for key, value in expected_topology.items()):
            raise ValueError(f"wrong topology in official mesh receipt: {receipt_path}")
        mesh = receipt.get("mesh", {})
        ep_ranks = list(range(rank // 8 * 8, (rank // 8 + 1) * 8))
        if (
            mesh.get("ep_ranks") != ep_ranks
            or len(mesh.get("ep_hosts", [])) != 8
            or len(set(mesh["ep_hosts"])) != 1
            or mesh.get("expert_fsdp_ranks") != list(range(rank % 8, 32, 8))
            or not mesh.get("engrams")
            or any(item.get("owner_ranks") != list(range(32)) for item in mesh["engrams"])
        ):
            raise ValueError(f"wrong ownership in official mesh receipt: {receipt_path}")
        if (
            not receipt.get("unique_rank_windows")
            or receipt.get("adapter", {}).get("backend") != expected_backend
            or receipt.get("adapter", {}).get("variant", "simplicial") != adapter_variant
            or not receipt.get("checkpoint_replay", {}).get("passed")
            or not receipt.get("checkpoint_replay", {}).get("exact_state_restoration")
            or receipt.get("checkpoint_replay", {}).get("protocol") != REPLAY_PROTOCOL["version"]
            or receipt.get("checkpoint_replay", {}).get("protocol_config") != REPLAY_PROTOCOL
            or not receipt.get("frozen_base", {}).get("unchanged")
            or not receipt.get("frozen_base", {}).get("no_gradients")
        ):
            raise ValueError(
                f"incomplete update/reload checks in official mesh receipt: {receipt_path}"
            )
        updates = receipt.get("updates", [])
        if len(updates) < 2 or any(
            not math.isfinite(update.get("loss", math.nan))
            or not all(
                update.get(key)
                for key in (
                    "replicated_fp32_masters",
                    "world_gradient_agreement",
                    "world_parameter_agreement",
                )
            )
            for update in updates
        ):
            raise ValueError(f"incomplete gradient checks in official mesh receipt: {receipt_path}")
        measured_runtime = {key: receipt.get(key) for key in runtime_keys}
        if any(value is None for value in measured_runtime.values()):
            raise ValueError(f"missing runtime in official mesh receipt: {receipt_path}")
        if expected_runtime is None:
            expected_runtime = measured_runtime
        if measured_runtime != expected_runtime:
            raise ValueError(f"runtime differs from official mesh receipt: {receipt_path}")
        receipts.append(sha256_file(receipt_path))
    return receipts


def run(args):
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    config = OfficialV41Config(**yaml.safe_load(args.recipe.read_text()))
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    kernels = select_container_kernel_packages(resolve_path(config.container_kernel_packages))
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_official_execution import (
        build_official_base,
        configure_official_reproducibility,
        runtime_identity,
    )
    from archlab.automodel.deepseek_v41_official_qualification import (
        prepare_official_reference,
        qualify_official_forward,
    )
    from archlab.automodel.deepseek_v41_official_training import (
        REPLAY_PROTOCOL,
        OfficialV41TrainingSession,
    )
    from archlab.automodel.deepseek_v41_training import emit

    if not os.environ.get("NGA_CONTAINER_DIGEST") or not os.environ.get("NGA_EXPECTED_COMMIT"):
        raise ValueError("record the unchanged container identity and immutable project source")
    source_root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != os.environ["NGA_EXPECTED_COMMIT"]:
        raise ValueError("project source does not match launch identity")
    if subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("launch from the clean immutable source snapshot")
    configure_official_reproducibility()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=120),
        device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
    )
    output = resolve_path(config.output)
    output_created = False
    try:
        if dist.get_world_size() != config.world_size:
            raise ValueError("the production launch requires all 32 GPUs")
        hashes = official_core_hashes()
        mesh_receipts = admit_mesh(
            resolve_path(config.mesh_qualification),
            hashes,
            runtime=runtime_identity(),
            adapter_variant=config.adapter_variant,
        )
        status = [None]
        if dist.get_rank() == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                status[0] = str(error)
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise ValueError(f"use a fresh production output directory: {status[0]}")
        output_created = True
        train = MathPilot(
            resolve_path(config.train_data),
            expected_split="train",
            expected_budget=config.supervised_tokens,
        )
        validation = MathPilot(
            resolve_path(config.validation_data),
            expected_split="validation",
            expected_budget=config.validation_tokens,
        )
        if train.context != config.context or validation.context != config.context:
            raise ValueError("both sealed pilots must use the recipe's 16K context")
        prepared_reference = prepare_official_reference(
            assets=resolve_path(config.assets),
            weights=resolve_path(config.weights),
            pilot=train,
            output=output / "forward-qualification",
        )
        model, setup, loading = build_official_base(
            weights=resolve_path(config.weights),
            assets=resolve_path(config.assets),
            ep_size=config.ep_size,
        )
        atomic_write_json(
            output / f"rank-{dist.get_rank():02d}-loading.json", loading, allow_nan=False
        )
        parity, baseline = qualify_official_forward(
            model,
            setup,
            pilot=train,
            output=output / "forward-qualification",
            prepared_reference=prepared_reference,
        )
        del prepared_reference
        adapter_backend = (
            "deterministic"
            if config.adapter_variant == "simplicial"
            else "flash-attn-deterministic"
        )
        adapters = install_official_adapters(
            model, device="cuda", backend=adapter_backend, variant=config.adapter_variant
        )
        inputs, _, _ = train.batch(
            dist.get_rank(), device="cuda", smoke_context=128, pad_to_full=True
        )
        with torch.no_grad():
            zero = model(input_ids=inputs, return_hidden_states=True).hidden_states
        identical = torch.tensor(
            int(torch.equal(zero.cpu(), baseline)), device="cuda", dtype=torch.int32
        )
        dist.all_reduce(identical, op=dist.ReduceOp.MIN)
        if not identical.item():
            raise ValueError("zero-initialized adapters changed the official base output")
        del baseline, zero, inputs
        emit("official_zero_adapter_identity_passed")
        # Checkpoint compatibility uses stable experiment/source/runtime identity.
        # Invocation paths, timestamps and probe timings belong in provenance.
        contract = {
            "schema": "archlab-official-v41-adapters-v1",
            "model_revision": MODEL_REVISION,
            "checkpoint_index_sha256": sha256_file(
                resolve_path(config.weights) / "model.safetensors.index.json"
            ),
            "base_config_sha256": sha256_file(resolve_path(config.weights) / "config.json"),
            "reference_config_sha256": sha256_file(
                resolve_path(config.assets) / "inference/config.json"
            ),
            "project_commit": commit,
            "implementation_sha256": hashes,
            "upstream_commit": UPSTREAM_COMMIT,
            "backend": config.backend,
            "recipe_sha256": sha256_file(args.recipe),
            "training": {
                key: value for key, value in asdict(config).items() if isinstance(value, int)
            },
            "train_manifest_sha256": sha256_file(
                resolve_path(config.train_data) / "PILOT_READY.json"
            ),
            "validation_manifest_sha256": sha256_file(
                resolve_path(config.validation_data) / "PILOT_READY.json"
            ),
            "data_order_seed": 2234,
            "adapter_variant": config.adapter_variant,
            "adapter_config": asdict(next(iter(adapters.values())).config),
            "adapter_layers_0based": list(adapters),
            "trainable_parameters": sum(
                p.numel() for a in adapters.values() for p in a.parameters()
            ),
            "container_image": loading["container_image"],
            "runtime": loading["packages"],
            "cuda": loading["cuda"],
            "nccl": loading["nccl"],
            "reproducibility": loading["reproducibility"],
            "optimizer": "headwise-Muon-QK/matrix-Muon/AdamW-nonmatrix",
            "forward_qualification": {
                "contexts": [128, 2048, 16384],
                "kl_limit": 0.02,
                "ce_delta_limit": 0.05,
            },
            "checkpoint_restore": "exact-serialized-state-and-RNG",
            "checkpoint_replay_protocol": REPLAY_PROTOCOL,
        }
        all_contracts = [None] * config.world_size
        dist.all_gather_object(all_contracts, contract)
        if any(item != contract for item in all_contracts):
            raise ValueError("ranks disagree on the production contract")
        if dist.get_rank() == 0:
            atomic_write_json(output / "RUN_CONTRACT.json", contract, allow_nan=False)
            atomic_write_json(
                output / "RUN_PROVENANCE.json",
                {
                    "source_root": str(source_root),
                    "recipe": str(args.recipe.resolve()),
                    "mesh_qualification": str(resolve_path(config.mesh_qualification)),
                    "mesh_receipts_sha256": mesh_receipts,
                    "kernel_packages": kernels,
                    "forward_receipts": "forward-qualification/rank*.json",
                    "resume_from": None
                    if args.resume_from is None
                    else str(args.resume_from.resolve()),
                },
                allow_nan=False,
            )
        session = OfficialV41TrainingSession(
            model, adapters, train, validation, output, contract, config
        )
        report = session.qualify_mesh()
        atomic_write_json(
            output / f"rank-{dist.get_rank():02d}-training-qualification.json",
            report,
            allow_nan=False,
        )
        dist.barrier()
        if dist.get_rank() == 0:
            atomic_write_json(output / "TRAINING_ADMITTED.json", {"passed": True, "world_size": 32})
        session.run(resume_from=args.resume_from)
    except BaseException:
        failure = {"rank": dist.get_rank(), "traceback": traceback.format_exc()}
        print(json.dumps({"event": "official_run_failed", **failure}), flush=True)
        if output_created:
            atomic_write_json(output / f"rank-{dist.get_rank():02d}-failure.json", failure)
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
