"""Fail-closed launch admission for the GPU-resident RL contract."""

import argparse
import json
import math
from pathlib import Path


def verify_numerical(root, momentum_dtype="float32"):
    root = Path(root)
    for rank in range(2):
        directory = {"float16": "fp16-stable-admission", "float32": "fp32-streaming-restore-admission",
                     "bfloat16": "numerical-admission"}[momentum_dtype]
        path = root / directory / f"rank-{rank}.json"
        receipt = json.loads(path.read_text())
        error = receipt.get("max_update_relative_error", float("inf"))
        cosine = receipt.get("min_update_cosine", -1)
        if momentum_dtype == "float16":
            direction_error = receipt.get("max_direction_relative_error", float("inf"))
            direction_cosine = receipt.get("min_direction_cosine", -1)
            if (receipt.get("orthogonalization_dtype") != "float32"
                    or not math.isfinite(direction_error) or not math.isfinite(direction_cosine)
                    or direction_error > .01 or direction_cosine < .999
                    or receipt.get("streaming_checkpoint") is not True
                    or receipt.get("destructive_restore_verified") is not True):
                raise ValueError("float16 direction/checkpoint numerical admission rejected")
        if (not math.isfinite(error) or not math.isfinite(cosine)
                or receipt.get("rank") != rank or receipt.get("momentum_dtype") != momentum_dtype
                or receipt.get("passed") is not True
                or error > .01
                or cosine < .999
                or receipt.get("sinkhorn_distributed") is not True
                or receipt.get("resident_resume_exact") is not True
                or (momentum_dtype == "float32" and receipt.get("streaming_checkpoint") is not True)
                or (momentum_dtype == "float32" and receipt.get("destructive_restore_verified") is not True)
                or receipt.get("distributed_checkpoint_roundtrip") is not True):
            raise ValueError(f"{momentum_dtype} momentum numerical admission rejected rank {rank}")
    if momentum_dtype == "float16":
        for rank in range(2):
            receipt = json.loads((root / "fp16-expert-shape-admission" / f"rank-{rank}.json").read_text())
            if (receipt.get("rank") != rank or receipt.get("matrix_shape") != [2304, 5120]
                    or receipt.get("momentum_dtype") != "float16"
                    or receipt.get("orthogonalization_dtype") != "float32"
                    or receipt.get("passed") is not True):
                raise ValueError("expert-size momentum numerical admission rejected")
            for name, limit, lower in (("max_direction_relative_error", .01, False),
                                        ("max_update_relative_error", .01, False),
                                        ("min_direction_cosine", .999, True),
                                        ("min_update_cosine", .999, True)):
                value = receipt.get(name, float("nan"))
                if not math.isfinite(value) or (value < limit if lower else value > limit):
                    raise ValueError(f"expert-size momentum numerical admission rejected: {name}")


def verify(root, momentum_dtype="float32", variants=("normal", "simplicial")):
    root = Path(root)
    verify_numerical(root, momentum_dtype)
    # Numerical component receipts never establish support for a full model.
    for variant in variants:
        receipt = json.loads((root / f"full-model-{variant}.json").read_text())
        if (receipt.get("world_size") != 32 or receipt.get("variant") != variant
                or receipt.get("momentum_dtype") != momentum_dtype
                or receipt.get("offload_bytes") != 0
                or receipt.get("completed_updates", 0) < 2
                or receipt.get("minimum_free_hbm_fraction", 0) < .1
                or receipt.get("checkpoint_restore") is not True
                or receipt.get("policy_parity") is not True):
            raise ValueError(f"full-model admission rejected {variant}")


def record_full_model(root, output, variant, momentum_dtype):
    """Aggregate measured pilot artifacts; never synthesize success flags."""
    root, output = Path(root), Path(output)
    verify_numerical(root, momentum_dtype)
    updates, minima = [], []
    for rank in range(32):
        state = json.loads((output / f"resident-rank-{rank:02d}.json").read_text())
        restored = json.loads((output / f"checkpoint-restore-rank-{rank:02d}.json").read_text())
        parity = json.loads((output / f"policy-parity-rank-{rank:02d}.json").read_text())
        if (state["rank"] != rank or state["world_size"] != 32
                or state["momentum_dtype"] != f"torch.{momentum_dtype}"
                or not state["gradients_discarded"] or not restored["checkpoint_restore"]
                or restored["rank"] != rank or not parity["policy_parity"]):
            raise ValueError(f"incomplete full-model evidence for rank {rank}")
        updates.append(min(state["completed_updates"], restored["completed_updates"]))
    for node in ("master-0", "worker-0", "worker-1", "worker-2"):
        observed = json.loads((output / f"hbm-{node}.json").read_text())
        if len(observed["minimum_free_hbm_fractions"]) != 8 or observed["samples"] < 100:
            raise ValueError(f"incomplete physical memory observation: {node}")
        minima.extend(observed["minimum_free_hbm_fractions"])
    if min(updates) < 2 or min(minima) < .1:
        raise ValueError(f"full-model admission failed: updates={min(updates)}, free_fraction={min(minima)}")
    receipt = dict(world_size=32, variant=variant, momentum_dtype=momentum_dtype,
                   completed_updates=min(updates), minimum_free_hbm_fraction=min(minima),
                   checkpoint_restore=True, policy_parity=True, offload_bytes=0,
                   evidence_directory=str(output), offload_basis="guarded no-backup discard lifecycle")
    (root / f"full-model-{variant}.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    verify(args.root)
    print("RESIDENT_RL_ADMITTED")


if __name__ == "__main__":
    main()
