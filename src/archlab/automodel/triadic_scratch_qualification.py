"""Portable matched-mixer manifest and adapter admission; reuse the scratch trainer.

The two-rank probe admits only the inserted branch and optimizer. The existing
eight-rank scratch qualification separately admits the complete backbone. A
prepared manifest is never evidence of GPU qualification or a launch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file

FORMAT = "archlab-triadic-scratch-study-v1"
VARIANTS = ("linear", "linsimp", "gdn", "triadic")
LAYERS = (2, 4, 7, 9, 12, 14, 17, 19)
TRAINER = "archlab.automodel.deepseek_v41_scratch_training"
REVISION = "2caa4098073da92c2ba0d573df6453ceb7dbce45"
COMMON_MIXER_TENSORS = ("read_logits", "write_logits", "input_norm", "q", "q_norm", "k",
                        "k_norm", "v", "output_gate", "output")
DELTA_SHARED_TENSORS = ("beta_projection", "qkv_causal_conv")


def validate_matched_contract(value):
    """Reject changes that invalidate the preregistered scientific controls."""
    value = copy.deepcopy(value)
    if value.get("format") != FORMAT or tuple(value.get("variants", ())) != VARIANTS:
        raise ValueError("matched study requires its four registered mixer variants")
    geometry = value["geometry"]
    expected = dict(width=640, depth=20, streams=4, query_heads=2, kv_heads=2,
                    head_dim=128, short_window=32, long_window=512)
    if any(type(geometry.get(k)) is not int or geometry[k] != v for k, v in expected.items()):
        raise ValueError("matched study requires common width640/depth20/Q2/KV2/D128 geometry")
    if tuple(geometry.get("branch_layers", ())) != LAYERS:
        raise ValueError("matched branch placement changed")
    initial = value["initialization"]
    if (initial.get("random_seed") != 42 or initial.get("pretrained_weights") is not False
            or initial.get("zero_core_output") is not True
            or initial.get("zero_compensation_output") is not True):
        raise ValueError("matched random initialization and zero output contract changed")
    if (tuple(initial.get("common_mixer_tensors", ())) != COMMON_MIXER_TENSORS
            or tuple(initial.get("delta_shared_tensors", ())) != DELTA_SHARED_TENSORS
            or initial.get("delta_named_rng_streams") != ["beta", "qkv-conv"]
            or initial.get("require_constructor_hash_receipts") is not True):
        raise ValueError("common mixer and delta initialization controls changed")
    mixer = value["mixer_contract"]
    if (mixer.get("feature_rank"), mixer.get("gdn_features"), mixer.get("triadic_features"),
            mixer.get("official_revision")) != (512, 1, 4, REVISION):
        raise ValueError("RF rank, GDN axes, or official source identity changed")
    compensation = mixer["compensation"]
    if (compensation.get("target_reference"), compensation.get("reference_intermediate_size"),
            compensation.get("alignment"), compensation.get("maximum_relative_parameter_error")) != (
                "linsimp", 1024, 16, 0.01):
        raise ValueError("aligned branch parameter compensation changed")
    training = value["training"]
    expected_training = dict(world_size=8, expert_parallel=8, micro_batch_size=2, accumulation=4,
                             global_windows_per_update=64, sequence_length=2048, supervised_tokens=10_000_000_000,
                             validation_tokens=1_000_000, validation_interval_tokens=100_000_000,
                             checkpoint_interval_tokens=2_000_000_000)
    if any(type(training.get(k)) is not int or training[k] != v for k, v in expected_training.items()):
        raise ValueError("matched data, batch, token, or checkpoint contract changed")
    if (training.get("module") != TRAINER
            or training.get("checkpoint_tokens") != [2_000_000_000 * i for i in range(1, 6)]
            or training.get("cross_document_packing") is not False
            or training.get("all_parameters_trainable") is not True):
        raise ValueError("existing trainer, document isolation and five full-state milestones required")
    if training.get("execution") != dict(optimized_synchronization=True, trim_alignment=128,
                                         indexer_sample_grid="fixed_original_context_cpu"):
        raise ValueError("matched arms require common synchronization and 128-aligned tail trimming")
    if value["matching"].get("compute_match") is not False or value["matching"].get("state_match") is not False:
        raise ValueError("equal tokens and aligned parameters do not establish equal compute or state")
    if (value["qualification"].get("adapter_world_size"),
            value["qualification"].get("full_model_world_size"),
            value["qualification"].get("allow_missing_receipts")) != (2, 8, False):
        raise ValueError("separate adapter2/full-model8 admission is required")
    return value


def read_matched_contract(path):
    import yaml

    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("matched recipe must be a mapping")
    return validate_matched_contract(value)


def mixer_specification(contract, variant):
    contract = validate_matched_contract(contract)
    if variant not in VARIANTS:
        raise ValueError("unknown matched mixer variant")
    geometry = contract["geometry"]
    config = {k: geometry[k] for k in ("width", "streams", "query_heads", "kv_heads", "head_dim",
                                      "short_window", "long_window")}
    return dict(variant=variant, adapter_config=config, feature_rank=512,
                feature_dim={"gdn": 1, "triadic": 4}.get(variant),
                compensation=copy.deepcopy(contract["mixer_contract"]["compensation"]),
                backbone=dict(width=640, depth=20), layers=list(LAYERS))


def state_accounting(contract):
    """Logical per-head persistent values; not bytes, FLOPs, or cache peaks."""
    validate_matched_contract(contract)
    matrix = 512 * 128
    return dict(linear=dict(global_matrix=matrix, denominator=512, short_axis=0, total=matrix + 512),
                linsimp=dict(global_matrix=matrix, denominator=512, short_axis=2 * 32 * 128,
                             total=matrix + 512 + 2 * 32 * 128),
                gdn=dict(global_matrix=128 * 128, denominator=0, short_axis=0, total=128 * 128),
                triadic=dict(global_matrix=4 * 128 * 128, denominator=0, short_axis=0,
                             total=4 * 128 * 128),
                convention="per head, recurrent matrix plus RF denominator and LinSimp K/V ring; excludes temporary activations and convolution state")


def build_manifest(contract, *, recipe, source, revision, output_root, data_contract, container_image):
    contract = validate_matched_contract(contract)
    recipe, source, output_root, data_contract = map(Path, (recipe, source, output_root, data_contract))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("immutable source needs a full git revision")
    if source.joinpath("SOURCE_REVISION").read_text().strip() != revision:
        raise ValueError("immutable source revision differs")
    source_files = {str(p.relative_to(source)): sha256_file(p)
                    for p in sorted((source / "src/archlab").rglob("*.py"))}
    if not source_files:
        raise ValueError("immutable source has no project implementation")
    data = json.loads(data_contract.read_text())
    if (data.get("training_targets"), data.get("validation_targets"), data.get("sequence")) != (
            10_000_000_000, 1_000_000, 2048):
        raise ValueError("sealed corpus must have the matched 10B/1M/2048 contract")
    stages = []
    for variant in VARIANTS:
        arm = output_root / variant
        common = ["--variant", variant, "--width", "640", "--context", "2048",
                  "--microbatch", "2", "--accumulation", "4", "--matched-mixer-contract", str(recipe)]
        stages.append(dict(name=f"{variant}-adapter2", scope="adapter-only", world=2,
                           module="archlab.automodel.triadic_scratch_qualification",
                           args=["adapter", "--recipe", str(recipe), "--variant", variant,
                                 "--output", str(arm / "adapter2"), "--backend", "official"],
                           required_receipt=str(arm / "adapter2/QUALIFIED.json")))
        stages.append(dict(name=f"{variant}-full8", scope="full-backbone", world=8, module=TRAINER,
                           args=common + ["--mode", "qualify", "--output", str(arm / "qualification")],
                           required_receipt=str(arm / "qualification/QUALIFIED.json")))
        stages.append(dict(name=f"{variant}-train", scope="full-backbone", world=8, module=TRAINER,
                           args=common + ["--mode", "train", "--qualification", str(arm / "qualification"),
                                          "--output", str(arm / "production")],
                           requires=[str(arm / "adapter2/QUALIFIED.json"),
                                     str(arm / "qualification/QUALIFIED.json"),
                                     str(output_root / "MATCHED_REPORT.json")]))
    # All four actual budgets/initializations must be admitted before the first
    # production arm. Otherwise the first train gate would wait for later stages.
    stages = [s for s in stages if not s["name"].endswith("-train")] + [
        s for s in stages if s["name"].endswith("-train")]
    return dict(format="archlab-triadic-scratch-manifest-v1", status="prepared-not-qualified",
                source=str(source), source_revision=revision, source_file_sha256=source_files,
                recipe=str(recipe), recipe_sha256=sha256_file(recipe), data_contract=str(data_contract),
                data_contract_sha256=sha256_file(data_contract), container_image=container_image,
                token_budget_per_arm=10_000_000_000, study=contract, state_accounting=state_accounting(contract),
                stages=stages, scheduling=contract["scheduling"], launches_performed=False)


def _validate_mixer_initialization(proof, variant):
    if (not isinstance(proof, dict)
            or proof.get("format") != "archlab-matched-mixer-initialization-v2"
            or type(proof.get("seed")) is not int):
        raise ValueError("constructor mixer initialization receipt missing or invalid")
    expected_delta = set(DELTA_SHARED_TENSORS) if variant in ("gdn", "triadic") else set()
    for field, names in (("common_core_sha256", set(COMMON_MIXER_TENSORS)),
                         ("delta_shared_sha256", expected_delta)):
        hashes = proof.get(field, {})
        if (not isinstance(hashes, dict) or set(hashes) != names
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                       for h in hashes.values())):
            raise ValueError("shared mixer initialization tensor hashes missing or invalid")
    if proof.get("delta_rng_streams") != (["beta", "qkv-conv"] if expected_delta else []):
        raise ValueError("shared delta initialization does not use the registered named streams")


def _compare_mixer_initialization(arms):
    if set(arms) != set(VARIANTS) or not arms["linear"]:
        raise ValueError("four-arm constructor mixer initialization evidence missing")
    layers = set(arms["linear"])
    for variant in VARIANTS:
        if set(arms[variant]) != layers:
            raise ValueError("constructor mixer initialization layer identities differ")
        for layer, proof in arms[variant].items():
            _validate_mixer_initialization(proof, variant)
            baseline = arms["linear"][layer]
            if (proof["seed"] != baseline["seed"]
                    or proof["common_core_sha256"] != baseline["common_core_sha256"]):
                raise ValueError("matched arms do not share common mixer initialization")
    for layer in layers:
        if arms["gdn"][layer]["delta_shared_sha256"] != arms["triadic"][layer]["delta_shared_sha256"]:
            raise ValueError("GDN and Triadic shared beta/convolution initialization differs")


def verify_admission(manifest):
    """Verify actual receipts before any production stage; metadata alone fails."""
    if sha256_file(manifest["recipe"]) != manifest["recipe_sha256"]:
        raise ValueError("sealed recipe changed")
    if sha256_file(manifest["data_contract"]) != manifest["data_contract_sha256"]:
        raise ValueError("sealed data changed")
    source = Path(manifest["source"])
    if source.joinpath("SOURCE_REVISION").read_text().strip() != manifest["source_revision"]:
        raise ValueError("sealed source revision changed")
    actual = {str(p.relative_to(source)): sha256_file(p) for p in sorted((source / "src/archlab").rglob("*.py"))}
    if actual != manifest["source_file_sha256"]:
        raise ValueError("sealed implementation changed")
    reports = {}
    common_fingerprints = []
    adapter_initialization, mixer_initialization = {}, {}
    for stage in manifest["stages"]:
        if stage["scope"] == "adapter-only" or stage["name"].endswith("-full8"):
            receipt = Path(stage["required_receipt"])
            report = json.loads(receipt.read_text())
            if report.get("passed") is not True or report.get("world_size") != stage["world"]:
                raise ValueError("GPU admission failed or uses a different mesh")
            variant = stage["name"].removesuffix("-adapter2").removesuffix("-full8")
            if stage["scope"] == "adapter-only":
                if report.get("source_revision") != manifest["source_revision"] or report.get("variant") != variant:
                    raise ValueError("adapter receipt source or variant differs")
                adapter_initialization[variant] = {"adapter": report.get("initialization_contract")}
            if stage["scope"] == "full-backbone":
                declared = report["contract"]
                if (declared.get("project_commit") != manifest["source_revision"]
                        or declared.get("variant") != variant
                        or declared.get("matched_study") != manifest["study"]):
                    raise ValueError("full-model receipt uses a different source")
                if declared.get("data_contract") != json.loads(Path(manifest["data_contract"]).read_text()):
                    raise ValueError("full-model receipt uses a different corpus")
                mixer_initialization[variant] = declared.get("runtime", {}).get("matched_initialization_contracts", {})
                if set(mixer_initialization[variant]) != {str(i) for i in LAYERS}:
                    raise ValueError("full-model constructor mixer initialization layers missing")
                if report["ranks"] != [f"rank-{r:02d}-qualified.json" for r in range(8)]:
                    raise ValueError("full-model qualification rank identities differ")
                for name in report["ranks"]:
                    rank = json.loads((receipt.parent / name).read_text())
                    if (rank.get("passed") is not True or rank.get("exact_checkpoint_restore") is not True
                            or rank.get("identical_next_update_state") is not True):
                        raise ValueError("full-model restore/next-update qualification failed")
                    common_fingerprints.append(rank["common_initial_sha256"])
            reports[str(receipt)] = sha256_file(receipt)
    match_path = Path(next(s for s in manifest["stages"] if s["name"].endswith("-train"))["requires"][-1])
    match = json.loads(match_path.read_text())
    if match.get("passed") is not True or match.get("source_revision") != manifest["source_revision"]:
        raise ValueError("actual matched parameter/backbone report is missing")
    if match.get("recipe_sha256") != manifest["recipe_sha256"] or set(match.get("variants", {})) != set(VARIANTS):
        raise ValueError("matched report has a different recipe or variant set")
    for arm in match["variants"].values():
        error = arm.get("branch_relative_parameter_error", float("inf"))
        if (type(arm.get("nonembedding_parameters")) is not int or arm["nonembedding_parameters"] <= 0
                or type(arm.get("branch_parameters")) is not int or arm["branch_parameters"] <= 0
                or not isinstance(error, (int, float)) or not math.isfinite(error) or abs(error) > 0.01):
            raise ValueError("actual branch/nonembedding parameter audit failed")
    # Sharded fingerprints differ by rank; compare the rank-aligned four arms.
    if len(common_fingerprints) != 32 or any(len(set(common_fingerprints[r::8])) != 1 for r in range(8)):
        raise ValueError("matched arms do not share rank-aligned backbone initialization")
    _compare_mixer_initialization(adapter_initialization)
    _compare_mixer_initialization(mixer_initialization)
    if (match.get("mixer_initialization") != mixer_initialization
            or match.get("adapter_initialization") != adapter_initialization):
        raise ValueError("matched report differs from constructor mixer initialization evidence")
    reports[str(match_path)] = sha256_file(match_path)
    return dict(passed=True, scope="four-arm full-model8 admission", receipt_sha256=reports)


def assemble_matched_report(manifest):
    """Derive the budget audit from actual eight-rank construction/restore runs."""
    validate_matched_contract(manifest["study"])
    variants, proofs, initializations = {}, {}, {}
    adapter_initialization, mixer_initialization = {}, {}
    for variant in VARIANTS:
        stage = next(s for s in manifest["stages"] if s["name"] == variant + "-full8")
        qualified_path = Path(stage["required_receipt"])
        run_path = qualified_path.parent / "RUN_CONTRACT.json"
        qualified = json.loads(qualified_path.read_text())
        run = json.loads(run_path.read_text())
        if (qualified.get("passed") is not True or qualified.get("world_size") != 8
                or qualified.get("contract") != run
                or run.get("variant") != variant
                or run.get("project_commit") != manifest["source_revision"]
                or run.get("matched_study") != manifest["study"]
                or run.get("data_contract") != json.loads(Path(manifest["data_contract"]).read_text())):
            raise ValueError("actual full-model construction/qualification contract differs")
        runtime = run["runtime"]
        mixer_initialization[variant] = runtime.get("matched_initialization_contracts", {})
        if set(mixer_initialization[variant]) != {str(i) for i in LAYERS}:
            raise ValueError("full-model constructor mixer initialization layers missing")
        adapter_stage = next(s for s in manifest["stages"] if s["name"] == variant + "-adapter2")
        adapter_path = Path(adapter_stage["required_receipt"])
        adapter = json.loads(adapter_path.read_text())
        if (adapter.get("passed") is not True or adapter.get("world_size") != 2
                or adapter.get("variant") != variant
                or adapter.get("source_revision") != manifest["source_revision"]):
            raise ValueError("actual adapter construction/qualification contract differs")
        adapter_initialization[variant] = {"adapter": adapter.get("initialization_contract")}
        proofs[str(adapter_path)] = sha256_file(adapter_path)
        layers = runtime["matched_parameter_contracts"]
        if isinstance(layers, dict):
            layers = list(layers.values())
        if len(layers) != 8 or any(layer.get("variant") != variant for layer in layers):
            raise ValueError("actual matched branch layer contracts differ")
        actual = sum(layer["total_parameters"] for layer in layers)
        target = sum(layer["target_parameters"] for layer in layers)
        if (runtime["branch_parameters"] != actual or target <= 0
                or type(runtime["nonembedding_parameters"]) is not int
                or runtime["nonembedding_parameters"] <= actual):
            raise ValueError("actual branch/nonembedding parameter counts differ")
        error = abs(actual - target) / target
        if error > 0.01 or any(not math.isfinite(layer["relative_parameter_error"])
                               or layer["relative_parameter_error"] > 0.01 for layer in layers):
            raise ValueError("actual branch compensation exceeds its parameter tolerance")
        ranks = qualified["ranks"]
        if ranks != [f"rank-{r:02d}-qualified.json" for r in range(8)]:
            raise ValueError("actual qualification rank identities differ")
        hashes = []
        timing = None
        for name in ranks:
            path = qualified_path.parent / name
            rank = json.loads(path.read_text())
            if (rank.get("passed") is not True or rank.get("exact_checkpoint_restore") is not True
                    or rank.get("identical_next_update_state") is not True):
                raise ValueError("actual checkpoint/optimizer next-update proof failed")
            hashes.append(rank["common_initial_sha256"])
            from archlab.automodel.deepseek_v41_scratch_training import summarize_training_timings

            observed = rank.get("real_data_timing")
            if (not isinstance(observed, dict) or observed.get("warmup_updates") != 2
                    or observed.get("measured_updates") != 3
                    or observed != summarize_training_timings(observed.get("records", []), warmup_updates=2)):
                raise ValueError("matched throughput requires warmed, sealed real-data evidence")
            if timing is not None and observed != timing:
                raise ValueError("rank-aligned real-data timing evidence differs")
            timing = observed
            proofs[str(path)] = sha256_file(path)
        initializations[variant] = hashes
        proofs[str(qualified_path)], proofs[str(run_path)] = sha256_file(qualified_path), sha256_file(run_path)
        variants[variant] = dict(branch_parameters=actual, target_branch_parameters=target,
                                 branch_relative_parameter_error=error,
                                 nonembedding_parameters=runtime["nonembedding_parameters"],
                                 total_parameters=runtime["parameters"], per_layer_contracts=layers,
                                 qualification_training_timing=timing)
    if any(initializations[v] != initializations["linear"] for v in VARIANTS):
        raise ValueError("actual arms do not share the same rank-aligned backbone initialization")
    _compare_mixer_initialization(adapter_initialization)
    _compare_mixer_initialization(mixer_initialization)
    # Check source/data/recipe pins and both meshes through the existing admission
    # routine after this report is saved; this report alone never admits training.
    return dict(passed=True, source_revision=manifest["source_revision"],
                recipe_sha256=manifest["recipe_sha256"], data_contract_sha256=manifest["data_contract_sha256"],
                variants=variants, rank_aligned_backbone_sha256=initializations,
                mixer_initialization=mixer_initialization, adapter_initialization=adapter_initialization,
                evidence_sha256=proofs, state_accounting=state_accounting(manifest["study"]),
                compute_matched=False, state_matched=False)


def run_adapter_smoke(contract, variant, output, *, backend):
    """Two-rank branch probe; complete-backbone admission belongs to the trainer."""
    import torch
    import torch.distributed as dist

    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.architectures.deepseek_v41_matched_mixer import MatchedMixerAdapter
    from archlab.optimizers.sharded_adafactor import ShardedAdafactor

    contract = validate_matched_contract(contract)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("this branch-only probe requires two ranks")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    spec = mixer_specification(contract, variant)
    torch.manual_seed(42)
    module = MatchedMixerAdapter(V41AdapterConfig(**spec["adapter_config"]), variant=variant,
                                 seed=42, backend=backend if variant in ("gdn", "triadic") else None,
                                 feature_dim=spec["feature_dim"] or 4,
                                 compensation_anchor=1024, compensation_alignment=16).cuda()
    initialization_contract = module.initialization_contract()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 128, 4, 640, device="cuda", dtype=torch.bfloat16)
    if not torch.equal(module(x), x):
        raise AssertionError("zero-initialized branch changes initial backbone activations")
    # Activate both output paths to expose gradients behind their initial zeros.
    with torch.no_grad():
        for _, p in module.named_parameters():
            if p.ndim == 2 and bool((p == 0).all()):
                p.normal_(0, 0.002)
    module.eval()
    with torch.no_grad():
        original = module(x)
        if not torch.equal(original, module(x)):
            raise AssertionError("identical forward input is not deterministic")
        future = x.clone()
        future[:, 64:] = torch.randn_like(future[:, 64:])
        if not torch.equal(original[:, :64], module(future)[:, :64]):
            raise AssertionError("future tokens affect earlier outputs")
        other = torch.randn_like(x)
        batched = module(torch.cat((x, other), dim=0))
        separate = module(other)
        if not torch.allclose(batched[1:], separate, rtol=0.002, atol=0.002):
            raise AssertionError("independent documents share recurrent state")
    module.train()
    optimizer = ShardedAdafactor(module.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    (module(x).float().square().mean()).backward()
    gradients = {}
    for name, p in module.named_parameters():
        if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
            raise AssertionError("missing or nonfinite trainable gradient: " + name)
        dist.all_reduce(p.grad)
        p.grad.div_(world)
        gradients[name] = p.grad.clone()
    optimizer.step()
    state = copy.deepcopy(dict(model=module.state_dict(), optimizer=optimizer.state_dict(),
                              cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    torch.save(state, output / f"rank-{rank:02d}-state.pt")

    def replay():
        for name, p in module.named_parameters():
            p.grad = gradients[name].clone()
        optimizer.step()
        return copy.deepcopy(dict(model=module.state_dict(), optimizer=optimizer.state_dict(),
                                  cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))

    expected = replay()
    restored = torch.load(output / f"rank-{rank:02d}-state.pt", map_location="cuda", weights_only=False)
    module.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["cpu_rng"].cpu())
    torch.cuda.set_rng_state(restored["cuda_rng"].cpu())
    actual = replay()

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
        return left == right

    if not equal(expected, actual):
        raise AssertionError("checkpoint optimizer/RNG replay is not exact")
    digest = hashlib.sha256()
    for name, p in module.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    replicas = [None] * world
    dist.all_gather_object(replicas, digest.hexdigest())
    if len(set(replicas)) != 1:
        raise AssertionError("branch replicas diverged")
    receipt = dict(passed=True, scope="adapter-only", world_size=world, variant=variant,
                   zero_output_identity=True, causal_future_isolation=True, independent_document_reset=True,
                   finite_all_trainable_gradients=True, identical_gradient_next_optimizer_update=True,
                   exact_checkpoint_optimizer_rng_restore=True, replica_agreement=True,
                   source_revision=os.environ.get("ARCHLAB_SOURCE_REVISION", os.environ.get("NGA_EXPECTED_COMMIT")), backend=module.backend,
                   parameters=module.parameter_contract(), initialization_contract=initialization_contract,
                   checkpoint_sha256=sha256_file(output / f"rank-{rank:02d}-state.pt"))
    atomic_write_json(output / f"rank-{rank:02d}-qualified.json", receipt)
    dist.barrier()
    if rank == 0:
        atomic_write_json(output / "QUALIFIED.json", dict(receipt, ranks=[f"rank-{r:02d}-qualified.json" for r in range(world)]))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest")
    for name in ("recipe", "source", "output-root", "data-contract", "output"):
        manifest.add_argument("--" + name, type=Path, required=True)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--container-image", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    adapter = sub.add_parser("adapter")
    adapter.add_argument("--recipe", type=Path, required=True)
    adapter.add_argument("--variant", choices=VARIANTS, required=True)
    adapter.add_argument("--output", type=Path, required=True)
    adapter.add_argument("--backend", choices=("official", "reference"), default="official")
    args = parser.parse_args()
    if args.command == "manifest":
        value = build_manifest(read_matched_contract(args.recipe), recipe=args.recipe, source=args.source,
                               revision=args.revision, output_root=args.output_root,
                               data_contract=args.data_contract, container_image=args.container_image)
        atomic_write_json(args.output, value)
    elif args.command == "verify":
        atomic_write_json(args.output, verify_admission(json.loads(args.manifest.read_text())))
    elif args.command == "report":
        atomic_write_json(args.output, assemble_matched_report(json.loads(args.manifest.read_text())))
    else:
        run_adapter_smoke(read_matched_contract(args.recipe), args.variant, args.output, backend=args.backend)


if __name__ == "__main__":
    main()
