"""Index a bounded, reproducible SWE subset for MiMo's existing agent runner.

Keep task/test JSON intact. The only conversion replaces unpublished local image
names with digest-pinned images from the publisher's Docker Hub repository.
Tokenization belongs to the multi-turn runner; hidden tests never become prompts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
REGISTRY = "xiaomimimo/mimo-v2.6-rl-oss"
REQUIRED = (
    "cwd",
    "docker_image",
    "instance_id",
    "problem_statement",
    "test_command",
    "test_patch",
    "verifier_timeout_sec",
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def instance(row: dict) -> dict:
    """Validate the upstream executable-task contract without executing it."""
    task = json.loads(row["extra_info"]["instance_json"])
    if (
        row.get("data_source") != "opensource-code"
        or row.get("ability") != "swe"
        or row.get("agent_name") != "mimo_swe_agent"
        or row["extra_info"].get("dataset_type") != "opensource-code"
        or task.get("dataset_type") != "opensource-code"
    ):
        raise ValueError("Only MiMo software-engineering agent tasks are permitted")
    if any(not task.get(key) for key in REQUIRED):
        raise ValueError("Incomplete executable task: required environment/test metadata missing")
    if row["extra_info"]["instance_id"] != task["instance_id"]:
        raise ValueError("Task identity mismatch")
    if not task["cwd"].startswith("/") or int(task["verifier_timeout_sec"]) <= 0:
        raise ValueError("Invalid sandbox working directory or verifier timeout")
    if row.get("prompt") != [{"role": "user", "content": task["problem_statement"]}]:
        raise ValueError("Unexpected prompt; hidden test metadata must stay outside the prompt")
    if row.get("reward_model") != {"style": "rule", "ground_truth": ""}:
        raise ValueError("SWE rewards must come from the executable verifier")
    return task


def select(rows: list[dict], train: int, validation: int, seed: int) -> list[dict]:
    """Hash-rank unique tasks, independent of source row order; fail on bad rows."""
    if train < 1 or validation < 1 or train + validation > 128:
        raise ValueError("Pilot requires positive train/validation counts, at most 128 total")
    seen_ids, seen_prompts, candidates = set(), set(), []
    for row in rows:
        task = instance(row)
        key = task["instance_id"]
        prompt_hash = digest(task["problem_statement"].encode())
        if key in seen_ids or prompt_hash in seen_prompts:
            raise ValueError("Duplicate task identity or prompt; resolve before splitting")
        seen_ids.add(key)
        seen_prompts.add(prompt_hash)
        candidates.append((digest(f"{seed}\0{key}".encode()), key, row))
    if len(candidates) < train + validation:
        raise ValueError("Insufficient unique tasks")
    return [row for _, _, row in sorted(candidates)[: train + validation]]


def image_metadata(task: dict) -> dict:
    """Resolve the publisher's tag into an immutable linux/amd64 image reference."""
    import requests

    tag = task["instance_id"]
    if task["docker_image"] != f"{tag}:latest" or not tag.startswith("format-code-task-"):
        raise ValueError("Unknown source image naming; refuse an inferred registry mapping")
    response = requests.get(
        f"https://hub.docker.com/v2/repositories/{REGISTRY}/tags/{tag}", timeout=60
    )
    response.raise_for_status()
    metadata = response.json()
    images = [x for x in metadata["images"] if x["os"] == "linux" and x["architecture"] == "amd64"]
    if len(images) != 1 or not images[0]["digest"].startswith("sha256:"):
        raise ValueError(f"Expected one immutable linux/amd64 image for {tag}")
    image = images[0]
    return {
        "source_image": task["docker_image"],
        "published_tag": f"{REGISTRY}:{tag}",
        "resolved_image": f"{REGISTRY}@{image['digest']}",
        "compressed_bytes": image["size"],
        "platform": "linux/amd64",
    }


def bind_image(row: dict, image: dict) -> dict:
    result = copy.deepcopy(row)
    task = instance(result)
    task["docker_image"] = image["resolved_image"]
    result["extra_info"]["instance_json"] = json.dumps(task, ensure_ascii=False, sort_keys=True)
    return result


def prepare(
    output: Path, *, train: int = 64, validation: int = 8, seed: int = 42,
    candidates: Path | None = None,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a dataset index: {output}")
    info = HfApi().get_paths_info(REPO, ["code.parquet"], repo_type="dataset", revision=REVISION)[0]
    if info.size > 20_000_000:
        raise ValueError("Code source exceeds pilot download budget")
    source = Path(hf_hub_download(REPO, "code.parquet", repo_type="dataset", revision=REVISION))
    source_hash = digest(source.read_bytes())
    if not info.lfs or source_hash != info.lfs.sha256:
        raise ValueError("Source Parquet checksum does not match publisher metadata")
    table = pq.read_table(source)
    rows = table.to_pylist()
    eligibility = None
    if candidates is not None:
        eligibility = json.loads(candidates.read_text())
        ids = eligibility["instance_ids"]
        if not eligibility.get("selection_basis") or not ids or len(set(ids)) != len(ids):
            raise ValueError("Candidate selection needs a stated basis and unique task identities")
        rows = [row for row in rows if instance(row)["instance_id"] in set(ids)]
        if len(rows) != len(ids):
            raise ValueError("Candidate identities are missing from the pinned source")
    rows = select(rows, train, validation, seed)
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(image_metadata, map(instance, rows)))
    bound = [bind_image(row, image) for row, image in zip(rows, images, strict=True)]
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for position, (row, image) in enumerate(zip(rows, images, strict=True)):
        task = instance(row)
        index.append(
            {
                "split": "train" if position < train else "validation",
                "instance_id": task["instance_id"],
                "source_index": row["extra_info"]["index"],
                "prompt_sha256": digest(task["problem_statement"].encode()),
                "source_instance_sha256": digest(row["extra_info"]["instance_json"].encode()),
                "test_patch_sha256": digest(task["test_patch"].encode()),
                "cwd": task["cwd"],
                "test_command": task["test_command"],
                "verifier_timeout_sec": task["verifier_timeout_sec"],
                **image,
            }
        )
    (output / "index.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in index)
    )
    for name, subset in [("train", bound[:train]), ("validation", bound[train:])]:
        pq.write_table(
            pa.Table.from_pylist(subset, schema=table.schema), output / f"{name}.parquet"
        )
    card = Path(hf_hub_download(REPO, "README.md", repo_type="dataset", revision=REVISION))
    (output / "SOURCE_README.md").write_bytes(card.read_bytes())
    manifest = {
        "schema_version": 1,
        "dataset": REPO,
        "revision": REVISION,
        "source_file": "code.parquet",
        "source_rows": table.num_rows,
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "license": "apache-2.0",
        "image_repository_licenses": "Retained inside source images; not relabeled",
        "selection": {
            "method": "sha256(seed + NUL + instance_id)",
            "seed": seed,
            "train": train,
            "validation": validation,
            "total_limit": 128,
        },
        "transformations": ["Resolve local Docker image alias to official immutable digest"],
        "tokenization": "Deferred to native multi-turn runner; no pre-truncation or text-only conversion",
        "runtime": {
            "dataset_class": "recipes.code.dataset.MimoAgentSWEDataset",
            "runner": "recipes.code.mimoagent_runner.mimoagent_runner",
            "environment": "mimoagent.environments.datasets.opensource_code.OpenSourceCodeEnvironment",
            "tools": "bash-only",
            "reward": "test_patch applied after rollout; test_command exit 0 = 1",
            "verl_revision": "a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
            "mimoagent_revision": "467f0a19016f0ac4d63b8d17a1f0da9ba07f232c",
        },
        "qualification": "index_only: image execution and policy/tool trajectory still require qualification",
        "files": {
            p.name: {"sha256": digest(p.read_bytes()), "bytes": p.stat().st_size}
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    }
    if eligibility is not None:
        manifest["selection"]["eligibility"] = eligibility
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(manifest_path: Path) -> dict:
    """Check indexed payloads and split identity offline; does not qualify a sandbox."""
    import pyarrow.parquet as pq

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("dataset"), manifest.get("revision")) != (REPO, REVISION):
        raise ValueError("Unexpected dataset source/revision")
    expected_files = {"train.parquet", "validation.parquet", "index.jsonl", "SOURCE_README.md"}
    if set(manifest["files"]) != expected_files:
        raise ValueError("Dataset payload catalog is incomplete or unexpected")
    for name, metadata in manifest["files"].items():
        payload = (manifest_path.parent / name).read_bytes()
        if len(payload) != metadata["bytes"] or digest(payload) != metadata["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    selection = manifest["selection"]
    eligibility = selection.get("eligibility")
    if eligibility is not None:
        ids = eligibility["instance_ids"]
        if not eligibility.get("selection_basis") or not ids or len(set(ids)) != len(ids):
            raise ValueError("Invalid candidate eligibility contract")
    rows = []
    for split in ("train", "validation"):
        part = pq.read_table(manifest_path.parent / f"{split}.parquet").to_pylist()
        if len(part) != selection[split]:
            raise ValueError(f"Dataset split count mismatch: {split}")
        rows.extend(part)
    index = [
        json.loads(line) for line in (manifest_path.parent / "index.jsonl").read_text().splitlines()
    ]
    if len(index) != len(rows):
        raise ValueError("Dataset index count mismatch")
    for position, (row, entry) in enumerate(zip(rows, index, strict=True)):
        task = instance(row)
        if eligibility is not None and task["instance_id"] not in eligibility["instance_ids"]:
            raise ValueError("Selected task is outside the declared candidate set")
        expected = {
            "instance_id": task["instance_id"],
            "split": "train" if position < selection["train"] else "validation",
            "source_index": row["extra_info"]["index"],
            "prompt_sha256": digest(task["problem_statement"].encode()),
            "test_patch_sha256": digest(task["test_patch"].encode()),
            "resolved_image": task["docker_image"],
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError("Dataset index disagrees with task payload")
        if not re.fullmatch(re.escape(REGISTRY) + r"@sha256:[0-9a-f]{64}", task["docker_image"]):
            raise ValueError("Task image is not pinned to the official immutable registry")
    if select(rows, selection["train"], selection["validation"], selection["seed"]) != rows:
        raise ValueError("Task partition/order disagrees with deterministic selection")
    return {
        "dataset": REPO,
        "revision": REVISION,
        "train": selection["train"],
        "validation": selection["validation"],
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "qualification": "data_index_verified; sandbox and policy trajectory not evaluated",
    }


def audit_tokenization(manifest_path: Path, model_path: Path, prompt_limit: int,
                       *, tokenizer=None) -> dict:
    """Audit immutable prompt inputs; generated EOS belongs to rollout telemetry.

    The agent later adds its system/tool messages. These prompt-only statistics
    do not certify the length of a future multi-turn conversation.
    """
    import pyarrow.parquet as pq

    manifest_path, model_path = Path(manifest_path), Path(model_path)
    manifest = json.loads(manifest_path.read_text())
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    generation = json.loads((model_path / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    if not eos_ids or any(type(x) is not int or x < 0 for x in eos_ids) or prompt_limit < 1:
        raise ValueError("invalid native EOS or prompt budget")
    splits = {}
    for split in ("train", "validation"):
        rows = pq.read_table(manifest_path.parent / f"{split}.parquet").to_pylist()
        lengths, eos_counts = [], []
        for row in rows:
            instance(row)
            ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                                return_dict=False)
            if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("prompt audit requires actual token IDs, not tokenizer metadata")
            if not ids or len(ids) > prompt_limit:
                raise ValueError(f"{split}: empty or over-budget prompt; no implicit truncation")
            lengths.append(len(ids))
            eos_counts.append(sum(token in eos_ids for token in ids))
        splits[split] = dict(rows=len(rows), tokens=sum(lengths), minimum=min(lengths),
                             maximum=max(lengths), native_eos_tokens=sum(eos_counts),
                             prompts_with_native_eos=sum(n > 0 for n in eos_counts),
                             truncated_prompts=0, removed_tokens=0)
    assets = {p.name: digest(p.read_bytes()) for p in sorted(model_path.iterdir())
              if p.is_file() and (p.name.startswith("tokenizer") or p.name in {
                  "chat_template.jinja", "special_tokens_map.json", "generation_config.json"})}
    return dict(format="archlab-agentic-data-audit-v1", source_manifest_sha256=digest(manifest_path.read_bytes()),
                tokenizer_assets=assets, native_eos_ids=sorted(eos_ids), tokenizer_eos_id=tokenizer.eos_token_id,
                prompt_scope="native user prompt plus generation prefix; agent system/tools added at runtime",
                generated_eos_scope="measured on model-generated spans by RolloutContract",
                curriculum={"kind": "static", "selection": manifest["selection"],
                            "reward_adaptive_selection": False},
                verifier=manifest.get("runtime", {}), splits=splits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=Path, help="Explicit eligibility snapshot; recorded in manifest")
    args = parser.parse_args()
    manifest = prepare(args.output, train=args.train, validation=args.validation,
                       seed=args.seed, candidates=args.candidates)
    print(json.dumps({"output": str(args.output), "selection": {
        key: manifest["selection"][key] for key in ("train", "validation", "seed", "method")
    }}))


if __name__ == "__main__":
    main()
