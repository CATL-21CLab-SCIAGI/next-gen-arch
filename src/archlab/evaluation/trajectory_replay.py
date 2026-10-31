"""Paired, generated-token NLL on fixed windows of saved agent trajectories.

This is an offline diagnostic with reset state at each window, not a measurement
of executable agent success or long-context serving. Run with a JSON contract.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from archlab.artifacts import sha256_file


def replay_window(path: Path, length: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep up to half a window of the initial prompt and score only generated targets."""
    if length < 2:
        raise ValueError("Window needs at least two tokens")
    with np.load(path, allow_pickle=False) as data:
        prompt, response, generated = (data[key] for key in
                                       ("traj0_prompt_ids", "traj0_response_ids", "traj0_response_mask"))
        if len(response) != len(generated) or not np.isin(generated, [0, 1]).all():
            raise ValueError("Invalid saved generation mask")
        start = max(0, len(prompt) - length // 2)
        ids = np.concatenate((prompt, response))[start:start + length].astype(np.int64)
        mask = np.concatenate((np.zeros(len(prompt), dtype=np.int8), generated))[start:start + length].copy()
    mask[0] = 0  # no predecessor exists for the first token of a reset window
    if len(ids) < 2 or not mask.any():
        raise ValueError("Window has no generated targets")
    return ids, mask, start


def main(argv=None):
    import torch
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModelForCausalLM

    from archlab.architectures.qwen35_switch import SwitchConfig
    from archlab.training.latent_match import text_model
    from archlab.training.stage2_distill import install_student_switches

    config_path = Path((argv or sys.argv[1:])[0])
    config = json.loads(config_path.read_text())
    output = Path(config["output"])
    output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    windows, membership = [], []
    for item in config["examples"]:
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Replay input changed")
        ids, mask, start = replay_window(path, config["sequence_length"])
        windows.append((ids, mask))
        membership.append({**item, "start": start, "length": len(ids), "targets": int(mask.sum())})
    (output / "membership.json").write_text(json.dumps(membership, indent=2) + "\n")
    torch.cuda.set_device(0)
    torch.manual_seed(config["seed"])
    geometry = AutoConfig.from_pretrained(config["model"], local_files_only=True).get_text_config()
    all_results = {}
    for arm in config["arms"]:
        arm_start = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(config["model"], dtype=torch.bfloat16,
                                                    device_map={"": "cpu"}, local_files_only=True)
        model.config.use_cache = False
        checkpoint = arm.get("checkpoint")
        if checkpoint:
            if sha256_file(checkpoint) != arm["checkpoint_sha256"]:
                raise ValueError("Replay checkpoint changed")
            install_student_switches(model, geometry.layer_types, arm["variant"],
                                     SwitchConfig(**arm["switch"]), seed=config["seed"], stage1_ckpt=checkpoint)
        model.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
        head, rows = model.get_output_embeddings(), []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for item, (ids, mask) in zip(membership, windows, strict=True):
                tokens = torch.tensor(ids, device="cuda")[None]
                targets = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
                hidden = text_model(model)(input_ids=tokens, use_cache=False).last_hidden_state[0, :-1]
                positions = targets.nonzero().flatten()
                loss_sum = 0.0
                for positions_chunk in positions.split(config["head_chunk_tokens"]):
                    logits = head(hidden[positions_chunk]).float()
                    loss_sum += float(F.cross_entropy(logits, tokens[0, positions_chunk + 1], reduction="sum"))
                row = dict(task=item["task"], seed=item["seed"], targets=len(positions), nll_sum=loss_sum,
                           nll=loss_sum / len(positions))
                rows.append(row)
                print(json.dumps({"arm": arm["label"], **row}), flush=True)
        torch.cuda.synchronize()
        result = dict(arm=arm, examples=rows, targets=sum(r["targets"] for r in rows),
                      nll=sum(r["nll_sum"] for r in rows) / sum(r["targets"] for r in rows),
                      elapsed_seconds=time.monotonic() - arm_start)
        all_results[arm["label"]] = result
        (output / f'{arm["label"]}.json').write_text(json.dumps(result, indent=2) + "\n")
        del model, head, hidden, logits
        gc.collect()
        torch.cuda.empty_cache()
    baseline = all_results[config["arms"][0]["label"]]
    for result in all_results.values():
        delta = np.array([b["nll"] - a["nll"] for a, b in
                          zip(baseline["examples"], result["examples"], strict=True)])
        draws = np.random.default_rng(42).choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
        result["task_paired_mean_nll_delta"] = float(delta.mean())
        result["task_paired_95ci"] = np.quantile(draws, [.025, .975]).tolist()
    summary = dict(complete=True, measurement="offline window NLL; lower is better; not agent success",
                   contract_sha256=sha256_file(config_path), results=all_results,
                   elapsed_seconds=time.monotonic() - started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
