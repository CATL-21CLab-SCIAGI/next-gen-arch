import json

import numpy as np
import pytest

from archlab.training import distill


def _write_traj(root, step, session, dataset_index, prompt_ids):
    d = root / f"step_{step}" / f"session-{session}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps({"dataset_index": dataset_index}))
    path = d / "trajectory.npz"
    np.savez(
        path,
        traj0_prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
        traj0_response_ids=np.asarray([10, 11], dtype=np.int64),
        traj0_response_mask=np.asarray([1, 1], dtype=np.int8),
    )
    return path


def test_split_groups_by_prompt_tokens_without_cross_source_index_leakage(tmp_path):
    paths = []
    # Two distinct sources share dataset_index 0..19 with distinct prompts; rollouts of the same prompt stay together.
    for idx in range(25):
        p_a = [1000 + idx, 2000 + idx, 3000 + idx]
        p_b = [9000 + idx, 8000 + idx, 7000 + idx]
        paths.append(_write_traj(tmp_path / "src_a", 1, f"a-{idx}-r0", idx, p_a))
        paths.append(_write_traj(tmp_path / "src_a", 2, f"a-{idx}-r1", idx + 500, p_a))
        paths.append(_write_traj(tmp_path / "src_b", 1, f"b-{idx}-r0", idx, p_b))
    train, evaluation = distill.split(paths, eval_fraction=0.3)
    assert train and evaluation and len(train) + len(evaluation) == len(paths)
    train_prompts = {distill.prompt_sha256(p) for p in train}
    eval_prompts = {distill.prompt_sha256(p) for p in evaluation}
    assert not (train_prompts & eval_prompts)


def test_distill_main_refuses_non_empty_output_directory(tmp_path):
    out = tmp_path / "existing_run"
    out.mkdir()
    (out / "metrics.jsonl").write_text("{}\n")
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"output": str(out)}))
    with pytest.raises(FileExistsError, match="non-empty output directory"):
        distill.main([str(cfg)])
