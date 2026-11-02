import numpy as np

from archlab.evaluation.trajectory_replay import replay_window


def test_replay_keeps_identical_context_and_masks_observations(tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(path, traj0_prompt_ids=np.arange(10), traj0_response_ids=np.array([10, 11, 12, 13]),
             traj0_response_mask=np.array([1, 0, 0, 1], dtype=np.int8))
    ids, mask, start = replay_window(path, 8)
    assert start == 6
    assert ids.tolist() == list(range(6, 14))
    assert mask.tolist() == [0, 0, 0, 0, 1, 0, 0, 1]
