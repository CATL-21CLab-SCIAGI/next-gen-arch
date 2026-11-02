import json

from archlab.rl import rollouts


def session(root, step, index, task, win):
    path = root / f"step_{step}" / f"session-{index}"
    path.mkdir(parents=True)
    (path / "session.json").write_text(json.dumps({"dataset_index": task, "reward_score": 1.0 if win else 0.0,
                                                   "num_turns": 3}))


def test_learning_compares_later_visits_with_the_first_visit_per_task(tmp_path):
    for i, win in enumerate([False, False]):          # task 0: 0/2 first, then 2/2
        session(tmp_path, 1, f"a{i}", 0, win)
    for i, win in enumerate([True, True]):
        session(tmp_path, 5, f"b{i}", 0, win)
    for i, win in enumerate([True, False]):           # task 1: 1/2 first, then 1/2
        session(tmp_path, 2, f"c{i}", 1, win)
    for i, win in enumerate([True, False]):
        session(tmp_path, 6, f"d{i}", 1, win)
    session(tmp_path, 3, "e0", 2, True)               # task 2: one visit only, not counted
    result = rollouts.learning(rollouts.sessions(tmp_path))
    assert result["tasks_revisited"] == 2 and result["later_minus_first"] == 0.5
    assert rollouts.learning(rollouts.sessions(tmp_path, 4, 9))["tasks_revisited"] == 0  # one window per data period


def test_waste_and_learnable_rows(tmp_path):
    session(tmp_path, 1, "a", 0, False)
    session(tmp_path, 1, "b", 0, False)
    session(tmp_path, 1, "c", 1, True)
    rows = rollouts.sessions(tmp_path)
    assert rollouts.waste(rows)["all_fail_group_share"] == 0.5
    kept = rollouts.learnable_rows(rows, total=4, label="x", recheck=0.0)
    assert kept == [1, 2, 3]  # task 0 failed its last visit; 2 and 3 were never visited
