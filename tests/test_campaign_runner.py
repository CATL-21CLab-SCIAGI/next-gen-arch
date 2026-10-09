import json

import pytest

from archlab.speedrun.campaign_runner import (
    Task,
    _cache_directory,
    _is_matching_complete_run,
    _task_queues,
)
from archlab.speedrun.campaigns import TEN_M_SEEDS, TEN_M_VARIANTS


def _flatten(queues):
    return [task for queue in queues for task in queue]


def test_seed_partition_is_complete_and_disjoint() -> None:
    tasks_by_node = [_flatten(_task_queues("full", node, 3, 5, "seed")) for node in range(3)]
    names = [task.name for tasks in tasks_by_node for task in tasks]

    assert len(names) == len(TEN_M_VARIANTS) * len(TEN_M_SEEDS)
    assert len(names) == len(set(names))
    assert [len(tasks) for tasks in tasks_by_node] == [16, 16, 16]
    assert [{task.seed for task in tasks} for tasks in tasks_by_node] == [
        {42},
        {43},
        {44},
    ]


def test_variant_partition_collocates_all_seeds() -> None:
    queues_by_node = [_task_queues("full", node, 3, 5, "variant") for node in range(3)]
    tasks = [task for queues in queues_by_node for task in _flatten(queues)]

    assert len(tasks) == len(TEN_M_VARIANTS) * len(TEN_M_SEEDS)
    assert len({task.name for task in tasks}) == len(tasks)
    for queues in queues_by_node:
        owners = {
            task.variant: queue_index for queue_index, queue in enumerate(queues) for task in queue
        }
        for variant, owner in owners.items():
            assert {task.seed for task in queues[owner] if task.variant == variant} == set(
                TEN_M_SEEDS
            )


def test_variant_probe_uses_one_seed_per_variant() -> None:
    tasks = _flatten(_task_queues("probe", 0, 1, 5, "variant"))

    assert len(tasks) == len(TEN_M_VARIANTS)
    assert {task.seed for task in tasks} == {TEN_M_SEEDS[0]}


def test_matching_complete_run_can_be_reused(tmp_path) -> None:
    task = Task("baseline", 42)
    (tmp_path / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "variant": {"name": "baseline"},
                "seed": 42,
                "mode": "full",
                "backend_profile": {"name": "compile"},
                "optimization_recipe": {"name": "baseline"},
            }
        ),
        encoding="utf-8",
    )

    assert _is_matching_complete_run(tmp_path, task, "full", "compile", "baseline")


def test_complete_run_contract_mismatch_fails_closed(tmp_path) -> None:
    task = Task("baseline", 42)
    (tmp_path / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "variant": {"name": "baseline"},
                "seed": 42,
                "mode": "full",
                "backend_profile": {"name": "legacy"},
                "optimization_recipe": {"name": "baseline"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="contract mismatch"):
        _is_matching_complete_run(tmp_path, task, "full", "compile", "baseline")


def test_explicit_warm_cache_root_uses_original_task_names(tmp_path) -> None:
    output = tmp_path / "output"
    cache = tmp_path / "cold-cache"
    task = Task("baseline", 42)

    assert _cache_directory(output, cache, 2, task, "seed") == cache / "baseline-seed42"
    assert _cache_directory(output, None, 2, task, "variant") == (
        output / "cache" / "node-2" / "baseline"
    )
