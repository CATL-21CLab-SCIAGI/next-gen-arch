from __future__ import annotations

import pytest

from archlab.megatron.collective_probe import validate_topology


def test_validate_topology_accepts_four_nodes_with_eight_ranks_each() -> None:
    hosts = [f"node-{node}" for node in range(4) for _ in range(8)]
    topology = validate_topology(hosts, world_size=32, expected_nodes=4, gpus_per_node=8)
    assert topology == {f"node-{node}": 8 for node in range(4)}


def test_validate_topology_rejects_rank_placement_drift() -> None:
    with pytest.raises(RuntimeError, match="ranks per host"):
        validate_topology(
            ["node-a"] * 7 + ["node-b"] * 9,
            world_size=16,
            expected_nodes=2,
            gpus_per_node=8,
        )
