import json
import sys
from pathlib import Path

import pytest

from archlab.automodel.limite_adapter_queue import check_handoff, main


@pytest.mark.parametrize(
    "tokens,ready_passed,expected",
    [
        (10_000_000_000, True, ["warmup", "rl"]),
        (123, True, ["warmup"]),
        (10_000_000_000, False, ["warmup"]),
    ],
)
@pytest.mark.parametrize("rl_status", ["complete", "stopped"])
def test_handoff_requires_exact_budget_and_qualified_source(
    tmp_path, monkeypatch, tokens, ready_passed, expected, rl_status
):
    output = tmp_path / "normal"
    output.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "SOURCE_REVISION").write_text("pinned\n")
    spec = dict(source=str(source), revision="pinned", port=12345, module="warmup", arguments=[])
    ready = tmp_path / "RL_READY.json"
    ready.write_text(
        json.dumps(dict(passed=ready_passed, variants=dict(normal=dict(spec, module="rl"))))
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                output=str(output),
                variant="normal",
                master_addr="localhost",
                runtime_overlay="overlay",
                warmup=spec,
                rl_ready=str(ready),
            )
        )
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command[-1])
        assert kwargs["env"]["ARCHLAB_SOURCE_REVISION"] == "pinned"
        assert kwargs["cwd"] == source
        if calls[-1] == "warmup":
            (output / "WARMUP_COMPLETE.json").write_text(json.dumps(dict(tokens=tokens)))
        else:
            (output / "rl").mkdir()
            (output / "rl" / "FINISHED.json").write_text(json.dumps(dict(status=rl_status)))

    monkeypatch.setattr("archlab.automodel.limite_adapter_queue.subprocess.run", run)
    monkeypatch.setattr(sys, "argv", ["queue", "--plan", str(plan), "--node-rank", "0"])
    if tokens != 10_000_000_000 or not ready_passed:
        with pytest.raises(ValueError):
            main()
    else:
        main()
        status = json.loads((output / "queue-node0.json").read_text())["status"]
        assert status == ("complete" if rl_status == "complete" else "paused")
    assert calls == expected


def test_completed_rl_queue_does_not_launch_the_finished_run_again(tmp_path, monkeypatch):
    output = tmp_path / "normal"
    (output / "rl").mkdir(parents=True)
    (output / "WARMUP_COMPLETE.json").write_text(json.dumps(dict(tokens=10_000_000_000)))
    (output / "rl" / "FINISHED.json").write_text(json.dumps(dict(status="complete")))
    ready = tmp_path / "READY.json"
    ready.write_text(json.dumps(dict(passed=True, variants=dict(normal=dict()))))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(output=str(output), variant="normal", rl_ready=str(ready))))

    def forbidden(*args, **kwargs):
        raise AssertionError("completed queue relaunched training")

    monkeypatch.setattr("archlab.automodel.limite_adapter_queue.subprocess.run", forbidden)
    monkeypatch.setattr(sys, "argv", ["queue", "--plan", str(plan), "--node-rank", "0"])
    main()
    assert json.loads((output / "queue-node0.json").read_text())["status"] == "complete"


@pytest.fixture
def full_handoff(tmp_path):
    checkpoint = tmp_path / "full-checkpoint"
    checkpoint.mkdir()
    parent = dict(
        tokens=10_000_000_000,
        trainable_mode="full",
        adapter=dict(variant="simplicial", attention_backend="tilelang"),
    )
    (checkpoint / "COMPLETE.json").write_text(json.dumps(parent))
    plan = dict(
        output=str(tmp_path / "simplicial"),
        variant="simplicial",
        trainable_mode="full",
        attention_backend="tilelang",
        warmup=dict(revision="matched-full-source"),
    )
    marker = dict(tokens=10_000_000_000, checkpoint=str(checkpoint))
    ready = dict(
        passed=True,
        trainable_mode="full",
        attention_backend="tilelang",
        variants=dict(
            simplicial=dict(
                revision="matched-full-source",
                arguments=[
                    "--trainable-mode", "full", "--attention-backend", "tilelang",
                    "--variant", "simplicial", "--warmup", plan["output"],
                ]
            )
        ),
    )
    return plan, marker, ready, parent


def test_full_handoff_accepts_own_completed_full_parent(full_handoff):
    plan, marker, ready, _ = full_handoff
    check_handoff(plan, marker, ready)


@pytest.mark.parametrize("field,value", [("tokens", 2_000_158_720), ("trainable_mode", "adapter")])
def test_full_handoff_rejects_adapter_or_partial_parent(full_handoff, field, value):
    plan, marker, ready, parent = full_handoff
    parent[field] = value
    (Path(marker["checkpoint"]) / "COMPLETE.json").write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="parent differs"):
        check_handoff(plan, marker, ready)


def test_full_handoff_rejects_other_variant_parent(full_handoff):
    plan, marker, ready, parent = full_handoff
    parent["adapter"]["variant"] = "normal"
    (Path(marker["checkpoint"]) / "COMPLETE.json").write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="parent differs"):
        check_handoff(plan, marker, ready)


def test_full_handoff_rejects_stale_adapter_ready(full_handoff):
    plan, marker, ready, _ = full_handoff
    ready["trainable_mode"] = "adapter"
    with pytest.raises(ValueError, match="qualification differs"):
        check_handoff(plan, marker, ready)


def test_full_handoff_rejects_other_variant_warmup_arguments(full_handoff):
    plan, marker, ready, _ = full_handoff
    ready["variants"]["simplicial"]["arguments"][-1] = "other-warmup"
    with pytest.raises(ValueError, match="arguments differ"):
        check_handoff(plan, marker, ready)


def test_full_handoff_rejects_other_qualified_revision(full_handoff):
    plan, marker, ready, _ = full_handoff
    ready["variants"]["simplicial"]["revision"] = "other-source"
    with pytest.raises(ValueError, match="sealed finetuning revision"):
        check_handoff(plan, marker, ready)


def test_explicit_qualified_rl_upgrade_preserves_historical_warmup_revision(full_handoff):
    plan, marker, ready, _ = full_handoff
    plan["rl_revision"] = "qualified-rl-fix"
    ready["variants"]["simplicial"]["revision"] = "qualified-rl-fix"
    check_handoff(plan, marker, ready)
    assert plan["warmup"]["revision"] == "matched-full-source"


def test_eval_gate_requires_both_completed_pinned_stages(tmp_path):
    from archlab.automodel.limite_adapter_queue import evaluation_ready, release_evaluation_pause

    state = tmp_path / "state.json"
    pause = tmp_path / "STOP_REQUEST"
    gate = dict(state=str(state), stages={"base": "base-sha", "violetto": "violetto-sha"},
                pause_file=str(pause), pause_contents="wait for evaluation\n")
    assert not evaluation_ready(gate)
    data = dict(status="running", stages={"base": dict(status="complete", plan_sha256="base-sha")})
    state.write_text(json.dumps(data))
    assert not evaluation_ready(gate)
    data["stages"]["violetto"] = dict(status="complete", plan_sha256="wrong")
    state.write_text(json.dumps(data))
    assert not evaluation_ready(gate)
    data["stages"]["violetto"]["plan_sha256"] = "violetto-sha"
    state.write_text(json.dumps(data))
    assert evaluation_ready(gate)
    pause.write_text("new user stop\n")
    with pytest.raises(ValueError, match="pause request changed"):
        release_evaluation_pause(gate)
    pause.write_text(gate["pause_contents"])
    release_evaluation_pause(gate)
    release_evaluation_pause(gate)  # Second node observes the already released pause.
    assert not pause.exists()
    data["status"] = "memory_overflow"
    state.write_text(json.dumps(data))
    assert not evaluation_ready(gate)


@pytest.mark.parametrize("blocked", ["running", "failed", "memory_overflow", "wrong_hash", "missing"])
def test_composite_evaluation_gate_requires_every_controller(tmp_path, blocked):
    from archlab.automodel.limite_adapter_queue import evaluation_ready

    gates = []
    for index in range(4):
        state = tmp_path / f"state-{index}.json"
        gates.append(dict(state=str(state), stages={"evaluation": f"hash-{index}"}))
        state.write_text(json.dumps(dict(
            status="complete", stages={"evaluation": dict(status="complete", plan_sha256=f"hash-{index}")}
        )))
    gate = dict(gates=gates)
    assert evaluation_ready(gate)
    state = Path(gates[2]["state"])
    data = json.loads(state.read_text())
    if blocked == "missing":
        state.unlink()
    else:
        if blocked in ("failed", "memory_overflow"):
            data["status"] = blocked
        elif blocked == "wrong_hash":
            data["stages"]["evaluation"]["plan_sha256"] = "wrong"
        else:
            data["stages"]["evaluation"]["status"] = "running"
        state.write_text(json.dumps(data))
    assert not evaluation_ready(gate)


@pytest.mark.parametrize("gates", [[], None, {}])
def test_composite_gate_rejects_empty_or_invalid_list(gates):
    from archlab.automodel.limite_adapter_queue import evaluation_ready

    with pytest.raises(ValueError, match="nonempty gates list"):
        evaluation_ready(dict(gates=gates))
