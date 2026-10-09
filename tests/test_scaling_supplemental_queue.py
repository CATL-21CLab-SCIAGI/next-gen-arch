import hashlib
import json
from pathlib import Path

import pytest

from archlab.automodel.deepseek_v41_scaling_campaign import (
    finish_handoff,
    phase_environment,
    training_queue,
)


def test_supplemental_plan_is_read_at_boundary_and_does_not_block_other_arm(tmp_path):
    runs = []
    for variant, nodes in (("normal", ["a", "b"]), ("simplicial", ["c", "d"])):
        for i, width in enumerate((640, 128, 384, 1280)):
            runs.append(
                dict(
                    variant=variant,
                    nodes=nodes,
                    width=width,
                    queue_index=i,
                    master_port=28000 + i * 2,
                    output=str(tmp_path / f"{variant}-{width}"),
                    qualification=str(tmp_path / f"q-{variant}-{width}"),
                )
            )
    extension = tmp_path / "supplemental.json"
    plan = dict(
        runs=runs,
        nodes=dict.fromkeys("abcd"),
        token_budget_per_run=10_000_000_000,
        supplemental_plan_path=str(extension),
    )
    queue = training_queue(plan, "a")
    assert next(queue)["width"] == 640
    assert next(queue)["width"] == 128
    # A future experiment can be prepared while d640/d128 are still training.
    extra = {
        **runs[1],
        "supplemental": True,
        "master_port": 29000,
        "output": str(tmp_path / "extra"),
        "qualification": str(tmp_path / "q-extra"),
        "training_token_budget": 1_000_000_000,
    }
    extension.write_text(json.dumps({"runs": [extra]}))
    assert next(queue) == extra
    assert [r["width"] for r in queue] == [384, 1280]
    extension.unlink()
    assert [r["width"] for r in training_queue(plan, "c")] == [640, 128, 384, 1280]


def test_run_specific_source_environment_preserves_qualified_base_contract():
    plan = dict(environment={"PYTHONPATH": "/new/src"}, source_commit="new")
    run = dict(environment={"PYTHONPATH": "/original/src"}, source_commit="qualified")
    result = phase_environment(plan, run)
    assert result["PYTHONPATH"] == "/original/src"
    assert result["NGA_EXPECTED_COMMIT"] == "qualified"


@pytest.mark.parametrize("exit_status", [0, 256, 9])
@pytest.mark.parametrize("mode", ["qualify", "train"])
@pytest.mark.parametrize("variant", ["normal", "simplicial"])
@pytest.mark.parametrize("peer_archived", [False, True])
@pytest.mark.parametrize("checkpointed_teardown", [False, True])
def test_handoff_requires_real_successful_child_exit_before_retiring_parent(
    tmp_path, monkeypatch, exit_status, mode, variant, peer_archived, checkpointed_teardown
):
    from archlab.automodel import deepseek_v41_scaling_campaign as campaign

    qualification = tmp_path / "qualification"
    qualification.mkdir()
    (qualification / "QUALIFIED.json").write_text('{"passed":true,"contract":{"version":"pinned"}}')
    output = tmp_path / "training"
    recovery = output / "recovery" / "step-000200"
    recovery.mkdir(parents=True)
    (recovery / "COMPLETE.json").write_text(
        '{"contract":{"version":"pinned"},"cursor":{"step":200}}'
    )
    stop_bytes = b'{"purpose":"loop qualification"}'
    (output / "STOP_REQUEST").write_bytes(stop_bytes)
    (output / "rank-00-initial.json").write_text('{"step":0}')
    handoff = dict(
        worker_pid=101,
        phase_pid=102,
        worker_start_ticks="123",
        phase_start_ticks="456",
        mode=mode,
        variant=variant,
        width=1280,
        minimum_step=200,
        stop_request_sha256=hashlib.sha256(stop_bytes).hexdigest(),
    )
    if checkpointed_teardown:
        handoff["checkpointed_teardown"] = dict(
            phase_pid=102,
            ranks=[dict(pid=200+i, in_destroy=True, returncode=0) for i in range(8)],
            verification={"ranks": 16},
        )
    (tmp_path / "LOOP_HANDOFF-a.json").write_text(json.dumps(handoff))
    parent, child = ["0"] * 52, ["0"] * 52
    parent[2], parent[21] = "T", "123"
    child[2], child[3], child[21], child[51] = "Z", "101", "456", str(exit_status)
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/101/stat":
            return " ".join(parent)
        if str(path) == "/proc/102/stat":
            return " ".join(child)
        return original(path, *args, **kwargs)

    if peer_archived:
        original_rename = Path.rename

        def peer_rename(path, target):
            original_rename(path, target)
            raise FileNotFoundError("peer already archived the identical request")

        monkeypatch.setattr(Path, "rename", peer_rename)
    signals = []
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(campaign.os, "kill", lambda *args: signals.append(args))
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    plan = dict(
        root=str(tmp_path),
        runs=[
            dict(variant=variant, width=1280, qualification=str(qualification), output=str(output))
        ],
    )
    if exit_status and not (checkpointed_teardown and mode == "train"):
        with pytest.raises(RuntimeError, match="qualification exited"):
            finish_handoff(plan, "a")
        assert signals == []
        assert (output / "STOP_REQUEST").read_bytes() == stop_bytes
    else:
        finish_handoff(plan, "a")
        assert signals == [(101, campaign.signal.SIGTERM), (101, campaign.signal.SIGCONT)]
        assert json.loads((tmp_path / "LOOP_HANDOFF-a.json").read_text())["phase_exit_code"] == campaign.os.waitstatus_to_exitcode(exit_status)
        if mode == "train":
            assert plan["runs"][0].pop("resume_checkpoint") == str(recovery)
            assert not (output / "STOP_REQUEST").exists()
            assert (output / "MAINTENANCE_STOP-200.json").read_bytes() == stop_bytes
            assert (output / "INITIALIZATION_BEFORE_MAINTENANCE" / "rank-00-initial.json").is_file()
            finish_handoff(plan, "a")
            assert plan["runs"][0]["resume_checkpoint"] == str(recovery)
            assert len(signals) == 2


def test_d128_priority_preserves_sweep_insertion_and_remaining_runs(tmp_path):
    runs = []
    for variant, nodes in (("normal", ["a", "b"]), ("simplicial", ["c", "d"])):
        for i, width in enumerate((640, 128, 384, 1280)):
            runs.append(dict(variant=variant, nodes=nodes, width=width, queue_index=i,
                             master_port=28000+i*2, output=str(tmp_path/f"{variant}-{width}"),
                             qualification=str(tmp_path/f"q-{variant}-{width}")))
    extension = tmp_path / "supplemental.json"
    extra = {**runs[1], "supplemental": True, "master_port": 29000,
             "output": str(tmp_path/"extra"), "qualification": str(tmp_path/"q-extra"),
             "training_token_budget": 1_000_000_000}
    extension.write_text(json.dumps({"runs": [extra]}))
    plan = dict(runs=runs, nodes=dict.fromkeys("abcd"), token_budget_per_run=10_000_000_000,
                supplemental_plan_path=str(extension), production_width_order=[128,640,384,1280])
    normal = list(training_queue(plan, "a"))
    assert [r["width"] for r in normal] == [128,128,640,384,1280]
    assert normal[1] == extra
    assert [r["width"] for r in training_queue(plan, "c")] == [128,640,384,1280]
    plan["production_width_order"] = [128,128,384,1280]
    with pytest.raises(ValueError, match="every base width"):
        list(training_queue(plan, "a"))


def test_natural_completion_handoff_never_signals_child_and_requires_five_checkpoints(
    tmp_path, monkeypatch
):
    from archlab.automodel import deepseek_v41_scaling_campaign as campaign

    qualification = tmp_path / "q"
    qualification.mkdir()
    (qualification / "QUALIFIED.json").write_text('{"passed":true}')
    output = tmp_path / "out"
    output.mkdir()
    (output / "COMPLETE.json").write_text('{"passed":true,"supervised_tokens":10000000000}')
    handoff = dict(worker_pid=101, phase_pid=102, worker_start_ticks="123",
                   phase_start_ticks="456", mode="complete", variant="normal", width=128)
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(handoff))
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) in ("/proc/101/stat", "/proc/102/stat"):
            fields = ["0"] * 52
            fields[2], fields[3], fields[21] = (
                ("T", "0", "123") if str(path) == "/proc/101/stat" else ("Z", "101", "456")
            )
            return " ".join(fields)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    signals = []
    monkeypatch.setattr(campaign.os, "kill", lambda *args: signals.append(args))
    run = dict(variant="normal", width=128, qualification=str(qualification), output=str(output))
    plan = dict(root=str(tmp_path), runs=[run])
    with pytest.raises(FileNotFoundError):
        campaign.finish_handoff(plan, "a", handoff_path=path)
    assert signals == []
    rows = []
    for i in range(1, 6):
        target = tmp_path / f"oss-{i}"
        target.mkdir()
        link = output / f"checkpoint-{i}"
        link.symlink_to(target, target_is_directory=True)
        rows.append(dict(path=str(link), oss_path=str(target), milestone_tokens=i*2_000_000_000))
    (output / "EVAL_CHECKPOINTS.json").write_text(json.dumps({"checkpoints": rows}))
    campaign.finish_handoff(plan, "a", handoff_path=path)
    assert signals == [(101, campaign.signal.SIGTERM), (101, campaign.signal.SIGCONT)]
    assert "resume_checkpoint" not in run
    assert json.loads(path.read_text())["finished"]


def test_revised_pair_qualifies_both_variants_on_own_nodes_without_peer_wait(tmp_path, monkeypatch):
    from archlab.automodel import deepseek_v41_scaling_campaign as campaign

    normal = dict(variant="normal", width=1280, nodes=["a", "b"], master_port=28000,
                  qualification=str(tmp_path / "normal"))
    simplicial = {**normal, "variant": "simplicial", "nodes": ["c", "d"],
                  "qualification": str(tmp_path / "simplicial")}
    plan = dict(runs=[normal, simplicial], environment={}, source_commit="fixed")
    calls = []
    monkeypatch.setattr(campaign, "run_phase", lambda p, r, n, m, e, s: calls.append(r))
    monkeypatch.setattr(campaign, "validate_qualification_pair", lambda *pair: pair)
    pair = campaign.qualify_local_pair(plan, simplicial, "c", tmp_path / "state")
    assert [r["variant"] for r in calls] == ["normal", "simplicial"]
    assert all(r["nodes"] == ["c", "d"] and r["master_port"] == 28000 for r in calls)
    assert pair[0]["qualification"] != normal["qualification"]
    assert normal["nodes"] == ["a", "b"]
    normal["source_commit"] = "wrong"
    with pytest.raises(ValueError, match="source_commit"):
        campaign.qualify_local_pair(plan, simplicial, "c", tmp_path / "state")
