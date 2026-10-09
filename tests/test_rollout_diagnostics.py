"""Opt-in host observations preserve async payloads/RNG and bound trace writes."""

import json

import pytest
import torch

from archlab.rl.async_rollout import AsyncRolloutQueue, PolicySnapshot
from archlab.rl.rollout_diagnostics import RolloutDiagnostics


@pytest.fixture
def timer(monkeypatch):
    calls = []
    monkeypatch.setattr("archlab.rl.rollout_diagnostics.faulthandler.dump_traceback_later",
                        lambda interval, **kwargs: calls.append(("arm", interval, kwargs)))
    monkeypatch.setattr("archlab.rl.rollout_diagnostics.faulthandler.cancel_dump_traceback_later",
                        lambda: calls.append(("cancel",)))
    return calls


def test_default_disabled_does_not_open_files_or_arm_timer(monkeypatch, timer):
    monkeypatch.delenv("ARCHLAB_ROLLOUT_DIAGNOSTICS_DIR", raising=False)
    assert RolloutDiagnostics.from_environment() is None
    assert not timer


def test_progress_refresh_and_end_cancel_without_cuda(tmp_path, timer, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must never touch CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    diagnostics = RolloutDiagnostics(tmp_path, rank=6, interval=30, max_dumps=2)
    try:
        diagnostics.begin(policy_version=140)
        diagnostics.mark("actor", "prefill", prompt_tokens=123, batch=4)
        diagnostics.mark("learner", "consume_wait", policy_version=141)
        diagnostics.mark("actor", "decode", token_index=512, decode_batch=2)
        state = json.loads(diagnostics.path.read_text())
        assert state["components"]["actor"]["phase"] == "decode"
        assert state["components"]["actor"]["token_index"] == 512
        assert state["components"]["learner"]["phase"] == "consume_wait"
        assert state["cuda_synchronization_added"] is False
        arms = [row for row in timer if row[0] == "arm"]
        assert len(arms) == 3  # learner observations cannot reset actor progress
        assert all(row[2]["repeat"] is False and row[2]["exit"] is False for row in arms)
        diagnostics.end()
        assert diagnostics.armed is False
    finally:
        diagnostics.close()


def test_observed_trace_writes_exhaust_bounded_dump_budget(tmp_path, timer):
    diagnostics = RolloutDiagnostics(tmp_path, rank=0, interval=30, max_dumps=2)
    try:
        diagnostics.begin()
        diagnostics.trace.write(b"first stalled actor traceback\n")
        diagnostics.mark("actor", "decode", token_index=512)
        assert diagnostics.dumps == 1 and diagnostics.armed
        diagnostics.trace.write(b"second stalled actor traceback\n")
        diagnostics.mark("actor", "decode", token_index=1024)
        assert diagnostics.dumps == 2 and not diagnostics.armed
        before = len(timer)
        diagnostics.mark("actor", "decode", token_index=1536)
        assert len(timer) == before
    finally:
        diagnostics.close()


def test_optional_io_failure_disables_observation_not_sampling(tmp_path, timer, monkeypatch):
    diagnostics = RolloutDiagnostics(tmp_path, rank=1, interval=30)
    monkeypatch.setattr("archlab.rl.rollout_diagnostics.atomic_write_json",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("NAS unavailable")))
    try:
        diagnostics.begin()
        assert diagnostics.failed and not diagnostics.armed
        diagnostics.mark("actor", "decode", token_index=512)
    finally:
        diagnostics.close()


def test_environment_contract_and_no_cuda_initialization(tmp_path, timer, monkeypatch):
    monkeypatch.setenv("ARCHLAB_ROLLOUT_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("ARCHLAB_ROLLOUT_DIAGNOSTICS_INTERVAL", "60")
    monkeypatch.setenv("ARCHLAB_ROLLOUT_DIAGNOSTICS_MAX_DUMPS", "3")
    before = torch.cuda.is_initialized()
    diagnostics = RolloutDiagnostics.from_environment()
    try:
        assert diagnostics.rank == 6 and diagnostics.interval == 60 and diagnostics.max_dumps == 3
        assert torch.cuda.is_initialized() == before
    finally:
        diagnostics.close()


@pytest.mark.parametrize("interval,max_dumps", [(0, 4), (float("nan"), 4), (30, 0), (30, 17)])
def test_invalid_limits_reject_before_open(tmp_path, interval, max_dumps):
    with pytest.raises(ValueError, match="limits"):
        RolloutDiagnostics(tmp_path, rank=0, interval=interval, max_dumps=max_dumps)
    assert not list(tmp_path.iterdir())


def test_async_payload_and_rng_identical_with_observation(tmp_path, timer):
    def run(diagnostics):
        generator = torch.Generator().manual_seed(123)

        def generate(prompts, snapshot, rng):
            return dict(tokens=torch.randint(100, (4,), generator=rng).tolist(), finish_reason=["eos"]), {}

        queue = AsyncRolloutQueue(generate, generator, PolicySnapshot(3, None), lambda: False,
                                  diagnostics=diagnostics)
        try:
            queue.prefetch(["prompt"])
            result, timing = queue.consume(["prompt"], 4)
            return result, generator.get_state().clone(), set(timing)
        finally:
            queue.close()

    plain, plain_rng, plain_keys = run(None)
    observed, observed_rng, observed_keys = run(RolloutDiagnostics(tmp_path, rank=0, interval=30))
    assert observed == plain
    assert torch.equal(observed_rng, plain_rng)
    assert observed_keys == plain_keys


def test_async_exception_cancels_timer_and_close_releases_file(tmp_path, timer):
    diagnostics = RolloutDiagnostics(tmp_path, rank=0, interval=30)

    def fail(*args):
        raise RuntimeError("generation failed")

    queue = AsyncRolloutQueue(fail, torch.Generator(), PolicySnapshot(0, None), lambda: False,
                              diagnostics=diagnostics)
    queue.prefetch(["prompt"])
    with pytest.raises(RuntimeError, match="generation failed"):
        queue.consume(["prompt"], 0)
    assert not diagnostics.armed
    with pytest.raises(RuntimeError, match="generation failed"):
        queue.close()
    assert diagnostics.trace.closed


def test_environment_missing_directory_is_optional(monkeypatch, timer):
    monkeypatch.setenv("ARCHLAB_ROLLOUT_DIAGNOSTICS_DIR", "/dev/null/not-a-directory")
    assert RolloutDiagnostics.from_environment() is None


def test_generate_progress_uses_host_shapes_not_tensor_values():
    # Source-level contract complements the CPU no-CUDA/RNG oracle: progress
    # never reads a CUDA scalar or invokes synchronization in the token loop.
    import ast
    import inspect

    from archlab.rl import limite_generation

    tree = ast.parse(inspect.getsource(limite_generation.graph_generate))
    marks = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "mark"]
    assert len(marks) == 4
    for node in marks:
        attributes = [call.func.attr for call in ast.walk(node) if isinstance(call, ast.Call)
                      and isinstance(call.func, ast.Attribute)]
        assert not set(attributes).intersection({"item", "tolist", "cpu", "synchronize", "sum"})
