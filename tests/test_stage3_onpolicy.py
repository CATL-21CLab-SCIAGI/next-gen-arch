"""Tests for Stage-3 on-policy reverse-KL distillation."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_stage2_distill import TinyCausalLM

from archlab.architectures.qwen35_switch import SwitchConfig
from archlab.training import stage2_distill as s2
from archlab.training import stage3_onpolicy as s3
from archlab.training.latent_match import SourceConfig, rng_state


def test_wsd_lr_schedule_and_curriculum_doubling():
    peak = 2e-5
    # Warmup over first 100 of 1000 tokens, stable over [100, 800], linear decay over [800, 1000] to 10%
    assert abs(s3.wsd_lr_at(50, 1000, peak, warmup_fraction=0.1, decay_fraction=0.2, final_lr_fraction=0.1) - 1e-5) < 1e-12
    assert abs(s3.wsd_lr_at(100, 1000, peak, warmup_fraction=0.1, decay_fraction=0.2, final_lr_fraction=0.1) - peak) < 1e-12
    assert abs(s3.wsd_lr_at(500, 1000, peak, warmup_fraction=0.1, decay_fraction=0.2, final_lr_fraction=0.1) - peak) < 1e-12
    mid_decay = peak * 0.55
    assert abs(s3.wsd_lr_at(900, 1000, peak, warmup_fraction=0.1, decay_fraction=0.2, final_lr_fraction=0.1) - mid_decay) < 1e-12
    assert abs(s3.wsd_lr_at(1000, 1000, peak, warmup_fraction=0.1, decay_fraction=0.2, final_lr_fraction=0.1) - peak * 0.1) < 1e-12

    cur = s3.CurriculumState(gen_tokens=64)
    assert not cur.update(1.0, max_gen_tokens=256, ema_alpha=1.0, improvement_rel=0.02, plateau_steps=3)
    # 1% improvement (< 2%) counts as stale step 1, 2, 3 -> advances to 128
    assert not cur.update(0.99, max_gen_tokens=256, ema_alpha=1.0, improvement_rel=0.02, plateau_steps=3)
    assert not cur.update(0.985, max_gen_tokens=256, ema_alpha=1.0, improvement_rel=0.02, plateau_steps=3)
    assert cur.update(0.985, max_gen_tokens=256, ema_alpha=1.0, improvement_rel=0.02, plateau_steps=3)
    assert cur.gen_tokens == 128 and cur.phase == 1 and cur.stale_steps == 0 and cur.best_smoothed_kl is None

    # Non-power-of-two max_gen_tokens or indivisible gen_batch_tokens must fail validation
    with pytest.raises(ValueError):
        s3.Stage3Config(
            teacher="t",
            init_checkpoint="c",
            output="o",
            variant="simplicial",
            sources=(SourceConfig("m.json", 1.0),),
            prompt_length=16,
            min_gen_tokens=64,
            max_gen_tokens=192,
            gen_batch_tokens=384,
            tokens=768,
            eval_windows=2,
        )


@pytest.mark.parametrize("tied", [False, True])
def test_chunked_reverse_kl_matches_unchunked_autograd_and_excludes_prompt(tied):
    torch.manual_seed(17)
    teacher = TinyCausalLM(tied=tied).eval().requires_grad_(False)
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}

    student_a = copy.deepcopy(teacher).train()
    switches_a = s2.install_student_switches(
        student_a,
        ["full_attention", "full_attention"],
        "simplicial",
        SwitchConfig(backend="reference", window_1=None, window_2=3),
        seed=23,
    )
    opt_a, params_a, _ = s2.build_two_timescale_optimizer(student_a, switches_a, lr=1e-3, backbone_lr_ratio=1.0)

    student_b = copy.deepcopy(student_a).train()
    switches_b = {0: student_b.model.layers[0].self_attn.switch, 1: student_b.model.layers[1].self_attn.switch}
    opt_b, params_b, _ = s2.build_two_timescale_optimizer(student_b, switches_b, lr=1e-3, backbone_lr_ratio=1.0)

    prompt_len = 4
    gen_len = 5
    seqs = torch.randint(0, 19, (4, prompt_len + gen_len))
    res_a = s3.chunked_reverse_kl_step(
        teacher,
        student_a,
        opt_a,
        params_a,
        seqs,
        prompt_length=prompt_len,
        microbatch_prompts=2,
        chunk_tokens=2,
        clip_norm=10.0,
    )

    opt_b.zero_grad(set_to_none=True)
    with torch.no_grad():
        t_h = teacher.model(seqs).last_hidden_state[:, prompt_len - 1 : prompt_len + gen_len - 1]
        t_logp = teacher.lm_head(t_h).float().log_softmax(-1)
    s_h = student_b.model(seqs).last_hidden_state[:, prompt_len - 1 : prompt_len + gen_len - 1]
    s_logp = student_b.lm_head(s_h).float().log_softmax(-1)
    loss = (s_logp.exp() * (s_logp - t_logp)).sum() / (seqs.shape[0] * gen_len)
    loss.backward()
    norm_b = torch.nn.utils.clip_grad_norm_(params_b, 10.0)
    opt_b.step()

    assert abs(res_a["train_reverse_kl"] - float(loss.detach())) < 1e-6
    assert abs(res_a["grad_norm"] - float(norm_b)) < 1e-5
    for pa, pb in zip(params_a, params_b, strict=True):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_before[k])


def test_stage3_onpolicy_rollout_reduces_reverse_kl_and_roundtrips_checkpoint(tmp_path):
    torch.manual_seed(29)
    teacher = TinyCausalLM().eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train()
    switches = s2.install_student_switches(
        student,
        ["full_attention", "full_attention"],
        "simplicial",
        SwitchConfig(backend="reference", init_std=0.25),
        seed=5,
    )
    opt, params, _ = s2.build_two_timescale_optimizer(student, switches, lr=5e-3, backbone_lr_ratio=1.0)

    prompt_len = 4
    gen_len = 4
    heldout = torch.randint(0, 19, (6, prompt_len + gen_len))
    before = s3.evaluate_on_policy(
        teacher, student, heldout, prompt_length=prompt_len, gen_tokens=gen_len, device=torch.device("cpu"), chunk_tokens=2
    )

    gen = torch.Generator().manual_seed(77)
    for _ in range(14):
        seqs = s3.rollout_on_policy(
            student, heldout[:, :prompt_len], gen_len, temperature=1.0, microbatch_prompts=3, generator=gen
        )
        assert seqs.shape == (6, prompt_len + gen_len)
        assert torch.equal(seqs[:, :prompt_len], heldout[:, :prompt_len])
        s3.chunked_reverse_kl_step(
            teacher,
            student,
            opt,
            params,
            seqs,
            prompt_length=prompt_len,
            microbatch_prompts=3,
            chunk_tokens=2,
        )

    after = s3.evaluate_on_policy(
        teacher, student, heldout, prompt_length=prompt_len, gen_tokens=gen_len, device=torch.device("cpu"), chunk_tokens=2
    )
    assert after["onpolicy_reverse_kl"] < before["onpolicy_reverse_kl"] * 0.6

    rng = np.random.default_rng(9)
    cur = s3.CurriculumState(gen_tokens=8, phase=1, smoothed_kl=0.12, best_smoothed_kl=0.11, stale_steps=4)
    local_state = rng_state(rng, torch.device("cpu"))
    local_state["rollout_generator"] = gen.get_state()
    ckpt = s3.save_stage3_checkpoint(
        tmp_path / "s3out", 14, 336, cur, student, switches, opt, [local_state], "fp-stage3"
    )

    fresh = copy.deepcopy(teacher).train()
    fresh_sw = s2.install_student_switches(
        fresh, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=1
    )
    fresh_opt, _, _ = s2.build_two_timescale_optimizer(fresh, fresh_sw, lr=5e-3, backbone_lr_ratio=1.0)
    fresh_gen = torch.Generator().manual_seed(1)
    step, tok, loaded_cur = s3.load_stage3_checkpoint(
        ckpt, fresh, fresh_opt, np.random.default_rng(0), fresh_gen, "fp-stage3"
    )
    assert step == 14 and tok == 336 and loaded_cur == cur
    assert torch.equal(fresh_gen.get_state(), gen.get_state())
    reloaded = s3.evaluate_on_policy(
        teacher, fresh, heldout, prompt_length=prompt_len, gen_tokens=gen_len, device=torch.device("cpu"), chunk_tokens=2
    )
    assert reloaded == after


def _gloo_stage3_worker(rank, world, port, teacher_sd, student_sd, seqs, out_path):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        teacher = TinyCausalLM().eval().requires_grad_(False)
        teacher.load_state_dict(teacher_sd)
        student = TinyCausalLM().train()
        switches = s2.install_student_switches(
            student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=42
        )
        student.load_state_dict(student_sd)
        opt, params, _ = s2.build_two_timescale_optimizer(student, switches, lr=2e-3, backbone_lr_ratio=1.0)
        shard = seqs[rank * 2 : (rank + 1) * 2]
        res = s3.chunked_reverse_kl_step(
            teacher, student, opt, params, shard, prompt_length=3, microbatch_prompts=1, chunk_tokens=2
        )
        if rank == 0:
            torch.save(
                {"rkl": res["train_reverse_kl"], "sd": {k: v.detach().cpu() for k, v in student.state_dict().items()}},
                out_path,
            )
    finally:
        dist.destroy_process_group()


def test_stage3_two_rank_gloo_matches_single_rank_global_batch(tmp_path):
    torch.manual_seed(53)
    teacher = TinyCausalLM().eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train()
    switches = s2.install_student_switches(
        student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=42
    )
    t_sd = copy.deepcopy(teacher.state_dict())
    s_sd = copy.deepcopy(student.state_dict())
    seqs = torch.randint(0, 19, (4, 7))

    opt, params, _ = s2.build_two_timescale_optimizer(student, switches, lr=2e-3, backbone_lr_ratio=1.0)
    single = s3.chunked_reverse_kl_step(
        teacher, student, opt, params, seqs, prompt_length=3, microbatch_prompts=1, chunk_tokens=2
    )

    out_path = str(tmp_path / "rank0_s3.pt")
    mp.spawn(_gloo_stage3_worker, args=(2, 29789, t_sd, s_sd, seqs, out_path), nprocs=2)
    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    assert abs(single["train_reverse_kl"] - payload["rkl"]) < 1e-6
    for k, v in student.state_dict().items():
        torch.testing.assert_close(v, payload["sd"][k], atol=1e-6, rtol=1e-5)


def test_rollout_on_policy_chunk_padding_is_causally_identical_and_enforces_multiple():
    torch.manual_seed(61)
    student = TinyCausalLM().eval()
    s2.install_student_switches(
        student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=3
    )
    prompts = torch.randint(0, 19, (3, 5))

    seen_lengths: list[int] = []
    orig_forward = student.model.forward

    def spy_forward(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        seen_lengths.append(int(ids.shape[1]))
        return orig_forward(*args, **kwargs)

    unpadded = s3.rollout_on_policy(
        student, prompts, 7, temperature=1.0, microbatch_prompts=2, generator=torch.Generator().manual_seed(99), pad_to=1
    )
    student.model.forward = spy_forward
    try:
        padded = s3.rollout_on_policy(
            student,
            prompts,
            7,
            temperature=1.0,
            microbatch_prompts=2,
            generator=torch.Generator().manual_seed(99),
            pad_to=8,
        )
    finally:
        student.model.forward = orig_forward

    assert torch.equal(unpadded, padded)
    assert seen_lengths and all(length % 8 == 0 for length in seen_lengths)


def test_stage3_cuda_checkpoint_roundtrip_restores_cuda_rollout_generator(tmp_path):
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    torch.manual_seed(71)
    student = TinyCausalLM().to(device).train()
    switches = s2.install_student_switches(
        student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=11
    )
    student.to(device)
    opt, _, _ = s2.build_two_timescale_optimizer(student, switches, lr=1e-3, backbone_lr_ratio=1.0)
    gen = torch.Generator(device=device).manual_seed(12345)
    _ = torch.rand(17, device=device, generator=gen)
    expected_next = torch.rand(8, device=device, generator=gen).clone()
    # Re-create state right before expected_next
    gen_saved = torch.Generator(device=device).manual_seed(12345)
    _ = torch.rand(17, device=device, generator=gen_saved)

    rng = np.random.default_rng(19)
    local_state = rng_state(rng, device)
    local_state["rollout_generator"] = gen_saved.get_state()
    cur = s3.CurriculumState(gen_tokens=64, phase=0, smoothed_kl=0.2, best_smoothed_kl=0.2, stale_steps=0)
    ckpt = s3.save_stage3_checkpoint(
        tmp_path / "cuda_s3", 2, 4096, cur, student, switches, opt, [local_state], "fp-cuda-s3"
    )

    fresh = TinyCausalLM().to(device).train()
    fresh_sw = s2.install_student_switches(
        fresh, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=99
    )
    fresh.to(device)
    fresh_opt, _, _ = s2.build_two_timescale_optimizer(fresh, fresh_sw, lr=1e-3, backbone_lr_ratio=1.0)
    fresh_gen = torch.Generator(device=device).manual_seed(999)
    step, tok, loaded_cur = s3.load_stage3_checkpoint(
        ckpt, fresh, fresh_opt, np.random.default_rng(0), fresh_gen, "fp-cuda-s3", device=device
    )
    assert step == 2 and tok == 4096 and loaded_cur == cur
    actual_next = torch.rand(8, device=device, generator=fresh_gen)
    assert torch.equal(expected_next, actual_next)



def test_curriculum_blocks_horizon_doubling_when_kl_diverges_upward():
    cur = s3.CurriculumState(gen_tokens=64)
    # Initial phase baseline is 0.20; upward drift to 0.30 (> 0.20 * 1.05) must never double horizon.
    assert not cur.update(0.20, max_gen_tokens=512, ema_alpha=0.5, improvement_rel=0.02, plateau_steps=3, max_divergence_rel=0.05)
    for _ in range(5):
        assert not cur.update(0.30, max_gen_tokens=512, ema_alpha=0.5, improvement_rel=0.02, plateau_steps=3, max_divergence_rel=0.05)
    assert cur.gen_tokens == 64 and cur.phase == 0

    # Once smoothed KL recovers below the non-divergent ceiling (<= 0.21) and plateaus, horizon doubles to 128.
    for _ in range(6):
        advanced = cur.update(0.18, max_gen_tokens=512, ema_alpha=0.5, improvement_rel=0.02, plateau_steps=3, max_divergence_rel=0.05)
        if advanced:
            break
    assert cur.gen_tokens == 128 and cur.phase == 1 and cur.phase_initial_kl is None


def test_anchored_reverse_kl_matches_unchunked_combined_objective():
    torch.manual_seed(89)
    teacher = TinyCausalLM().eval().requires_grad_(False)
    student_a = copy.deepcopy(teacher).train()
    sw_a = s2.install_student_switches(
        student_a, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference", window_2=3), seed=7
    )
    opt_a, params_a, _ = s2.build_two_timescale_optimizer(student_a, sw_a, lr=1e-3, backbone_lr_ratio=0.1)

    student_b = copy.deepcopy(student_a).train()
    sw_b = {0: student_b.model.layers[0].self_attn.switch, 1: student_b.model.layers[1].self_attn.switch}
    opt_b, params_b, _ = s2.build_two_timescale_optimizer(student_b, sw_b, lr=1e-3, backbone_lr_ratio=0.1)

    sequences = torch.randint(0, 19, (4, 8))
    anchor_windows = torch.randint(0, 19, (2, 6))
    alpha = 0.25

    res = s3.chunked_reverse_kl_step(
        teacher,
        student_a,
        opt_a,
        params_a,
        sequences,
        prompt_length=3,
        microbatch_prompts=2,
        chunk_tokens=2,
        clip_norm=10.0,
        anchor_windows=anchor_windows,
        anchor_kl_weight=alpha,
        anchor_microbatch_windows=1,
    )

    # Full unchunked reference
    opt_b.zero_grad(set_to_none=True)
    with torch.no_grad():
        t_gen = teacher.lm_head(teacher.model(sequences).last_hidden_state[:, 2:7]).float().log_softmax(-1)
        t_anc = teacher.lm_head(teacher.model(anchor_windows).last_hidden_state[:, :-1]).float().log_softmax(-1)
    s_gen = student_b.lm_head(student_b.model(sequences).last_hidden_state[:, 2:7]).float().log_softmax(-1)
    s_anc = student_b.lm_head(student_b.model(anchor_windows).last_hidden_state[:, :-1]).float().log_softmax(-1)
    rkl = (s_gen.exp() * (s_gen - t_gen)).sum() / (4 * 5)
    fkl = (t_anc.exp() * (t_anc - s_anc)).sum() / (2 * 5)
    loss = (1.0 - alpha) * rkl + alpha * fkl
    loss.backward()
    norm_b = torch.nn.utils.clip_grad_norm_(params_b, 10.0)
    opt_b.step()

    assert abs(res["train_reverse_kl"] - float(rkl.detach())) < 1e-6
    assert abs(res["train_anchor_forward_kl"] - float(fkl.detach())) < 1e-6
    assert abs(res["grad_norm"] - float(norm_b)) < 1e-5
    for k, v in student_a.state_dict().items():
        torch.testing.assert_close(v, student_b.state_dict()[k], atol=1e-6, rtol=1e-5)
