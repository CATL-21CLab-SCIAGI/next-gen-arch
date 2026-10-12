"""Opt-in tests against real GPU package workers, never production training."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import pytest
import torch

pytestmark = pytest.mark.slow


def _deployment():
    path = os.getenv("ARCHLAB_QUALIFICATION_CONFIG")
    if not path:
        return None
    from omegaconf import OmegaConf

    return OmegaConf.load(path)


def _tiny_policy(snapshot):
    from archlab.architectures.limite_adapter import (
        LimiteAdapterConfig,
        install_adapters,
        set_trainable_mode,
    )
    from archlab.architectures.limite_loader import upstream_classes

    config_class, model_class = upstream_classes(snapshot)
    config = config_class.from_pretrained(snapshot, local_files_only=True)
    for name, value in dict(
        hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=64, padded_vocab_size=64, tokenizer_vocab_size=64,
        max_position_embeddings=256, global_layers=[1, 3],
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=9, _serialized_sliding_window=8, rope_n_pairs=16,
        attn_gate_channels=32, ve_dim=64, ve_stored_heads=2, ve_layers=[1, 2],
        ve_gate_channels=4, xsa_layers=[0, 1, 2, 3], mudd_layers=[2, 3], mudd_at=[2, 3],
        mudd_tap_idx={"2": [0, 1, 2], "3": [0, 2, 3]}, mudd_inter=4,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ).items():
        setattr(config, name, value)
    config._attn_implementation = "sdpa"
    torch.manual_seed(73)
    model = model_class(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel() == 1:
                parameter.fill_(1.)
            else:
                parameter.normal_(std=.1)
            if parameter.ndim == 2 and "mudd" not in name:
                parameter.data = parameter.data.to(torch.bfloat16)
    model = install_adapters(model, LimiteAdapterConfig(variant="normal"))
    set_trainable_mode(model, "full")
    with torch.no_grad():
        for adapter in model.model.adapters:
            adapter.native.o_proj.weight.normal_(std=.01)
    return model


def _error_stats(values):
    if not values:
        return dict(count=0)
    data = torch.tensor(values, dtype=torch.float64)
    return dict(count=len(values), maximum=float(data.max()), mean=float(data.mean()),
                p99=float(data.quantile(.99)))


def _logratio_stats(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return dict(absolute=_error_stats(values.abs().tolist()),
                p95_absolute=float(values.abs().quantile(.95)),
                k3_kl_estimate=float((values.exp() - values - 1).mean()))


def _assert_response_contract(details, evidence):
    for item in details:
        # Compare against the independent publisher FA2 cache on the same
        # sampled tokens. Both reductions return BF16; TileLang and FA2 may
        # round differently. Separately bound serving versus that cache.
        comparisons = (
            (item["response_logratio"], item["serving_precision_cache_vs_learner_replay"]),
            (item["vllm_vs_serving_precision_cache"], item["serving_precision_cache_vs_replay"]),
        )
        if "conversion_cache" in item:
            control = item["original_base_control_cache_vs_replay"]
            comparisons = tuple((item[name], control) for name in (
                "response_logratio", "native_cache_vs_replay_same_tokens",
                "vllm_vs_serving_precision_cache"))
            assert item["conversion_cache"]["cached_argmax_agreement"] == 1., evidence
        for metric, baseline in comparisons:
            assert metric["absolute"]["mean"] <= 1.25 * baseline["absolute"]["mean"] + .01, evidence
            assert metric["p95_absolute"] <= 1.25 * baseline["p95_absolute"] + .01, evidence
            assert metric["absolute"]["maximum"] <= .5, evidence
            assert metric["k3_kl_estimate"] <= .005, evidence
    if any("conversion_cache" in item for item in details):
        audits = evidence["conversion_raw_reduction"]
        assert audits and all(audit["live_slot_byte_ranges_disjoint"] for audit in audits), evidence
        for audit in audits:
            assert len(audit["owners"]) == 96 and len(audit["live_slots"]) == 96, evidence
            assert all(slot["exact_values_preserved"] for slot in audit["live_slots"]), evidence
            checks = audit["raw_checks"]
            for kind in ("prefill_last_query_vs_decode", "paged_decode_vs_contiguous_history"):
                assert {item["layer"] for item in checks if item["kind"] == kind} == set(range(48)), evidence
            assert all(item["relative_l2"] <= 1e-3 for item in checks), evidence


@contextmanager
def _native_serving_precision(actor, attention_config):
    if (attention_config.get("backend") != "FLASH_ATTN"
            or attention_config.get("flash_attn_version") != 2):
        raise ValueError("the independent native serving oracle requires explicit FlashAttention-2")
    if getattr(actor.model, "conversion_config", None):
        attentions = [layer.self_attn for layer in actor.model.layers]
        if any(attention.conversion_config.backend != "tilelang" for attention in attentions):
            raise ValueError("the replacement oracle requires its actual TileLang conversion")
        # Keep the publisher's converted forward and its own bounded long/short
        # DynamicCache. Ordinary FA2 would change the learned architecture.
        yield
        return
    attentions = [attention for adapter, layer in zip(
        actor.model.adapters, actor.model.base.layers, strict=True)
        for attention in (adapter.native, layer.self_attn)]
    # Prelude masters remain FP32. eval() folds/casts their actual projection
    # weights to BF16; checking master storage rejects the correct native path.
    if any(getattr(attention, "_inference_qkv_weight", None) is None
           or attention._inference_qkv_weight.dtype != torch.bfloat16 for attention in attentions):
        raise ValueError("the native FA2 serving oracle requires BF16 cached QKV projections")
    forwards = [(attention, attention.forward) for attention in attentions]
    configs = {id(config): (config, config._attn_implementation) for config in
               [actor.model.base.config, *(attention.config for attention in attentions)]}
    try:
        for attention in attentions:
            attention.forward = MethodType(type(attention).forward, attention)
        for config, _ in configs.values():
            config._attn_implementation = "flash_attention_2"
        yield
    finally:
        for attention, forward in forwards:
            attention.forward = forward
        for config, implementation in configs.values():
            config._attn_implementation = implementation


def _cached_response_logprobs(actor, full, prompt_length, response, *, full_history=False,
                             replay_logits=None, diagnostics=None):
    kwargs = {}
    converted = bool(getattr(getattr(actor, "model", None), "conversion_config", None))
    if full_history and not converted:
        from transformers import DynamicCache

        # Keep absolute key positions for the independent paged-serving oracle.
        # The publisher's FA2 calls still enforce each layer's attention window.
        # Physically slicing the cache changes FA2 tiles at the window boundary.
        cache = DynamicCache()
        cache.archlab_native_preludes = DynamicCache()
        kwargs["past_key_values"] = cache
    if converted:
        kwargs.update(position_ids=torch.arange(prompt_length, device=full.device)[None],
                      attention_mask=dict(full_attention=None, sliding_attention=None), logits_to_keep=1)
    cached = actor(input_ids=full[:, :prompt_length], use_cache=True, **kwargs)
    values, argmax = [], []
    squared_error, squared_reference = 0., 0.
    for index, token in enumerate(response):
        raw = cached.logits[0, -1].float()
        argmax.append(int(raw.argmax()))
        values.append(float(raw.log_softmax(-1)[token]))
        if replay_logits is not None:
            expected = replay_logits[prompt_length - 1 + index]
            squared_error += float((raw - expected).double().square().sum())
            squared_reference += float(expected.double().square().sum())
        if index + 1 < len(response):
            next_kwargs = (dict(position_ids=torch.tensor([[prompt_length + index]], device=full.device),
                                attention_mask=dict(full_attention=None, sliding_attention=None),
                                logits_to_keep=1) if converted else {})
            cached = actor(input_ids=full[:, prompt_length + index:prompt_length + index + 1],
                           past_key_values=cached.past_key_values, use_cache=True, **next_kwargs)
    if diagnostics is not None:
        diagnostics["logits_relative_l2"] = (squared_error / squared_reference) ** .5
        diagnostics["argmax_token_ids"] = argmax
    return values


def _check_worker_parity(source, root, prompts, context, tmp_path, description, deployment=None):
    from vllm import LLM, SamplingParams

    from archlab.training.serving import register

    register()
    converted = bool(getattr(source.model, "conversion_config", None))
    layer_count = len(source.model.layers) if converted else len(source.model.adapters)
    attention_config = json.loads(os.getenv(
        "ARCHLAB_VLLM_ATTENTION_CONFIG", '{"backend":"FLASH_ATTN","flash_attn_version":2}'))
    max_num_seqs = int(os.getenv("ARCHLAB_PARITY_MAX_NUM_SEQS", "1" if converted else "4"))
    memory_utilization, prefill_budget = .15, 8192
    execution = dict(enforce_eager=True)
    cache_execution = dict(enable_chunked_prefill=False) if converted else {}
    if deployment is not None:
        from omegaconf import OmegaConf

        attention_config = OmegaConf.to_container(
            deployment.actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config, resolve=True)
        assert deployment.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
        rollout = deployment.actor_rollout_ref.rollout
        execution = dict(enforce_eager=rollout.enforce_eager,
                         compilation_config=OmegaConf.to_container(
                             rollout.engine_kwargs.vllm, resolve=True).get("compilation_config"))
        assert not deployment.actor_rollout_ref.rollout.enable_prefix_caching
        max_num_seqs = int(deployment.actor_rollout_ref.rollout.max_num_seqs)
        memory_utilization = float(deployment.actor_rollout_ref.rollout.gpu_memory_utilization)
        prefill_budget = int(deployment.actor_rollout_ref.rollout.max_num_batched_tokens)
        cache_execution = dict(enable_chunked_prefill=rollout.enable_chunked_prefill)
        if converted:
            assert max_num_seqs == 1 and execution["enforce_eager"]
            assert not cache_execution["enable_chunked_prefill"]
            assert prefill_budget >= context
    llm = LLM(model=str(root), skip_tokenizer_init=True, dtype="bfloat16", max_model_len=context,
              **execution, **cache_execution, gpu_memory_utilization=memory_utilization, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=prefill_budget,
              enable_prefix_caching=False, seed=42, tensor_parallel_size=1,
              worker_extension_cls=("archlab.training.qualification_serving.ConversionQualificationWorker"
                                    if converted else "archlab.training.serving.QualificationWorker"),
              attention_config=attention_config)
    if converted:
        llm.collective_rpc("archlab_start_cache_audit")
    if layer_count == 48:
        from transformers import AutoTokenizer

        serving_vocab_size = len(AutoTokenizer.from_pretrained(root, local_files_only=True))
    else:
        serving_vocab_size = int(source.config.vocab_size)
    verl_initialization = llm.collective_rpc("archlab_initialize_verl_model",
                                             kwargs={"vocab_size": serving_vocab_size})
    actor = source.to(device="cuda").eval()
    original_control = None
    if converted:
        from archlab.architectures.limite_loader import load_model

        config = json.loads((root / "config.json").read_text())
        original_control = load_model(config["publisher_snapshot"], attn_implementation="flash_attention_2")
        original_control = original_control.to(device="cuda").eval()
    temperature = float(os.getenv("ARCHLAB_PARITY_TEMPERATURE", "0"))
    generated_tokens = int(os.getenv("ARCHLAB_PARITY_GENERATED_TOKENS", "4"))
    if deployment is not None:
        temperature = float(deployment.actor_rollout_ref.rollout.temperature)
        generated_tokens = 128
    params = SamplingParams(temperature=temperature, max_tokens=generated_tokens, ignore_eos=True, logprobs=1,
                            prompt_logprobs=1, detokenize=False)
    errors, details = [], []
    for phase in range(2):
        outputs = llm.generate([dict(prompt_token_ids=ids) for ids in prompts], params, use_tqdm=False)
        for ids, output in zip(prompts, outputs, strict=True):
            response = list(output.outputs[0].token_ids)
            full = torch.tensor([ids + response], device="cuda")
            with torch.inference_mode():
                if converted:
                    replay_ids = full[:, :-1]
                    raw_replay = actor(input_ids=replay_ids,
                                       position_ids=torch.arange(replay_ids.shape[1], device=full.device)[None],
                                       attention_mask=dict(full_attention=None, sliding_attention=None),
                                       use_cache=False).logits[0].float()
                    logits = raw_replay.log_softmax(-1)
                else:
                    logits = actor(input_ids=full, use_cache=False).logits[0, :-1].float().log_softmax(-1)
            pairs = [(i - 1, ids[i], item[ids[i]].logprob)
                     for i, item in enumerate(output.prompt_logprobs) if i]
            pairs.extend((len(ids) - 1 + i, token, item[token].logprob)
                         for i, (token, item) in enumerate(zip(response, output.outputs[0].logprobs, strict=True)))
            local = [(pos, abs(float(logits[pos, token]) - value)) for pos, token, value in pairs]
            errors.extend(value for _, value in local)
            report = dict(phase=phase, prompt_tokens=len(ids),
                          prefill=_error_stats([v for pos, v in local if pos < len(ids) - 1]),
                          decode=_error_stats([v for pos, v in local if pos >= len(ids) - 1]),
                          before_window=_error_stats([v for pos, v in local if pos < 1024]),
                          after_window=_error_stats([v for pos, v in local if pos >= 1024]))
            tail_mass = logits[len(ids) - 1:, serving_vocab_size:].logsumexp(-1).exp()
            report["native_probability_removed_by_stock_oov_mask"] = dict(
                mean=float(tail_mass.mean()), maximum=float(tail_mass.max()))
            if os.getenv("ARCHLAB_DIAGNOSE_PARITY") == "1" and not converted:
                from types import MethodType
                originals = [(adapter.native.forward, adapter.native.config._attn_implementation)
                             for adapter in actor.model.adapters]
                try:
                    for adapter in actor.model.adapters:
                        adapter.native.forward = MethodType(type(adapter.native).forward, adapter.native)
                        adapter.native.config._attn_implementation = "sdpa"
                    with torch.inference_mode():
                        native = actor(input_ids=full, use_cache=False).logits[0, :-1].float().log_softmax(-1)
                    report["native_sdpa_vs_vllm"] = _error_stats(
                        [abs(float(native[pos, token]) - value) for pos, token, value in pairs])
                    report["tilelang_vs_native_sdpa"] = _error_stats(
                        [abs(float(native[pos, token]) - float(logits[pos, token]))
                         for pos, token, _ in pairs])
                finally:
                    for adapter, (original, backend) in zip(actor.model.adapters, originals, strict=True):
                        adapter.native.forward = original
                        adapter.native.config._attn_implementation = backend
                with torch.inference_mode():
                    prefill = actor(input_ids=full[:, :len(ids)], use_cache=False).logits[0, :-1].float().log_softmax(-1)
                report["native_same_prompt_different_gemm_shape"] = _error_stats(
                    [abs(float(prefill[pos, token]) - float(logits[pos, token]))
                     for pos, token, _ in pairs if pos < len(ids) - 1])
            with torch.inference_mode():
                conversion_cache = {} if converted else None
                cached_values = _cached_response_logprobs(
                    actor, full, len(ids), response,
                    replay_logits=raw_replay if converted else None, diagnostics=conversion_cache)
                with _native_serving_precision(actor, attention_config):
                    precision_logits = (logits if converted else
                                        actor(input_ids=full, use_cache=False).logits[0, :-1].float().log_softmax(-1))
                    precision_replay = [float(precision_logits[len(ids) - 1 + i, token])
                                        for i, token in enumerate(response)]
                    precision_cache = _cached_response_logprobs(
                        actor, full, len(ids), response, full_history=True)
                    del precision_logits
            if converted:
                report["conversion_cache"] = conversion_cache
                with torch.inference_mode():
                    control_raw = original_control(input_ids=full[:, :-1], use_cache=False).logits[0].float()
                    control_replay = control_raw.log_softmax(-1)
                    control_diagnostics = {}
                    control_cache = _cached_response_logprobs(
                        original_control, full, len(ids), response, replay_logits=control_raw,
                        diagnostics=control_diagnostics)
                control_values = [float(control_replay[len(ids) - 1 + i, token])
                                  for i, token in enumerate(response)]
                report["original_base_control_cache_vs_replay"] = _logratio_stats(
                    [a - b for a, b in zip(control_values, control_cache, strict=True)])
                report["original_base_control_cache"] = control_diagnostics
                conversion_cache["cached_argmax_agreement"] = sum(
                    token == next(t for t, value in item.items() if value.rank == 1)
                    for token, item in zip(conversion_cache["argmax_token_ids"],
                                           output.outputs[0].logprobs, strict=True)) / len(response)
            learner_values = [float(logits[len(ids) - 1 + i, token]) for i, token in enumerate(response)]
            serving_values = [item[token].logprob for token, item in
                              zip(response, output.outputs[0].logprobs, strict=True)]
            for label, left, right in (
                ("serving_precision_cache_vs_learner_replay", learner_values, precision_cache),
                ("serving_precision_cache_vs_replay", precision_replay, precision_cache),
                ("vllm_vs_serving_precision_cache", serving_values, precision_cache),
            ):
                report[label] = _logratio_stats([a - b for a, b in zip(left, right, strict=True)])
            native_delta = torch.tensor([
                float(logits[len(ids) - 1 + i, token]) - cached_values[i]
                for i, token in enumerate(response)], dtype=torch.float64)
            serving_delta = torch.tensor([
                item[token].logprob - cached_values[i]
                for i, (token, item) in enumerate(zip(response, output.outputs[0].logprobs, strict=True))],
                dtype=torch.float64)
            for label, values in (("native_cache_vs_replay_same_tokens", native_delta),
                                  ("vllm_vs_native_cache_same_tokens", serving_delta)):
                report[label] = dict(absolute=_error_stats(values.abs().tolist()),
                                     p95_absolute=float(values.abs().quantile(.95)),
                                     k3_kl_estimate=float((values.exp() - values - 1).mean()))
            if converted:
                report["frozen_cache_v1"] = dict(
                    logits_relative_l2_limit=.02, selected_logprob_maximum_limit=.1,
                    candidate_passed=(conversion_cache["logits_relative_l2"] <= .02 and
                        report["native_cache_vs_replay_same_tokens"]["absolute"]["maximum"] <= .1),
                    original_base_passed=(control_diagnostics["logits_relative_l2"] <= .02 and
                        report["original_base_control_cache_vs_replay"]["absolute"]["maximum"] <= .1),
                    role="preserved historical diagnostic; control-calibrated v3 is the explicit release contract")
            logratios = torch.tensor([
                float(logits[len(ids) - 1 + i, token]) - item[token].logprob
                for i, (token, item) in enumerate(zip(response, output.outputs[0].logprobs, strict=True))
            ], dtype=torch.float64)
            report["response_logratio"] = dict(
                mean=float(logratios.mean()), absolute=_error_stats(logratios.abs().tolist()),
                p95_absolute=float(logratios.abs().quantile(.95)),
                k3_kl_estimate=float((logratios.exp() - logratios - 1).mean()),
                argmax_disagreement=sum(
                    int(logits[len(ids) - 1 + i].argmax()) != next(token for token, value in item.items() if value.rank == 1)
                    for i, item in enumerate(output.outputs[0].logprobs)) / len(response),
            )
            # Preserve the actual comparison, including failures. Aggregate
            # quantiles alone cannot distinguish a different sampled trajectory
            # from a token-local numerical or attention-window discrepancy.
            report["response_tokens"] = [dict(
                index=i, position=len(ids) + i, token_id=token,
                native_replay_logprob=float(logits[len(ids) - 1 + i, token]),
                native_cache_logprob=cached_values[i],
                serving_precision_replay_logprob=precision_replay[i],
                serving_precision_cache_logprob=precision_cache[i],
                vllm_logprob=item[token].logprob,
                **(dict(original_base_replay_logprob=control_values[i],
                        original_base_cache_logprob=control_cache[i]) if converted else {}),
            ) for i, (token, item) in enumerate(zip(response, output.outputs[0].logprobs, strict=True))]
            details.append(report)
        if phase == 0:
            from safetensors.torch import load_model, save_file

            updated = os.getenv("ARCHLAB_PARITY_UPDATED_MODEL")
            if updated:
                # Stock verl's HF export follows the same native key contract.
                # This test keeps the immutable starting export untouched.
                reload_path = Path(updated) / "model.safetensors"
                wrapper = torch.nn.Module()
                wrapper.add_module("policy", actor)
                load_model(wrapper, reload_path, strict=True, device="cpu")
            else:
                changed = {}
                with torch.no_grad():
                    if converted:
                        actor.model.layers[0].self_attn.native.o_proj.weight.add_(.02)
                        changed["policy.model.layers.0.self_attn.native.o_proj.weight"] = (
                            actor.model.layers[0].self_attn.native.o_proj.weight.detach().cpu())
                        for index, layer in enumerate(actor.model.layers):
                            layer.self_attn.short_kv_delta.add_(.001)
                            changed[f"policy.model.layers.{index}.self_attn.short_kv_delta"] = (
                                layer.self_attn.short_kv_delta.detach().cpu())
                    else:
                        actor.model.adapters[0].native.o_proj.weight.add_(.02)
                        changed["policy.model.adapters.0.native.o_proj.weight"] = (
                            actor.model.adapters[0].native.o_proj.weight.detach().cpu())
                reload_path = tmp_path / "changed-weights.safetensors"
                save_file(changed, reload_path)
            actor.train().eval()
            reports = llm.collective_rpc("archlab_reload_weights", args=(str(reload_path),))
            assert all(report["cache_layers"] == 2 * layer_count for report in reports)
    # Exercise exact publisher stopping in the actual package worker. Limite
    # Base's tokenizer EOS differs from its generation EOS; only the latter
    # terminates the native model's turn.
    native_eos = int(source.generation_config.eos_token_id)
    native_stop = llm.generate(
        [dict(prompt_token_ids=prompts[-1])],
        SamplingParams(max_tokens=2, ignore_eos=True, stop_token_ids=[native_eos],
                       allowed_token_ids=[native_eos], detokenize=False), use_tqdm=False,
    )[0].outputs[0]
    assert native_stop.finish_reason == "stop" and native_stop.stop_reason == native_eos
    tokenizer_eos_continues = None
    if native_eos == 151645:
        continuation = llm.generate(
            [dict(prompt_token_ids=prompts[-1])],
            SamplingParams(max_tokens=2, ignore_eos=True, stop_token_ids=[native_eos],
                           allowed_token_ids=[151643], detokenize=False), use_tqdm=False,
        )[0].outputs[0]
        assert continuation.finish_reason == "length" and len(continuation.token_ids) == 2
        tokenizer_eos_continues = True
    receipt = json.loads((root / "EXPORT.json").read_text())
    from archlab.artifacts import sha256_file

    evidence = dict(reload_kind="package_optimizer_update" if updated else "stress_o_proj_plus_0.02",
                    reloaded_weights_sha256=sha256_file(reload_path),
                    source_state_sha256=receipt["state_sha256"],
                    export_config_sha256=receipt["files"]["config.json"],
                    native_eos=native_eos, tokenizer_eos_continues=tokenizer_eos_continues,
                    max_logprob_error=max(errors), compared_logprobs=len(errors),
                    phases=2, prompts_per_phase=len(prompts), cache_layers=2 * layer_count,
                    context=context, source=description, details=details,
                    response_limits=dict(contract=("original_base_control_calibrated_v3" if converted else
                                                   "same_token_serving_precision_relative_v2"),
                                         baseline_multiplier=1.25, baseline_slack=.01,
                                         maximum=.5, k3_kl=.005),
                    conversion_cache_limits=(dict(raw_reduction_relative_l2=.001, cached_argmax_agreement=1.)
                                             if converted else None),
                    conversion_raw_reduction=(llm.collective_rpc("archlab_cache_audit") if converted else None),
                    serving_precision_oracle=dict(
                        implementation=("publisher converted forward with actual TileLang joint softmax"
                                        if converted else "publisher native forward with flash_attention_2"),
                        attention_output_dtype="bfloat16", weights="identical",
                        cache=("native DynamicCache long windows and separate W16 short history" if converted
                               else "full history; native per-layer attention windows unchanged"),
                        bounded_cache="retained as separate diagnostic",
                        tokens="same recorded response; no resampling",
                        independent_checks=["learner versus serving", "serving versus precision-matched native cache"]),
                    native_attention=dict(
                        backbone="tilelang_conversion" if converted else source.model.base.config._attn_implementation,
                        intervention="replace_attention" if converted else "insert_attention",
                        adapters=None if converted else source.model.adapter_config["attention_backend"],
                        head_precision=getattr(source, "head_precision_mode", "publisher"),
                        window=source.model.config.sliding_window,
                        scaling=source.model.config.attention_softmax_scale,
                    ),
                    attention_config=attention_config, execution=execution,
                    temperature=temperature, generated_tokens=generated_tokens,
                    max_num_seqs=max_num_seqs, gpu_memory_utilization=memory_utilization,
                    verl_initialization=verl_initialization,
                    max_num_batched_tokens=prefill_budget)
    (tmp_path / "serving-parity.json").write_text(json.dumps(evidence, indent=2))
    # RL optimizes only assistant-response tokens; prompt differences remain
    # visible diagnostic evidence but are excluded from the training loss.
    _assert_response_contract(details, evidence)


def test_vllm_tiny_worker_parity(tmp_path):
    """One GPU, four layers; this is not a production admission test."""
    if os.getenv("ARCHLAB_RUN_DISTRIBUTED") != "1":
        pytest.skip("set ARCHLAB_RUN_DISTRIBUTED=1 on a free qualification GPU")
    from archlab.training.limite import export_model

    snapshot = Path(os.getenv("ARCHLAB_LIMITE_PUBLISHER", "/mnt/oss/models/limite-1b-base-cc612bafcd4a"))
    source = _tiny_policy(snapshot)
    root = tmp_path / "tiny-export"
    export_model(source, snapshot, root)
    prompts = [[1, 4, 8, 5, 6, 10, 13, 3, 9, 7, 4, 8], [1, 6, 3, 8, 10]]
    _check_worker_parity(source, root, prompts, 128, tmp_path, "four-layer tiny publisher fixture")


def test_vllm_full_checkpoint_short_context_parity(tmp_path):
    """Real checkpoint, one GPU, 1100-token prefill crossing the local window.

    This does not qualify native-context memory or a distributed learner.
    """
    deployment = _deployment()
    export = (deployment.actor_rollout_ref.model.path if deployment is not None
              else os.getenv("ARCHLAB_LIMITE_EXPORT"))
    if os.getenv("ARCHLAB_RUN_DISTRIBUTED") != "1" or not export:
        pytest.skip("set ARCHLAB_RUN_DISTRIBUTED=1 and ARCHLAB_LIMITE_EXPORT")
    if deployment is not None:
        updated = os.getenv("ARCHLAB_PARITY_UPDATED_MODEL")
        assert updated and (Path(updated) / "model.safetensors").is_file(), (
            "production serving qualification requires the preceding package optimizer update"
        )
    from archlab.training.limite import LimiteForCausalLM

    root = Path(export)
    source = LimiteForCausalLM.from_pretrained(root).policy
    assert len(source.model.layers if getattr(source.model, "conversion_config", None)
               else source.model.adapters) == 48
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    from importlib.resources import files

    import yaml

    fixtures = yaml.safe_load((files("archlab") / "prompts/limite_agentic_parity.yaml").read_text())["cases"]
    prompts = [[1 + i % 512 for i in range(1100)]]
    prompts.extend(tokenizer.encode(tokenizer.apply_chat_template(
        case["messages"], tokenize=False, add_generation_prompt=True)) for case in fixtures.values())
    _check_worker_parity(source, root, prompts, 2048, tmp_path,
                         "actual 48-layer checkpoint; short context only", deployment)


def test_native_context_forward_backward_memory(tmp_path):
    """Exercise the real export at its advertised context with package scoring.

    This is a single-GPU memory/numerics gate; distributed optimizer and rollout
    synchronization are qualified separately. No sequence cap is substituted.
    """
    if os.getenv("ARCHLAB_RUN_NATIVE_CONTEXT") != "1":
        pytest.skip("set ARCHLAB_RUN_NATIVE_CONTEXT=1 on a free GPU")
    import math
    import time
    from functools import partial
    from importlib import import_module
    from types import SimpleNamespace

    import torch.distributed as dist

    import_module("archlab.training.megatron")  # verified FA4 namespace before TE
    from megatron.core import parallel_state
    from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from archlab.architectures.limite_head import chunked_logits_processor
    from archlab.training.limite import LimiteForCausalLM
    from archlab.training.limite_artifacts import verify_export

    deployment = _deployment()
    root = Path(deployment.actor_rollout_ref.model.path if deployment is not None
                else os.environ["ARCHLAB_LIMITE_EXPORT"])
    receipt = verify_export(root)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=(tmp_path / "rendezvous").as_uri(),
                            rank=0, world_size=1)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model = LimiteForCausalLM.from_pretrained(root, device_map="cuda").train()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    length = int(model.config.max_position_embeddings)
    assert length == 131072, "qualification must use the native advertised context"
    if deployment is not None:
        assert deployment.actor_rollout_ref.rollout.max_model_len == length
        assert deployment.data.max_prompt_length + deployment.data.max_response_length == length
    ids = (torch.arange(length, device="cuda") % 5000 + 2000)[None]
    ids[:, 0] = model.config.bos_token_id
    processor = partial(
        MegatronEngineWithLMHead._lm_head_logits_processor,
        SimpleNamespace(engine_config=SimpleNamespace(entropy_from_logits_with_chunking=False)),
        calculate_sum_pi_squared=False, calculate_entropy=True,
        distillation_use_topk=False, distillation_only=False,
        logits_processor_func=None, batch=None, data_format="bshd",
    )
    if os.getenv("ARCHLAB_DETECT_ANOMALY") == "1":
        torch.autograd.set_detect_anomaly(True)
    if os.getenv("ARCHLAB_TRACE_GRADS") == "1":
        traced = set()

        def observe(name):
            def forward_hook(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    def report_gradient(gradient):
                        if name not in traced:
                            traced.add(name)
                            print(json.dumps(dict(gradient_boundary=name,
                                                  finite=bool(torch.isfinite(gradient).all()),
                                                  maximum=float(gradient.abs().max()),
                                                  nan=int(torch.isnan(gradient).sum()),
                                                  inf=int(torch.isinf(gradient).sum()))), flush=True)
                        return gradient
                    tensor.register_hook(report_gradient)
            return forward_hook

        for name, module in model.policy.named_modules():
            if name.endswith((".self_attn", ".native")):
                module.register_forward_hook(observe(name))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    backbone_kwargs = {}
    if model.config.conversion:
        mask, positions = model.conversion_inputs(ids, None, None)
        backbone_kwargs = dict(attention_mask=mask, position_ids=positions)
    hidden = model.policy.model(input_ids=ids, use_cache=False, **backbone_kwargs,
                                return_dict=True).last_hidden_state
    scores = chunked_logits_processor(
        model.policy._softcapped_logits, hidden, processor,
        dict(label=ids.roll(-1, 1), temperature=torch.ones_like(ids, dtype=torch.float32)),
    )
    entropy_coefficient = float(os.getenv(
        "ARCHLAB_CONTEXT_ENTROPY_COEFFICIENT",
        str(deployment.actor_rollout_ref.actor.entropy_coeff) if deployment is not None else ".001",
    ))
    loss = -scores["log_probs"].mean() - entropy_coefficient * scores["entropy"].mean()
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - started
    print(json.dumps(dict(phase="native_context_forward", tokens=length,
                          seconds=forward_seconds, loss=float(loss.detach()),
                          allocated_gib=torch.cuda.memory_allocated() / 2**30)), flush=True)
    assert torch.isfinite(loss)
    loss.backward()
    torch.cuda.synchronize()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    nonfinite = [name for name, parameter in model.named_parameters()
                 if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
    training_peak_allocated = torch.cuda.max_memory_allocated()
    training_peak_reserved = torch.cuda.max_memory_reserved()
    training_seconds = time.monotonic() - started
    # Native mixed-dtype autograd leaves BF16 matrix gradients; the deployment
    # reduces them in FP32. Qualify the actual stock clipping reduction on that
    # representation, as finite entries alone do not exclude norm overflow.
    norms64 = {
        name: float(torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float64))
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    global_norm64 = math.hypot(*norms64.values())
    gradients32 = [gradient.detach().float() for gradient in gradients]
    direct_norm32 = float(torch.stack([
        gradient.square().sum(dtype=torch.float32) for gradient in gradients32
    ]).sum().sqrt())
    stock_norm = get_grad_norm_fp32(gradients32, norm_type=2,
                                    grad_stats_parallel_group=dist.group.WORLD)
    stock_norm32 = float(stock_norm)
    clip_limit = (float(deployment.actor_rollout_ref.actor.optim.clip_grad)
                  if deployment is not None else 1.)
    stock_clip = clip_limit / (stock_norm + 1e-6)
    stock_clip_coefficient = (float(stock_clip.clamp_max(1.)) if isinstance(stock_clip, torch.Tensor)
                              else min(stock_clip, 1.))
    finite_norms = all(math.isfinite(value) and value > 0
                       for value in (global_norm64, direct_norm32, stock_norm32,
                                     stock_clip_coefficient))
    norm_report = dict(
        per_tensor_fp64=norms64, global_fp64=global_norm64,
        direct_square_sum_fp32=direct_norm32, stock_megatron_fp32=stock_norm32,
        stock_implementation="megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
        stock_input_dtype="float32", clip_limit=clip_limit,
        stock_clip_coefficient=stock_clip_coefficient, finite_positive_norms=finite_norms,
    )
    (tmp_path / "gradient-norms.json").write_text(json.dumps(norm_report, indent=2) + "\n")
    report = dict(
        format="archlab-native-context-v1", state_sha256=receipt["state_sha256"],
        export_config_sha256=receipt["files"]["config.json"], tokens=length,
        attention_backend=("tilelang" if model.config.conversion else model.config.adapter["attention_backend"]),
        normal_kernel=model.config.normal_kernel, normal_backward=model.config.normal_backward,
        parameters=sum(p.numel() for p in model.parameters()), gradient_tensors=len(gradients),
        forward_seconds=forward_seconds, total_seconds=training_seconds,
        peak_allocated_gib=training_peak_allocated / 2**30,
        peak_reserved_gib=training_peak_reserved / 2**30,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        full_context=True, finite_loss_and_gradients=bool(gradients) and not nonfinite,
        nonfinite_gradients=nonfinite, entropy_coefficient=entropy_coefficient,
        gradient_norms={key: value for key, value in norm_report.items() if key != "per_tensor_fp64"},
        per_tensor_norms_file="gradient-norms.json",
    )
    (tmp_path / "native-context.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    assert report["finite_loss_and_gradients"], report
    assert finite_norms, report


def test_vllm_native_context_capacity(tmp_path):
    """Real full-context paged prefill and decode, independently of the learner."""
    if os.getenv("ARCHLAB_RUN_NATIVE_CONTEXT") != "1":
        pytest.skip("set ARCHLAB_RUN_NATIVE_CONTEXT=1 on a free GPU")
    import time

    from vllm import LLM, SamplingParams

    from archlab.training.limite import LimiteConfig
    from archlab.training.limite_artifacts import verify_export
    from archlab.training.serving import register

    deployment = _deployment()
    root = Path(deployment.actor_rollout_ref.model.path if deployment is not None
                else os.environ["ARCHLAB_LIMITE_EXPORT"])
    receipt = verify_export(root)
    config = LimiteConfig.from_pretrained(root)
    length = int(config.max_position_embeddings)
    assert length == 131072
    register()
    attention_config = {"backend": "FLASH_ATTN", "flash_attn_version": 2}
    execution = dict(enforce_eager=True)
    converted = bool(config.conversion)
    capacity = dict(gpu_memory_utilization=.15, max_num_seqs=1,
                    max_num_batched_tokens=length if converted else 8192)
    if converted:
        execution["enable_chunked_prefill"] = False
    if deployment is not None:
        from omegaconf import OmegaConf

        rollout = deployment.actor_rollout_ref.rollout
        attention_config = OmegaConf.to_container(rollout.engine_kwargs.vllm.attention_config)
        assert rollout.max_model_len == length
        assert not rollout.enable_prefix_caching
        assert rollout.tensor_model_parallel_size == 1
        execution = dict(enforce_eager=rollout.enforce_eager,
                         compilation_config=OmegaConf.to_container(
                             rollout.engine_kwargs.vllm, resolve=True).get("compilation_config"))
        if converted:
            assert rollout.enforce_eager and rollout.max_num_seqs == 1
            assert not rollout.enable_chunked_prefill and rollout.max_num_batched_tokens >= length
            execution["enable_chunked_prefill"] = False
        capacity = dict(gpu_memory_utilization=float(rollout.gpu_memory_utilization),
                        max_num_seqs=int(rollout.max_num_seqs),
                        max_num_batched_tokens=int(rollout.max_num_batched_tokens))
    llm = LLM(model=str(root), skip_tokenizer_init=True, dtype="bfloat16",
              max_model_len=length, **execution, enable_prefix_caching=False,
              **capacity,
              tensor_parallel_size=1, attention_config=attention_config)
    started = time.monotonic()
    ids = [2000 + index % 5000 for index in range(length - 1)]
    output = llm.generate([dict(prompt_token_ids=ids)],
                          SamplingParams(temperature=0, max_tokens=1, ignore_eos=True,
                                         logprobs=1, detokenize=False), use_tqdm=False)[0]
    completion = output.outputs[0]
    assert len(output.prompt_token_ids) == length - 1 and len(completion.token_ids) == 1
    value = completion.logprobs[0][completion.token_ids[0]].logprob
    assert torch.isfinite(torch.tensor(value))
    report = dict(format="archlab-vllm-native-context-v1", source_state_sha256=receipt["state_sha256"],
                  export_config_sha256=receipt["files"]["config.json"],
                  context=length, prompt_tokens=len(ids), response_tokens=1,
                  selected_logprob=value, seconds=time.monotonic() - started,
                  attention_config=attention_config, execution=execution, capacity=capacity, tensor_parallel_size=1)
    (tmp_path / "serving-native-context.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
