"""Small native-model forward/backward qualification; NEVER launches training.

Requires the pinned downloaded reference, the existing NeMo checkout on
PYTHONPATH and compatible packages already present in the frozen container.
Outputs are synthetic-model tests, not pretrained checkpoint metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import traceback
from pathlib import Path


def run(checkpoint: Path, package_root: Path, *, pytorch=False):
    import torch

    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
    from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn
    from archlab.automodel.deepseek_v41_autograd import (
        attach_adapters,
        install_frozen_backward,
        training_hidden,
    )
    from archlab.automodel.deepseek_v41_execution import enable_activation_checkpointing
    from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
    from archlab.automodel.deepseek_v41_runtime import (
        implementation_hashes,
        load_native_reference,
        select_container_kernel_packages,
    )

    report = {"hostname": socket.gethostname(), "kind": "synthetic-single-GPU-qualification",
              "container": os.environ.get("NGA_CONTAINER_DIGEST"), "training_launched": False,
              "torch": torch.__version__, "tests": []}
    report["implementation_sha256"] = implementation_hashes()
    report["kernel_packages"] = select_container_kernel_packages(package_root)
    native = load_native_reference(checkpoint, module_name="_archlab_v41_oracle")
    bridge = load_native_reference(checkpoint, module_name="_archlab_v41_backward")
    report["native_row_padding"] = install_native_row_padding(native)
    install_native_row_padding(bridge)
    report["device"] = torch.cuda.get_device_name()
    report["capability"] = torch.cuda.get_device_capability()

    def check(name, function):
        try:
            evidence = function()
            torch.cuda.synchronize()
            record = {"name": name, "passed": True, "evidence": evidence}
        except Exception:
            record = {"name": name, "passed": False, "traceback": traceback.format_exc()}
        report["tests"].append(record)
        print(json.dumps(record), flush=True)

    def mhc():
        torch.manual_seed(1)
        mixes = torch.randn(1, 7, 24, device="cuda", dtype=torch.float32, requires_grad=True)
        scale = torch.randn(3, device="cuda", dtype=torch.float32)
        base = torch.randn(24, device="cuda", dtype=torch.float32)
        expected = native.hc_split_sinkhorn(mixes, scale, base)
        actual = hc_split_sinkhorn(mixes, scale, base)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-6)
        sum(x.square().sum() for x in actual).backward()
        assert mixes.grad.isfinite().all() and mixes.grad.count_nonzero()
        return {"native_requires_grad": [x.requires_grad for x in expected],
                "max_abs_error": max((a - b).abs().max().item() for a, b in zip(actual, expected, strict=True))}

    check("mhc_native_parity_and_input_gradient", mhc)
    if pytorch:
        from archlab.architectures.deepseek_v41_torch import (
            query_chunked_sparse_attention,
            rounded_activation,
        )

        def torch_numerics():
            errors = {}
            torch.manual_seed(83)
            for bits, group, e4m3 in ((8, 32, False), (4, 32, False), (4, 16, True)):
                x = torch.randn(2, 67, 128, device="cuda", dtype=torch.bfloat16)
                expected = x.clone()
                if bits == 8:
                    native.act_quant(expected, group, "ue8m0", torch.float8_e8m0fnu, True)
                else:
                    native.fp4_act_quant(expected, group, True,
                                         torch.float8_e4m3fn if e4m3 else torch.float8_e8m0fnu)
                actual = rounded_activation(x, bits=bits, block_size=group, e4m3_scale=e4m3)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                errors[f"FP{bits}-group{group}-e4m3{e4m3}"] = float((actual - expected).abs().max())
            q = torch.randn(1, 67, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            kv = torch.randn(1, 83, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            ids = torch.randint(-1, 83, (1, 67, 71), device="cuda", dtype=torch.int32)
            sink = torch.randn(16, device="cuda", dtype=torch.float32)
            actual = query_chunked_sparse_attention(q, kv, sink, ids, .125, query_chunk=11,
                                                    native_rounding=True)
            expected = native.sparse_attn(q, kv, sink, ids, .125)
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.01)
            actual.square().mean().backward()
            assert q.grad.isfinite().all() and kv.grad.isfinite().all()
            errors["sparse_relative_error"] = float((actual - expected).float().norm() / expected.float().norm())
            return errors

        check("pytorch_activation_rounding_and_sparse_forward_gradient", torch_numerics)

        def production_simplicial():
            from archlab.architectures.simplicial_attention import (
                reference_simplicial,
                simplicial_attention,
            )

            torch.manual_seed(92)
            inputs = [torch.randn(1, 513, heads, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
                      for heads in (8, 2, 2, 2, 2)]
            oracle = [value.detach().float().requires_grad_() for value in inputs]
            positions = [0, 1, 31, 32, 127, 511, 512]
            actual = simplicial_attention(*(value.float() for value in inputs), 32, 512)[:, positions]
            expected = reference_simplicial(*oracle, 32, 512, query_positions=positions)
            torch.testing.assert_close(actual.float(), expected, atol=.025, rtol=.03)
            # Both paths must receive the identical BF16-representable dOutput;
            # comparing BF16 dOutput against a different FP32 draw is not a
            # gradient parity test.
            upstream = torch.randn_like(actual, dtype=torch.bfloat16).float()
            actual.backward(upstream.to(actual.dtype))
            expected.backward(upstream)
            errors = []
            for value, reference_value in zip(inputs, oracle, strict=True):
                torch.testing.assert_close(value.grad.float(), reference_value.grad, atol=.025, rtol=.04)
                relative = float((value.grad.float() - reference_value.grad).norm()
                                 / reference_value.grad.norm().clamp_min(1e-20))
                assert relative < .025, relative
                errors.append(relative)
            return {"sequence": 513, "query_heads": 8, "kv_heads": 2, "head_dim": 128,
                    "core_precision": "float32", "projection_precision": "bfloat16",
                    "windows": [32, 512], "oracle_query_positions": positions,
                    "forward_relative_l2": float((actual.float() - expected).norm() / expected.norm()),
                    "q_k1_k2_v1_v2_gradient_relative_l2": errors}

        check("production_simplicial_geometry_window_edges_and_five_gradients", production_simplicial)
    install_frozen_backward(bridge)
    args = native.ModelArgs(
        max_batch_size=1, max_seq_len=64, vocab_size=256, dim=128, moe_inter_dim=128,
        n_layers=5, n_mtp_layers=0, n_heads=16, n_routed_experts=4, n_activated_experts=2,
        q_lora_rank=64, head_dim=64, rope_head_dim=32, o_groups=8, o_lora_rank=64,
        window_size=8, index_n_heads=8, index_head_dim=64, index_topk=8,
        dspark_block_size=0, dspark_target_layer_ids=(), dtype="fp8", expert_dtype="fp4",
        compress_ratios=(0, 2, 2, 1, 1), kv_source_layers=(1, 3), index_source_layers=(1, 3, 4),
        candidate_source_layer=3, candidate_topk_blocks=2, candidate_block_size=2,
        engram_layer_ids=(1, 3), engram_num_embeddings=(1, 1), engram_max_ngram_size=3,
        engram_n_heads=2, engram_head_dim=32, engram_vocab_size=11, engram_compressed_vocab_size=256,
    )
    layout = native.EngramLayout.from_args(args)
    args.engram_num_embeddings = tuple(sum(sum(h) for h in layer) for layer in layout.primes)

    class TinyTokenizer:
        def __init__(self):
            self.backend_tokenizer = self

        def __len__(self):
            return 256

        def decode(self, ids, **kwargs):
            return f"token{ids[0]}"

        def id_to_token(self, token_id):
            return f"token{token_id}"
    # Float8/float4 initialization is explicit; random_ on the packed dtype is
    # unsupported. This is synthetic data, never a checkpoint conversion.
    def initialize(model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dtype == torch.float4_e2m1fn_x2:
                    p.view(torch.uint8).copy_(torch.randint(0, 256, p.shape, device=p.device, dtype=torch.uint8))
                elif p.dtype == torch.float8_e8m0fnu:
                    p.copy_(torch.full(p.shape, .015625, device=p.device).to(p.dtype))
                elif "norm.weight" in name:
                    p.copy_(torch.ones_like(p))
                elif name.endswith("_scale"):
                    p.fill_(.01)
                else:
                    p.copy_((torch.randn(p.shape, device=p.device, dtype=torch.float32) * .02).to(p.dtype))
        model.requires_grad_(False)

    def tiny_model():
        previous_dtype = torch.get_default_dtype()
        previous_device = torch.get_default_device()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                torch.manual_seed(72)
                oracle = native.Transformer(args, tokenizer=TinyTokenizer())
                initialize(oracle)
                candidate = bridge.Transformer(args, tokenizer=TinyTokenizer())
                candidate.load_state_dict(oracle.state_dict(), strict=True)
                candidate.requires_grad_(False)
                tokens = torch.randint(0, args.vocab_size, (1, 17), device="cuda")
            if pytorch:
                from archlab.automodel.deepseek_v41_pytorch import (
                    dequantize_base_once,
                    install_pytorch_leaves,
                )

                dequantize_base_once(candidate)
                install_pytorch_leaves(bridge, query_chunk=8)
            config = V41AdapterConfig(width=128, query_heads=4, kv_heads=2, head_dim=16,
                                      short_window=2, long_window=8)
            torch.set_default_device("cuda")
            with torch.no_grad():
                bridge_baseline = candidate.head(training_hidden(bridge, candidate, tokens)).clone()
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)
            adapters = {i: V41SimplicialAdapter(config).cuda() for i in (0, 2, 4)}
            torch.set_default_dtype(torch.bfloat16)
            attach_adapters(candidate, adapters)
            enable_activation_checkpointing(candidate)
            # The official implementation intentionally creates positions on
            # the default device; generate.py sets CUDA globally as well.
            torch.set_default_device("cuda")
            captures = [{}, {}]

            def capture_modules(model, destination):
                def hook(name):
                    def record(module, inputs, output):
                        # Tiny-model diagnostics only. Copy away from mutable
                        # shared-KV buffers before another layer reuses them.
                        values = output if isinstance(output, tuple) else (output,)
                        for index, value in enumerate(values):
                            if isinstance(value, torch.Tensor):
                                destination[f"{name}:{index}"] = value.detach().float().cpu().clone()
                    return record
                return [module.register_forward_hook(hook(name))
                        for name, module in model.named_modules() if name]

            handles = capture_modules(oracle, captures[0]) if pytorch else []
            try:
                expected = oracle(tokens)[1]
            finally:
                for handle in handles:
                    handle.remove()
            handles = capture_modules(candidate, captures[1]) if pytorch else []
            try:
                hidden = training_hidden(bridge, candidate, tokens)
            finally:
                for handle in handles:
                    handle.remove()
            if pytorch:
                differences = []
                for name, value in captures[0].items():
                    other = captures[1].get(name)
                    if other is not None and value.shape == other.shape and not torch.equal(value, other):
                        differences.append({"module": name, "shape": list(value.shape),
                                            "max_abs": float((other - value).abs().max()),
                                            "relative_l2": float((other - value).norm() / value.norm().clamp_min(1e-20))})
                report["pytorch_forward_diagnostic"] = differences
                print(json.dumps({"event": "pytorch_forward_diagnostic", "differences": differences}), flush=True)
            actual = candidate.head(hidden)
            torch.testing.assert_close(actual, bridge_baseline, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.03)
            relative = (actual - expected).float().norm() / expected.float().norm().clamp_min(1e-10)
            actual.square().mean().backward()
            first = {str(i): float(a.output.weight.grad.float().norm()) for i, a in adapters.items()}
            assert all(v > 0 for v in first.values()), first
            with torch.no_grad():
                for a in adapters.values():
                    a.output.weight.add_(a.output.weight.grad, alpha=-0.1)
            candidate.zero_grad(set_to_none=True)
            candidate.head(training_hidden(bridge, candidate, tokens)).square().mean().backward()
            second = {}
            for i, a in adapters.items():
                for name, p in a.named_parameters():
                    assert p.grad is not None and p.grad.isfinite().all(), (i, name)
                    second[f"{i}.{name}"] = float(p.grad.float().norm())
                    assert second[f"{i}.{name}"] > 0, (i, name)
            assert all(p.grad is None for name, p in candidate.named_parameters() if "simplicial_adapter" not in name)
            with torch.no_grad():
                full = candidate.head(training_hidden(bridge, candidate, tokens), full_logits=True)[:, :9]
                prefix = candidate.head(training_hidden(bridge, candidate, tokens[:, :9]), full_logits=True)
                torch.testing.assert_close(full, prefix, rtol=.03, atol=.015)
            activation_drift = None
            if pytorch:
                with torch.no_grad():
                    rounded_logits = candidate.head(training_hidden(bridge, candidate, tokens))
                    bridge._archlab_activation_mode = "bf16"
                    bf16_logits = candidate.head(training_hidden(bridge, candidate, tokens))
                    activation_drift = float((bf16_logits - rounded_logits).norm() / rounded_logits.norm())
                    bridge._archlab_activation_mode = "native"
            return {"relative_logits_error": float(relative.detach()), "first_output_gradient_norms": first,
                    "pytorch_first": pytorch, "bf16_activation_ablation_relative_error": activation_drift,
                    "second_gradient_norms": second, "engram_covered": True,
                    "checkpoint_recomputation_covered": True, "causal_prefix_covered": True,
                    "zero_adapter_matches_bridge_exactly": True,
                    "distributed_covered": False, "full_checkpoint_covered": False}
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.set_default_device(previous_device)

    check("native_fp4_fp8_shared_kv_two_step_backbone", tiny_model)
    report["passed"] = all(test["passed"] for test in report["tests"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pytorch", action="store_true")
    args = parser.parse_args()
    # Reserve a new evidence artifact; never overwrite an earlier probe.
    with args.output.open("x") as stream:
        try:
            report = run(args.checkpoint, args.container_kernel_packages, pytorch=args.pytorch)
        except Exception:
            report = {"passed": False, "training_launched": False, "traceback": traceback.format_exc()}
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
