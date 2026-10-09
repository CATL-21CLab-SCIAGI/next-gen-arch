"""End-to-end admission for frozen native Limite and both inserted attentions."""

import argparse
import json
import time
from pathlib import Path

import torch

from archlab.architectures.limite_adapter import LimiteAdapterConfig, install_adapters
from archlab.architectures.limite_loader import load_model
from archlab.automodel.limite_adapter_common import frozen_fingerprint, loss_sum, save_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--attention-backend", choices=("native", "tilelang"), default="native")
    p.add_argument("--length", type=int, default=2048)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(a.model, attn_implementation="sdpa", device_map="cuda").eval()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    ids = tok(
        "Compute 13 + 29. The sum is 42. Therefore the answer is \\boxed{42}.", return_tensors="pt"
    ).input_ids.cuda()
    with torch.no_grad():
        parent = model(ids, use_cache=False).logits
    model = install_adapters(model, LimiteAdapterConfig(variant=a.variant, attention_backend=a.attention_backend))
    initial = frozen_fingerprint(model)
    with torch.no_grad():
        zero = model(ids, use_cache=False).logits
    zero_error = float((parent - zero).abs().max())
    assert zero_error == 0, zero_error
    train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train, lr=1e-4, weight_decay=0, betas=(0.9, 0.95), fused=True)
    ids = ids.repeat(1, a.length // ids.shape[1] + 2)[:, : a.length + 1]
    records = []
    for _step in range(3):
        t = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_sum(model, ids[:, :-1], ids[:, 1:]) / a.length
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(train, 1.0)
        assert torch.isfinite(norm)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
        torch.cuda.synchronize()
        records.append(dict(loss=float(loss), norm=float(norm), seconds=time.perf_counter() - t))
        print(records[-1], flush=True)
    assert frozen_fingerprint(model) == initial
    model.eval()
    short = ids[:, :160]
    with torch.no_grad():
        full = model(short, use_cache=False).logits[:, -1]
        out = model(short[:, :-1], use_cache=True)
        decoded = model(short[:, -1:], past_key_values=out.past_key_values, use_cache=True).logits[
            :, -1
        ]
        lp = full.log_softmax(-1)
        dp = decoded.log_softmax(-1)
        sample = lp.topk(128).indices
        cache_error = float((lp.gather(-1, sample) - dp.gather(-1, sample)).abs().mean())
    assert cache_error < 0.15, cache_error
    checkpoint = save_adapter(
        model, optimizer, a.output, 3, a.length * 3, extra=dict(frozen_sha256=initial)
    )
    before = {k: v.clone() for k, v in model.model.adapters.state_dict().items()}
    with torch.no_grad():
        train[0].add_(1)
    model.model.adapters.load_state_dict(torch.load(checkpoint / "adapter.pt", weights_only=True))
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=True))
    assert all(torch.equal(v, model.model.adapters.state_dict()[k]) for k, v in before.items())
    report = dict(
        passed=True,
        variant=a.variant,
        attention_backend=a.attention_backend,
        zero_output_error=zero_error,
        frozen_sha256=initial,
        trainable_parameters=sum(p.numel() for p in train),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        steps=records,
        cache_replay_top128_mean_error=cache_error,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        checkpoint_restore_exact=True,
    )
    (a.output / "QUALIFIED.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
