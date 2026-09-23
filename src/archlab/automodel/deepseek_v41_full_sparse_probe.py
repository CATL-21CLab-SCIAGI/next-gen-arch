"""Compare trainable sparse-sink gradients to the official native kernel."""
import torch
from archlab.automodel.deepseek_v41_official_sparse import deterministic_sparse_attention


def qualify_sparse_sink():
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(53)
        q = torch.randn(1, 33, 4, 64, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(1, 45, 64, device='cuda', dtype=torch.bfloat16)
        sinks = torch.randn(4, device='cuda', dtype=torch.float32)
        indices = torch.randint(-1, 45, (1, 33, 17), device='cuda', dtype=torch.int32)
        indices[:, -1] = -1
        cotangent = torch.randn_like(q)
        results = []
        for fn in (dsv4_sparse_attention, deterministic_sparse_attention):
            x, k, s = q.clone().requires_grad_(), kv.clone().requires_grad_(), sinks.clone().requires_grad_()
            y = fn(x, k, s, indices, 64 ** -.5, backend='tilelang', reference_rounding=True)
            y.backward(cotangent)
            results.append((y.detach(), x.grad, k.grad, s.grad))
        assert torch.equal(results[0][0], results[1][0])
        relative = float((results[1][-1] - results[0][-1]).norm() / results[0][-1].norm().clamp_min(1e-20))
        if relative > 2e-5 or not torch.isfinite(results[1][-1]).all():
            raise AssertionError(f'attention sink gradient differs from official kernel: {relative}')
        return {'passed':True, 'forward_byte_exact':True, 'sink_gradient_relative_l2':relative}
