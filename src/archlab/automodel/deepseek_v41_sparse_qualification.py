"""Independent numerical admission check for the batched sparse backend."""
from __future__ import annotations


def qualify_batched_sparse():
    import torch
    from archlab.automodel.deepseek_v41_scratch_high_mfu import high_mfu_sparse_attention
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention

    # Preserve the training initialization and stochastic-rounding RNG streams.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(703)
        b, t, h, d, n, slots = 1, 33, 64, 64, 45, 65
        q = torch.randn(b, t, h, d, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(b, n, d, device='cuda', dtype=torch.bfloat16)
        sinks = torch.randn(h, device='cuda', dtype=torch.float32)
        indices = torch.randint(-1, n, (b, t, slots), device='cuda', dtype=torch.int32)
        indices[:, -1] = -1
        indices[:, 0, 1:] = -1
        indices[:, 0, 0] = 0
        indices[:, 2, :4] = 1
        cotangent = torch.randn_like(q)
        x, k, s = [v.float().detach().requires_grad_() for v in (q, kv, sinks)]
        selected = k[torch.arange(b, device='cuda')[:, None, None], indices.clamp_min(0).long()]
        scores = torch.einsum('bthd,btkd->bthk', x, selected) * d**-.5
        scores = scores.masked_fill(indices[:, :, None, :] < 0, -torch.inf)
        probabilities = torch.cat((scores, s[None, None, :, None].expand(b, t, -1, -1)), -1).softmax(-1)[..., :-1]
        expected = torch.einsum('bthk,btkd->bthd', probabilities, selected)
        reference = [expected.detach(), *torch.autograd.grad(expected, (x, k, s), cotangent.float())]
        x, k, s = [v.detach().requires_grad_() for v in (q, kv, sinks)]
        actual = high_mfu_sparse_attention(x, k, s, indices, d**-.5,
                                          backend='tilelang', reference_rounding=True)
        values = [actual.detach(), *torch.autograd.grad(actual, (x, k, s), cotangent)]
        errors = {}
        for name, value, target in zip(('output', 'dq', 'dkv', 'dsink'), values, reference):
            if not bool(value.isfinite().all()):
                raise AssertionError(f'sparse qualification: {name} is nonfinite')
            error = float((value.float()-target).norm()/target.norm().clamp_min(1e-20))
            if error >= .01:
                raise AssertionError(f'sparse qualification: {name} relative error {error} >= .01')
            errors[name] = error
        with torch.no_grad():
            native = dsv4_sparse_attention(q, kv, sinks, indices, d**-.5,
                                            backend='tilelang', reference_rounding=True)
        if not torch.equal(native, actual):
            raise AssertionError('sparse qualification: forward changed native output')
        if bool(values[0][:, -1].count_nonzero()) or bool(values[1][:, -1].count_nonzero()):
            raise AssertionError('sparse qualification: masked row is not zero')
        return {'passed': True, 'native_forward_exact': True, 'relative_l2_to_fp32': errors,
                'relative_l2_limit': .01, 'masked_and_duplicate_indices': True,
                'rng_preserved': True, 'shape': [b, t, h, d], 'slots': slots}
