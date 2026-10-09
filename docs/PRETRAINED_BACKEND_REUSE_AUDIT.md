# Pretrained Qwen backend selection audit

**Type:** historical audit, 2026-09-07.
**Outcome:** NeMo AutoModel was subsequently selected under an updated experiment contract.

## Candidates inspected

| Candidate | Revision | Finding |
| --- | --- | --- |
| NeMo AutoModel | `a4ce87c003f08b74d68684d3627f6e6048bc0140` | Exact text backbone, QSA, PLE and checkpoint conversion; preferred integration reference |
| mcore-bridge | `7965f71e9d16a3ca35cdf5351f5dffaa71574e64` | Exact-model Megatron adapter; initializer performs runtime patches |
| Transformers | `fc5c5bde8e656dad91cbf34e61940d984b1c7b91`, `c93057d4835cd31752bb56f59989dd27696eb45b` | Exact model exists, but inspected versions required APIs missing from the frozen installation |

All three inspected projects use Apache-2.0 licensing; preserve attribution for derived work.

## Audit evidence

| Check | Result |
| --- | --- |
| Configuration-only comparison | All 32 audited architecture fields match |
| Checkpoint index | 1,658 entries: 1,293 language, 333 vision, 31 MTP, one head |
| Simplicial discovery | 18 passed; five CUDA/distributed checks skipped |
| Full weight loading | Not performed during this audit |

The initial attribute-only import check overstated the `transformers.initialization` incompatibility; explicit submodule import succeeded.

## Integration requirements

1. Reuse an existing exact-model implementation before writing a backbone.
2. Account for every checkpoint tensor, including explicitly inactive vision/MTP.
3. Preserve the attention-residual → adapter → MoE insertion boundary.
4. Qualify 16K forward/backward, frozen state, optimizer membership and true process groups.
5. Verify complete checkpoint restoration and next-update continuation.

The initial audit's DP-only/no-runtime-patching restrictions were later superseded by approval for AutoModel's integration settings, EP and omission of inactive MTP. Use the [actual pretrained experiment](PRETRAINED_QWEN_NEXT_SIMPLICIAL.md) for that contract.

The original audit established a reusable candidate, not a launch-ready backend.
