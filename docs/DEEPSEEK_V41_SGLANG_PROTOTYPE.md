# SGLang prototype for the full fine-tuned V4.1 pair

Status (2026-09-20): component prototype, **not an engine-ready export or deployed
SGLang service**. The existing inference service and training were not changed.
No throughput improvement is claimed.

## Implemented and checked

- `src/archlab/architectures/deepseek_v41_incremental.py`: request-local cached
  execution for both trained adapter types. It preserves the read/norm,
  projection, attention, gate, output, and residual-write equations. It retains
  only the last 32/512 short/long K/V entries, supports contiguous chunks and
  individual tokens, rejects replayed positions, and restores state on failure.
  This is an FP32 attention oracle, not a fused decode kernel. GPU BF16 precision
  and performance remain unqualified.
- `src/archlab/serving/v41_checkpoint_inventory.py`: read-only preflight for the
  reviewed 16-rank layout, plus checksum-verified reconstruction of bounded
  tensors. It distinguishes row shards, EP8/FSDP2 expert shards, and replicas.
  It rejects inconsistent rank metadata and replica checksums. It does not
  load optimizer state. Large tensors deliberately require a streaming exporter.
- `src/archlab/serving/sglang_v41_engram.py`: BF16 row-sharded lookup component
  compatible with the ordinary TP lookup interface. Tested with uneven and empty
  shards and exact BF16 values. The caller supplies the engine's reduction.
  This has not been installed or tested inside SGLang or across GPUs.
- Portable target/qualification contract:
  `recipes/experiments/deepseek_v41_sglang_prototype.yaml`.

Validation: 18 CPU tests passed. Coverage includes nonzero output weights,
chunked versus full execution, production window-boundary eviction, request
isolation, both checkpoint sharding axes, empty shards, corrupt payloads,
inconsistent replicas, and BF16 lookup preservation. Selected files passed Ruff.

Actual checkpoint probes used layer 4, 34 synthetic residual-stream tokens,
chunk sizes 17/16/1, and the reference full-prefix adapter equation. All loaded
adapter payloads passed SHA-256 checks and strict state-dict loading.

| Checkpoint | Supervised tokens | Unique tensor bytes | Max absolute adapter error |
|---|---:|---:|---:|
| Normal 4537 | 756,364,650 | 1,499,017,028,096 | 2.384185791015625e-7 |
| Simplicial 3620 | 603,590,955 | 1,499,100,918,272 | 2.384185791015625e-7 |

These are CPU FP32 adapter checks, not full-model logits, GPU BF16 parity,
held-out loss, or a token-matched architecture comparison. Full manifests were
checked; the full 1.5 TB payloads were not read or verified in this preflight.
Local receipts are in `.runtime/sglang-v41-prototype/CHECKPOINT_ADAPTER_PROBES.json`.

## Runtime findings affecting the port

The [published Dynamo preview](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0-deepseek-v4.1-flash-dev.1)
uses SGLang's preview container and quantized released weights. It does not
establish support for these full fine-tunes.

The inspected SGLang source revision is
`e54009240a84bf52eb7a21ec532ea49f1b9dd941`. This is a separately pinned main-tree
inspection, **not a claim about the exact source bundled in that preview image**.

1. V4.1 lives in
   [`deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/e54009240a84bf52eb7a21ec532ea49f1b9dd941/python/sglang/srt/models/deepseek_v4.py).
   The correct insertion is inside `forward_hc_pre_from_prev`, after attention
   HC expansion and before FFN coefficient prediction. A decoder-output hook
   would change the architecture. Fused precomputed FFN inputs must be invalidated
   at adapted boundaries, and fused paths that bypass the boundary must be disabled
   or explicitly integrated.
2. [`EngramEmbedding`](https://github.com/sgl-project/sglang/blob/e54009240a84bf52eb7a21ec532ea49f1b9dd941/python/sglang/srt/layers/engram.py)
   hardcodes FP8 storage plus scales, even though our checkpoint contains trained
   BF16 rows. Setting a general model dtype is insufficient. The replacement
   lookup component needs engine integration and distributed qualification.
3. [Published B300 recipes](https://github.com/sgl-project/sglang/blob/e54009240a84bf52eb7a21ec532ea49f1b9dd941/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx)
   use the quantized base. They do not validate BF16 expert execution under our
   original FP32 routing/projection/reduction policy.
4. External model packages are supported through `SGLANG_EXTERNAL_MODEL_PACKAGE`.
   Keep project integration under `src/archlab`; do not modify container-owned
   PyTorch, CUDA, NCCL, Transformer Engine, or NeMo sources.

## Work still required before service replacement

Implement the complete bounded-memory backbone export, including stacked expert
splitting/transposition, Engram rows, buffers, HC coefficients and adapter keys.
Verify coverage against the engine loader; do not fill missing trained tensors
from the released base. Preserve the tokenizer/rendering revision used by training.

Integrate both cached branches and BF16 Engram into a pinned engine package.
Initially disable speculation, prefix sharing, graphs, DP and CP; connect cache
allocation, cancellation, request-slot reuse, and completion cleanup. The current
Python cache is explicitly owned per request/layer, not a paged engine cache.

Then validate on B300 with all weights resident, compare prefill/decode logits
and held-out loss to the qualified service, and measure TTFT, inter-token latency,
and throughput at controlled context/output/concurrency settings. Only after
those gates pass should the existing MLflow endpoints be switched.
