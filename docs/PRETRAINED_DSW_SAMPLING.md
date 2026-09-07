# Pretrained additive-model sampling on DSW

Entry: `src/archlab/automodel/sample.py`. This is inference only. It does not
resume training, restore optimizer/RNG state, stop services, or modify installed
packages. Use the existing user-approved DSW sampling environment, not a newly
created environment. Its package versions differ from the frozen DLC training
container and are recorded in each result.

The base is the complete language backbone from Qwen3.8-Flash-Next, including
the original QSA, GDN, PLE, four-stream hyperconnections, and MoE. Only the
originally inactive vision/MTP components remain inactive. The checkpoint loader
is the pinned, unmodified NeMo AutoModel adapter, not a local tensor converter.

## Memory and inference policy

- No quantization. Frozen weights retain their training compute precision,
  including the upstream GDN FP32 exceptions and FP32 RoPE buffers.
- The complete PLE table remains in CPU RAM. The backend's original single-owner
  embedding performs the lookup, with Accelerate's `AlignDevicesHook` returning
  the result to the GPU. The table's shape, values, hashing, and checkpoint keys
  do not change. Everything else executes on one GPU, without EP/FSDP.
- Only the trained simplicial tensors are restored from the distributed
  checkpoint. Their FP32 master weights are checked and hashed before casting
  to BF16, matching training's FSDP compute precision. The full-adapter hash is
  **not** a training rank-state/optimizer hash.
- Sampling is an unpadded raw continuation, not chat formatting. The backend
  does not implement a generation cache: each token recomputes the full prefix.
  `logits_to_keep=1` avoids projecting all prefix positions to vocabulary logits.
- Growing prefixes can exhaust the upstream static FlexAttention wrapper's
  Dynamo recompilation limit. PyTorch then uses ordinary unfused attention for
  these short prefixes. A full-head-geometry test checks it against compiled
  attention; no debug bypass is enabled. This is not a long-context throughput
  benchmark or a claim of bitwise cross-runtime parity.
- Temperature 0.8, top-p 0.95, 128 new tokens maximum, per-prompt seeds 42–45.
  Prompts are loaded from the versioned `backbone_validation.yaml`, not copied
  into the sampler. EOS can stop a sample earlier.
- The sampler reserves GPU headroom and aborts if the placement budget does not
  fit. It never evicts Ollama or changes the running DLC job.

## Reproduction

Inject machine paths at launch. Run the new entry **by filename** with the
checkpoint's exact training source snapshot on `PYTHONPATH`; the entry checks
model/integration source hashes against the checkpoint contract. Also put the
qualified upstream AutoModel checkout on `PYTHONPATH`.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$TRAINING_SOURCE/src:$AUTOMODEL_SOURCE" \
"$SAMPLING_PYTHON" src/archlab/automodel/sample.py \
  --base "$VERIFIED_PRETRAINED_CACHE" \
  --checkpoint "$COMPLETE_ADAPTER_CHECKPOINT" \
  --prompts src/archlab/prompts/backbone_validation.yaml \
  --output "$NEW_SAMPLING_OUTPUT"
```

The output directory must not exist. `samples.json` is published atomically
after loading and after each completed prompt, with `complete: true` only after
all prompts finish. It includes token IDs, decoded continuations, runtime and
source provenance, checkpoint identity, offload policy, timings, and seeds.
Do not compare samples across environments as a bitwise numerical parity test.

## Qualification

`tests/test_automodel_sampling.py` checks exact adapter-only DCP restoration,
rejection of mismatched keys/shapes/dtypes, decoding configuration validation,
bitwise CPU-vs-GPU equivalence of the original PLE lookup, an exact single-owner
base-checkpoint roundtrip through the upstream loader, and uncached small-model
GDN/QSA/MoE forwards with nonzero simplicial additions. The full-width
leaf-kernel suite is `tests/test_simplicial_head256.py`, covering FP32/BF16 forward
and backward behavior and both window boundaries. Full loading additionally
audits checkpoint key coverage and detects missing/nonfinite weights using
NaN-poisoned destinations.
