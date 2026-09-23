# Pretrained Qwen sampling

**Type:** historical single-GPU inference protocol.
**Entry:** `src/archlab/automodel/sample.py`.

## Contract

| Item | Setting |
| --- | --- |
| Backbone | Released Qwen3.8-Flash-Next text model; original QSA/GDN/PLE/MoE/residual structure |
| Added weights | Exact adapter subset from a complete distributed checkpoint |
| Precision | Training compute precision; no quantization |
| Placement | PLE table in CPU RAM with original lookup; other execution on one GPU |
| Input | Unpadded raw continuation; no chat template |
| Sampling | Temperature 0.8; top-p 0.95; up to 128 response tokens; per-prompt seeds 42–45 |
| Cache | Full-prefix recomputation |
| Prompt asset | `src/archlab/prompts/backbone_validation.yaml` |

The recorded DSW runtime differs from training. Results record its versions and do not claim bitwise cross-runtime parity.

## Invocation

Use the checkpoint's exact architecture source on the import path:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$TRAINING_SOURCE/src:$AUTOMODEL_SOURCE" \
"$SAMPLING_PYTHON" src/archlab/automodel/sample.py \
  --base "$VERIFIED_PRETRAINED_CACHE" \
  --checkpoint "$COMPLETE_ADAPTER_CHECKPOINT" \
  --prompts src/archlab/prompts/backbone_validation.yaml \
  --output "$NEW_SAMPLING_OUTPUT"
```

The output must be fresh. Atomic `samples.json` records token IDs, text, identities, timings and completion state.

## Qualification and limits

Checks cover exact adapter restore, shape/dtype rejection, PLE lookup equality, base loading, nonzero branches and full head/window geometry. Complete loading audits missing/nonfinite tensors.

The ordinary unfused attention fallback at growing prefixes was separately checked. This is a bounded sampling tool, not a long-context throughput benchmark.

[Capability pilot](PRETRAINED_CAPABILITY_EVALUATION.md) · [Training lineage](PRETRAINED_QWEN_NEXT_SIMPLICIAL.md)
