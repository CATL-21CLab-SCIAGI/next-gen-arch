# Experiment contracts

**Purpose:** define which conclusions a run can support. **Reader guide:** [Experiment Design](wiki/Experiment-Design.md).

## Comparison regimes

| Regime | Budget | Required interpretation |
| --- | --- | --- |
| `controlled` | Matched effective training tokens | Component effect under the matched recipe |
| `fixed_compute` | Matched algorithmic model FLOPs | Quality at equal model-compute budget |
| `scaling` | Declared tokens per parameter | System behavior across model/training scale |

Discrete-step rounding, actual parameter counts, executed fallback FLOPs and measured throughput remain explicit.

## Paired axes

| Work | Match or declare |
| --- | --- |
| All comparisons | Seed, initialization, data order, tokenizer, context, batch, optimizer, precision, evaluation |
| Pretrained adaptation | Parent weights, insertion sites, initialization, trainable set and phase transition |
| RL | Behavior policy, sampling settings, reward, normalization, replay, generated tokens and applied updates |
| Systems comparison | Hardware, topology, compilation state and timing window |

```bash
PYTHONPATH=src python -m archlab.cli pair-check \
  --baseline /path/to/baseline/result.json \
  --variant /path/to/variant/result.json
```

## Promotion and reporting

1. Pass construction, numerical, optimizer and checkpoint-continuation gates.
2. Confirm with paired seeds at small scales; the historical confirmation set is 42–44.
3. Promote only under declared quality, stability and efficiency criteria.
4. Retain constituent controls when combining mechanisms.
5. Keep the largest scale separate when testing a scaling-law prediction.

Report paired deltas and seed variation. Evaluation-sample confidence intervals do not replace independent training seeds.

Failed seeds stay failed; favorable earlier checkpoints are not replacement endpoints. Changed numerical, capacity or scientific settings need a new contract.

## Historical boundary

The frozen 100M–1B manifest includes `--no-save-final-checkpoint` for reproduction. New research runs require complete final state. Fixed-token and tokens-per-parameter campaigns answer different questions; keep their absolute losses in separate comparisons.
