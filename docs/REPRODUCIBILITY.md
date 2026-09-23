# Reproducibility

**Purpose:** reproduce a named result without changing its scientific contract.
**Start:** [Runbook](wiki/Runbook.md) and [provenance](PROVENANCE.md).

## Procedure

1. Identify the result, source revision, runtime and complete artifact manifest.
2. Verify data/tokenizer content and the declared comparison budget.
3. Reconstruct the exact model, optimizer, trainable set and data order.
4. Run the required numerical and checkpoint-continuation checks.
5. Launch into a new attempt namespace.
6. Compare matching checkpoints, evaluation sets and timing windows.

Use the existing validated GPU runtime. Development setup is in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Inspect frozen evidence

```bash
PYTHONPATH=src python -m archlab.cli verify
PYTHONPATH=src python -m archlab.cli show --size 100m --variant engram --seed 42
PYTHONPATH=src python -m archlab.cli render \
  --config recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml \
  --path data_root=/path/to/prepared/data
```

The frozen manifest includes historical checkpoint flags. Use the research-policy recipe for a new reportable run.

## Backend reproduction

The accepted safe-autotune policy uses max-autotune for 13 variants and default compilation for KDA, Kimi K3 KDA and Qwen GDN. Keep the `baseline` optimizer recipe when reproducing backend effects.

```bash
PYTHONPATH=src python -m archlab.speedrun.campaign_compare \
  --megatron-root /path/to/megatron-10m \
  --reference src/archlab/data/speedrun-10m-reference.csv \
  --output-dir /path/to/backend-comparison
```

Historical recovery/override sets are explicit in their policy manifests. Do not merge duplicate keys or silently replace failures.

## Key source identities

| Role | Revision |
| --- | --- |
| Historical nanochat | `b9f5025652d51470e2c31117100d9ff48717b911` |
| Modded lineage | `f411b3d346aa52d3504324ca93c230fd84c6c07f` |
| Historical Megatron | `55ac7082517c3878ae653c07c09c534b8aed49f6` |
| Accepted safe-autotune training source | `f8bc91df1aa10a2e4fd193cb9acdc4df3cdba975` |
| Safe-autotune policy | `f79d77c8f81f953666e58d6dcb4f1b52194bba2c` |
| Original comparison primary pass | `e6d9b0b1153e74078dbb87d4c0e8b12c8d4df513` |
| DSA correction | `ed8336e5403d8da75082502a96a115f06ee17334` |

Full file hashes and accepted row identities remain in the [frozen manifest](recorded-results/parameter-scale-100m-1b-v1-manifest.json), [safe-autotune ledger](recorded-results/megatron-10m-safe-autotune-b300/runs.csv), and [original backend ledger](recorded-results/backend-10m-runs.csv).
