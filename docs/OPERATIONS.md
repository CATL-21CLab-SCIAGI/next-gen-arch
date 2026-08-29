# Training operations

## Preflight

Before a GPU launch:

1. resolve the exact DLC/DSW job and pod identities;
2. verify mounts, GPU type/count/free memory, active processes, and campaign locks;
3. use a clean immutable checkout and record its commit;
4. verify dataset manifest, tokenizer identity, runtime versions, and container identity;
5. render the full command and confirm comparison regime, seed, budget, output namespace, and parent checkpoint;
6. run one load/forward/backward batch and the required distributed collective probe;
7. refuse an output directory that already contains a different contract.

Never launch on a GPU merely because its utilization is momentarily low. It must have no compute process and must be explicitly allowlisted.

## Research-policy FineWeb run

```bash
python -m archlab.megatron.train \
  --dataset fineweb10b \
  --data-root /datasets/fineweb10B \
  --data-manifest /datasets/fineweb10B.manifest.json \
  --data-verification metadata \
  --scale 1m \
  --variant baseline \
  --seed 42 \
  --comparison-regime controlled \
  --target-train-tokens 7864320 \
  --artifact-policy research \
  --initialization-hash shared \
  --run-dir /runs/baseline-seed42 \
  --checkpoint-dir /runs/baseline-seed42/checkpoints \
  --save-interval 10 \
  --backend-profile compile \
  --optimization-recipe baseline
```

For scaling, replace the controlled budget with `--comparison-regime scaling --tokens-per-parameter 12`. For fixed compute, use `--comparison-regime fixed_compute --target-model-flops ...`.

## Research-policy speedrun recipe

The portable speedrun renderer removes the historical no-checkpoint switch and injects the content manifest, raw metrics, shared-initialization hash, and an output-local checkpoint directory:

```bash
export NGA_DATA_ROOT=/datasets/nanochat
export NGA_DATA_MANIFEST=/datasets/climbmix.manifest.json
export NGA_OUTPUT_DIR=/runs/speedrun-baseline-100m-seed42

next-gen-arch render \
  --config recipes/experiments/speedrun_research_baseline_100m_seed42.yaml
```

For ClimbMix, `NGA_DATA_ROOT` is the nanochat base directory and the manifest inventory is rooted at its `base_data_climbmix/` child. A research resume uses the same output and adds `--resume-from-step N`; it appends attempt-tagged metrics, checks the stable run ID, restores the exact packing cursor and per-rank RNG, and creates a new immutable attempt directory.

## Resume

Resume into a new attempt directory and point `--checkpoint-dir` at the complete checkpoint:

```bash
python -m archlab.megatron.train ... \
  --resume \
  --run-dir /runs/baseline-seed42-attempt2 \
  --checkpoint-dir /runs/baseline-seed42/checkpoints
```

The restored Megatron iteration drives the FineWeb microbatch cursor. Do not manually choose a “latest-looking” checkpoint or remove a stale running marker without first verifying the process is gone.

## Throughput

Training records:

- lifecycle wall time;
- post-warmup aggregate tokens/s;
- median and p90 step time;
- steady-state tokens/s;
- exact warmup and measurement interval counts.

Both backends expose the same predeclared warmup/measurement protocol. A speedrun attempt stores its own lifecycle wall time and steady-state window separately from legacy accumulated training time.

The default excludes the first ten optimizer-step intervals. Short performance probes should explicitly set a fixed measurement window. Compare isolated runs on the same topology and avoid treating throughput from a host running other NAS- or GPU-intensive work as a clean kernel benchmark.

## Failure handling

| Category | Identical retry? | Action |
| --- | --- | --- |
| operational | yes | resume from a complete checkpoint with a new attempt ID |
| numerical | no | retain failure; diagnose stability and create a new contract if changed |
| capacity | no | change batch/topology as a newly named contract |
| contract | no | repair provenance, path, or harness precondition |
| unknown | no | manual triage |

All ranks fail together on non-finite loss or gradients. A divergent seed remains failed; an earlier favorable checkpoint is not the endpoint.

## Long-running jobs

Treat a running training checkout and output namespace as immutable. Development happens in a separate local checkout and is synced to a new commit-named NAS directory only for a new smoke test or launch. Pushing Git does not authorize replacing a running checkout, restarting a job, or changing its shared output files.
