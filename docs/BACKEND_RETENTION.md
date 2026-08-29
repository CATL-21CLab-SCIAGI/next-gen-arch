# Backend retention decision

Updated: 2026-08-30 (UTC+8)

## Decision

Megatron is the default backend for scaling, distributed training, and the active 7B
line. The speedrun execution path is frozen as the 1M/10M cold-start screen and
regression oracle. Do not delete it yet.

This is deliberately narrower than keeping two evolving trainers. New architecture
math stays shared; new distributed/runtime work targets Megatron. A controlled campaign
uses one backend throughout and never treats a cross-backend absolute loss difference
as an architecture improvement.

Deleting the whole `archlab.speedrun` package would also delete code currently reused
by the Megatron comparison wrapper: campaign contracts, model construction, ClimbMix
packing, tokenizer handling, runtime helpers, and optimizer lineage. Retiring the
standalone `base_train` lane therefore requires moving those shared contracts into a
backend-neutral package first.

## Fresh smallest full-scope check

One fresh seed-42 run per backend used the same scientific contract:

- ClimbMix content manifest identity
  `8f50ff9f3ce41d29cca1d0676d400f851f622e62777e094e8d6023e25452c109`;
- tokenizer SHA-256
  `77cd24eae5d4a1c580dcf6af113caceb1c58de99a45c64b5c73abd4c2f329c31`;
- baseline d5/w56 model, 9,363,488 parameters, seed 42, BF16, sequence length 2,048;
- 286 steps, 393,216 tokens/step, and 112,459,776 training tokens;
- 3,932,160 fixed validation tokens.

| Metric | speedrun | Megatron `compile` |
| --- | ---: | ---: |
| Final validation BPB | 1.549953 | 1.544295 |
| Final training loss | 5.107197 | not emitted by the wrapper |
| Process/lifecycle wall | 199.8 s | 228.9 s |
| Aggregate measured tok/s | 761,313 | 688,163 |
| Median steady tok/s | 750,939 | 676,623 |
| Peak reserved memory | 2.56 GiB | 6.68 GiB |

The host was contended during these two full runs, so their throughput is validation
evidence, not the release performance comparison. The two backends also produced
different initialization hashes and data-order IDs. They share the dataset, tokenizer,
seed, budget, and architecture contract, but are not bitwise the same run. In
particular, Megatron's `-0.005658` absolute BPB difference is a backend effect and must
not be reported as an architecture gain.

The stronger exclusive-card evidence remains the 16-variant, three-seed 10M campaign:
paired architecture deltas correlate at `0.971361`, mean absolute delta gap is
`0.009241 BPB`, and direction agrees for 13 of 15 non-baseline variants. Under the
measured safe-autotune policy, Megatron's baseline reaches 1,576,266 median steady
tok/s and 1,529,304 aggregate tok/s versus the historical speedrun aggregate of
1,360,228 tok/s. Fresh cold max-autotune process wall is much worse, 567.0 seconds
versus 79.8 seconds, which is why the small cold-start lane remains useful.

## Checkpoint evidence and repairs

The speedrun run resumed from step 143 to 286 with the same stable run ID. Its final
BPB was `1.549956`, only `+0.00000279` above the uninterrupted `1.549953`; final loss
changed by `+0.00002575`. Model, optimizer, per-rank RNG, and packing cursor artifacts
are complete.

The first real Megatron resume audit found two independent defects:

1. the custom optimizer adapter had replaced MCore setup without calling native
   `load_checkpoint`, so `--resume` silently restarted at iteration zero;
2. after native loading was restored, historical custom optimizer groups lacked the
   stable metadata MCore needs to reorder groups.

The fixes are:

- `acecdb0`: restore the exact ClimbMix microbatch cursor from loaded iteration;
- `ee336b0`: restore native model, optimizer, RNG, and schedule checkpoint state and
  reject a non-positive or inconsistent restored iteration;
- `e883f2e`: persist stable optimizer-group IDs and migrate older group metadata only
  in memory after group count, parameter count, kind, LR, and canonical-group
  fingerprints match.

The failed and false-positive runs remain preserved. They are not accepted as resume
evidence.

A clean same-commit probe at `e883f2e` then compared an uninterrupted 20-step run with
a run restored from its own iteration-10 checkpoint. Both attempts have the same run
ID, shared-initialization hash, data-order ID, and source commit. The restored attempt
reported iteration 10, started training at iteration 11, and did not invoke legacy
optimizer-group migration. Step-11 loss matched exactly; the maximum absolute loss
difference over steps 11--20 was `0.00004760`, and final validation BPB changed from
`2.01305300` to `2.01305749` (`+0.00000450`). Final RNG tensors match bit-for-bit.
Model and optimizer tensors do not, so this is strong numerical resume parity under
the compiled runtime, not bitwise model-state parity.

## Operating policy

| Workload | Default | Reason |
| --- | --- | --- |
| 1M/10M one-off screen | speedrun | lowest cold-start cost and historical oracle |
| 1M/10M backend qualification | both | detects harness-dependent architecture signal |
| 100M+ or repeated compiled runs | Megatron | better steady-state/scaling path |
| multi-node DP/TP/PP/CP/EP | Megatron | container-owned distributed lifecycle |
| active 7B training | immutable launched checkout | never changed by small tests |

The standalone speedrun trainer is now maintenance-only: correctness, provenance,
checkpoint compatibility, and security fixes are allowed; new architecture mechanisms
must be implemented in shared architecture modules rather than a second trainer fork.

## Gates before deleting the speedrun trainer

All gates are required:

1. Megatron resume parity passes from a complete intermediate checkpoint, including
   model, optimizer, RNG, schedule, and exact data cursor. The 20-step probe passes
   numerical parity; repeat this at the complete 10M budget before deletion.
2. Three paired seeds at 10M and at least one larger scale reproduce the baseline and
   selected architecture deltas under one documented Megatron contract.
3. Variant direction agreement reaches at least 95%, or every exception has a
   mechanism-specific explanation and larger-scale control.
4. Cold-start wall at 10M is at most 1.25x speedrun, or compiler caches are provisioned
   before the measured lifecycle.
5. Shared model/data/tokenizer/optimizer contracts have moved out of
   `archlab.speedrun`, and historical speedrun checkpoints remain readable.
6. The published 1M--1B reports can be regenerated without executing the retired
   trainer.

Until then, “drop speedrun” means drop it as the scaling default, not delete the code or
its artifacts.

## Durable evidence

- Fresh full pair and speedrun resume:
  `/mnt/nas/evergreen/next-gen-arch/backend-retention-d8c2e9e`
- Megatron resume-repair probes:
  `/mnt/nas/evergreen/next-gen-arch/backend-retention-acecdb0`
- Native-load and optimizer-compatibility probes:
  `/mnt/nas/evergreen/next-gen-arch/backend-retention-ee336b0`
  and `/mnt/nas/evergreen/next-gen-arch/backend-retention-e883f2e`
- Frozen content manifest:
  `/mnt/nas/evergreen/next-gen-arch/backend-retention-d8c2e9e/climbmix-171.manifest.json`

See [BACKEND_COMPARISON.md](BACKEND_COMPARISON.md) for the complete three-seed campaign
and [OPTIMIZATION_AUDIT.md](OPTIMIZATION_AUDIT.md) for the Marin/Modded optimization
ledger.
