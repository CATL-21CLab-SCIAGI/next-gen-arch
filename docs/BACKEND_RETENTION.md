# Backend retention decision

**Type:** historical decision, 2026-08-30. **Decision:** retain speedrun as the frozen small-model oracle; use Megatron for scaling.

## Evidence

| Check | Result | Interpretation |
| --- | --- | --- |
| Three-seed 10M campaign | Delta correlation 0.971361; 13/15 directions agree | Broad ranking transfer, not numerical identity |
| Cold max-autotune lifecycle | 567.0 s Megatron versus 79.8 s speedrun | Retain cold-screen reference |
| Fresh contended full pair | BPB 1.549953 speedrun; 1.544295 Megatron | Different initialization/data-order identities; no architecture claim |
| Speedrun resume | Final BPB delta +0.00000279 | Complete state/cursor recovery |
| Megatron 10→20 resume probe | Step-11 loss exact; max loss difference 0.00004760; BPB delta +0.00000450 | Numerical continuation, not bitwise model/optimizer parity |

The fresh full pair used the same 9,363,488-parameter model, 112,459,776 training tokens and 3,932,160 validation tokens. Its contended-host timing is not the release speed comparison.

## Recovery repairs

| Commit | Repair |
| --- | --- |
| `acecdb0` | Restore ClimbMix microbatch cursor |
| `ee336b0` | Invoke native checkpoint loading and reject invalid restored iteration |
| `e883f2e` | Stable optimizer-group identities and validated in-memory legacy migration |

Failed and false-positive attempts remain excluded from successful resume evidence.

## Conditions for retirement

Retiring the standalone speedrun trainer requires mature resume parity, paired multi-seed/larger-scale reproduction, explained ranking differences, acceptable cold-start cost, shared-contract extraction and historical report regeneration.

Shared primitives have since moved to neutral modules, but frozen packing/tokenizer semantics still require care. Do not infer retirement readiness from one short probe.

[Backend comparison](BACKEND_COMPARISON.md) · [Infrastructure boundaries](TRAINING_INFRASTRUCTURE.md)
