# Resident RL cache status — 2026-09-23

The main branch contains an **experimental, disabled-by-default** resident-policy
cache. The running v3 jobs still execute their recorded `source-v2` checkout with
512 response tokens. This commit does not admit the cache for production.

## Implementation

- `automodel/deepseek_v41_rl_cache.py` captures prompt state, then executes the
  existing decoder blocks with incremental CSA2 attention, bounded adapter KV
  windows, and the original Engram and MoE modules.
- Normal adapters use container FlashAttention. Simplicial adapters use a
  one-query form of the existing deterministic Triton forward arithmetic.
- Cached rollouts record actual execution shapes separately from the padded
  full-prefix shapes used by policy-gradient replay. Sampling distributions,
  EOS handling, and the 0.02-nat replay limit are preserved.
- The opt-in recipe flag is `cache_policy`. Cached evaluation currently requires
  one prompt per rank because local prompt rows must have equal lengths.
- Actual-actor qualification compares hidden states and all vocabulary log
  probabilities against padded full-prefix forwards. A separate real rollout
  and gradient-replay audit is required before training admission.

## Evidence so far

The focused CPU suites passed **95 tests** using the DLC image's existing
`/opt/venv/bin/python` runtime. These cover source layout, artifacts, the existing
proposal and Qwen modules, rollout/evaluation, and RL training/update protocols.
The cache protocol tests include unequal EOS times and reconstruction of every
full replay prefix.

Small B300 checks compared the new adapter kernels with their full-sequence
counterparts at lengths 1, 5, 17, and 33; maximum absolute error was zero. A tiny
CSA2 source-attention check also matched across five compression boundaries.
A six-layer random model with nonzero adapter output weights was checked at four
incremental positions: simplicial hidden outputs matched exactly; normal maximum
absolute hidden error was 0.0078125 at one position and zero at the others.
These small GPU checks used bare system Python and do not constitute a receipt
for the full pretrained, distributed production actor.

## Required before restart

1. Qualify the distributed FSDP/EP/Engram cache path and ownership cleanup using
   the existing container runtime, then qualify the actual step-4537 actors.
2. Replay a real 512-token cached rollout within 0.02 nats and measure generation
   speed and memory.
3. Measure completion rates and replay memory at 1,024 tokens and a larger admitted
   budget. The current 2,048-token context allows a uniform response budget of at
   most 1,489 for the longest training prompt. A larger context needs its own
   memory and numerical qualification.

The existing sparse-attention backward already uses TileLang. Its project wrapper
gives each query private KV-gradient storage and performs an ordered reduction.
The cache is an inference change and does not replace that backward path.
