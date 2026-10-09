# Concurrent matched reasoning evaluation

The evaluation uses the existing frozen normal/simplicial checkpoints while the
independent d128 scratch trainers continue. It creates no optimizer and cannot
signal a trainer. No new top-level MLflow experiment is created: the paired run
is filed under **DeepSeek V4.1 — Full fine-tuning**.

| Pair | Step | Matched supervised tokens | Backbone |
|---|---:|---:|---|
| Full fine-tuned, primary | 4537 | 756,364,650 | Trained along with adapters |
| Adapter-only, secondary | 307 | 50,119,869 | Frozen released backbone |

Each pair is evaluated sequentially on the same 16 GPUs, with a fresh process per
variant. The two variants are concurrent with training, not resident together.
Archived model source is loaded from its clean, checkpoint-qualified checkout;
the new driver is executed as a file. Both source identities are recorded. The
adapter pair uses its exact original source, including its original sparse
attention implementation, and must qualify its 32-to-16-rank inference mesh.

## Registered protocol

`recipes/deepseek_v41/eval_reasoning_eval_v1.yaml` fixes 280 native-chat prompts:
40 calibration cases and 240 held-out test cases. Five task families each have
two levels, with 12 counterfactual test pairs per level. Cases cover function
composition, two-hop relational joins, modular pair search, and one-hop retrieval
and specified-pair arithmetic controls. Function tables are random permutations;
modular YES/NO twins preserve both coordinate marginals. Levels 8/16 increase
table/search size; arithmetic uses the same distribution at both levels and is
an independently drawn control, not a difficulty manipulation.

Seeds, templates, prompts, labels, encoder/tokenizer identities, and token IDs
are sealed before model outputs. Labels are in a separate file never read by
the inference driver. Entire twin groups remain in the same split. Native-token
offsets audit the earliest/latest facts and query against the 512/32 windows at
the first answer position. The instantiated suite has all 280 cases eligible.

The decoder uses native non-thinking chat, greedy uncached full-prefix forwards,
and at most four new tokens. Every rank handles a different case but executes
the same collective schedule; completed EOS ranks keep participating. The last
partial batch is padded with unscored cases. Strict trimmed YES/NO exact accuracy
is primary; punctuation/explanation/invalid answers count as incorrect and are
reported separately. This evaluates short-answer reasoning, not extended CoT.

Reports include case accuracy, bootstrap intervals resampling whole twins, and
paired both-twins success with conservative Wilson intervals and exact McNemar
statistics. Calibration controls must reach 90% in both arms before interpreting
a search deficit. The held-out test is not tuned based on calibration outcomes.
Multiple task comparisons are descriptive; this small, single-checkpoint study
cannot establish broad intelligence or reproduce a paper's trained synthetic
task result. See `SIMPLICIAL_REASONING_RESEARCH.md` for motivation and limits.

## Admission and coexistence

1. Run the same distributed decoder on tiny normal and simplicial models.
2. Restore every full-checkpoint weight chunk with SHA-256 verification, or check
   the complete adapter payload and released-base config/index identities.
3. Re-evaluate the same sealed 1M-target math validation set and require CE to
   match the historical reference within 1e-4 before any scored prompt.
4. Pace evaluation at approximately 10% duty cycle, increasing rest when median
   training update time exceeds 1.30 times its pre-evaluation baseline.

PyTorch allocation is capped at 180 GiB/GPU, with at least 205 GiB free required
at admission. The node supervisor monitors total GPU and host memory, both
trainers' metric freshness, training completion, and update latency. It exits
only its own evaluation process group if GPU headroom falls below 55 GiB, host
headroom below 32 GiB, metrics go stale for five minutes, training completes,
or slowdown exceeds 75% for three minutes. Each phase is bounded to eight hours
and the queue to 24 hours. A shared evaluation STOP file propagates failure to
the other evaluation node. No training signals, checkpoint writes, or queue
mutations are part of this evaluator.

Median update ratios are a practical contention indicator, not causal GPU
utilization or MFU measurements. NAS checkpoint reads also contend for storage;
there is no claim that co-location has zero training cost. Numerical qualification
and inference use the validated uncached implementation; the previously rejected
incremental cache is not enabled.
