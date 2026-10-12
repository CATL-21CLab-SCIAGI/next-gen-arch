# Limite agentic attention conversion

This experiment replaces attention in the **original Limite base** rather than
inserting another attention block. All backbone weights are trainable. The
scientific target is higher verified task-solving ability at an explicitly
measured additional compute cost; lower cross entropy alone is insufficient.
The former 2B/10B adapter checkpoints are historical comparisons, not parents.

## Transfer and bootstrap

[OPAL](https://arxiv.org/html/2609.31947v1) motivates transferring native weights,
checking the converted function, and recovering behavior on student-generated
trajectories. Its published intervention replaces 75% of attention with Gated
DeltaNet for efficiency. Our intervention replaces all 48 native reductions
with joint-softmax 2-simplicial attention, retaining the publisher's projections,
MLPs, Q/K normalization, local RoPE/global NoPE pattern, gates, value embeddings,
MUDD, and logit transformation. These are different experiments.

Each short key/value is `1 + delta(x)` with initially zero projection weights.
Duplicating an identical key/value over the valid short positions cancels in
joint softmax, recovering the inherited ordinary attention mathematically.
Floating-point kernels need separate numerical checks. Original-teacher hidden
matching/KL is initially zero in exact arithmetic, so it does not by itself
bootstrap agentic capability. The new short projections add 31,457,280 parameters
at native width 1280 and KV width 256, instead of another full attention stack.

The first recovery stage is a small, assistant-only CE bootstrap on complete
successful public bash trajectories, rendered by the actual MiMo agent codec
and unchanged Limite tokenizer/template. The initial budget is five passes over
135 training trajectories, with 14 held-out trajectories, no clipping, and
validation/checkpoints each epoch. The two-update smoke run is separate from
the completed 80-update bootstrap. See [data contract](agentic-sft-data.md)
for filtering, provenance and evaluation exclusions.

Native token IDs require explicit care: the pinned model generation EOS is
151645; tokenizer EOS/PAD is 151643. The bootstrap supervises assistant turns,
including their true EOS, and masks user/system/tool observations. It does not
remove thinking or relabel truncated outputs as complete answers.

## Teachers

The local Limite base and Violetto snapshots have byte-identical tokenizer JSON
(SHA-256 `296e081e2f5ecf9d87814aa9b0f4b12d670ed2b2e2be6c84e01a9466c953afb7`).
Violetto is therefore a candidate for directly aligned token distributions,
but its published strength is mathematical reasoning, not a demonstrated code
agent curriculum.

The inspected Qwen3, QwQ, DeepSeek-distilled-Qwen and MiMo candidates do not have
the same complete special-token mapping. In the local MiMo-V2.6-Flash snapshots,
ordinary vocabulary, merges, normalizer, pre-tokenizer and decoder match. Every
one of Limite's 151,667 token strings exists in MiMo: only the two thinking-token
IDs move, and MiMo adds eight special tokens. These are not admitted for naive
index-aligned logit KL. The extra tool-response tokens also change segmentation
of agent histories, so ID remapping alone does not establish sequence alignment.
A mapped-distribution teacher is a promising separate qualification target.
A larger teacher can still
produce text/tool demonstrations that are retokenized for Limite; exact
vocabulary equality is not required for text distillation. This is a bounded
search, not proof that no compatible large teacher exists.

## Observed bootstrap behavior

The initial five-pass run completed 80 updates and 4,664,565 logged input tokens;
held-out trajectory CE decreased to 1.3001. All 48 short projections updated,
and all five full-state checkpoints were exported without changing their trained
weight bytes. This establishes supervised learning, not task-solving ability.

The three held-out MiMo first-turn probes each exhausted 1,024 generated tokens
without a valid bash call or native EOS. Exact source reconstruction, runtime
prefix parity and supervised delimiter/EOS coverage passed. Teacher-forced
content predictions had low CE, but tool delimiters and EOS remained around
probability 0.00001 even at training positions. Original control-token output
rows are nearly identical (cosine above 0.999998); their nonzero training updates
were only about 0.034% of row norm. This supports inadequate protocol recovery,
without proving that it is the only cause. Violetto also failed a bounded native
tool-use control and is not yet an admitted agentic teacher.

A separate fresh bootstrap revision uses the same original converted parent and
data, learning rate 0.0001, ten epochs / 160 updates, validation every 16 updates,
and five full-state checkpoints every 32 updates. It completed 9,384,127 input
tokens. Validation CE reached 1.1437 at update 32 and ended at 1.4101, while
training CE continued decreasing. Update 32 still failed all six bounded
greedy/sampled first-turn tool probes; final behavior remains a separate check.
It must demonstrate generated tool behavior before a costly RL pilot. The
initial run and failed diagnostics remain preserved; neither qualifies RL.

An additional native-head audit found a genuinely broad vocabulary distribution
at protocol boundaries: over 85,000 tokens within two logits of the maximum in
one training example, versus one such token for ordinary content. Manual head
calculation and native forward matched exactly. Output gradients were nonzero,
so a saturated negative-logit floor does not explain this failure. A targeted
protocol-supervision fallback preserves the complete source conversations and
splits, selecting only assistant EOS and tool-delimiter targets. It is prepared
as a separate dataset, not silently mixed into the original loss or claimed as
task-solving evidence.

Staged SFT must declare `parent_stage=package_sft`; it verifies the actual trained
export, original publisher lineage and full replacement geometry. The default
`original_base` stage retains the zero-initialized conversion checks. Both start
fresh optimizers and record the exact supervision manifest.

## Admission and RL health

`archlab.training.bootstrap` composes the pinned upstream SFT engine; verl owns
CE, distributed Adam, checkpointing and monitoring. DP8 uses one unpacked native
sequence per GPU. Full package checkpoints retain model, optimizer/master
weights, scheduler, RNG and dataloader state. A strict finalizer checks and
receipts the package's trained HF export without reconstructing the model.
The subsequent RL experiment starts a fresh optimizer from that trained export.

Production RL requires actual paged serving and learner parity, correct native
termination and real tool/observation cycles, executable verifier results,
finite nonzero updates, useful reward groups, and checkpoint/resume evidence.
All-zero reward and gradient after context-length outputs is a failed pilot,
not healthy training. Asynchrony must remain bounded by explicit policy lag and
importance correction; it does not remove capability or verifier failures.

The first converted GPU probe had finite gradients in all 48 short projections.
The cached/full-prefix test did **not** pass its predeclared precision bounds:
relative-logit L2 reached 0.0713 versus 0.0678 in the unchanged FA2 base control;
raw captured TileLang reduction differences were at most 0.000102. This points
to native BF16 evaluation sensitivity but does not qualify cached serving.
Failures and acceptance bounds remain in ignored run receipts.

A separate control-calibrated admission version is declared before execution.
On the same sampled response tokens, compare paged serving and native cached
inference against learner replay, with an independently loaded original FA2
control. Mean and 95th-percentile absolute log-probability differences must not
exceed 1.25 times the native control plus 0.01; maximum absolute difference must
stay at or below 0.5 and the K3 KL estimate at or below 0.005. Captured raw
TileLang prefill/decode reductions must agree within relative L2 0.001. Request
isolation, cache windows, weight reload and native termination remain separate
requirements. This evaluates the probabilities actually used by RL, while
retaining the failed full-vocabulary 0.02/0.1 check explicitly. It is a new
versioned criterion, not a retroactive pass under the original bounds.

After 2-simplicial RL is stable with repeatable held-out reward signal, the other
DLC node may run a triadic replacement under a new matched contract. Match the
original parent, bootstrap data/order, trainable scope, rollout harness, reward,
held-out tasks and optimization/token budget; report parameter and wall-time
costs separately. A triadic run is not authorized for automatic promotion merely
because SFT loss decreases. The existing math RL job is preserved meanwhile.

## Maintained boundaries

- `architectures/limite_conversion.py`: native replacement and numerical contract.
- `data/agentic_sft.py`: reproducible prepared assistant targets.
- `training/bootstrap.py`: thin upstream SFT composition/admission.
- `training/limite_artifacts.py`: export checksums without a model dependency.
- `training/limite.py` and `megatron/limite.py`: native model/checkpoint boundaries.
- `training/serving_conversion.py`: stock vLLM paged-cache bridge.
- `recipes/sft/limite_agentic_bootstrap.yaml`: portable initial bootstrap contract.

Run-specific sources, logs, failures, qualification and trainer handles belong
under ignored `results/limite-conversion-20261013/`; datasets belong in OSS.
No maintained module imports the archived scheduler or legacy launchers.
