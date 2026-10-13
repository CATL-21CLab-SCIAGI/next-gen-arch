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
and unchanged Limite tokenizer/template. The initial run used five passes over
135 training trajectories, with 14 held-out trajectories, no clipping, and
validation/checkpoints each epoch. The maintained recipe now uses the tested
ten-pass revision below. The two-update smoke run is separate from
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

The local Qwen3.6-35B-A3B teacher is supported by the existing serving runtime
with its upstream Triton MoE backend. All 26 weight shards and seven runtime /
license assets match the pinned publisher revision; the local README differs.
It uses another tokenizer, so this is a text-distillation candidate. Two actual
training-task controls failed the executable verifier, including one naturally
completed 18-turn episode. Neither trace is accepted as successful supervision.

The current bootstrap distills successful teacher text/tool trajectories,
not teacher logits. On-policy logit distillation is not enabled: the upstream
agent-framework rollout adapter rejects a teacher client, and token/history
alignment still needs qualification. A separate MiMo service would also need
the [supported serving stack](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL):
its packed MXFP4 checkpoint requires vLLM 0.31.0 or later, whereas the qualified
Limite runtime uses 0.20.2. That training runtime is kept unchanged.

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
greedy/sampled first-turn tool probes. At update 160, all three greedy probes
and two of three sampled probes produced a valid bash call and native EOS in
57–67 tokens. The remaining sampled response exhausted its 512-token probe
budget, despite containing five parseable calls, and was rejected as unfinished.
These six first-turn checks establish protocol recovery, not executable reward
or sustained RL learning. All five full-state checkpoints are preserved; actual
serving, verifier and nonzero-update qualification remain required.

A bounded real episode on training task `000311` completed two EOS-terminated
turns at temperature 1.0, but emitted malformed tool transcripts: zero actual
bash executions, no source patch, and executable reward zero. Parser error
feedback is not a real tool cycle; the admission gate requires executed bash
commands and their matching observations, rather than malformed-call placeholders
or argument errors returned before execution.
This diagnostic used 16K context, 1,024 tokens
per turn and 12 steps, and does not qualify the full production contract.

At temperature 0.6, the same bounded task produced five real bash executions
and observations across seven turns. It inspected the repository but made no
source edit; an attempted edit script failed with a syntax error. A broad search
also exhausted the diagnostic's 16K context, and the final turn was correctly
marked unfinished. Verifier reward remained zero. This is evidence of usable
tool execution, not successful repair, and motivates more diverse complete
teacher trajectories before a long RL pilot.

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

The historical bootstrap actually used Adam betas `(0.9, 0.999)`: the pinned
Megatron configuration translation ignored the generic `optim.betas` setting.
Its weights and receipts remain unchanged. New SFT/RL stages explicitly select
the intended `(0.9, 0.95)` through upstream `override_optimizer_config` fields;
admission checks the constructed optimizer's actual groups. A fresh RL start is
selected with `trainer.resume_mode=disable`. Scheduler restoration is enabled
separately so a later checkpoint resume retains the saved learning-rate step.

The broader staged curriculum contains 510 publisher-resolved successful
histories from 97 repositories: 461 training and 49 validation, retaining all
previous validation assignments. It supplies 5,349,111 input tokens and 815,686
assistant targets within 32K, without clipping assistant turns. Its 426 source
tool-output excerpts retain the publisher's explicit warning, head, omission
count and tail; omitted stdout is never invented or represented as complete.
These are complete delivered observations, distinct from unfinished model
responses. Source success labels have not been independently re-executed.
The continuation from update 160 completed one pass at LR 0.00005 with a fresh
optimizer: 57 finite stock DP8 updates, 4,832,967 logged input tokens and five
full-state saves at 12/24/36/48/57. Final validation CE was 0.934689 on the new
49-history split; it is not directly comparable with the earlier 14-history
validation set. All five trained exports were finalized without changing weight
bytes or resume state. Training used frozen source v4; the separately receipted
v5 finalizer handles the upstream writer's omitted tied-embedding alias metadata
by verifying the complete parent export and state digest, without rewriting it.

At update 57, all six bounded held-out first-turn probes produced native EOS
and at least one parsed bash call in 47–150 tokens, with no length caps. One
sample also contained malformed nested tool-call text. An actual training-task
episode subsequently executed 16 bash commands, all with native EOS and real
return-code observations, but repeated inspection instead of editing code and
reached the diagnostic's 16-turn ceiling. A diagnostic-only policy-version
metadata error occurred after environment cleanup and before the verifier result
was persisted; that verdict is unavailable and must not be inferred. Neither
first-turn protocol recovery nor lower CE establishes a successful repair or
healthy RL. Actual repair rewards determine readiness for RL.

A focused recovery continuation reuses 60 reviewed repair-and-retest histories
from that training split, without adding unique demonstrations or any of the
11 pilot tasks or their repositories. Three passes completed 21 finite updates
and 2,603,435 input tokens at LR 0.00002 with fresh Adam `(0.9, 0.95)`; complete
states were saved at 7/14/21. Stock distributed sampling used its actual default
seed 0, separately from trainer/model seed 42, and exposed 168 rows covering all
60 unique histories. Validation kept the same 49 histories (48 evaluated by the
stock loader): CE was 0.9370 at update 14 and 0.9769 at 21. This is a behavior
recovery curriculum, not evidence of successful repairs or intelligence gains.

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
The eight training and three validation code tasks form an availability pilot,
not a benchmark of architectural intelligence. Passing it establishes an
executable learning loop; an advantage requires broader held-out evaluation
under the matched comparison contract.
The conversion harness explicitly uses upstream tool-output excerpting at
12,000 characters, with its visible warning, while retaining the full 131,072
token context and native generated-response termination checks. This prevents
one broad shell search from unnecessarily filling the context. Harness hashes
and executable sandbox baselines are renewed together when that setting changes.

The task sandboxes have no internet access. Their package managers now receive
explicit offline settings, so an unavailable dependency produces a normal tool
error instead of waiting for a blocked download and losing the episode. In an
actual SQLGlot sandbox, installing the absent pytest package returned exit 1
in 0.35 seconds; the installed native unittest runner still passed 38 tests.
SDK/network failures remain infrastructure errors, never negative code rewards.

The first converted GPU probe had finite gradients in all 48 short projections.
The cached/full-prefix test did **not** pass its predeclared precision bounds:
relative-logit L2 reached 0.0713 versus 0.0678 in the unchanged FA2 base control;
raw captured TileLang reduction differences were at most 0.000102. This points
to native BF16 evaluation sensitivity but does not qualify cached serving.
Failures and acceptance bounds remain in ignored run receipts.

The trained update-160 checkpoint separately passed full 131,072-token serving
and backward memory checks with finite gradients. Its DP8 gradient comparison
failed the fixed 1% global / 2% layer bounds, including repeated native controls,
despite exact forward outputs and weights. Captured backward replays trace tiny
FP32 shared-gradient atomic differences across BF16 rounding boundaries, which
amplify through the layers. Accumulating the four shared K/V gradients in FP64
removed these differences: all 647 parameter gradients matched exactly across
six actual trained-model backward repeats, with unchanged forward outputs.
Seven focused B300 kernel checks and 34 CPU architecture/export checks passed.
The true replacement selects this precision explicitly; existing inserted
adapters retain their previous precision contract.

Only shared-gradient accumulation changes precision; tensor products and
returned gradients retain BF16, with FP32 softmax. Isolated native model
forward/backward time increased from 0.286 to 0.405 seconds at 2K tokens, and
from 0.605 to 0.952 seconds at 4K. These are diagnostic costs, not end-to-end
learner slowdowns. The stock DP8 learner baseline is 2.677 seconds per update
including optimizer/offload, or about 6,119 model tokens/second, excluding
generation and tools. The FP64 candidate measured 2.727 seconds per update,
1.85% slower, passing the unchanged 20% throughput-regression bound on all
eight ranks. The separate DP8 native-gradient comparison also passed its
unchanged bounds: all 647 parameter gradients matched exactly on every rank,
including repeated unchanged controls. This measurement used the historical
optimizer configuration. A fresh matched comparison with the corrected Adam
betas and scheduler settings measured 2.740 seconds for FP64 versus 2.869 for
FP32, again passing the unchanged throughput bound on all eight ranks. Five
samples per path establish the regression budget, not a reliable speedup.
The corrected stock checkpoint test also passed on all eight ranks: model and
master weights, moments, RNG and scheduler restored, and the subsequent update
met the unchanged numerical bounds. Constructed optimizer groups used Adam
betas `(0.9, 0.95)`. Empty local optimizer groups can acquire a broadcast step
counter on load; only that nonnumerical empty-group metadata is excluded from
the fingerprint. Populated-group counters and all tensor state remain checked.
Real-policy reward gates remain separate requirements. Production remains
gated; full-context stability alone does not waive them.

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

An update-57 serving diagnostic isolated a separate prefill/decode discrepancy:
simplicial prefill used 128-key tiles while cached decode used 64. BF16 online
probability rounding depends on that partition. Matching decode to 128 makes
all six fixed native replay cases pass the unchanged control-calibrated bounds;
ordinary attention retains 64. Twenty focused B300 kernel checks also pass.
This changes cached inference only, preserving training and checkpoint bytes.
Actual full serving and RL admission remain separate requirements.

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
