# Pretrained architecture conversion for reasoning

**Research review and proposed next experiment, 2026-10-10.** Architecture
conversion after pretraining is an established method, including conversions
that improve capability. Our useful target is more specific: **does a higher-order
attention component improve verified reasoning and learning of unseen
compositions beyond an equally trained ordinary-attention control?**

The present Limite experiment adds an attention module before each original
block. It tests augmentation. Attention-to-SSM conversion usually replaces an
existing computation and then recovers its behavior; dense-to-MoE upcycling
usually adds conditional capacity. These interventions have different controls.
This proposal does not change the active RL recipe or authorize a new launch.

## Late-stage patching and the MAPLE lead

The linked [late-stage patching article](https://hunchings.github.io/late-stage-patching/)
describes **[On-Policy Attention Linearization, OPAL](https://arxiv.org/html/2609.31947v1)**.
It replaces 27 of 36 attention layers with Gated DeltaNet, followed by about
1B alignment/off-policy distillation tokens and 2B student-generated tokens
with teacher supervision. The final stage is on-policy distillation, not RL
from verifiable rewards. Inherited weights are unfrozen after initial alignment.

Its MiMo result recovers 93.4% of an aggregate across eight math metrics,
including both pass@1 and pass@8. This is not 93.4% AIME accuracy. MiMo AIME24
pass@1 changes from 73.3 to 65.4; AIME25 from 67.9 to 47.5. The useful controlled
finding is that student-generated-prefix distillation substantially improves
recovery relative to offline distillation. This supports testing recovery on
the model's own trajectories before judging reasoning gains.

The privately reported CMU MAPLE submission remains unidentified. Neither public
homonym is a likely match: [expert-allocation MAPLE](https://arxiv.org/html/2608.15299v1)
lists Bristol authors and reallocates experts in an existing MoE without
retraining; [Masked Adapter Prototype Learning](https://openreview.net/pdf?id=YxVebk7kIw)
routes among frozen LoRA adapters. Its [author lab](https://x-up-lab.github.io/)
identifies the Surrey-led work and lists it under EMNLP 2026. We can accept the
reported submission as a research lead without attributing either public
paper's method or results to it. Reassess when its manuscript becomes public.

## Attention and recurrent conversions

Budgets below are those reported in the linked versions. They are not estimates
of the cost of adapting Limite or of performance on B300.

| Study | Intervention and evidence | Consequence for our design |
| --- | --- | --- |
| [LoLCATs, ICLR 2025](https://arxiv.org/html/2410.10254v3) | Output matching then LoRA; hybrid linear attention plus a short softmax window. About 40M tokens and under 0.2% updated parameters for 7/8B conversions. Llama3 six-task mean 72.0→70.7, but MMLU 66.6→52.8. | A cheap conversion can preserve an average while losing a specific capability. Inspect task-level retention. |
| [The Mamba in the Llama](https://arxiv.org/html/2408.15237v3) | Reuses projections, progressively converts attention, distills and applies DPO. The 50%-attention model improves AlpacaEval LC 22.90→29.61 while MT-Bench falls 8.00→7.35. | Some source metrics improve, but extra preference training and stronger teachers in later variants confound architecture-only attribution. |
| [MOHAWK](https://arxiv.org/html/2408.10189v1) | Matrix matching, hidden-state alignment, then end-to-end distillation. Phi-Mamba and hybrid variants use 3B/5B conversion tokens; hybrid task mean 64.0 versus teacher 64.9. | Separates local numerical recovery from whole-model behavioral recovery. |
| [RADLADS, COLM 2025](https://arxiv.org/html/2505.03005v1) | Qwen 7/32/72B to RWKV variants with 350–700M tokens. Its ablations find frozen inherited weights and aggressive LoRA can hurt whole-model recovery. | Freezing is a useful controlled phase, not an assumption that the final architecture will be learned best with a permanently frozen backbone. |
| [Jet-Nemotron and PostNAS](https://arxiv.org/html/2508.15884v3) | Preserves MLPs while searching attention architecture. Appendix lists 200B search plus 400B final-training tokens. Same Stage-2 corpus: continued Qwen2.5-1.5B MMLU/math 56.7/37.6 versus Jet2B 59.6/40.2. | Capability-oriented architecture search exists. Added parameters and substantial training must be included in the comparison. |
| [HALO and HypeNet](https://arxiv.org/html/2601.22156v1) | About 2.3B training tokens plus selection work. Exceeds its teacher on particular long-context retrieval tasks, while acknowledging lost alignment behavior. | Retrieval, instruction following and long mathematical generation need separate evaluations. |
| [Retrieval-Aware Distillation, ICML 2026](https://proceedings.mlr.press/v306/bick26a.html) | Selects retrieval-critical attention heads before distilling other heads to SSMs; 12B tokens. Retaining 10/512 heads recovers about 95% of retrieval-heavy teacher performance. | Use component-sensitive tasks and selection controls; uniform replacement is not the only sensible intervention. |

The most relevant independent criticism is
[Untangling Component Imbalance in Hybrid Linear Attention Conversion Methods](https://arxiv.org/html/2510.05901v2).
For a released LoLCATs checkpoint, full hybrid accuracy averages 71.08, while
sliding-window-only scores 71.45 and linear-only 33.96. A model can perform well
while hardly using its novel component. Consequently, aggregate improvement
alone cannot establish that our second interaction axis is useful. Post-training
ablation is informative but can introduce distribution shift; it complements,
rather than replaces, a separately trained matched control.

## Expansion and MoE conversion for capability

| Study | Evidence beyond merely retaining the source model | Main control limitation or lesson |
| --- | --- | --- |
| [Sparse Upcycling, ICLR 2023](https://arxiv.org/html/2212.05055) | Dense checkpoints become MoEs; comparisons include continued dense training, scratch MoE and depth expansion, with compute-time curves and downstream evaluation. | Direct precedent for capability-seeking migration. T5/ViT results and substantial extra compute do not predict a modern 1B reasoner's adaptation budget. |
| [Sparse Upcycling: Inference Inefficient Finetuning](https://arxiv.org/html/2411.08968v1) | Cosine-annealed 436M/1.4B parents; dense continuation versus 8-expert/top-2 upcycling at matched additional FLOPs. Gains on a 35-task suite, but measured peak serving throughput falls roughly 35–45%. | Especially useful methodology: match additional training compute and measure actual sampling cost. Equal active FLOPs do not guarantee equal runtime. |
| [Drop-Upcycling, ICLR 2025](https://arxiv.org/html/2502.19261v1) | Partial coordinated expert reinitialization improves specialization. At 1.5B parent scale, task mean 40.3 versus naive upcycling 37.4 and scratch MoE 34.5 under the same 500B MoE-token budget. | Initialization matters; copying weights is not enough. Hundreds of billions of tokens and extra expert capacity limit transfer to our small pilot. |
| [Innovator](https://arxiv.org/html/2507.18671v1) | Qwen2.5-7B→53.3B total/13.3B active MoE; specialist preparation, router warmup, 300B joint training, then GRPO. Science mean 29.4→36.6; general mean 66.4→65.7. | Closest science-oriented precedent. A dense continuation control exists for general tasks, but is missing from the main science table. Capacity, specialist data and RL all contribute. |
| [LLaMA Pro, ACL 2024](https://aclanthology.org/2024.acl-long.352.pdf) | Identity-initialized new blocks; original weights frozen during 80B code/math tokens, then full SFT. Llama2-7B→8.3B; HumanEval 13.05→28.66 and GSM8K 14.48→17.89 in the pretrained comparison. | Very close to our frozen insertion methodology. Extra capacity and domain training require ordinary-block and unchanged-architecture continuation controls. |
| [SOLAR, NAACL 2024](https://aclanthology.org/2024.naacl-industry.3.pdf) | Mistral weights expanded from 32 to 48 blocks, continued pretraining, then instruction/preference tuning. | Establishes practical depth expansion, but its final benchmark advantage mixes architecture, additional data and post-training. |
| [MiniCPM-MoE](https://arxiv.org/html/2404.06395v1) | Stable-phase dense checkpoint→eight experts/top-2; about 520B continuation tokens. MMLU 53.46→58.90 and GSM8K 53.83→61.56. | Useful open capability example, with much larger stored/active capacity. The report's DPO extension is separate from its MoE experiment. |
| [LLaMA-MoE](https://arxiv.org/html/2406.16554v1) | Splits dense FFN channels into experts; 200B continuation tokens yields variants around 3.5B active parameters. | Primarily efficient transfer/compression; comparison with smaller dense models does not establish superiority to the equally trained 7B parent. |
| [MoLEx, ICLR 2025](https://arxiv.org/html/2503.11144v1) | Reuses pretrained layers as conditional experts with LoRA. Five-seed experiments and cross-task transfer; GPT-2-medium E2E BLEU 70.7±0.4 versus LoRA 70.0±0.5. | A structural change can be tested with small trainable overhead. Tasks are classification/data-to-text, not hard mathematical RL. |
| [BAM](https://arxiv.org/html/2408.08274v1) | Domain-specialized dense descendants supply both attention and FFN experts. Includes data-matched and TPU-core-day-matched comparisons against BTX. | Transferring attention structure can matter; specialist preparation and larger expert capacity belong in the total budget. |

There is meaningful counterevidence. In
[OLMoE appendix B.1.2](https://proceedings.iclr.cc/paper_files/paper/2025/file/9b224ace8963c9385ad5e2b5c9039b97-Paper-Conference.pdf),
a scratch MoE catches an upcycled, heavily pretrained 1B parent after roughly
500B additional tokens and overtakes it near 600B. Inheritance can accelerate
an experiment without being the best route to its eventual capability ceiling.

## Changes to reasoning computation and RL transfer

[Net2Net](https://arxiv.org/abs/1511.05641) provides the foundation for
function-preserving width/depth expansion. For our zero-output insertions,
preserving the parent's initial function is a numerical property to test,
including cached decoding; it does not establish learning quality.

[OpenAI Five](https://cdn.openai.com/dota-2.pdf) used network surgery throughout
long-running RL, allowing high-skill evaluation without restarting after every
change. Its final clean rerun nevertheless exceeded 98% win rate against the
surgery-trained agent. This supports surgery as an iteration method while
warning that inherited training history can constrain the final result.

[Reincarnating Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/ba1c5356d9164bb64c446a4b690226b0-Paper-Conference.pdf)
explicitly studies reusing an existing policy and experience while improving
the learner. Its QDagger experiments transfer to stronger architectures and
wean off teacher imitation. The comparison principle is directly useful: all
arms should receive the same prior computational work and workflow. Atari
and control results are methodological precedents, not evidence about Limite.

[Coconut](https://arxiv.org/html/2412.06769v2) changes a pretrained GPT-2's
computation by feeding continuous hidden states back as inputs. In this version,
ProsQA rises from CoT's 77.5 to 97.0, while GSM8K falls from 42.9 to 34.1.
Curriculum and no-thought controls are crucial: some simpler variants also solve
the synthetic tasks. This motivates targeted planning/composition tests without
assuming a universal benefit from extra latent computation.

[Relaxed Recursive Transformers](https://arxiv.org/html/2410.20672v3) converts
pretrained models to repeated shared blocks with layer-specific LoRA. Its aim
is compactness and recovery. Our previously reviewed
[recurrence scaling work](LOOP_SCALING_RESEARCH.md) tests a different question:
regularization across separately trained recursion configurations. Neither
should be conflated with evidence that changing attention improves RL.

Published follow-up ablations are the strongest feedback in this review:
component collapse, loss of long-generation quality, serving overhead and
scratch catch-up are experimentally testable objections. Conference acceptance
alone is not independent replication. We do not infer reviewer consensus from
social posts or inaccessible OpenReview discussions.

## What the Limite comparison already establishes

The [SFT study](wiki/Limite-SFT-Comparison.md) records geometry-matched
ordinary/simplicial insertions, followed by 2B adapter-only tokens and 8B
full-weight continuation. The simplicial model has 2.57% more total parameters
and 16.66% more adapter parameters. This is not parameter or compute matching.
Observed AIME26 success counts are 25→19 for normal and 27→25 for simplicial
between the 2B and 10B endpoints, out of 120 responses on 30 problems. These
single-pair observations motivate studying the training phase; they do not
establish a general degradation law. Historical dropped document endings and
repetition also affect how much RL must repair before it can improve reasoning.

The [current RL contract](../recipes/limite/full_math_rl_native_context.yaml)
restarts from each variant's 10B weights and permits the native 131,072-token
context minus the actual prompt. Both inherit the same historical SFT recipe.
Finite gradients and correct probability accounting qualify execution; they
cannot by themselves show a capability advantage. See the
[RL evidence and limitations](wiki/Limite-RL-Health-and-Performance.md).

The present pair shares the original parent and insertion count, controlling
original source identity and generic added depth for the simplicial-minus-normal
comparison. Its parameter and compute differences remain relevant. Comparison
against the unmodified parent additionally needs unchanged-architecture
continued training. RL-only attribution must account for each variant's
different post-SFT starting behavior, rather than treating the final score
difference as the improvement produced by RL.

## Proposed next target

**Test whether higher-order attention expands the set of verifiable problems a
strong pretrained model can learn, particularly unseen compositions, at a
declared additional training and inference budget.** Report an accuracy versus
token/compute curve, rather than calling a lower training loss intelligence.

Use the released, independently evaluated Violetto checkpoint as the primary
new parent, after qualifying its exact template and inserted-layer numerical
equivalence. It supplies strong reasoning behavior without first repeating the
historical 10B SFT. AIME is near ceiling for this parent, so it is a retention
and external-transfer benchmark, not the architecture-selection objective.
Retain the current Limite-base lineage as a separate replication setting.

| Arm | Intervention | What it controls |
| --- | --- | --- |
| Parent reference | Original released checkpoint, no new training | Initial competence and retention |
| Unchanged architecture | Same parent, supervised data, RL prompt stream and declared budget | Improvement due to the recipe alone |
| Ordinary attention | Native-shaped, zero-output attention insertions | Generic extra depth; a geometry-matched baseline |
| Higher-order attention | Same placement and shared geometry, with the declared second interaction axis | Incremental contribution of the proposed mechanism |

The primary ordinary/higher-order pair should preserve geometry. A separate
ordinary control must match added parameter capacity before attributing a small
gain to interaction order. Plot equal-token and equal-measured-compute cuts
separately; one configuration cannot generally match geometry, stored
parameters, active FLOPs and runtime simultaneously. Include sampling,
distillation teacher and evaluation costs in resource accounting.

Start with sparse insertions, provisionally 12 of 48 blocks, using one
predeclared placement rule stratified by the native local/global schedule
(nine local and three global sites for the existing 3:1 schedule). Uniform
every-fourth placement would select only one attention type. Verify and record
the actual site list and cost before freezing the contract. This is a
new experiment, not a continuation of the 48-insertion comparison. It reduces
the cost of rejection and makes a placement ablation practical. Keep exact
2-simplicial and linearized/triadic mechanisms separately named and qualified;
their state, score and value interactions are not interchangeable.

### Training and promotion

1. **Admission:** parent-equivalent initial logits and cached decoding; finite
   gradients; full optimizer/RNG/data resume; matched loss masking and token
   counts, with declared trainable sets. Measure prefill, decode and update costs at representative
   contexts. Preserve original container dependencies.
2. **Short adaptation:** start with 10M and 30M supervised-token probes, then
   100M if healthy. Preserve complete answers/EOS and disclose window overlap.
   In the primary comparison, every trained arm sees the same data from token
   zero and trains inherited weights with the same small learning rate and
   fresh optimizer state. Inserted weights have a separately declared learning
   rate. An insertion-only frozen-backbone phase is a separate recovery
   ablation; the unchanged arm must not be denied its training allowance.
   If recovery needs on-policy teacher supervision, compare that treatment in
   all applicable arms, counting generated tokens and teacher compute.
3. **Decision checkpoints:** consider 100M, 300M and at most 1B adaptation
   tokens for the first pilot. These are proposed ceilings, not evidence that
   1B is sufficient. Reject or revise a treatment that cannot preserve source
   behavior or show useful learning on development tasks. Do not automatically
   spend another 10B tokens to rescue an uninformative comparison.
4. **RL pilot:** first qualify 8 useful updates, review again at 16, then expand
   under a new budget if learning is measurable. Record attempted/applied
   updates, sampled responses, generated tokens, useful reward groups and GPU
   hours. Keep a common training-only curriculum; arm-specific solvability
   filtering would change the data distribution. Evaluate curves at matched
   generated-token budgets and matched compute, not wall-clock checkpoint age.
5. **Confirmation:** repeat the selected protocol with at least three paired
   training seeds and a fixed untouched test set before a component-level
   superiority claim. Declare the comparison and stopping rule before outcomes
   are inspected. A small pilot selects a design; it is not the final test.

Layer-output distillation is not automatically required for identity-initialized
insertions: at initialization their output is intentionally zero. Copying an
entire replacement-recovery recipe would change the meaning of the intervention.

### Tasks and metrics

Reuse the upstream task generators identified in the
[higher-order reasoning review](SIMPLICIAL_REASONING_RESEARCH.md), after auditing
their generators and verifiers for this decoder setting: relational joins,
variable binding, function composition and planning with distractors. That
review is a task proposal, not a qualified local implementation.
Split by graph/function structure before paraphrase or permutation augmentation.
Include unseen compositions and longer dependency chains, alongside one-hop
retrieval as a negative control. Control operand distances relative to the
second-axis window so failure outside its reach is not misread as a mechanism
test. Calibrate difficulty on training/development sets only.

A later science-facing extension can use generated equation systems or reaction
networks with exact symbolic checks and held-out dependency structures. It must
first pass verifier and ambiguity audits. Such tasks test compositional skills;
they do not establish real-world scientific discovery.

Report held-out exact success, pass@1 and pass@8, unique problems solved,
confidence intervals clustered by problem, and variation across training seeds.
Use the same inference-token allowances for all arms and retain full-context
evaluation as a separate view. Do not choose architecture or curriculum using
AIME answers. Report general retention and long-generation termination as well
as the targeted task.

Evaluate removal of the inserted branch and, separately, removal or causal
shuffling of its second-axis information with norm/mask controls. A trained
ordinary control and a mechanism-sensitive task are still needed to interpret
any ablation loss.

For RL, track natural EOS, native-context exhaustion, repetition stops,
reward contrast, entropy, importance-weight health and learning progress.
Shorter generation limits are a changed scientific contract, not an invisible
performance optimization. Retain the native limit for the existing runs.

## Implementation reuse and next decisions

| Upstream | Reusable part |
| --- | --- |
| [HazyResearch LoLCATs](https://github.com/HazyResearch/lolcats) | Attention matching, conversion evaluations and low-cost control design |
| [NVlabs Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron) | Placement/search and inherited-backbone comparison methodology |
| [MambaInLlama](https://github.com/jxiw/MambaInLlama), [RADLADS](https://github.com/recursal/RADLADS-paper) | Projection transfer and staged behavioral recovery |
| [T5X MoE](https://github.com/google-research/t5x/tree/main/t5x/contrib/moe), [Drop-Upcycling](https://github.com/Taishi-N324/Drop-Upcycling) | Dense-to-expert initialization references if an MoE comparator is added later |
| [MoLEx](https://github.com/rachtsy/molex) | Low-parameter structural controls and repeated-seed evaluation |
| [XiaomiMiMo verl](https://github.com/XiaomiMiMo/verl) | Already pinned RL components; full executor compatibility still requires qualification |

Prefer existing tested primitives after source/license and numerical checks.
Do not vendor container runtimes or mistake a paper's unverified repository
name for a usable implementation. No public implementation was verified for
OPAL in the author-linked sources reviewed here.

The immediate research priority is the unchanged-architecture control and a
small verifiable composition suite. An MoE comparator is useful later, but
introducing routing, higher-order attention and a different RL recipe together
would make a positive result difficult to explain. The next launch contract
should select one treatment and preserve the controls above.
