# Where higher-order attention has helped

Primary-source review, 2026-09-28. The actionable hypothesis for our small V4.1
models is better learning of relations and unseen compositions. Evidence for a
universal language-model advantage is mixed. Exact windowed 2-simplicial attention,
linearized approximations, persistent edge states, and tree attention are distinct
architectures; results from one are not direct validation of another.

| Primary study | Data and setting | Finding and limits |
|---|---|---|
| [Fast and Simplex, Roy et al., 2025](https://arxiv.org/pdf/2507.02754), §8/Table 1 | MoE at 1B/2B/3.5B active parameters; fixed token budget; every fourth layer simplicial; 512×32 windows; GQA 64 | No gain below 2B active in the tested setting. At 3.5B, GSM8K NLL .2781→.2718 and MMLU-Pro .7858→.7689. These are likelihoods, not solved-question accuracies. The reviewed version does not identify the training corpus or numeric token budget. Our models are much smaller in active compute. |
| [Logic and the 2-Simplicial Transformer, Clift et al., 2020](https://arxiv.org/pdf/1909.00668), §5 | Bridge BoxWorld; two interacting solution paths, lengths 1–3; half of episodes have a bridge; 7×9 grid; IMPALA/RMSProp; four trials/arm, up to 5.5B environment steps | Simplicial agents outperform relational agents on conjunction/planning. No curriculum or episode time cap. Uses virtual entities and tensor-product values, different from our 2025-style local branch. The 93% standard-BoxWorld result is a baseline sanity check, not the comparative gain. |
| [Linearized 2-Simplicial Attention, Das et al., 2026](https://arxiv.org/html/2608.09307v1), §4–7 | d1024/L24, ~330M non-embedding parameters; FineWeb-Edu 350M-reference tokens, FineMath-4+ 3B; 2k/16k context; parameter matching within 0.5% | At 2k web, KDA+LinSimp loses slightly to KDA under analytic equal-FLOPs (.3446 vs.3458 mean accuracy); exact windowed attention has the worst validation loss after its token budget is reduced. At 2k math, .3900 vs.3895 is tiny. At 16k math, .3888 vs.3809 is stronger, but uses 3.14B vs 2.70B tokens; measured throughput 20.2k vs 24.5k tokens/s is slower. Single seed. The seven-task mean tests general downstream accuracy, not math-solution correctness. Approximate global/local attention plus KDA is not our exact local mechanism. |
| [Strassen Attention, Kozachinskiy et al., 2025](https://arxiv.org/html/2501.19215v2), §6/Appendix A | Function composition, binary-relation composition, Match3, quotient relation composition; generated examples; shallow networks | Third-order and Strassen mechanisms succeed across these tasks. The reported random 90/10 train/validation split, also used for testing, is not evidence of unseen-composition generalization. Match3 uses permutation augmentation; split underlying instances before augmentation to avoid related instances crossing our splits. Strassen's scoring rule differs from our trilinear score. |
| [Systematic Generalization with Edge Transformers, Bergen et al., 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/0a4dc6dae338c9cb08947c07581f77a2-Paper.pdf), §4 | CLUTRR graph relations: train on 2–3 or 2–4 facts, test up to 10; CFQ maximum-compound-divergence splits; COGS; models as small as d64/L3 | COGS 87.4±0.4 versus 78.4 for the cited graph-prediction baseline. CFQ semantic exact match 24.69±1.27 vs 21.26±1.06; large dependency-parsing gains too. These models maintain pairwise edge states. Value-product ablation hurts CLUTRR/CFQ, but not COGS, so every gain cannot be attributed solely to multiplication. |
| [Poly-attention, 2026](https://arxiv.org/html/2602.02422v1), §5/Appendix H | Function composition and COGS; COGS d64/L3, four heads, 200 epochs, 10 seeds | Tree attention learns composition faster and improves COGS generalization while in-distribution accuracy is similar. Strong seed variation remains. This uses a different factorization; its benefit motivates a task and split, not replacing our mechanism mid-study. |
| [Representational Strengths and Limitations of Transformers, Sanford et al., 2023](https://arxiv.org/pdf/2306.02896), Theorems 7/18 | Formal Match3 expressivity | Separation between one-layer ordinary and third-order attention motivates a controlled diagnostic. The stronger multilayer impossibility is a conjecture in this source. This is not a theorem that our 20-layer baseline cannot solve the task, nor proof of learnability. |
| [Conditional Memory via Scalable Lookup, 2026](https://arxiv.org/html/2601.07372v1), §2–3 | Engram conditional memory and MoE allocation under parameter/compute controls | Engram also changes reasoning performance; it is not merely an inert parameter counter. Fixing capacity is an experimental control requested here, not a claim that the paper prescribes constant tables. Keep its channels, hash geometry, placement policy, optimizer, and data equal within each pair. |

The repositories reviewed provide reusable kernels, generators, or protocols:

| Repository | Use for this study |
|---|---|
| [pytorch/FBGEMM simplicial_attention](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/simplicial_attention) | Primary exact-kernel reference, numerical tests and benchmarks. README's TLX installation is not permission to replace the container runtime. Reuse code only after our head dimensions, GQA, causal windows, backward and numerical tolerances pass qualification. |
| [dmurfet/2simplicialtransformer](https://github.com/dmurfet/2simplicialtransformer) | Original bridge-BoxWorld environment and agents; useful mechanistic reference, not a drop-in current training backend. |
| [furrutiav/strassen-attention-neurips25](https://github.com/furrutiav/strassen-attention-neurips25) | Existing composition/Match3 datasets, model references and experiment scripts. Prefer its task definitions and audited generators over independently inventing them. |
| [bergen/EdgeTransformer](https://github.com/bergen/EdgeTransformer) | CLUTRR/CFQ/COGS experiments and split conventions. Preserve the distinction between graph prediction and decoder text generation. |
| [qlabs-eng/scaling-exponents](https://github.com/qlabs-eng/scaling-exponents) | Existing confirmed recurrence study and evaluation protocol. Repeated-data regularization is a different hypothesis from simplicial reasoning; retain it as a separate sweep. Its GPT-2 tokenizer cannot be silently substituted into V4.1. |
| [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) | Official memory mechanism/demo and paper; retain upstream hash semantics. Our revision changes declared dimensions rather than inventing a new embedding implementation. |

Our proposed evaluation protocol below is an inference from this evidence, not
an already-run experiment or a change to the active pretraining data:

1. Start with a small, answer-supervised composition diagnostic: function
   composition, two-hop relation joins and Match3. Include matched one-hop
   retrieval as a negative control. Use the existing public task definitions,
   deterministic generators, balanced labels and a held-out test set. Split by
   underlying graph/function/multiset before paraphrase or permutation augmentation.
   Train with equal data and tuning budgets for both variants, then test unseen
   compositions, renamed entities, longer chains and distractors separately.
   Decoder text versions are adaptations; report that distinction.
2. Test CLUTRR-style train depth 2–3/test depth 4–10 and COGS/CFQ compositional
   splits after the synthetic sanity check. Report exact answers/structures,
   per-depth accuracy and confidence intervals, alongside in-distribution results.
   Ordinary random-split accuracy can hide memorization and is insufficient.
3. Respect our 512×32 window. First put both operands within 32 tokens of the
   query; then one within 32 and the other within 512, counterbalancing order.
   Add distance buckets outside 32 and 512 as explicit stress tests. A local branch
   that cannot directly see the required pair is not a clean test of pair reasoning.
   Full-model native attention still propagates information, so distinguish direct
   branch visibility from full-model reachability. Evaluate using actual tokenizer
   positions, not word counts.
4. At the saved pretraining checkpoints, add held-out GSM8K/MATH-style answer
   correctness and MBPP execution where model ability permits. Report solution-only
   NLL separately from exact-answer/pass@1 results, and exclude evaluation examples
   from any later supervised adaptation. Chance-level zero-shot scores in a tiny
   model are uninformative; use equal-budget task training as a diagnostic.
5. A separate FineMath-4+ continuation/pretraining pair is a reasonable follow-up,
   but the current evidence does not justify declaring math a universal win for
   exact 2-simplicial attention. Keep the active FineWeb-Edu cohort sealed. Likewise,
   a 16k linearized-attention comparison would require a new architecture/context
   contract and is outside this revision.
6. Retain the 10B equal-token comparison and also compare checkpoints by measured
   GPU-seconds and reported FLOPs. Record Engram and vocabulary embeddings apart
   from active non-embedding computation. Replicate a promising diagnostic with
   at least three seeds and equal hyperparameter-search budgets before interpreting
   a small mean gain. Inspect branch output norms, gates and gradients to ensure
   the treatment is actually used; measure ablations only on evaluation copies.

Current d128 uses the same lookup geometry as the revised cohort. Future d384,
d1280 and all unstarted recurrence cells use that constant 153.7M-element sharded
budget. Historical d640 retains 3.073B Engram elements and belongs to a separate
geometry cohort: compare its two attention variants, but exclude it from the
constant-memory scaling fit. At d1280 the active experts contain 294.9M weights
across 20 layers (including the shared experts); billions of inactive experts or
lookup entries do not move it into a paper's 2B-active regime. Our additive eight-site
branch, tiny heads and native sparse backbone also differ from the large-model
study. These limits should accompany any positive or negative result.
