# Matched DeepSeek V4.1 math RL continuation

The user approved the pretrained normal/simplicial pair at **step 4537 /
756,364,650 supervised tokens each**. The four scratch trainers were separately
checkpointed and stopped; they are not the RL parents.

The experiment lives under `results/deepseek-v41-math-rl-20260922/`. Each actor
uses its original 16-GPU ownership on two existing B300 nodes. `LAUNCH_PLAN.json`,
`recipe.resolved.json` and each production directory's `ACTIVITY.json` record
the actual stage. Restoration or numerical qualification is not an RL update.

## Data and objective

The approved split contains **8,192 training / 512 held-out problems**. All five
original source files had contributed to earlier pilots; the user approved
new shards containing unused problems within those files. Selection excludes
all touched prepared parts and 138,049 prior normalized problem hashes plus
123,957 UUIDs, with cross-mode deduplication. This does not certify absence of
paraphrases or unknown original backbone-pretraining exposure.

Only problem, expected-answer and UUID columns were read. Teacher solutions
are not policy targets. Expected answers are used only by the versioned exact
rational/scalar reward verifier. Unsupported references were filtered before
splitting. Native-tokenizer admission retained every record; maximum prefix
lengths are 559 training tokens and 299 held-out tokens. Seals and verification
are in `results/rl-nemotron-preparation-20260922/approved-v1/`.

The algorithm is online **REINFORCE with a leave-one-out baseline (RLOO)**:

- One distinct prompt per rank, four current-policy samples per prompt.
- Temperature 1, top-p 1; exactly one update per fresh rollout batch.
- Correct final answer earns 1; incorrect, absent or unsupported answers earn 0.
- Each trajectory's advantage subtracts the other three rewards' mean.
- The objective averages advantage-weighted sums of completion log probabilities
  across trajectories. Prompt and padding targets are masked.
- Globally flat rewards skip optimization honestly. No teacher-answer CE,
  reward-filtered SFT, indexer auxiliary loss or router auxiliary update is used.

All text weights are trainable. Parent weights are checksum-restored; a fresh
memory-bounded Adafactor uses relative learning rate 1e-6 and global gradient
clipping at 1. The initial phase has no reference-policy KL penalty, and absent
KL is not reported as zero. The bound is 128 fresh rollout batches per arm;
the early two-batch pilot must demonstrate a real parameter update before
continuing. Matching prompt/sample budgets does not imply equal generated
token counts or equal numbers of non-skipped updates.

## Qualification, evaluation and efficiency

The frozen snapshot passed **103 CPU tests** in the training container. A
separate source-bound **16-GPU NCCL/FSDP** probe verified signed head/hidden
gradients against a dense global-sum oracle and sampled/replayed token scores.
Actual actors additionally require real-model replay/backward qualification.
Synthetic qualification advantages never reach an optimizer step. A genuine
reward-driven update is marked separately by `REAL_UPDATE_VERIFIED.json`.

Greedy evaluation uses the same frozen 64 held-out problems before training,
after the pilot and every eight rollout batches. It reports pass@1, valid-answer
rate and truncation; dummy rows are excluded, and RNG/module modes are restored.
The eight-case numerical-admission evaluation is recorded separately. These
small subsets are diagnostics, not a broad capability benchmark.

Implemented efficiency changes are distinct rank-local rollouts, four local
samples, deterministic length-bucketed global prompt batches, dynamic sequence
buckets, four evaluation examples per rank, and a chunked differentiable
selected-logprob head. Native rollouts remain uncached. The profiler warms and
measures batch sizes 1 and 4, using actual generated tokens and the slowest
rank's elapsed time. `PROFILE.json` supplies measured throughput evidence;
neither GPU utilization nor total parameter count is reported as MFU.

## Tracking and recovery

Dedicated MLflow experiment: **deepseek-v41-nemotron-rloo-20260922**. Run names:
`full-normal-4537-nemotron-rloo-v1` and `full-simplicial-4537-nemotron-rloo-v1`.
The independent watcher records reward, policy loss, skipped updates, answer
validity, throughput, update time and held-out accuracy, plus source/data/parent
identities. It does not overwrite supervised histories or relabel policy loss
as cross entropy. Live sync state: `results/rl-monitor/RL_SYNC_STATUS.json`.

`STOP_REQUEST` requests a safe boundary stop and full checkpoint. Checkpoints
retain optimizer, Torch/Python/NumPy RNG states, prompt cursor and policy version.
Resume preflights rank manifests against the completed checkpoint identity.
The first pilot checkpoint has critical-state readback evidence, distinguished
from a full model checkpoint restore/replay proof.
