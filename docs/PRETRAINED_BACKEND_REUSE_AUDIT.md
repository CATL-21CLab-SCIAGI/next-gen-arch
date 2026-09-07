# Pretrained Qwen Next: reuse-first backend audit

Audit date: 2026-09-07. Status: **no inspected backend is qualified for launch**.
No new backbone implementation, package installation, runtime patch, node
restart, full-checkpoint load, or finetuning was performed during this audit.

## Decision and scope

Use the following order for the frozen-pretrained/additive-simplicial experiment:

1. Search maintained off-the-shelf implementations of the exact architecture.
2. Inspect the existing backend source, dependencies, weight conversion and tests.
3. Reuse suitable implementation code. Write new code only for an uncovered gap,
   with an explicit source reference and a numerical/behavioral test.
4. Pass full integration qualification before any finetuning launch.

An unknown model type in the installed Transformers is not sufficient reason to
write a backbone from scratch. Conversely, a model appearing in a support table
does not establish support for our frozen runtime, DP-only topology, 16K context,
or checkpoint-retention contract.

## Pinned primary-source candidates

All three source projects use Apache-2.0 licensing; preserve applicable notices
and attribution for any derived code. Audit downloads remain uncommitted source
snapshots, not dependencies on the production Python path.

| Candidate | Inspected revision | Evidence and disposition |
| --- | --- | --- |
| NVIDIA NeMo AutoModel | `a4ce87c003f08b74d68684d3627f6e6048bc0140` | Exact-model trainable text backbone, QSA, PLE and checkpoint conversion; preferred training reference, but not a drop-in runtime. |
| ModelScope mcore-bridge | `7965f71e9d16a3ca35cdf5351f5dffaa71574e64` | Exact-model Megatron layer/weight adapter, used by ms-swift; ordinary import patches the container-owned runtime. |
| Hugging Face Transformers | initial model `fc5c5bde8e656dad91cbf34e61940d984b1c7b91`; current `c93057d4835cd31752bb56f59989dd27696eb45b` | Exact-model implementation exists; both inspected versions need APIs absent from frozen Transformers 5.8.1. |

### NeMo AutoModel

Relevant source:

- [Model construction and forward](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/components/models/qwen3_8_flash_next/model.py).
- [Decoder layers](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/components/models/qwen3_8_flash_next/layers.py).
- [QSA implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/components/models/qwen3_8_flash_next/qsa.py).
- [PLE/Engram implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/components/models/qwen3_8_flash_next/engram.py).
- [Checkpoint conversion](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/components/models/qwen3_8_flash_next/state_dict_adapter.py).
- [Configuration](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/components/models/qwen3_8_flash_next/config.py) and [upstream unit tests](https://github.com/NVIDIA-NeMo/Automodel/tree/a4ce87c003f08b74d68684d3627f6e6048bc0140/tests/unit_tests/models/qwen3_8_flash_next).

The decoder has the desired insertion boundary: in `layers.py`, line 764 combines
the original attention branch into the residual streams; line 766 begins the
existing MoE hyper-connection read. Preserve both and insert the independent
module between them. Calling the old replacement-attention pilot is incorrect.

The [published validation](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/docs/model-coverage/llm/qwen/qwen3-8-flash-next.mdx)
provides useful QSA/SGLang parity evidence and training/checkpoint tests. Its
EP-based, 4K validation is not proof of our DP-only, 16K experiment.

Remaining incompatibilities:

- The package is absent from the container. Its dependency declaration pins
  Transformers 5.12.1, unlike installed 5.8.1; a working configuration import
  does not qualify the whole dependency graph.
- Its [package initializer](https://github.com/NVIDIA-NeMo/Automodel/blob/a4ce87c003f08b74d68684d3627f6e6048bc0140/nemo_automodel/__init__.py)
  installs a hook that changes Transformers' allowed layer types. This was
  inspected, not executed. Bypassing dependency constraints or patching the
  installed runtime is not an approved fallback.
- The model is language-only, sets `mtp=None`, and its checkpoint converter
  explicitly drops `mtp.*`. Our integration must account for every original
  tensor, retaining inactive vision/MTP weights for restoration/export without
  adding an auxiliary training objective.
- PLE uses owner-sharded lookup with All-to-All, separately from normal FSDP
  weight all-gathers. Do not label the upstream recipe "DP-only" by setting
  `ep_size=1`: embedding ownership and collectives also require qualification.
  A strictly DP-only solution must not silently introduce another form of model
  parallel execution. Replicated frozen PLE with local CPU lookup is a candidate
  to assess, not an implemented or validated fallback.

### ModelScope mcore-bridge / ms-swift

The [exact-model implementation](https://github.com/modelscope/mcore-bridge/blob/7965f71e9d16a3ca35cdf5351f5dffaa71574e64/src/mcore_bridge/model/gpts/qwen4_exp.py)
and [multimodal adapter](https://github.com/modelscope/mcore-bridge/blob/7965f71e9d16a3ca35cdf5351f5dffaa71574e64/src/mcore_bridge/model/mm_gpts/qwen4_exp.py)
are useful references for existing Megatron integration. The loader rejects MTP;
QSA falls back to full attention for packing or CP, so those paths must not be
silently reused. Our CP=1 setting alone does not prove all execution paths faithful.

Its package initializer calls
[runtime patching code](https://github.com/modelscope/mcore-bridge/blob/7965f71e9d16a3ca35cdf5351f5dffaa71574e64/src/mcore_bridge/patcher.py),
including assignments to Megatron/Transformer Engine classes. It was not imported.
The [ms-swift example](https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen4_exp/megatron_sft.sh)
also requests new packages and TP/PP/EP, so it is not our launch recipe.

### Transformers and downstream wrappers

Both the [initial Qwen4Exp model](https://github.com/huggingface/transformers/blob/fc5c5bde8e656dad91cbf34e61940d984b1c7b91/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
and the [current revision](https://github.com/huggingface/transformers/blob/c93057d4835cd31752bb56f59989dd27696eb45b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
have seven unavailable import entries in the installed runtime: `initialization`,
`use_kernel_func_from_hub_with_fallback`, `force_accelerate_hooks`,
`create_recurrent_attention_mask`, `accepts_precomputed_kwargs`, `get_max_seqlen`,
and the `vision_utils` module. Simply copying the model file is not sufficient.

The [ms-swift HF loader](https://github.com/modelscope/ms-swift/blob/main/swift/model/models/qwen.py)
depends on Transformers' Qwen4Exp class. [Axolotl's exact-model support](https://docs.axolotl.ai/docs/models/qwen3.8-flash-next.html)
is another finetuning lead, but its documented quantized/LoRA routes are not a
qualification of this unchanged-weight additive-module experiment. Serving
backends can provide forward references; they cannot be assumed to supply the
input gradients needed through a frozen suffix.

## Checks actually executed

- Frozen-container package inventory: AutoModel, ms-swift, mcore-bridge, Axolotl
  and Unsloth are not installed. Existing FLA and native PyTorch FSDP2 are present.
- Loaded only the inspected upstream AutoModel configuration file in an isolated
  module namespace, without importing its package initializer. All 32 audited
  text-architecture fields matched the actual source configuration; the installed
  Transformers layer-type allow-list was unchanged. Model-type and mRoPE-key
  warnings were retained; this is not a model-loading/parity test.
- Read the checkpoint index: 1,658 entries over 131 shards, comprising 1,293
  language-model tensors, 333 vision tensors, 31 MTP tensors and one LM-head
  tensor. No weight payload was loaded by this audit.
- Compared the imports of both pinned Transformers model files against installed
  APIs; both fail compatibility as described above.
- Re-ran the project's simplicial test discovery: 18 tests passed and five
  CUDA/native-distributed tests skipped in the local restricted process.
  Previous isolated frozen-container GPU results remain leaf-module evidence,
  not full-model evidence.

## Qualification required before finetuning

1. Choose a reusable backend/integration route that obeys the frozen-runtime and
   DP-only contract. Preserve source provenance; do not rederive existing model
   equations or import a backend that patches container-owned packages.
2. Audit all original checkpoint entries as either loaded for execution or
   explicitly retained inactive weights. No unexpected missing/unmapped tensor,
   random base initialization, quantization, or unrecorded conversion is allowed.
3. Compare the unmodified pretrained forward with its numerical reference,
   including PLE hash/EOS cases and QSA routing beyond the 2,048-token budget.
   Then compare before/after insertion at zero adapter output, including 16K.
4. Test adapter-only optimizer membership, finite gradients through frozen
   suffixes, nonzero earlier-adapter gradients after the zero-output warm start,
   and unchanged original weights. No whole-model `no_grad` shortcut is valid.
5. Verify native sharded-DP numerical behavior against the same combined batch,
   inspect actual process groups/collectives (including PLE), and run bounded
   16K memory/throughput probes in the existing container.
6. Restore adapter, optimizer, scheduler, RNG and data cursor from a checkpoint;
   compare the next update and confirm all original weights remain recoverable.
7. Only after these pass, launch the single agreed FineWeb-Edu job. No additional
   comparison training job, old-run resume, runtime change, or node restart.

This document corrects the earlier premature conclusion that a new backbone port
was necessarily required. Existing model implementations have been found;
production integration and seamlessness testing remain unfinished.
