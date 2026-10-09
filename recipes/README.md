# Experiment recipes

Start with the model family or study. A study contract may describe multiple
matched arms; a catalog keeps related model, scale or launch profiles together.

| Directory / file | Contents |
| --- | --- |
| [`limite/`](limite/) | Matched normal/2-simplicial warmup, full finetuning, RL and evaluation |
| [`deepseek_v41/`](deepseek_v41/) | Scratch scaling, adapter comparisons, RL and performance contracts |
| [`qwen/`](qwen/) | Qwen studies plus shared model, launch and proposal catalogs |
| [`reference/`](reference/) | Frozen speedrun and Megatron topology comparisons |
| [`triadic/`](triadic/) | Triadic scratch study and historical-prefix comparison |
| [`defaults.yaml`](defaults.yaml) | Shared backend and scale defaults used by reference profiles |

## Limite comparison

- [Matched full finetuning](limite/full_finetune_math.yaml): matched 2B adapter
  warmup checkpoints, then full-weight continuation to 10B total.
- [Original adapter warmup](limite/native_adapters_math.yaml): earlier proposal;
  the actual sequence is explained in the [study](../docs/wiki/Limite-SFT-Comparison.md).
- [Historical paired RL](limite/full_math_rl_async.yaml).
- [Native-context Violetto successor](limite/violetto_math_rl_native_context.yaml).
- [AIME26 evaluation](limite/eval_aime26_pipeline_v3.yaml).

## Select a catalog profile

Catalogs contain a `profiles` mapping. Select a named entry explicitly with
`FILE.yaml#PROFILE`; the loader never guesses the model from an output name.
Quote recipe references in shell commands. For example:

```bash
PYTHONPATH=src python -m archlab.cli render \
  --config 'recipes/reference/speedrun_100m.yaml#research_baseline'

PYTHONPATH=src python -m archlab.megatron.launch_config \
  --recipe 'recipes/qwen/launches.yaml#flash_next_w320_e32' --bindings
```

The first command requires the data/output environment bindings in the selected
contract. The second renders allowlisted bindings; it does not launch training.

| Catalog | Profiles |
| --- | --- |
| [`qwen/models.yaml`](qwen/models.yaml) | Seven recorded model geometries |
| [`qwen/launches.yaml`](qwen/launches.yaml) | Six explicit launch/topology profiles |
| [`qwen/proposals.yaml`](qwen/proposals.yaml) | Four Qwen research proposals |
| [`deepseek_v41/proposals.yaml`](deepseek_v41/proposals.yaml) | Two DeepSeek research proposals |
| [`reference/megatron_native_100m.yaml`](reference/megatron_native_100m.yaml) | Dense/MoE parallelism profiles |
| [`qwen/qwen2p5_pretrain.yaml`](qwen/qwen2p5_pretrain.yaml) | DP8/DP32 profiles for the same pretraining recipe |

`extends` can reference another catalog profile, including a profile in the same
file. Model definitions and proposals are descriptive contracts, not admitted
launch commands. Their status and qualification requirements remain explicit
inside the selected profile. Historical profiles are retained for reproducibility;
their presence does not indicate a current queue or production recommendation.

Supply machine paths and runtime identities through the declared environment
bindings. Changes to geometry, optimizer, budget or data order require a new
experiment contract. The folder cleanup preserves those settings.
