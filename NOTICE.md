# Provenance and third-party notice

Next-Gen Architecture Lab is derived from the MIT-licensed [nanochat](https://github.com/karpathy/nanochat) codebase and retains speedrun optimizer/kernel lineage from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). The upstream license and copyright notice are retained in [`LICENSE`](LICENSE). The frozen campaign recorded nanochat commit `b9f5025652d51470e2c31117100d9ff48717b911`; the local modded-nanogpt reference used for provenance is `f411b3d346aa52d3504324ca93c230fd84c6c07f`.

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is an external runtime dependency supplied by the execution environment. Its license and notices remain with that installation. This repository does not vendor, rewrite, or relicense Megatron-LM source files. Historical benchmark artifacts retain the exact upstream commit `55ac7082517c3878ae653c07c09c534b8aed49f6` used for those runs.

The optimization audit also studies public experiments and reports from
[Marin](https://github.com/marin-community/marin) at
`299c7f3245e2e6998345980cadad75f45088f63f` and the current Modded-NanoGPT
history at `ecbb586296d3dac36fd206211f25d63bad4a6b35`. Marin/Levanter source is not
vendored or used as a runtime backend; portable hypotheses are independently
expressed as small, attributed experiment recipes.

Architecture modules in this repository are research adaptations informed by the primary sources listed in [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md). They are not represented as official implementations, endorsements, or exact reproductions of the authors' complete systems.

“nanochat”, “Engram”, “Kimi”, “Qwen”, “DeepSeek”, “GLM”, “Inkling”, and other project names belong to their respective owners. No model weights, dataset shards, tokenizer binaries, or third-party trademarks are distributed as project assets.

The machine-readable manifest includes historical source hashes for provenance. Its public copy replaces internal paths and hostnames with portable placeholders; those substitutions do not change the run contract, source hashes, data fingerprint, or tokenizer hashes.
