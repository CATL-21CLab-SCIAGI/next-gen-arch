# Provenance and third-party notice

Next-Gen Architecture Lab is derived from the MIT-licensed [nanochat](https://github.com/karpathy/nanochat) codebase and retains speedrun optimizer/kernel lineage from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). The upstream license and copyright notice are retained in [`LICENSE`](LICENSE). The frozen campaign recorded nanochat commit `b9f5025652d51470e2c31117100d9ff48717b911`; the local modded-nanogpt reference used for provenance is `f411b3d346aa52d3504324ca93c230fd84c6c07f`.

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is consumed only as the Git submodule `third_party/Megatron-LM`, pinned to `55ac7082517c3878ae653c07c09c534b8aed49f6`. Its own license and notices remain inside the submodule. This repository does not vendor, rewrite, or relicense Megatron-LM source files.

Architecture modules in this repository are research adaptations informed by the primary sources listed in [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md). They are not represented as official implementations, endorsements, or exact reproductions of the authors' complete systems.

“nanochat”, “Engram”, “Kimi”, “Qwen”, “DeepSeek”, “GLM”, “Inkling”, and other project names belong to their respective owners. No model weights, dataset shards, tokenizer binaries, or third-party trademarks are distributed as project assets.

The machine-readable manifest includes historical source hashes for provenance. Its public copy replaces internal paths and hostnames with portable placeholders; those substitutions do not change the run contract, source hashes, data fingerprint, or tokenizer hashes.
