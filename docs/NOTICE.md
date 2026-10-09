# Provenance and third-party notice

Copyright 2026 CATL 21C Lab SCIAGI contributors.

Project-owned contributions in this publication are licensed under the Apache
License, Version 2.0; see [`LICENSE`](LICENSE). Third-party portions retain their
original licenses and attributions. The earlier MIT notice is preserved in
[`docs/licenses/legacy-MIT.txt`](licenses/legacy-MIT.txt); previously granted
MIT permissions are not withdrawn by this publication.

Next-Gen Architecture Lab is derived from the MIT-licensed [nanochat](https://github.com/karpathy/nanochat) codebase and retains speedrun optimizer/kernel lineage from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). The inherited MIT license and copyright notices are retained in [`docs/licenses/legacy-MIT.txt`](licenses/legacy-MIT.txt). The frozen campaign recorded nanochat commit `b9f5025652d51470e2c31117100d9ff48717b911`; the local modded-nanogpt reference used for provenance is `f411b3d346aa52d3504324ca93c230fd84c6c07f`.

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

`src/archlab/rl/limite_protocol.py` adapts the word 4-gram collapse detector in
[`recipes/design/repetition.py`](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/design/repetition.py)
and the linear soft-overlong reward formula in `verl/workers/reward_manager/dapo.py`
from XiaomiMiMo's verl fork at `a2ad9f6160b03ff2d47e59832bfb6b289f37c917`.
Copyright 2025–2026 Bytedance Ltd. and/or its affiliates. These adapted portions
retain Apache License 2.0; see `docs/licenses/verl-Apache-2.0.txt` (packaged in wheels as
`archlab/licenses/verl-Apache-2.0.txt`). The unfinished
math penalty, native graph decoder, and shared training-only curriculum are local
experiment choices, not claimed to reproduce MiMo's complete agentic recipe.


`src/archlab/architectures/tilelang_attention.py` adapts scheduling and online
softmax patterns from TileLang's FlashAttention/GQA examples at
[41b25527cd672434f88eeea7e056d8d7c0d4faa4](https://github.com/tile-ai/tilelang/tree/41b25527cd672434f88eeea7e056d8d7c0d4faa4/examples/flash_attention).
TileLang remains an external, version-pinned compiler dependency. Its upstream
license for the adapted example portions is retained below:

`src/archlab/architectures/tilelang_gqa.py` additionally adapts the key-owned
GQA backward at TileLang revision
[`a35f8ddf45eba16c21211ec8822d56ce5363036f`](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f):
`examples/flash_attention/example_gqa_bwd.py`. It receives the project's
unchanged ordinary forward as a callback. Only necessary backward kernel pieces
are adapted; upstream example CLIs are not a runtime dependency. The same
Tile-AI MIT license below applies to those portions.

    MIT License

    Copyright (c) Tile-AI.
    **During the period from December 1, 2024, to Mar 14, 2025, this project is
    subject to additional collaboration terms with Microsoft Corporation.**

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
