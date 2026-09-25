# DeepSeek Experiments

Current DeepSeek RL launch and qualification entry: [Miles baseline](../MILES_BASELINE.md).

## Lineages and controls

| Phase | Trainable weights | Comparison |
| --- | --- | --- |
| Frozen-backbone math adaptation | Added branches | Ordinary versus simplicial attention |
| Full-weight continuation | Text backbone and adapters | Each arm continues its own matched adapter checkpoint |
| Scratch studies | Randomly initialized backbone and branches | Ordinary, simplicial, linear and linearized-simplicial candidates |
| Math RL | Declared policy parameter set | Outcome rewards from matched SFT parents |

The final matched supervised pair is step **4537/4537**, with **756,364,650** supervised tokens each: 307 adapter updates followed by 4,230 full-weight updates.

## Mechanism

Adapters enter after attention expands into the four residual streams and before the FFN read. The simplicial branch uses exact joint softmax over 32×512 causal key windows. The ordinary control uses the 512-token window.

The adapters match geometry and shared initialization, not parameter count:
normal has 146.84M parameters; simplicial has 167.82M.

## Results and boundaries

Normal slightly wins final held-out math CE and has higher recorded throughput. The earlier 91M-token simplicial advantage reversed later.

The original scratch pair stopped at 240.875M tokens of a planned 10B; it cannot establish the converged ranking. Its first-option-biased MMLU predictions are not evidence of strong knowledge.

Native sparse attention already uses TileLang. Serving and incremental-cache work require separate full-model qualification.

**Records:** `docs/DEEPSEEK_V41_FULL_COMPARISON.md`, `docs/DEEPSEEK_V41_NORMAL_CONTROL.md`, `docs/DEEPSEEK_V41_DEBUG_20260922.md`.

**Next:** [[Math RL|Math-RL]] · [[Evaluation]]
