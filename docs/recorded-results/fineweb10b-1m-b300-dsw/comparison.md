# FineWeb10B 1M architecture screen on one B300

All 17 seed-42 Megatron runs completed on one PAI DSW node under the same clean
commit, data prefix, optimizer, and batch contract. Each optimizer step contains 192
sequences as 12 accumulated microbatches of 16. The campaign completed in 1,231.1
seconds with zero failed or non-finite runs.

| Variant | Final BPB | Δ vs baseline | Steady tok/s | Speed |
| --- | ---: | ---: | ---: | ---: |
| Engram | **2.167429** | **-0.334663** | 3,861,844 | 0.961× |
| Kimi K3 KDA update | 2.178213 | -0.323880 | 1,211,831 | 0.301× |
| GLM MLA | 2.179100 | -0.322992 | 3,889,982 | 0.968× |
| Relative attention | 2.179734 | -0.322358 | 3,031,932 | 0.754× |
| Partial RoPE 25% | 2.180228 | -0.321865 | 3,948,640 | **0.982×** |
| AttnRes | 2.180521 | -0.321571 | 3,797,451 | 0.945× |
| mHC | 2.180575 | -0.321518 | 2,622,664 | 0.653× |
| sconv-KV | 2.180781 | -0.321312 | 3,797,174 | 0.945× |
| sconv residual | 2.181504 | -0.320589 | 3,859,975 | 0.960× |
| KDA | 2.184269 | -0.317823 | 1,208,195 | 0.301× |
| Gated attention | 2.227635 | -0.274458 | 3,915,089 | 0.974× |
| SiTU-GLU | 2.229148 | -0.272945 | 3,874,863 | 0.964× |
| DSA | 2.231178 | -0.270914 | 1,992,895 | 0.496× |
| xIELU | 2.253218 | -0.248874 | 3,927,003 | 0.977× |
| Qwen GDN | 2.282735 | -0.219357 | 1,164,314 | 0.290× |
| Baseline | 2.502092 | — | 4,019,370 | 1.000× |
| CoLU | 3.010535 | +0.508443 | 3,891,596 | 0.968× |

Engram is the best quality result and retains 96.1% of baseline steady throughput.
Kimi K3 KDA and KDA also score well but run at about 30% of baseline. CoLU is the
only negative result in this screen.

This is a one-seed, extremely small-model screen, not a mature-scale leaderboard. The
baseline moved materially when the single-GPU run changed from one 192-sequence
microbatch to mathematically equivalent 16-sequence gradient accumulation, showing
that this 37-step regime is numerically sensitive. Relative attention, mHC, and DSA
also contradict their larger-scale evidence. Promote mechanisms only after matched
multi-seed 10M/100M confirmation.

Full precision values are in [summary.csv](summary.csv), and the frozen runtime and
provenance contract is in [campaign.json](campaign.json). Raw per-run results remain
on durable campaign storage; the public artifact intentionally omits its
host-specific path.
