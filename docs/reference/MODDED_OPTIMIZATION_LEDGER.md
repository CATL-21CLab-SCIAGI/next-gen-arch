# Modded-NanoGPT optimization ledger

**Type:** historical reference, 2026-08-26. **Source:** `ecbb586296d3dac36fd206211f25d63bad4a6b35`.
Records are cumulative; grouping follows the first introduction of each mechanism.

| Records | Mechanisms | Disposition here |
|---|---|---|
| 1–5 | modern GPT, RoPE, QK norm, ReLU², padding, zero projections, Muon | retained |
| 6–10 | distributed Muon, PyTorch upgrade, untied head, value/x0 paths, BF16 | retained or delegated to container-owned Megatron/PyTorch |
| 11–21 | U-Net skips, attention/window tuning, value embeddings, softcap, FP8 head, fused QKV, batch tuning | retained where contract-compatible; FP8 head is a B300 capability probe |
| 22–24 | faster all-reduce, overlap, reduce-scatter | delegated to Megatron distributed optimizer/DDP; requires a multi-rank scale probe |
| 25–33 | runtime upgrade, BOS alignment, transposed MLP kernel, attention gate, FA3, layer dropping, YaRN, BF16 cleanup, async data | BOS/BF16/gates retained; FA3 is Hopper-only; shape/schedule changes stay explicit |
| 34–43 | smear, dropped layers, fused Muon comms, BF16 CE, Polar Express, Adam every two, backout, NorMuon, cautious WD | retained or isolated recipes |
| 44–50 | optimizer hooks/overlap, refined skips, batch schedule, lambda placement, Muon reshape, PKO, cautious Adam WD | Megatron/compile or recipes; batch schedule is a contract change |
| 51–57 | retie/split embeddings, scalar schedules, MTP, asymmetric logits, value gates, compiled Adam, mixed weights/interleaving | MTP/value gates/compiled Adam retained or existing arms; dynamic tying is a contract-changing parameter schedule |
| 58–61 | paired-head attention, fused ReLU², fused softcapped CE, unified/transposed optimizer layout | seven heads make exact pairing impossible; kernels are capability-probed; optimizer is already unified |
| 62–71 | bigram hash/sign precursor, untied/fused value embeddings, mimetic V/O init, Torch upgrades, kernel tuning, sparse bigram comms | Engram/smear cover the controlled lexical-memory comparison; H100/vocab-50304 kernels do not fit the 56-wide, vocab-32768 contract |
| 72–80 | sequence schedule, partitioned/simplified hyperconnections, flattened forward, CE/transpose kernels, varlen bounds, paired-head Muon | schedules are explicit contract changes; mHC is an existing arm; compile handles flattening; per-head/full controls are measured |
| 81–89 | MUDD, learnable/algebraic XSA, bigram sign, FP8 MLP, dynamic MHA, fused ReLU², prefix loss, FP8 down projection | XSA/MUDD/SConv/GDN relatives remain architecture arms; prefix loss changes the objective; FP8/fused kernels require aligned larger shapes |

“Retained,” “recipe,” and “scale candidate” describe this repository's disposition, not universal performance claims. Hardware/shape assumptions require independent qualification.

[Measured optimization audit](../OPTIMIZATION_AUDIT.md)
