# Marin optimization ledger

**Type:** historical reference, 2026-08-26. **Source:** `299c7f3245e2e6998345980cadad75f45088f63f`.
The audit classified 80 Agent-MoE experiments; MoE-only evidence is not silently generalized to dense controls.

| Area and issue IDs | Disposition |
|---|---|
| PKO/RoPE #4802 #4849 #4899 #4900 #4946 #4976 #5152 #5160 | portable portions are recipes; MHA is already the 10M baseline |
| attention layouts #4907 #4951 #5047 #5151 | paired heads are incompatible with seven heads; wide/MHA changes geometry; MHA already active |
| QK controls #5114 #5227 #5230 #5373 #5381 | QK gain is a recipe; known-negative removal/multiplier sweeps are not promoted; QK norm retained for scaling stability |
| value path #4986 | retained in the inherited baseline for comparison continuity despite Marin's negative MoE result |
| expert topology #4901 #5387 #5399 #5409 #5477 #5486 #5491 #5501 #5502 #5515 | MoE-only scale candidates, outside the dense 10M backend |
| router changes #5409 #5477 #5486 #5491 #5501 #5502 #5507 #5512 | upstream negative or MoE-only; no dense analogue |
| Muon family #5115 #5134 #5167 #5517 #5585 #5596 #6388 #6404 #6505 #8131 | MuonH/MuonEq recipes and LR retuning; factorized gains/row norms remain negative or higher-scale candidates |
| AdamH/numerics #5178 #5203 #5235 #5238 #5250 #5251 | AdamH, init, and clipping recipes; negative epsilon/gate/gradient-aware variants documented |
| precision/pipeline #6431 #6486 | delayed pipeline is a PP scale candidate; BF16-master negative is directly reproduced by backend profiles |
| residuals #4806 #4807 #4905 #4906 #4987 #5110 #5113 #7409 #8196 | x0/backout retained; cached and midpoint K/V are recipes; block/identity HC need deeper models; full residual is upstream negative |
| convolution #8377 | two SConv architecture arms already exist; CUDA/PyTorch backend remains independent of Pallas |
| depth/block #5002 #5154 #5423 #5938 | geometry changes or upstream negative; not folded into backend |
| head/norm #4803 #4973 #5222 #5224 #6442 | softcap retained; z-loss is a scale-stability candidate; negative init/split norms not repeated |
| activations #5407 #5460 #6519 #7255 | xIELU/SiTU arms already exist; GEGLU/SwiGLU would change the architecture comparison |
| data/objective/eval #5306 #5493 #6509 #6570 #7181 | data, sequence, serving, or evaluation questions—not backend optimizations under fixed ClimbMix |
| synthesis #4999 #5371 plus #5184 #5292 #5763 #6882 | dense portable compound is a recipe; expert count/router parts are MoE-only |

Disposition records the inspected evidence and experimental fit. A proposed recipe or a larger-scale candidate is not a demonstrated gain in the current backend.

[Measured optimization audit](../OPTIMIZATION_AUDIT.md)
