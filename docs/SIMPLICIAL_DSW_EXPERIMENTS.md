# Controlled simplicial attention: DSW preflight

Only the mechanism preflight is implemented and run. Full-model A/B/C training
has **not** been launched. This is not yet a supported Megatron architecture.
The current DLC training code, process, allocation and container are unchanged.

## Reproduce the bounded preflight

Use the existing container Python with project `src` on `PYTHONPATH`:

```sh
python -m archlab.benchmarks.simplicial_attention \
  --output <new-evidence-json> --batch 4 --sequence 2048 \
  --repetitions 20 --include-flash
```

The output must not exist. Source hashes, runtime/container metadata, GPU process
snapshots, all numerical errors and timing samples are saved. No packages are
installed. Both no-grad forward and forward/backward paths are warmed up before
timing. GPU memory is capped at 25%. Other services are not stopped; timings are
preliminary if a co-resident service is active.

CPU tests need only existing PyTorch and the standard library:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_simplicial_attention.py -v
```

## Code boundaries

- `src/archlab/architectures/simplicial_attention.py`: input contract, independent
  FP64-capable oracle and lazy CUDA dispatch. No trainer/optimizer dependencies.
- `src/archlab/architectures/simplicial_kernels.py`: project-owned forward and
  backward kernels using stock Triton, not TLX. Both KV groups are local. A query
  owns 12 valid heads in a masked 16-head tile. There is no TP or EP requirement.
- `src/archlab/benchmarks/simplicial_attention.py`: bounded execution adapter.
- `recipes/proposals/qwen38_w320_simplicial_dsw.yaml`: descriptive proposed
  A/B/C contract, not an executable training recipe.

The kernel is the coordinate-wise trilinear score with joint softmax over two
causal sliding-window axes and elementwise products of the two value branches.
It adds no KV biases and does not itself apply projections, Q/K normalization,
gating, residual mixing, or positional embeddings. Those remain required in
the future enclosing model adapter. It is not determinant-based attention.

Backward recomputes scores and uses FP32 atomics for shared K/V gradients.
Consequently it is not bitwise deterministic. Higher-order differentiation is
not validated. The preflight does not establish full-model optimizer grouping,
checkpointing, distributed equivalence, convergence or training throughput.

## Decisions before full-model training

1. Choose a positional encoding explicitly. Retaining ordinary RoPE on Q and
   both keys defines a named RoPE+simplicial variant, without claiming ordinary
   RoPE relative-position invariance. A no-RoPE experiment needs matched B/C
   controls in the six changed slots, not an undocumented omission in C.
2. Choose the existing runtime. Container Python has PyTorch/Triton and Apex,
   but lacks FLA for the unchanged GDN layers. The pre-existing sampling
   environment has FLA and different package versions. Do not silently switch
   environments or install packages.
3. Choose the short/long windows using actual forward/backward and full-block
   cost. The paper's large window is not a throughput-neutral change at 2K.
4. Implement and test the native model/optimizer/checkpoint integration, paired
   common-weight initialization, identical data order and held-out evaluation.
   DP1 requires 1024 microbatches at microbatch 4 to retain the 4096-sequence
   effective batch. All three arms must use the same DSW runtime and settings;
   the DLC curve is not a matched DSW baseline.

## References

- https://arxiv.org/html/2507.02754v1
- https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/

This implementation was written against the mathematical operation; it does
not vendor or patch Triton, Megatron, PyTorch, Transformer Engine or CUDA.
