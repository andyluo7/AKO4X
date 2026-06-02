# FlyDSL — Detailed Reference

Companion to `SKILL.md`. FlyDSL is a Python-embedded DSL for AMD preshuffle GEMM, targeting gfx942 and gfx950.

Repo: `github.com/ROCm/FlyDSL` (currently access-restricted; contact AMD).

## Operator coverage

Primary: **preshuffle GEMM** — a GEMM variant where the K dimension is permuted (preshuffled) ahead of time so the LDS load pattern is conflict-free. Used widely in modern attention/MoE pipelines where the K-side activation tensor can be permuted at allocation time (avoiding the cost online).

Secondary: any GEMM you can express via the same `preshuffle_gemm.py` template (BF16, FP16, FP8 inputs).

## Fused-epilogue support (PR #404)

`body_row` slot in the GEMM output store loop accepts a small Python-AST callable. Supported epilogues:

- `bias` (vector add)
- `ReLU`, `SiLU`, `GeLU` (activations)
- chained: `bias + activation` is one extra instruction in the inner loop

Example:
```python
@flydsl.preshuffle_gemm(
    m=BS*S, n=N, k=K,
    dtype="bf16",
    epilogue=flydsl.Epilogue.BIAS_SILU,  # bias + SiLU
)
def my_gemm(a, b, bias, out):
    ...
```

Compared to running stock GEMM then a separate epilogue kernel: ~5-20% E2E saving, depending on the shape (larger savings on small-M / small-N).

## gfx950 preload tuning (PR #411)

`preshuffle_gemm.py` has a `_preload_table` lookup keyed by GPU arch. The gfx950 entries were missing in the initial release — using gfx942 entries on gfx950 hardware regresses ~14.6% on DeepSeek-R1 MoE workloads (measured 109.05 → 95.5 tok/s).

Make sure your FlyDSL install includes the gfx950 preload table — verify with:
```python
import flydsl
print(flydsl.preshuffle_gemm._preload_table.keys())
# Should include 'gfx950', not just 'gfx942'.
```

## Persistent-kernel codegen

FlyDSL emits persistent kernels by default (one block per CU, looping over work items). This avoids the ~10 µs HIP launch overhead and is the main reason it beats hipBLASLt at small-N. The persistent loop is configurable via the `persist=True` decorator argument — turn it off only when the work doesn't divide cleanly across CUs.

## Beating FlyDSL

Hard. FlyDSL is highly tuned for the preshuffle pattern on AMD. Realistic wins:

- **Shapes outside its sweet spot** — non-power-of-2 N, K not aligned to 256, very-small M (< 32).
- **Multi-op fusion beyond bias+activation** — e.g., GEMM + LayerNorm + GEMM as one persistent kernel.
- **Hybrid with MFMA-direct intrinsics** — bypass FlyDSL's codegen for the inner loop while using its preload scheduling.

## Coupling notes

FlyDSL outputs HIP source that gets compiled by hipcc. You can dump the generated source with:
```python
flydsl.compile(my_gemm, debug=True)  # writes /tmp/flydsl_out_*.hip
```

Useful to study what FlyDSL emits before deciding to bypass it.

## Versioning

FlyDSL is under active development; pin the commit in `baseline.json` (`flydsl_tuned` row's `commit` field). Re-baseline on bumps.
