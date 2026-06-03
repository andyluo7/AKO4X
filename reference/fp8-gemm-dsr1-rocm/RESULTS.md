# Phase 2 baselines — FP8 GEMM (DSR1 FFN shape) on MI350X gfx950

**Hardware:** DigitalOcean MI350X box, AMD Instinct MI350X VF (gfx950, CDNA4)
**Software:** `rocm/pytorch-training:v25.9_gfx950` (torch 2.9.0+rocm7.0.0, AITER 25.9)
**Operator:** FP8 GEMM, `C[m,n] = (sum_k A[m,k]*B[n,k]) * scale_a[m] * scale_b[n]`
**Dtypes:** A, B in `float8_e4m3fn` (OCP); C in `bf16`; scales fp32 (per-token A + per-channel B)
**Shape:** M=4096, N=7168, K=2048 (DSR1 FFN-like), 120.3 GFLOP per call
**Methodology:** 5 warmup + 200 timed iterations, correctness atol = max(0.5, 0.05·out_amax)

## Baseline table

| backend | latency (µs) | TFLOP/s | % MI350X FP8 peak (~5 PF) | correctness |
|---|---:|---:|---:|---|
| **`hipblaslt_scaled_mm`** (torch._scaled_mm)     | **77.7** | **1548** | **31%** | ✅ atol 0.5 |
| `aiter_a8w8` (CK FP8 path)                       | 87.1     | 1381     | 28%      | ✅ atol 0.5 |
| `aiter_a8w8_bpreshuffle` (needs preshuffled W)   | —        | —        | —        | ❌ wrong (W not shuffled) |
| `hip_stock` (scalar one-output-per-thread)       | 114700   | 1.0      | 0.02%    | ✅ atol 0.0 (exact match) |

## What the baselines tell us

**hipBLASLt is the bar to beat: 77.7 µs / 1548 TFLOP/s.** It beats AITER's generic path
by 12% on this shape. AITER's `gemm_a8w8_bpreshuffle` (the dedicated FP8 fast-path on
MI350) requires the weight tensor to be pre-shuffled into a specific tile layout; calling
it with a row-major weight produces wrong results. With proper preshuffle, it would
likely beat both.

**31% of FP8 peak is not the ceiling.** K=2048 is relatively small — the operator
has less arithmetic intensity than a square 4096³ GEMM, so the *practical* peak (given
DRAM bandwidth for A) sits well below 5 PFLOP/s. A roofline estimate puts the achievable
ceiling around 2.0–2.5 PFLOP/s for this shape; hipBLASLt is at ~62–77% of that practical
peak, AITER at ~55–69%.

**hip_stock is hilariously slow as expected** — 1 TFLOP/s. Its only job is to provide
an exact correctness anchor (atol 0.0 against the fp32 reference) so we trust the
variant comparisons. The 5-orders-of-magnitude gap shows the size of the win available
to a proper MFMA tile.

## Operator-archive layout

```
reference/fp8-gemm-dsr1-rocm/
├── README.md  RESULTS.md  baseline.json
├── benchmark_fp8_gemm.py  (driver: hipBLASLt + AITER ×2 + hip_stock, auto-discovers variants/)
├── baselines/
│   ├── torch_naive.py      (fp32 reference)
│   ├── hip_stock.hip       (scalar one-output-per-thread, manual e4m3fn decode)
│   └── hip_stock_binding.cpp
└── variants/  (empty)
```

## What's next — v1 directions

Three plausible v1 attempts, in increasing effort:

1. **`v1_bpreshuffle`** — shuffle the weight tensor once with AITER's helper, then call
   `gemm_a8w8_bpreshuffle`. Tests whether AITER's fast FP8 path beats hipBLASLt with
   the right input layout. Low effort (1-2 hours), tests an AITER tuning trick rather
   than a custom kernel.
2. **`v1_mfma_tile`** — handwritten HIP kernel using `mfma_f32_16x16x32_fp8_fp8`,
   BLOCK_M=128 × BLOCK_N=128 × BLOCK_K=64, LDS double-buffered, scales fused into
   the epilogue. Real custom-kernel work; multi-day to get right.
3. **`v1_tilelang`** — write the same MFMA tile in TileLang DSL and let the autotuner
   sweep tile/pipeline params. Lower hand-tuning burden, depends on TileLang's gfx950
   support being solid.
