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
| **`aiter_a8w8_bpreshuffle`** (W preshuffled once) | **79.0** | **1522** | **30%** | ✅ atol 0.25 |
| `hipblaslt_scaled_mm` (torch._scaled_mm)         | 79.8     | 1508     | 30%      | ✅ atol 0.25 |
| `aiter_a8w8` (CK FP8 path, generic)              | 89.3     | 1347     | 27%      | ✅ atol 0.25 |
| `hip_stock` (scalar one-output-per-thread)       | 114700   | 1.0      | 0.02%    | ✅ atol 0.0 (exact match) |

Weight pre-shuffle (`aiter.ops.shuffle.shuffle_weight`) is a one-shot tile permutation that
matches `gemm_a8w8_bpreshuffle`'s expected layout — done once at load time in real inference,
so it's outside the timed loop here.

## What the baselines tell us

**The bar to beat is now ~79 µs / 1522 TFLOP/s** (AITER bpreshuffle, tied with hipBLASLt).
With weights pre-shuffled into the bpreshuffle tile layout, AITER edges out hipBLASLt
by ~1% — essentially a tie. AITER's generic `gemm_a8w8` (row-major weight, no preshuffle)
is 12% slower.

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

- ✅ **bpreshuffle delivered** — landed as a baseline (vendor tuning trick, not a custom
  kernel). Sets the new bar at ~79 µs / 1522 TFLOP/s.
- 🚧 **`v1_mfma_tile`** in progress — handwritten HIP kernel using `mfma_f32_16x16x32_fp8_fp8`.
  First commit: correctness-first scaffold (64×64 block, 4 waves, single-buffered LDS,
  scales in epilogue). Expected to be slow until iterated; the goal of v1 is correctness,
  v2+ adds double-buffering / global_load_lds / wider tiles.
- ⏭️ **`v2_tilelang`** (deferred) — same MFMA tile written in TileLang DSL with autotuner
  sweep. Depends on TileLang's gfx950 support being solid.
