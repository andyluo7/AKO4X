# Phase 2 baselines — FP8 GEMM (DSR1 FFN shape) on MI355X gfx950

**Hardware:** `smci355-ccs-aus-m15-17`, AMD Instinct MI355X (gfx950, CDNA4)
**Software:** `rocm/pytorch-private:mxfp8-gfx950-v26.6` (torch 2.12.0.dev+rocm7.1, AITER 25.9)
**Operator:** FP8 GEMM, `C[m, n] = (sum_k A[m,k]*B[n,k]) * scale_a[m] * scale_b[n]`
**Dtypes:** A, B in `float8_e4m3fn` (OCP); C in `bf16`; scales fp32 (per-token A + per-channel B)
**Shape:** M=4096, N=7168, K=2048 (DSR1 FFN-like), 120.3 GFLOP per call
**Methodology:** 5 warmup + 100 timed iterations, correctness atol = max(0.5, 0.05·out_amax)

## Baseline table (MI355X)

| backend | latency (µs) | TFLOP/s | correctness |
|---|---:|---:|---|
| **`aiter_a8w8_bpreshuffle`** (W preshuffled once) | **74.1** | **1622.7** | ✅ atol 0.25 |
| `aiter_a8w8` (CK FP8 path, generic)              | 93.4     | 1287.3     | ✅ atol 0.25 |
| `hip_stock` (scalar one-output-per-thread)        | 104923   | 1.1        | ✅ atol 0.0  |

hipBLASLt (`torch._scaled_mm`) crashes on the small correctness-check shape with `HIPBLAS_STATUS_INVALID_VALUE`
on torch 2.12+rocm7.1 — excluded.

## Variants (6 attempts, 2 wins, 3 dead-ends)

Latest run on MI355X, all variants ran for 100 iters after 5 warmup:

| variant | latency | TFLOP/s | vs v1 | result |
|---|---:|---:|---:|---|
| v1_mfma_tile (64×64 block, 4 waves, single LDS)         | 430.3 µs | 279   | 1.00× | ✅ baseline (correctness anchor) |
| v2_wider_tile (128×128 block, 4 waves)                  | 242.9 µs | 495   | 1.77× | ✅ was anchor (beat by v6) |
| v3_double_buffer (v2 + 2-stage LDS pipeline)            | 252.4 µs | 476   | 1.70× | ❌ tied v2 (compiler already overlaps) |
| v4_bigger_mfma (v2 + mfma_32x32x16)                     | 243.1 µs | 495   | 1.77× | ❌ tied v2 (dispatch ≠ bottleneck) |
| v5_block_256x128 (256×128 block, 8 waves)               | 280.3 µs | 429   | 1.54× | ❌ slower (occupancy collapsed) |
| v6_lds_swizzle (v2 + 8-byte LDS row pad)                | 138.5 µs | 868   | 3.10× | ✅ was anchor (beat by v7) |
| **v7_8wave** (v6 + 8 waves/block, 24 waves/CU)          | **102.3 µs** | **1175** | **4.20×** | ✅ **THE ANCHOR** |
| v8_wider_n (BN=256, 25% less DRAM, 16 waves/CU)         | 109.8 µs | 1095  | 3.91× | ❌ slower (MFMA latency exposed at 16 waves/CU) |

**Headline:** v7 → 1.35× over v6, 4.2× over v1; **still 1.37× behind AITER bpreshuffle** (102.3 µs vs 74.7 µs).

## What the results taught us

### Dead-ends

- **v3 (double-buffer):** with regular global loads, the HIP compiler already issues
  loads ahead of MFMA work via instruction scheduling. Explicit software double-buffering
  doesn't add overlap that wasn't already there.
- **v4 (32x32x16 MFMA):** halved MFMA dispatch count, zero speedup. MFMA dispatch
  overhead is not the bottleneck at this tile size.
- **v5 (256×128):** reduced redundant DRAM traffic ~25%, got *slower*. The 24KB LDS
  footprint forced 1 block/CU (was 2), and total block count dropped (896 vs 1792)
  — the wavefront scheduler had less to interleave.

### Win: v7 — more waves per block

**v7's 1.35× win over v6** confirms MFMA latency exposure at 12 waves/CU.
Doubling to 8 waves/block gives 24 waves/CU; with MFMA latency ≈ 64 cycles and
4-cycle issue rate, 24×4=96 cycles between re-issues on the same wave — enough to
fully hide the 64-cycle latency. The occupancy-per-block impact is zero (LDS unchanged
at 18KB → 3 blocks/CU); the gain is purely from better MFMA pipeline utilization.

**Remaining ~1.37× gap to AITER bpreshuffle** may be DRAM-bandwidth limited:
v7 reads 917MB at ~9 TB/s ≈ 102µs which matches our measurement. AITER at 74.7µs
reads significantly less (better L2 reuse or larger tiles).

### Win: v6 — bank conflict elimination

**v6's 1.76× win over v2 confirms the root cause:** v2's `A_lds[128][64]` has a 64-byte
row stride = 16 LDS banks. With 32 banks total, 16 consecutive-row MFMA-operand reads
(one wave's 8-byte loads at the same column) all aliased into 2 bank-pair groups —
an 8-way bank conflict. Adding 8 bytes of padding (stride → 72 bytes = 18 banks per row)
makes consecutive rows walk the bank space with stride 2 (gcd(18,32)=2), giving 16 distinct
bank-pair offsets for 16 rows → conflict-free.

Occupancy dropped from 4 blocks/CU to 3 blocks/CU (18 KB × 3 = 54 KB ≤ 64 KB), but the
conflict savings (8× serialization eliminated for every MFMA operand load) outweighed the
occupancy loss.

**Remaining ~1.86× gap to AITER bpreshuffle** likely lives in:

- **`__builtin_amdgcn_global_load_lds`** — true async DRAM→LDS (no VGPR staging); could
  overlap tile loads with MFMA compute across K iterations.
- **Multi-stage software pipelining** (3- or 4-stage, with manual `s_waitcnt`).
- **LDS swizzle for the tile-load (store) phase** — the 16-byte uint4 stores during
  tile load also have bank conflicts (though less severe than the 8-byte MFMA loads).
- **Register pressure / occupancy tuning** — with 3 blocks/CU (12 waves), there may be
  residual latency exposure.
