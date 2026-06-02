# Phase 1 results — RMSNorm on MI350X gfx950

**Hardware:** DigitalOcean contracted MI350X box, 8× AMD Instinct MI350X VF (gfx950, CDNA4)
**Software:** ROCm 7.2 host kernel + `rocm/pytorch-training:v25.9_gfx950` (torch 2.9.0+rocm7.0.0)
**Operator:** RMSNorm BF16, three hidden dims (h=128 MLA-per-head, h=4096 Llama, h=7168 DSR1)
**Methodology:** 5 warmup + 200 timed iterations per shape, all rows pass correctness vs torch reference (atol=0.0312 in BF16)

## Latency table (µs)

| n_rows | hidden | torch_naive | aiter_tuned | hip_stock | **v1_vector_loads** | best | speedup vs aiter |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 128 | 40.9 | 11.4 | 4.6 | 4.7 | hip_stock | **2.4×** |
| 128 | 128 | 43.8 | 11.9 | 4.7 | 4.9 | hip_stock | **2.5×** |
| 1024 | 128 | 44.4 | 10.4 | 4.7 | 4.8 | hip_stock | **2.2×** |
| 8192 | 128 | 44.7 | 12.0 | 4.8 | 4.8 | tie | **2.5×** |
| 1 | 4096 | 43.9 | 9.2 | 5.4 | **4.8** | v1 | **1.9×** |
| 128 | 4096 | 43.6 | 10.1 | 5.5 | **4.9** | v1 | **2.1×** |
| 1024 | 4096 | 64.5 | 11.1 | 7.3 | **5.0** | v1 | **2.2×** |
| 8192 | 4096 | 354.6 | 23.0 | 42.2 | **22.4** | v1 | **1.03×** |
| 1 | 7168 | 43.3 | 9.1 | 7.5 | **4.7** | v1 | **1.9×** |
| 128 | 7168 | 43.4 | 10.1 | 8.1 | **4.9** | v1 | **2.1×** |
| 1024 | 7168 | 91.7 | 11.0 | 13.0 | **6.6** | v1 | **1.7×** |
| 8192 | 7168 | 638.5 | **40.0** | 72.9 | 47.0 | aiter | 0.85× |

**Geomean speedup over AITER across all 12 shapes:** v1_vector_loads at **1.81×**.

## Summary

- **11 of 12 shapes won by AKO4X-ROCm variants** (`hip_stock` for h=128, `v1_vector_loads` for h≥4096)
- **AITER's `aiter.rms_norm` runs at ~9-12 µs constant** across most n_rows / hidden combinations → strongly launch-bound for small workloads
- **AITER scales** on the largest shape `(8192, 7168)`: 40 µs vs torch's 638 µs (16× over reference) → that's where AITER's tuning effort lives, and where we still have a 1.18× gap

## What's in each variant

### `hip_stock` (baseline, naive HIP)
- One block per row, `BLOCK=128` for `h=128` (one elem/thread); `BLOCK=256` chunked for other hiddens
- Single LDS reduction, no vector loads, no MFMA, no persistent kernels
- Header at `baselines/hip_stock.hip`

### `v1_vector_loads` (this commit's winner)
- **Only change from `hip_stock`**: 128-bit (16-byte = 8×bf16) vector loads + writes via
  `union { uint4 u; __hip_bfloat16 bf[8]; }` reinterpret
- Reduction loop unchanged
- Gets memory throughput up 8× per thread → **closes ~70-100% of the gap to AITER** on every shape
- Header at `variants/v1_vector_loads/kernel.hip` (full Identity / Delta / Lessons / Dead-ends per the AKO4X variant convention)
- Note: clang's `ext_vector_type(8)` rejects `__hip_bfloat16` as an element — the
  union+uint4 idiom is the AMD-correct pattern (documented as a Dead-end in the header)

## What's next (Phase 1.x — open)

The remaining `(8192, 7168)` gap of 1.18× is the v2 target. Likely techniques (from the `hip` SKILL):

1. **Persistent kernel** — grid sized to CU count (256 on gfx950), each block processes
   multiple rows. Wins on small `n_rows` AND amortizes launch on large.
2. **Wave64 cross-lane reduction** via `__builtin_amdgcn_ds_bpermute` — eliminates the
   LDS reduction pass (saves ~2 µs at large hidden).
3. **Multi-block reduction at large hidden** — split each row across multiple blocks,
   use atomicAdd for the per-row final sum. Lets us scale beyond one block's threads.
4. **Direct DRAM→LDS path** via `__builtin_amdgcn_global_load_lds` — skips VGPR staging
   for the K/V tiles.

Each is a candidate v2/v3/v4 in the same `variants/<name>/{kernel.hip,binding.cpp}` shape.

## Phase 1 success criteria — met

- [x] **MVP scaffolding**: AKO4X-rocm running end-to-end on MI350X (build, run, correctness, latency)
- [x] **Baseline matrix**: torch / AITER / hand-HIP all measured side-by-side with correctness gate
- [x] **First "agent" variant**: `v1_vector_loads` produced, correctness ✅, beats AITER on 9/12 shapes
- [x] **Variant auto-discovery**: drop a `variants/<name>/` folder, next benchmark run picks it up
- [x] **Result archival**: `baseline.json` written in the AKO4X reference schema

## Reproduction

On the MI350X box (root@134.199.194.185):

```bash
cd /workspace/AKO4X
git pull origin rocm-port
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v /workspace/AKO4X:/work \
    -w /work/reference/rmsnorm-h128-rocm \
    rocm/pytorch-training:v25.9_gfx950 \
    python benchmark_rmsnorm_h128.py \
        --n-rows 1,128,1024,8192 --hidden 128,4096,7168 \
        --iters 200 --warmup 5 --out baseline.json
```

Output: `baseline.json` with per-shape, per-impl latency rows.
