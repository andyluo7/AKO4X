# fp8-gemm-dsr1-rocm — Phase 2: compute-bound operator

DSR1 FFN-ish FP8 GEMM. **Single shape, real scales.**

- **Shape:** M=4096, N=7168, K=2048 (override with `--m/--n/--k`)
- **Dtypes:** A, B in `float8_e4m3fnuz` (ROCm-native); C in `bf16`
- **Layout:** A `[M, K]`, B `[N, K]` (weight layout — K-contig for both)
- **Scales:** per-token `scale_a [M]` + per-channel `scale_b [N]`, fp32
- **Math:** `C[m, n] = (sum_k A[m,k] * B[n,k]) * scale_a[m] * scale_b[n]`

Compute = 2·M·N·K = **120 GFLOP per call**. MI350X peak FP8 ~5 PFLOP/s → theoretical floor ~24 µs.

## Goal

The Phase 1 operators (RMSNorm, LayerNorm) hit a bandwidth wall fast. This operator is **compute-bound** — the variant search has real room to move (MFMA tile shape, software pipelining, swizzle, scale-fold strategy). Target: parity with hipBLASLt / AITER `gemm_a8w8` on this single shape.

## Baselines

| name | source | what it tests |
|---|---|---|
| `torch_naive` | `baselines/torch_naive.py` | fp32 reference, correctness anchor |
| `hipblaslt_scaled_mm` | `torch._scaled_mm` | hipBLASLt-backed FP8 GEMM |
| `aiter_tuned` | probed `aiter.gemm_a8w8*` | AMD-tuned FP8 path |
| `hip_stock` | `baselines/hip_stock.hip` | naive scalar HIP, one output per thread |

## Variants (Phase 2 work-in-progress)

`variants/<name>/{kernel.hip, binding.cpp}` are auto-discovered by the driver. Planned first attempt: `v1_mfma_tile` — `mfma_f32_16x16x32_fp8_fp8`, BLOCK_M=128 × BLOCK_N=128 × BLOCK_K=64, LDS double-buffered, scales applied in the epilogue.

**Design deferred until baselines measured** — the variant design space (tile shape, pipeline depth, swizzle) is too wide to commit blind; we want to see where hipBLASLt + AITER land first so v1 has a clear bar to beat.

## Reproduction

```bash
ssh root@134.199.194.185
cd /workspace/AKO4X && git pull origin rocm-port
docker run --rm \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v /workspace/AKO4X:/work \
    -w /work/reference/fp8-gemm-dsr1-rocm \
    rocm/pytorch-training:v25.9_gfx950 \
    python benchmark_fp8_gemm.py --iters 200 --warmup 5 --out baseline.json
```
