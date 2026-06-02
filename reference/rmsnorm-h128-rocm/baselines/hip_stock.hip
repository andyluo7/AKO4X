// hip_stock.hip — naive HIP RMSNorm reference for h=128.
//
// Second baseline row for ako4x-rocm Phase 1. Intentionally simple:
//   * One block per row, BLOCK=128 threads (one thread per hidden element).
//   * Two passes (load → reduce → broadcast → scale → store), single LDS reduction.
//   * No __builtin_amdgcn intrinsics; no MFMA; no fused epilogue.
//
// Compile via PyTorch's load_inline (which routes through hipcc on ROCm) — see
// the companion bench harness ``benchmark_rmsnorm_h128.py`` for the call site.
//
// Expectation: this should be CLEARLY beatable by:
//   - LDS bank-conflict-free variant (~1.2× win, agent's first easy target)
//   - persistent kernel with grid sized to CU count (~1.3-1.5× win on small batches)
//   - wave64 cross-lane reduction via __builtin_amdgcn_ds_bpermute (~1.1× win)
//   - fused epilogue (load + reduce + scale all in one kernel — already done here)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HIDDEN 128
#define BLOCK 128

__global__ void rmsnorm_h128_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
          __hip_bfloat16* __restrict__ y,
    int n_rows,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n_rows) return;

    // Load + accumulate sum-of-squares in fp32 (bf16 reduction has too much error).
    __shared__ float ssq[BLOCK];
    float val = static_cast<float>(x[row * HIDDEN + tid]);
    ssq[tid] = val * val;
    __syncthreads();

    // Tree reduction. Simple log2(BLOCK)=7 steps; no warp shuffle, no skew.
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) ssq[tid] += ssq[tid + s];
        __syncthreads();
    }

    // Broadcast the inverse-RMS to every thread (LDS read).
    float inv_rms = rsqrtf(ssq[0] / static_cast<float>(HIDDEN) + eps);

    // Scale + store.
    float w_val = static_cast<float>(w[tid]);
    y[row * HIDDEN + tid] = static_cast<__hip_bfloat16>(val * inv_rms * w_val);
}

extern "C" void rmsnorm_h128_launch(
    const void* x, const void* w, void* y,
    int n_rows, float eps, hipStream_t stream)
{
    dim3 grid(n_rows);
    dim3 block(BLOCK);
    rmsnorm_h128_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __hip_bfloat16*>(x),
        static_cast<const __hip_bfloat16*>(w),
        static_cast<__hip_bfloat16*>(y),
        n_rows, eps);
}
