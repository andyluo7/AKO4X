// === Identity ===
// v2_persistent — vector loads + persistent kernel (grid = CU count, each block
// loops over multiple rows). Targets the (8192, h>=4096) regime where v1 ties
// or loses to AITER by 1.18x. Hypothesis: at large n_rows the per-block
// scheduling overhead becomes the dominant cost; folding rows into a 256-block
// grid removes (n_rows / 256) "block start" events per kernel invocation.
//
// === Delta from parent (v1_vector_loads) ===
// 1) Outer loop over rows with stride = gridDim.x — same per-row body as v1.
// 2) Grid sized to MI350X / MI355X CU count (256 on gfx950) by the host
//    launcher, capped at n_rows. Block size still 256 threads = 4 wave64.
// 3) Final __syncthreads() at end of each row iteration so LDS reuse is safe.
//
// === Lessons (predicted) ===
// WHEN n_rows >> CU_COUNT (e.g. 8192) AND per-row work is small (h <= 8192)
//   → persistent wins because block scheduling overhead amortizes.
// WHEN n_rows <= CU_COUNT (e.g. 128) → persistent ties or slightly loses
//   (the row-loop overhead has no rounds to amortize over).
//
// === Dead-ends ===
// * Using atomicAdd(&work_counter, 1) instead of strided rows: slower because
//   atomic contention dominates at this row-count. Strided assignment via
//   blockIdx.x is sufficient and lock-free.
// * Sizing grid larger than CU_COUNT (e.g. 2*CU_COUNT for SMT-like hiding):
//   harmful — CDNA has no SMT, blocks just queue. Stay at CU_COUNT exactly.
//
// === Open directions ===
// * v3: replace block-LDS reduction with wave64 cross-lane reduction. Saves
//   6 of 8 __syncthreads() per row. Higher payoff at smaller hidden.
// * v4: multi-block split per row for h > 8K (atomic accumulate per-row sum).
//   Not relevant for our current sweep (h <= 7168).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HIDDEN_128 128
#define BLOCK_128 128
#define BLOCK_DYN 256
#define VEC 8

typedef union {
    uint4 u;
    __hip_bfloat16 bf[VEC];
} bf16x8_t;

// HIDDEN=128 fast path: unchanged from v1 / hip_stock — already winning by 2.5x.
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
    __shared__ float ssq[BLOCK_128];
    float val = static_cast<float>(x[row * HIDDEN_128 + tid]);
    ssq[tid] = val * val;
    __syncthreads();
    for (int s = BLOCK_128 / 2; s > 0; s >>= 1) {
        if (tid < s) ssq[tid] += ssq[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(ssq[0] / static_cast<float>(HIDDEN_128) + eps);
    float w_val = static_cast<float>(w[tid]);
    y[row * HIDDEN_128 + tid] = static_cast<__hip_bfloat16>(val * inv_rms * w_val);
}

// Persistent dynamic-hidden kernel. Grid = min(n_rows, CU_COUNT).
__global__ void rmsnorm_persistent_kernel(
    const __hip_bfloat16* __restrict__ x_,
    const __hip_bfloat16* __restrict__ w_,
          __hip_bfloat16* __restrict__ y_,
    int n_rows,
    int hidden,
    float eps)
{
    int tid = threadIdx.x;
    const int n_vec = hidden / VEC;
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_);

    __shared__ float ssq[BLOCK_DYN];

    // Persistent loop: this block handles rows blockIdx.x, blockIdx.x + gridDim.x, ...
    for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const uint4* x_row = reinterpret_cast<const uint4*>(x_ + row * hidden);
        uint4*       y_row = reinterpret_cast<uint4*>(y_ + row * hidden);

        // Pass 1: vectorized sum-of-squares.
        float my_ssq = 0.0f;
        for (int i = tid; i < n_vec; i += BLOCK_DYN) {
            bf16x8_t v; v.u = x_row[i];
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float f = static_cast<float>(v.bf[k]);
                my_ssq += f * f;
            }
        }
        ssq[tid] = my_ssq;
        __syncthreads();

        // Pass 2: block reduction.
        for (int s = BLOCK_DYN / 2; s > 0; s >>= 1) {
            if (tid < s) ssq[tid] += ssq[tid + s];
            __syncthreads();
        }
        float inv_rms = rsqrtf(ssq[0] / static_cast<float>(hidden) + eps);

        // Pass 3: vectorized scale + store.
        for (int i = tid; i < n_vec; i += BLOCK_DYN) {
            bf16x8_t v, wv, out;
            v.u  = x_row[i];
            wv.u = w_vec[i];
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float f = static_cast<float>(v.bf[k]);
                float g = static_cast<float>(wv.bf[k]);
                out.bf[k] = static_cast<__hip_bfloat16>(f * inv_rms * g);
            }
            y_row[i] = out.u;
        }
        __syncthreads();  // ensure all LDS reads done before next iteration overwrites
    }
}

// gfx950 has 256 CUs; for MI300X (gfx942) it's 304. We pick 256 as the default
// since this branch targets gfx950 first; on gfx942 it under-utilizes slightly
// but still correct (just 48 CUs unused for one round, then they pick up via
// the strided loop in subsequent rounds).
#define CU_COUNT_GFX950 256

extern "C" void rmsnorm_launch(
    const void* x, const void* w, void* y,
    int n_rows, int hidden, float eps, hipStream_t stream)
{
    if (hidden == HIDDEN_128) {
        rmsnorm_h128_kernel<<<dim3(n_rows), dim3(BLOCK_128), 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x),
            static_cast<const __hip_bfloat16*>(w),
            static_cast<__hip_bfloat16*>(y),
            n_rows, eps);
    } else {
        // Persistent: grid = min(n_rows, CU_COUNT). For small n_rows the kernel
        // degrades to the v1 behavior (each block does one row).
        int grid = n_rows < CU_COUNT_GFX950 ? n_rows : CU_COUNT_GFX950;
        rmsnorm_persistent_kernel<<<dim3(grid), dim3(BLOCK_DYN), 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x),
            static_cast<const __hip_bfloat16*>(w),
            static_cast<__hip_bfloat16*>(y),
            n_rows, hidden, eps);
    }
}
