// === Identity ===
// v1_vector_loads — RMSNorm with 128-bit (8×bf16) vector loads + writes.
// Targets the (n_rows >= 1024, hidden >= 4096) regime where naive hip_stock loses to AITER.
//
// === Delta from parent (hip_stock) ===
// 1) Loads/stores 8 bf16 elements per instruction via the AMD ext-vector type,
//    cutting memory transactions 8×.
// 2) Reduction unchanged (single LDS pass) — first focus on memory throughput.
// 3) HIDDEN=128 fast path kept verbatim from hip_stock (already winning).
//
// === Lessons (predicted) ===
// WHEN hidden is a multiple of 8 AND aligned to 16 bytes → vector loads always win.
// WHEN data isn't aligned → fall back to scalar. (Not a concern here — torch
// allocates 256-byte-aligned tensors.)
//
// === Dead-ends to avoid ===
// Trying to vectorize the LDS reduction with banks of 8 floats: triggers bank
// conflicts. Keep LDS reduction scalar; only vectorize global memory access.
//
// === Open directions ===
// Next variant: combine vector loads with wave64 cross-lane reduction (avoid the
// LDS reduction entirely for the per-wave partial sums).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HIDDEN_128 128
#define BLOCK_128 128
#define BLOCK_DYN 256
#define VEC 8

// 8-wide bf16 vector type for 128-bit loads/stores.
typedef __hip_bfloat16 bf16x8 __attribute__((ext_vector_type(8)));

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

__global__ void rmsnorm_dynamic_kernel(
    const __hip_bfloat16* __restrict__ x_,
    const __hip_bfloat16* __restrict__ w_,
          __hip_bfloat16* __restrict__ y_,
    int n_rows,
    int hidden,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n_rows) return;

    __shared__ float ssq[BLOCK_DYN];
    const bf16x8* x_row = reinterpret_cast<const bf16x8*>(x_ + row * hidden);
    const bf16x8* w_vec = reinterpret_cast<const bf16x8*>(w_);
    bf16x8*       y_row = reinterpret_cast<bf16x8*>(y_ + row * hidden);
    int n_vec = hidden / VEC;  // hidden must be multiple of 8

    // Pass 1: vectorized sum-of-squares.
    float my_ssq = 0.0f;
    for (int i = tid; i < n_vec; i += BLOCK_DYN) {
        bf16x8 v = x_row[i];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float f = static_cast<float>(v[k]);
            my_ssq += f * f;
        }
    }
    ssq[tid] = my_ssq;
    __syncthreads();

    // Pass 2: block reduction (scalar LDS, no bank conflicts).
    for (int s = BLOCK_DYN / 2; s > 0; s >>= 1) {
        if (tid < s) ssq[tid] += ssq[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(ssq[0] / static_cast<float>(hidden) + eps);

    // Pass 3: vectorized scale + store.
    for (int i = tid; i < n_vec; i += BLOCK_DYN) {
        bf16x8 v = x_row[i];
        bf16x8 wv = w_vec[i];
        bf16x8 out;
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float f = static_cast<float>(v[k]);
            float g = static_cast<float>(wv[k]);
            out[k] = static_cast<__hip_bfloat16>(f * inv_rms * g);
        }
        y_row[i] = out;
    }
}

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
        rmsnorm_dynamic_kernel<<<dim3(n_rows), dim3(BLOCK_DYN), 0, stream>>>(
            static_cast<const __hip_bfloat16*>(x),
            static_cast<const __hip_bfloat16*>(w),
            static_cast<__hip_bfloat16*>(y),
            n_rows, hidden, eps);
    }
}
