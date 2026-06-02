// === Identity ===
// v1_vector_loads — RMSNorm with 128-bit (16-byte = 8×bf16) vector loads/writes.
// Targets the (n_rows >= 1024, hidden >= 4096) regime where naive hip_stock loses to AITER.
//
// === Delta from parent (hip_stock) ===
// 1) Loads/stores 8 bf16 elements per memory instruction via uint4 + union
//    reinterpret (AMD-idiomatic — clang's ext_vector_type rejects bf16 as elem).
// 2) Reduction unchanged (single LDS pass) — focus on memory throughput first.
// 3) HIDDEN=128 fast path kept verbatim from hip_stock.
//
// === Lessons ===
// WHEN hidden % 8 == 0 AND base ptr is 16-byte aligned → uint4 loads always win.
//   torch tensors are 256-byte aligned, so the alignment side is free.
//   For h=4096 and h=7168 both divide by 8 cleanly.
// WHEN hidden % 8 != 0 → must guard the tail with scalar loop (not needed here).
//
// === Dead-ends ===
// * clang's __attribute__((ext_vector_type(8))) — bf16 isn't a valid elem on AMD HIP.
//   Use union-based uint4 reinterpret instead.
// * Trying to vectorize LDS reduction with 8-wide banks — triggers bank conflicts.
//   Keep LDS reduction scalar; only vectorize global memory access.
//
// === Open directions ===
// Next variant: combine vector loads with wave64 cross-lane reduction (avoid the
// LDS reduction entirely for per-wave partial sums).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HIDDEN_128 128
#define BLOCK_128 128
#define BLOCK_DYN 256
#define VEC 8

// 128-bit packed bf16 type — uint4 is HIP-native 16-byte vector; union gives
// per-element access without losing the wide-load codegen.
typedef union {
    uint4 u;
    __hip_bfloat16 bf[VEC];
} bf16x8_t;

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
    const uint4* x_row = reinterpret_cast<const uint4*>(x_ + row * hidden);
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_);
    uint4*       y_row = reinterpret_cast<uint4*>(y_ + row * hidden);
    int n_vec = hidden / VEC;  // caller ensures hidden % 8 == 0 (true for 128, 4096, 7168)

    // Pass 1: per-thread sum-of-squares via 128-bit loads.
    float my_ssq = 0.0f;
    for (int i = tid; i < n_vec; i += BLOCK_DYN) {
        bf16x8_t v;
        v.u = x_row[i];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float f = static_cast<float>(v.bf[k]);
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
