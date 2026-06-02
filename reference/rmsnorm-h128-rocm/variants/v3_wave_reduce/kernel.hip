// === Identity ===
// v3_wave_reduce — v1's vector loads + wave64 cross-lane reduction.
// Targets the (8192, h>=4096) regime where v1 still loses to AITER by ~1.18x.
// Eliminates 6 of 8 __syncthreads() per row by reducing within-wave first via
// __shfl_xor, then cross-wave with a tiny 4-element LDS exchange.
//
// === Delta from parent (v1_vector_loads) ===
// 1) Per-thread sum-of-squares loop unchanged (vector loads).
// 2) Reduction REWRITTEN:
//    Step A (intra-wave): 6 cycles of __shfl_xor(sum, 32/16/8/4/2/1) — no sync,
//             every lane in the wave ends up holding the wave's total.
//    Step B (cross-wave): wave 0's lane 0 reads partials from each wave, does
//             one more 2-cycle xor reduction. One __syncthreads() before
//             cross-wave read, one __syncthreads() before broadcast.
//    Total syncs per row: 2 (was 8).
// 3) HIDDEN=128 fast path unchanged (already winning).
// 4) Grid is NOT persistent — kept at one block per row, since v2 proved
//    persistent kernels are a dead-end at large n_rows on CDNA.
//
// === Lessons (predicted) ===
// WHEN n_rows large AND per-row reduction depth >= 4 syncs (BLOCK >= 128)
//   → wave reduction wins because sync count drops 4x.
// WHEN BLOCK <= 64 (single wave) → wave reduction wins by even more (0 syncs).
//
// === Dead-ends (from prior variants) ===
// * v2_persistent: persistent kernel hurts at n_rows >> CU_COUNT on CDNA.
//   The wave scheduler already hides launch overhead of many short blocks;
//   long-lived blocks degrade memory locality. Avoid pattern at this scale.
// * In v1: clang's ext_vector_type doesn't accept bf16. Use uint4+union.
//
// === Open directions ===
// * v4: combine wave reduction with __builtin_amdgcn_global_load_lds (direct
//   DRAM->LDS path, skips VGPR staging).
// * v5: register-level scratch (no LDS at all) using DPP cross-lane permute.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define HIDDEN_128 128
#define BLOCK_128 128
#define BLOCK_DYN 256
#define WAVE 64
#define N_WAVES (BLOCK_DYN / WAVE)  // 4 waves per block
#define VEC 8

typedef union {
    uint4 u;
    __hip_bfloat16 bf[VEC];
} bf16x8_t;

// HIDDEN=128 fast path unchanged.
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

// Wave64 cross-lane reduction: 6 __shfl_xor calls, no LDS, no sync.
__device__ __forceinline__ float wave_reduce_sum(float v) {
    // HIP exposes __shfl_xor across wave64 on CDNA.
    v += __shfl_xor(v, 32);
    v += __shfl_xor(v, 16);
    v += __shfl_xor(v, 8);
    v += __shfl_xor(v, 4);
    v += __shfl_xor(v, 2);
    v += __shfl_xor(v, 1);
    return v;
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
    int lane_id = tid & (WAVE - 1);
    int wave_id = tid >> 6;  // 6 = log2(64)
    if (row >= n_rows) return;

    const uint4* x_row = reinterpret_cast<const uint4*>(x_ + row * hidden);
    const uint4* w_vec = reinterpret_cast<const uint4*>(w_);
    uint4*       y_row = reinterpret_cast<uint4*>(y_ + row * hidden);
    const int n_vec = hidden / VEC;

    // Pass 1: vectorized sum-of-squares (unchanged from v1).
    float my_ssq = 0.0f;
    for (int i = tid; i < n_vec; i += BLOCK_DYN) {
        bf16x8_t v; v.u = x_row[i];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            float f = static_cast<float>(v.bf[k]);
            my_ssq += f * f;
        }
    }

    // Pass 2a: intra-wave reduction (no sync, no LDS).
    float wave_sum = wave_reduce_sum(my_ssq);

    // Pass 2b: cross-wave via 4-elem LDS. Only wave_id 0 does the final reduce.
    __shared__ float wave_sums[N_WAVES];
    if (lane_id == 0) wave_sums[wave_id] = wave_sum;
    __syncthreads();

    // Broadcast inv_rms via LDS slot 0.
    __shared__ float inv_rms_shared;
    if (wave_id == 0) {
        float v = (lane_id < N_WAVES) ? wave_sums[lane_id] : 0.0f;
        v += __shfl_xor(v, 2);
        v += __shfl_xor(v, 1);
        if (lane_id == 0) {
            inv_rms_shared = rsqrtf(v / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();
    float inv_rms = inv_rms_shared;

    // Pass 3: vectorized scale + store (unchanged from v1).
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
