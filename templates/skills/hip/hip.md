# HIP — Detailed Reference

Companion to `SKILL.md`. Detailed reference for writing HIP C++ kernels on AMD Instinct GPUs (CDNA3 gfx942 = MI300X/MI325X, CDNA4 gfx950 = MI350X/MI355X).

## Entry points: TVM-FFI direct-export vs Python-binding

Two ways to expose a HIP kernel to the bench runtime:

### TVM-FFI direct export (preferred for new kernels)

```cpp
// kernel.hip
#include <hip/hip_runtime.h>
#include <tvm/ffi/function.h>

extern "C" void rmsnorm(DLTensor* x, DLTensor* w, DLTensor* y, float eps) {
    auto* x_ptr = static_cast<__hip_bfloat16*>(x->data);
    auto* w_ptr = static_cast<__hip_bfloat16*>(w->data);
    auto* y_ptr = static_cast<__hip_bfloat16*>(y->data);
    int n_rows = x->shape[0];
    int hidden = x->shape[1];
    hipStream_t s = reinterpret_cast<hipStream_t>(x->byte_offset);  // see "Stream binding" below
    rmsnorm_kernel<<<n_rows, 256, 0, s>>>(x_ptr, w_ptr, y_ptr, hidden, eps);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, rmsnorm);
```

Wire via `solution.json`'s `entry_point: "kernel.hip::kernel"`. The bench runtime calls the exported function directly, no Python binding overhead per call.

### Python binding (torch C++ extension)

```python
# binding.py
import torch
from torch.utils.cpp_extension import load_inline
mod = load_inline(name="myker", cpp_sources=["..."], cuda_sources=["..."])
# ^ load_inline routes through hipcc on ROCm hosts (PyTorch+ROCm wheel does this).
def kernel(x, w, eps):
    y = torch.empty_like(x)
    mod.rmsnorm(x, w, y, eps)
    return y
```

Use when the kernel needs Python-side tensor allocation or shape inference. Pays one extra Python call per launch.

## Stream binding is NOT optional under HIP graph capture

Same trap as CUDA, same shape. The HIP "null stream" (stream 0, passed as the 4th chevron arg) is **not** capture-aware: under `hipStreamBeginCapture`, kernels launched on stream 0 are silently dropped from the captured graph. Symptom is `hipGraphLaunch` returns OK but produces garbage / zero output.

**Fix**: always pass an explicit stream — either the one from the DLTensor metadata (TVM-FFI path above) or `torch.cuda.current_stream()._cdata` (Python path). Never rely on stream 0 inside a kernel that may be graph-captured.

Validation: run the kernel once with `HIP_LAUNCH_BLOCKING=1` + a hipDeviceSynchronize after each launch — if outputs differ from the graph-capture path, you have the null-stream bug.

## Wave size — fixed at 64 on CDNA

Unlike RDNA (which has Wave32/Wave64), all CDNA GPUs (gfx9xx) run with **fixed Wave64**. Block sizes you choose must be multiples of 64. Common pitfall: porting a CUDA kernel with `blockDim.x = 32` — on AMD this wastes 50% of every wavefront.

Recommended block sizes:
- Element-wise / reductions: 256 (4 waves), 512 (8 waves)
- Attention decode: 64 (1 wave per query head)
- Attention prefill: 256 (matches MFMA tile latency)
- MFMA-dominated GEMMs: 256 with 2-wave-per-block scheduling

## MFMA instructions — gfx942 vs gfx950 deltas

Both archs support the core `mfma_*_*x*x*_*` family for FP16, BF16, FP8 (E4M3/E5M2), INT8.

**gfx942 only** (not on gfx950): `mfma_f32_16x16x8_xf32` (some older xf32 variants).
**gfx950 only** (not on gfx942): new `mfma_f32_*_xf32` lowering, expanded FP6/FP4 packed variants, and microarchitectural changes to LDS read paths.

Pragmatic guide:
- For **portable** code targeting both archs, use the BF16/FP16/FP8 shared variants — these have identical encodings and similar timings on both.
- For **gfx950-only** kernels, the new MFMA encodings can give ~10-15% headroom on FP8 + FP6 paths; the agent should explore these in the per-arch tuning phase.
- For **gfx942-only**, fall back to the older instructions where they're faster.

The compiler auto-selects via `--offload-arch`, but in inline `__builtin_amdgcn_mfma_*` intrinsics you have to choose the variant manually — guard with `#if __gfx950__` / `#if __gfx942__`.

## LDS bank conflicts

LDS is 32 banks × 4 bytes wide on both archs (64 KiB total per CU). Same conflict rules as NV's shared memory but more impactful because:
- Wave64 means 64 threads collide on every load, twice the contention surface of NV's 32-lane warps.
- gfx950's LDS read path is slightly redesigned — some patterns that conflict on gfx942 are conflict-free on gfx950 (and vice versa).

**Pattern**: pad LDS arrays to avoid the common stride-conflict (`array[tid] = ...; array[tid + 64] = ...;` with `array` declared as `__shared__ T array[BLOCK + 1]` not `[BLOCK]`).

**Diagnose**: `rocprofv3 --counters LDS_BANK_CONFLICT` (the rocprof equivalent of NCU's `l1tex_data_bank_conflicts` counter).

## __builtin_amdgcn intrinsics — common ones

- `__builtin_amdgcn_ds_bpermute(addr, val)` — cross-lane permute (the AMD `__shfl_sync` equivalent).
- `__builtin_amdgcn_readlane(val, lane)` — broadcast from a specific lane.
- `__builtin_amdgcn_s_barrier()` — finer-grained than `__syncthreads()`; only synchronizes scalar unit, useful in some load/compute overlap patterns.
- `__builtin_amdgcn_global_load_lds_*` — direct DRAM → LDS path on gfx9x (CDNA), skips registers. **High impact**: bypasses VGPR pressure for large LDS-staged tiles. Use for attention prefill where K/V tiles dominate VGPR usage.
- `__builtin_amdgcn_mfma_*` — the matrix instructions (see MFMA section above).

## hipcc flags worth knowing

- `--offload-arch=gfx950` (or `gfx942`) — required; default is host arch.
- `-O3 -ffast-math` — usual perf flags.
- `-fgpu-rdc` — relocatable device code, needed for cross-TU device function calls.
- `--save-temps` — dumps `.s` (gfx ISA) and `.bc` (LLVM IR). Use to inspect MFMA selection.
- `-Rpass=inline` / `-Rpass=loop-vectorize` — see what the compiler did.
- `-mcumode` — CU mode (vs WGP mode); CDNA is always CU mode but the flag is sometimes needed to suppress warnings on tooling that defaults to RDNA.

## HIP-graph capture pitfalls

Same shape as CUDA: avoid stream 0, avoid `hipDeviceSynchronize` inside the captured region, ensure all allocations happen before capture begins.

Additional AMD-specific:
- `hipMallocAsync` is supported on ROCm 7+ but allocations made inside a captured region don't always behave the same as CUDA — test with `--print-graph` to verify the captured graph's node count matches expectations.
- AMD-specific `hipCooperativeLaunch` is the cooperative-groups equivalent; useful for grid-wide barriers in persistent-kernel patterns.

## Persistent kernel patterns

Decode-style attention with small per-batch work benefits from persistent kernels (one block per CU, looping over work items) — bypasses the ~10 µs launch overhead of HIP. On CDNA the overhead is higher than CUDA's, so persistent patterns pay off sooner.

Skeleton:
```cpp
__global__ void persistent_attn(...) {
    int cu_id = blockIdx.x;  // grid sized to 256 (= CU count on gfx950)
    int work_id;
    while ((work_id = atomicAdd(&work_counter, 1)) < total_work) {
        // process work item
    }
}
// host: launch with <<<256, 256>>>, no for-loop on host side.
```

Coordinate with `aiter` SKILL — AITER's MLA kernel uses this pattern.

## Decision table — Waves-Per-CU

CDNA has 4 SIMD units per CU, each with 10 wavefront slots → 40 waves/CU max. Achieving full occupancy on a 256-thread block (4 waves) means 10 blocks/CU — usually impossible due to VGPR or LDS pressure.

| Per-block VGPR | Per-block LDS | Achievable waves/CU | Use case |
|---|---|---|---|
| ≤ 64 | ≤ 8 KiB | 32-40 (high) | Element-wise, RMSNorm |
| 64-128 | 8-16 KiB | 16-24 (medium) | Attention decode |
| 128-256 | 16-32 KiB | 8-12 (low) | Attention prefill, GEMMs |
| > 256 | > 32 KiB | 4-8 (very low) | Heavy MFMA-saturated workloads (usually fine — MFMA hides latency) |

Check actual occupancy with `rocprofv3 --counters CU_OCCUPANCY`.

## See also

- `aiter/aiter.md` — pre-tuned kernel reference and tuning heuristics.
- `flydsl/flydsl.md` — preshuffle-GEMM DSL details.
- `ck/ck.md` — Composable Kernel template library.
- `profiler-rocprof/rocprof.md` — `rocprofv3` invocation + counter reference.
