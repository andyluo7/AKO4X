---
name: hip
description: HIP C++ (.hip / .cu-under-hipcc) kernel reference for AMD Instinct GPUs (CDNA3 gfx942 MI300X/MI325X, CDNA4 gfx950 MI350X/MI355X) — TVM-FFI direct-export vs Python-binding entry points, wave64 fixed-size, the LDS bank-conflict layout, MFMA tile selection per arch (gfx942 vs gfx950 instruction set delta), __builtin_amdgcn intrinsics, hipcc --offload-arch flags, and the HIP-graph-capture stream binding rules. Use when writing or debugging hand-written HIP C++ kernels, or whenever you need generic HIP-graph / sync-audit reasoning.
---

# HIP

Reference for HIP C++ kernel writers on AMD Instinct GPUs. Detailed guide: `hip.md`.

## When to consult

- Writing a HIP `.hip` (or `.cu`-under-hipcc) kernel — TVM-FFI direct export or Python binding via hipcc + JIT.
- Resolving a `RUNTIME_ERROR` traceable to chevron-launch stream binding (legacy null-stream IS NOT capture-aware under `hipGraphLaunch` — see `hip.md` "Kernel-launch stream is NOT optional under HIP graph capture").
- Reducing per-call overhead: GPU↔CPU sync audit, HIP-graph capture, persistent-kernel patterns — generic theory + the Waves-Per-CU decision table live here; per-DSL bindings live in the DSL skills (triton, tilelang, ck).
- Choosing MFMA instruction shape — gfx942 vs gfx950 differ on a small but important set (gfx950 has new `mfma_f32_*_xf32` variants; some gfx942 instructions are absent or differently-priced on gfx950).

## Architectures targeted

- **gfx942** — MI300X, MI325X (CDNA3). Wave64. Reference clock 2100 MHz boost. 304 CUs (MI300X) / 304 CUs (MI325X with larger HBM3e). LDS 64 KiB/CU.
- **gfx950** — MI350X, MI355X (CDNA4). Wave64. 256 CUs. Same 64 KiB LDS/CU. New MFMA variants for xf32 and some FP6 / FP4 layouts.

Use `hipcc --offload-arch=gfx942,gfx950` to emit a fat binary that runs on both; use `--offload-arch=gfx950` only when targeting MI355X exclusively (slightly smaller binary, lets the compiler use gfx950-only instructions).

## COUPLED references

None directly to runtime. Per-operator HIP wins / traps live under `docs/prior/` (when your operator has a prior archive). Sister SKILLs:
- `aiter` — the AMD-tuned reference kernel library; pre-tuned heuristics worth studying before writing from scratch.
- `flydsl` — the AMD preshuffle-GEMM DSL with fused epilogue; relevant when the operator decomposes to large GEMMs.
- `ck` — Composable Kernel templates (cute-dsl equivalent), useful for MFMA tile-level abstraction.
