---
name: aiter
description: AITER — AMD's optimized kernel library for Instinct GPUs. The closest analogue on AMD to FlashInfer-python on NVIDIA. Use when (a) studying a tuned reference implementation before writing your own kernel, (b) calling AITER ops as part of a hybrid solution (pre/post-process in HIP, MFMA loop via AITER's `aiter.ops.*`), or (c) measuring against the AITER baseline (which is one of the three "stock expert" rows ako4x-rocm scores against — alongside stock HIP and FlyDSL). Per-op tuned heuristics live in AITER's `op_tests/` reference impls.
---

# AITER

Reference for using and beating AITER. Detailed guide: `aiter.md`.

## When to consult

- Studying a tuned AMD reference impl before writing your own kernel (read `aiter/op_tests/<op>.py`).
- Calling AITER ops directly as part of a hybrid solution.
- Setting the baseline for a new operator — AITER is one of the three baselines ako4x-rocm beats.

## Operator coverage (as of 2026-05)

- **Attention**: MLA (decode/prefill), GQA paged, flash-attention v3 variants
- **MoE**: fused MoE (BF16, FP8 block-scale, FP8 per-tensor), top-k routing
- **GEMM**: BF16/FP16/FP8 GEMM with autotune, hipBLASLt wrappers
- **Norms**: RMSNorm, LayerNorm, group-norm
- **Activation+epilogue fused ops**: SiLU+mul, GeLU, etc.

## Key conventions

- All AITER ops live under `aiter.ops.<group>.<op>` (e.g., `aiter.ops.rmsnorm.rms_norm_fwd`).
- Most ops accept a `tune_config: TuneConfig` parameter — None means use the bundled best-known config for the input shape.
- AITER autotune tables ship as `.csv` files alongside the kernels; per-shape best configs are baked in for common shapes (MI300X, MI355X).
- Kernels are compiled lazily via `hipcc` at first call; expect a 2-10s first-launch JIT cost.

## COUPLED references

- `hip` — for the underlying HIP runtime semantics.
- `flydsl` — alternative DSL for the GEMM family.
- `ck` — Composable Kernel; AITER's GEMM family is built on top of CK.
- Per-operator AITER baselines under `reference/<family>/baseline.json` (the `aiter_tuned` row).
