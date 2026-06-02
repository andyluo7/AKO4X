---
name: flydsl
description: FlyDSL — AMD's preshuffle GEMM DSL (ROCm/FlyDSL), with fused-epilogue support (bias + ReLU/SiLU/GeLU) and gfx950-specific preload tuning tables. Use when the target operator decomposes to a large GEMM (BF16 / FP16 / FP8), when you need fused epilogue beyond hipBLASLt's stock support, or when you want to beat AITER's GEMM wrapper at small-N / unusual shapes. FlyDSL kernels run inside hipcc-compiled extensions and integrate cleanly with HIP and AITER pre/post-process kernels.
---

# FlyDSL

Reference for the FlyDSL preshuffle-GEMM DSL. Detailed guide: `flydsl.md`.

## When to consult

- Writing or tuning a preshuffle GEMM (`preshuffle_gemm.py`-style).
- Adding a fused epilogue (bias + ReLU / SiLU / GeLU) — the `body_row` epilogue slot lives at the GEMM output store loop, no separate epilogue kernel needed.
- Targeting gfx950 specifically — preload table is arch-specific (gfx942 and gfx950 have different LDS/VMEM latencies; using the wrong table can regress ~14% E2E).

## What it is

A Python-embedded DSL (similar to Triton/TileLang) that generates HIP/CK kernels specifically for preshuffle-GEMM shapes. Strengths: low launch overhead (persistent-kernel codegen), explicit preload scheduling, fusion of common epilogues into the store loop.

## Coupling with AITER

AITER's GEMM family (`aiter.ops.gemm`) wraps hipBLASLt for square shapes. FlyDSL has the advantage on:
- Small-N (n < 64) shapes where hipBLASLt under-uses MFMA bandwidth.
- Fused-epilogue cases where AITER would do a second pass (norm/activation).
- Long-K decomposable shapes (K > 8192) that benefit from split-K reduce.

## COUPLED references

- `hip` — for the surrounding runtime / launch / stream semantics.
- `aiter` — alternative GEMM path; choose per-shape via the autotune table.
- Per-operator FlyDSL baselines under `reference/<family>/baseline.json` (the `flydsl_tuned` row).
