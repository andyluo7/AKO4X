---
name: ck
description: Composable Kernel (CK) — AMD's CUTLASS-equivalent template library. Header-only C++ templates for MFMA tile-level abstraction over gfx942 / gfx950. Use when you want CUTLASS-style tile decomposition without dropping all the way to raw HIP intrinsics. AITER's GEMM and some attention paths are built on top of CK.
---

# CK (Composable Kernel)

Reference for using AMD's Composable Kernel template library. Detailed guide: `ck.md` (TODO — Phase 1).

## When to consult

- Writing a kernel that wants CUTLASS-style tile decomposition + MFMA tiles but not the full AITER framework.
- Modifying an AITER kernel — its inner loops typically dispatch to CK templates.
- Studying how AMD's reference impls compose MFMA blocks.

## Repo

`github.com/ROCm/composable_kernel` (open source, header-only). Already a dependency of AITER.

## Phase 0 status

Stub-level documentation only. Expand in Phase 1 of the ako4x-rocm port when an actual CK-based agent kernel is first attempted.

## COUPLED references

- `hip` — underlying runtime / launch semantics.
- `aiter` — primary consumer of CK in the AMD stack.
