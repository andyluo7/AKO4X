# AKO4X → ROCm Porting Plan

**Targets:** gfx950 (MI350X / MI355X) — high priority. gfx942 (MI300X / MI325X) — secondary.
**Goal:** drive Claude Code (or another agent) to produce HIP / AITER / FlyDSL kernels that beat their stock implementations on AMD Instinct hardware.

## What "stock" means concretely

For each operator, the agent's kernels are scored against three reference baselines (the strongest of the three is the floor to beat):

| Baseline row | What it is | Where it lives |
|---|---|---|
| `hip_stock` | Hand-written naive HIP impl | `reference/<family>/baselines/hip_stock.hip` |
| `aiter_tuned` | AITER's tuned op for the same shape | call via `aiter.ops.<group>.<op>` |
| `flydsl_tuned` | FlyDSL preshuffle-GEMM where applicable | `reference/<family>/baselines/flydsl_tuned.py` |

## Operator priority list

| Priority | Operator | Why |
|---|---|---|
| **P0** | RMSNorm (h=128, h=7168) | Simple, fast to iterate; NV branch showed 1.14× over expert |
| **P0** | GQA paged decode (h=32, kv=8, d=128) | Decode is launch-bound on ROCm; persistent-kernel headroom |
| **P0** | MLA paged decode (h=16, ckv=512, kpe=64) | DSR1 attention; AITER MLA is the AMD reference |
| **P1** | MoE FP8 block-scale | Kimi/DSR1 routing kernel; AITER MoE FP8 has known headroom on unusual shapes |
| **P1** | GDN decode (Gated Delta Net) | Qwen3-Next / GLM-5; uneven AMD coverage |
| **P2** | GEMM (n=2048, k=4096) | Expected honest negative — hipBLASLt is hard to beat (matches NV's "GEMM ties cuBLAS") |

## Phased roadmap

```
Phase 0 — Plumbing (3-5 days)                      ← THIS COMMIT
  ├─ Fork TongmingLAIC/AKO4X → andyluo7/AKO4X branch rocm-port
  ├─ pyproject.toml: ROCm deps (aiter, rocm-smi-lib, ROCm Triton)
  ├─ spawn.py: rocm-smi probe, .hip extension support
  ├─ Replace cuda/cute-dsl/profiler-ncu SKILLs with hip/aiter/flydsl/ck/profiler-rocprof
  ├─ benchmark_adapter.py: stub interface preserved, ROCm-PORT comments mark TODOs
  └─ templates/benchmark/evaluation.toml: ROCm tolerances + arch hint

Phase 1 — Single-operator MVP on gfx950 (1 week)
  ├─ Implement HIP builder in benchmark_adapter._build_runnable
  ├─ Implement rocprofv3 wrapper in scripts/run_local_profile.py
  ├─ RMSNorm operator: workload set + 3 baselines + frozen baseline.json
  ├─ Run Manual mode (Mode 1): single spawn + Claude Code session
  └─ MILESTONE: agent produces a kernel that beats the best of {hip, aiter, flydsl} by ≥10% on ≥1 workload

Phase 2 — Closed-loop on gfx950, 3 operators (2 weeks)
  ├─ Add GQA-paged-decode, MLA-paged-decode
  ├─ Implement master/ scaffolding for ROCm
  ├─ Run Mode 2 campaigns (5-10 rounds per operator)
  └─ MILESTONE: geomean ≥1.2× over best-of-three baselines on ≥2 of 3 operators

Phase 3 — Breadth + Mode 3 (2 weeks)
  ├─ Add MoE FP8, GDN decode, GEMM
  ├─ Run Mode 3 (sub agent rewrites its own SKILLs each round)
  └─ MILESTONE: ≥5 operators with ≥1.2× geomean; honest negatives documented

Phase 4 — gfx942 cross-arch (1 week)
  ├─ Re-baseline on MI300X + MI325X
  └─ MILESTONE: per-arch best-variant table; arch-specific tuning lessons documented

Phase 5 — Publish (1 week)
  ├─ Results writeup (modeled on AKO tech report)
  └─ Two PR options: (a) PR to TongmingLAIC/AKO4X adding ROCm SKILLs, or (b) standalone fork with link back to upstream
```

Total calendar estimate: **6-7 weeks** for the full deliverable; **2 weeks** to a credible first "AKO4X-ROCm beats AITER on RMSNorm" demonstration.

## Hardware plan

| Tier | Hardware | Where | Used for |
|---|---|---|---|
| Primary | gfx950 MI355X | Tensorwave `mia1-p01-g36` (amd-tw partition), AAC1 `smci355-ccs-aus-g12-*` | All P0 work |
| Secondary | gfx942 MI300X | Hotaisle `23.183.40.49`, Conductor `smci355-ccs-aus-m15-17` | Phase 4 cross-arch |
| Secondary | gfx942 MI325X | (confirm AAC1 sinfo) | Phase 4 |
| Local driver | macOS host | n/a | SSH orchestration (same as P1b model) |

## Files changed in this commit (Phase 0)

- `pyproject.toml` — ROCm deps, drops NVIDIA-only ones
- `spawn.py` — `resolve_gpu()` probes rocm-smi first; `infer_language()` knows `.hip`
- `templates/skills/cuda/` → deleted
- `templates/skills/cute-dsl/` → deleted
- `templates/skills/profiler-ncu/` → deleted
- `templates/skills/hip/{SKILL.md, hip.md}` — new, full detail level
- `templates/skills/aiter/{SKILL.md, aiter.md}` — new, full detail level
- `templates/skills/flydsl/{SKILL.md, flydsl.md}` — new, full detail level
- `templates/skills/profiler-rocprof/{SKILL.md, rocprof.md}` — new (replaces profiler-ncu)
- `templates/skills/ck/SKILL.md` — new, stub-level (expand in Phase 1)
- `templates/benchmark/evaluation.toml` — ROCm tolerances + `target_gpu_arch` field
- `scripts/benchmark_adapter.py` — rewritten with ROCm-PORT TODOs at each NV-specific call site
- `README.md` — fork banner above upstream content
- `docs/rocm-porting-plan.md` — this file

## What still doesn't work (Phase 0 deliberately leaves)

- `benchmark_adapter.profile()` raises `NotImplementedError` (Phase 1 wires rocprofv3)
- `benchmark_adapter.sanitize()` raises `NotImplementedError` (Phase 1 wires HIP_LAUNCH_BLOCKING + ASan)
- No SLURM / Tensorwave runners yet (Phase 1)
- No actual HIP / AITER / FlyDSL builder dispatch (Phase 1)
- `master/` Mode 2/3 scaffolding untouched — still NV-tuned prompt; will adapt in Phase 2

## How to test what IS working in Phase 0

```bash
# In a Linux + ROCm 7+ environment:
git clone https://github.com/andyluo7/AKO4X.git
cd AKO4X && git checkout rocm-port
pip install -e .

# Smoke test the GPU detection:
python -c "from spawn import resolve_gpu; print(resolve_gpu(None, 'local'))"
# → ('mi355x', 'MI355X') on a gfx950 host

# Inspect the SKILL catalog:
ls templates/skills/
# → aiter bench benchmark ck cpp flydsl hip profiler-rocprof sanitizer tilelang triton
```

`spawn.py` still works end-to-end against the upstream FIB engine, but kernel
execution against the bundled CUDA-only solutions will fail at build time.
That's expected for Phase 0 — Phase 1 lands the ROCm builder.

## Upstream credit

All of the architecture, the closed-loop design, the SKILL catalog shape, the master/sub
split, the `benchmark_adapter.py` plain-data seam, the reference-archive contract,
the lessons convention — all of it is from [`TongmingLAIC/AKO4X`](https://github.com/TongmingLAIC/AKO4X)
(authors: Shuxiao Xie, Shuyang Xie). This fork extends them to ROCm.
