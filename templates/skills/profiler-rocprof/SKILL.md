---
name: profiler-rocprof
description: Run AMD rocprofv3 per-kernel profiling for VGPR pressure, occupancy, stall-reason, memory throughput, and IPC analysis on Instinct GPUs. The ROCm analogue of NVIDIA Nsight Compute (NCU). Use BEFORE architecting an optimization fix when a hypothesis about microarchitectural behavior needs verification. rocprof `KernelTime` is NOT comparable to bench timing; use rocprof for ratios only.
---

# Profiler (rocprofv3)

Wrapper around `rocprofv3`. Command entry: `bash scripts/profile.sh`.

Detailed reference: `rocprof.md`. Top-level commands:

```bash
bash scripts/profile.sh --list                                    # list available counter sets
bash scripts/profile.sh --index 5                                 # default --set basic
bash scripts/profile.sh --index 5 --set memory                    # memory-heavy counters
bash scripts/profile.sh --index 5 --kernel-name ".*rmsnorm.*"     # filter kernels
bash scripts/profile.sh --index 5 --counters VALU_BUSY,LDS_BANK_CONFLICT
bash scripts/profile.sh --index 5 --env NO_GRAPH=1                # disable HIP-graph capture
```

## Key constraints

- **rocprof under HIP graph capture is unreliable** — graph-launched kernels go through `hipGraphLaunch` and the kernel filter often returns "No kernels were profiled". If your kernel uses `torch.cuda.CUDAGraph` (which maps to `hipGraph_t` on ROCm), install a module-level `_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))` gate AND set `NO_GRAPH=1` — same pattern as the NV branch's `profiler-ncu` skill. See `rocprof.md` "HIP graph capture interaction".
- **Don't profile the reference** — the unoptimized Python implementation launches dozens of small kernels; profiling produces unhelpful noise. Make at least one optimization pass first.
- **rocprof's overhead is higher than NCU's** — counter collection on AMD is heavier than on NVIDIA; expect ~2-5× slowdown when running profiling. Use it sparingly, not in the inner bench loop.

## Useful counter sets for kernel optimization

- `VALU_BUSY` — fraction of time the vector ALU was doing work (occupancy proxy)
- `LDS_BANK_CONFLICT` — LDS conflict count per kernel
- `MFMA_OCC` — MFMA unit occupancy
- `TCC_HIT_PCT` — L2 (TCC) cache hit rate
- `MEM_UNIT_BUSY` — DRAM bandwidth utilization
- `WAVES_PER_CU` — actual achieved wave occupancy (not theoretical)
- `VGPR_SPILL` — register spills (non-zero = something to fix)
- `SQ_INSTS_VALU` — VALU instruction count

Get the full list: `rocprofv3 --counter-list` or `bash scripts/profile.sh --list-counters`.

## COUPLED references

- Local backend: `scripts/run_local_profile.py`
- Shared runtime: `scripts/bench_utils.py` (workload loading, dataset resolution)
- For graph-capture interaction patterns: `hip/hip.md` "HIP-graph capture pitfalls"

Per-operator rocprof traps (e.g. LDS-staging regressions, MFMA-unit pricing surprises on gfx950 vs gfx942) live in `docs/prior/TRAPS.md` (when your operator has a prior archive).
