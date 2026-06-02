# rocprofv3 — Detailed Reference

Companion to `SKILL.md`. AMD's per-kernel profiler — the analogue of NVIDIA's Nsight Compute (NCU).

## Invocation

`rocprofv3` is the v3 CLI (replaces the older `rocprof` and `rocprofv2`). Most common forms:

```bash
# Aggregated counter run (one run = one counter set)
rocprofv3 --hip-trace --kernel-trace --output-format csv -- python my_app.py

# Targeted single-kernel deep dive
rocprofv3 --kernel-name "rmsnorm_kernel" \
          --counters VALU_BUSY,LDS_BANK_CONFLICT,WAVES_PER_CU \
          --output-format csv -- python my_app.py
```

In ako4x-rocm, both wrap inside `scripts/run_local_profile.py` and surface as `bash scripts/profile.sh`.

## Counter sets

There's no equivalent to NCU's `--set detailed` shorthand. Use named groups:

- `--counters VALU_BUSY,SALU_BUSY,MFMA_OCC,LDS_BANK_CONFLICT` → "basic"
- `--counters TCC_HIT_PCT,L2_HIT,MEM_UNIT_BUSY` → "memory"
- `--counters WAVES_PER_CU,VGPR_SPILL,SGPR_SPILL` → "occupancy"

Use `rocprofv3 --counter-list` to enumerate all available counters on your device. Note: not all counters work on all archs (some are gfx942-only or gfx950-only).

## HIP graph capture interaction

`hipGraphLaunch` replays kernels through a path that defeats `--kernel-name` filtering — you'll see "No kernels were profiled". Same trap as NCU under CUDA-graph.

**Pattern**:
```python
# In your kernel module
_NO_GRAPH = bool(os.environ.get("NO_GRAPH"))

def run(...):
    if _NO_GRAPH:
        # Direct launch path
        kernel<<<grid, block, 0, stream>>>(...)
    else:
        # Graph-replay fast path
        graph_exec.launch(stream)
```

Then profile with `NO_GRAPH=1`.

## Overhead

rocprofv3's per-kernel counter collection is heavier than NCU's:
- Single-counter runs: ~20-50% slowdown
- Full counter set (~30 counters): 3-5× slowdown
- With `--hip-trace --kernel-trace`: even more

**Don't put rocprof in the inner bench loop.** Use it for diagnostic runs, then go back to clean bench timing.

## Output formats

- `--output-format csv` — easy to grep, columnar.
- `--output-format json` — structured, programmatically parseable.
- `--output-format pftrace` — Chrome trace format, view in `chrome://tracing` or Perfetto. Useful for kernel-launch timeline analysis.

## omniperf — higher-level alternative

If counter-by-counter is too low-level, `omniperf` provides a dashboard view on top of rocprofv3 data:

```bash
omniperf profile --name my_run -- python my_app.py
omniperf analyze --workload workloads/my_run
```

Useful for getting a holistic per-CU view (memory pressure, occupancy, MFMA utilization) without deciding counter sets manually.

## Common diagnoses

| Symptom | Likely counter to check | Fix |
|---|---|---|
| Slow kernel, no obvious memory pressure | `VALU_BUSY` < 0.5 | Increase block size, more waves/CU |
| LDS-heavy kernel slower than expected | `LDS_BANK_CONFLICT` > 0 | Pad LDS arrays (see `hip/hip.md` "LDS bank conflicts") |
| Low MFMA throughput | `MFMA_OCC` < 0.6 | Increase block size to feed MFMA pipeline; check for stalls between MFMA waves |
| Register-spill warnings | `VGPR_SPILL` > 0 | Reduce VGPR pressure: smaller tile, more shared LDS staging |
| Memory-bound but DRAM not saturated | `MEM_UNIT_BUSY` < 0.6, `TCC_HIT_PCT` low | Coalesce access pattern; use `__builtin_amdgcn_global_load_lds_*` for direct DRAM→LDS |

## See also

- `scripts/run_local_profile.py` — local backend wrapper.
- `hip/hip.md` — generic HIP perf reasoning.
