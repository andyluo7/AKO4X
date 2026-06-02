# rmsnorm-h128-rocm

Phase 1 operator archive for ako4x-rocm. RMSNorm with hidden dim 128, bfloat16.

## Why this operator

- Simple (no complex memory pattern) → fast iteration for the agent's first round.
- AKO4X-NV showed 1.14× over expert on the same operator → there's known headroom.
- Maps to real workloads: MLA per-head RMSNorm (h=128 in DSR1, Kimi-K2.5), GQA per-head norm.

## Baselines (the three rows to beat)

| Source | Where | Reference |
|---|---|---|
| `torch_naive` | `baselines/torch_naive.py` | torch eager mean²+rsqrt — correctness anchor, intentionally slow |
| `aiter_tuned` | `aiter.ops.norm.rms_norm_fwd` | AITER's tuned op — the "expert" row, hardest of the three to beat |
| `hip_stock` | `baselines/hip_stock.hip` | Naive hand-written HIP kernel — easy to beat, included to set a floor |

`flydsl_tuned` will be added in Phase 2 — FlyDSL access is gated to internal users
and the operator-decomposition path for RMSNorm isn't a natural fit for preshuffle GEMM.

## Running the benchmark

```bash
# Inside rocm/pytorch container with --device /dev/kfd --device /dev/dri:
python benchmark_rmsnorm_h128.py --n-rows 1,8,32,128,1024,8192 \
                                  --iters 200 --warmup 5 \
                                  --dtype bfloat16 \
                                  --out baseline.json
```

Writes a `baseline.json` with per-shape, per-impl latency in the AKO4X
`reference/<family>/` schema (so `spawn.py` + master can read it later without
translation).

## Phase 1 goal

Get an agent (Claude Code, Mode 1) to produce a kernel under `variants/<name>/`
that beats the **best of {torch_naive, aiter_tuned, hip_stock}** by ≥10% on at
least one workload shape, with all workloads passing correctness.

## Known easy wins (the agent should find these)

1. **LDS bank-conflict padding** on the reduction array — ~1.2× over `hip_stock`.
2. **Wave64 cross-lane reduction** via `__builtin_amdgcn_ds_bpermute` — saves the
   final LDS broadcast — ~1.1× over `hip_stock`.
3. **Persistent kernel** with grid sized to CU count (256 on gfx950, 304 on gfx942)
   — wins on small-batch shapes where launch overhead dominates — ~1.3-1.5×.
4. **Fused load+reduce+scale in one pass** without separate LDS reduction (use
   subgroup ops to broadcast inv_rms) — ~1.1-1.2×.

(All of these would still need to beat `aiter_tuned`, which is the real bar.)

## Operator family registration (Phase 2 hook)

Once the standalone benchmark works, this folder gets registered into the AKO4X
reference/ archive with the standard contract:
  - `definition.json` — schema (inputs / outputs / axes)
  - `workloads.jsonl` — concrete shape instances with uuids
  - `baseline.json` — the three baseline rows (already in the schema above)
  - `variants/<name>/` — agent-produced kernels with `parent.txt` lineage

The benchmark above is the Phase 1 stepping stone; the formal definition
+ workloads + flashinfer-bench dataset integration comes in Phase 2.
