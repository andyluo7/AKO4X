# AITER — Detailed Reference

Companion to `SKILL.md`. AITER is AMD's optimized kernel library for Instinct GPUs — the closest analogue on AMD to FlashInfer-python on NVIDIA.

Repo: `github.com/ROCm/aiter` (open source). Install: included in `ako4x-rocm`'s `pyproject.toml`.

## Operator coverage (as of 2026-05)

| Family | Module path | Notes |
|---|---|---|
| **MLA (DeepSeek-R1/V3)** | `aiter.ops.mla` | Decode + prefill; gfx950-tuned for h=16, ckv=512, kpe=64. Persistent-kernel pattern. |
| **GQA paged attention** | `aiter.ops.attention` | Compatible with vLLM block-table format. h=32 kv=8 d=128 is most-tuned shape. |
| **MoE (BF16 / FP8 block-scale)** | `aiter.ops.moe` | Fused gate + expert. FP8 block-scale variant uses 128×128 scale tiles. |
| **GEMM** | `aiter.ops.gemm` | Wraps hipBLASLt internally; rarely worth beating for square shapes; the small-N regime (n < 64) has more headroom. |
| **RMSNorm / LayerNorm** | `aiter.ops.norm` | Best-tuned for h=128 (head-dim) and h=7168 (DSR1 hidden). Other sizes can be 20-30% slower than achievable. |
| **Activations** | `aiter.ops.activation` | SiLU, GeLU, fused SiLU+mul. |

## Reading AITER kernels

The most useful study material is `aiter/op_tests/<op>.py` — each contains:
1. A naive PyTorch reference
2. The AITER op call
3. A correctness check
4. A timing loop with config sweep

This is exactly the "expert reference" you want to beat. Read the autotune table (`.csv` next to the op) to see what tile / block sizes AITER picked for your shape.

## Hybrid usage (AITER kernel as a building block)

A common winning pattern: use AITER's MFMA inner loop for the heavy compute, then do pre/post-processing in your own HIP. Example:

```python
# kernel.py
import aiter.ops.mla as mla
def run(q, kv_cache, out, ...):
    # Custom pre-process: layout transform / quantize / fuse with previous op
    q_perm = my_hip_permute(q)  # custom .hip kernel
    # AITER for the heavy MFMA loop
    mla.mla_decode_fwd(q_perm, kv_cache, out, **best_known_config)
    # Custom post-process: epilogue / fuse with next op
    my_hip_epilogue(out)
```

This bypasses AITER's framework overhead while keeping its compute-loop quality.

## Beating AITER — known regimes

- **Small batch / launch-bound decode**: AITER's launch overhead is non-trivial; persistent kernels typically win 1.2-1.5×.
- **Unusual hidden sizes**: AITER's autotune table covers common sizes well (128, 256, 4096, 7168); off-the-grid sizes (e.g., 192, 5120) often run on a sub-optimal config and have 10-30% headroom.
- **Fused epilogues**: AITER op-by-op assumes Python-side composition; agent-written fused kernels (norm+activation+next-layer-pre-proc) routinely beat the unfused chain.
- **Cross-op redundancy elimination**: if two AITER ops in a chain redundantly load the same activation, a hand-fused HIP kernel that loads once can beat the chain by 1.3-1.8×.

## Tuning configs to study

AITER's per-op tuning table format (CSV columns):
```
input_shape, block_m, block_n, block_k, num_stages, num_warps, mfma_variant, gflops
```

Pull the best config for a given shape, then probe ±1 step in each dim to see if AITER's table is locally-optimal — often it isn't (the table is sparse) and small perturbations win.

## Caveats

- AITER ships kernels compiled for both gfx942 and gfx950 in the same wheel — check `aiter.__version__` and `aiter.list_supported_archs()` to confirm your target is covered.
- `aiter.ops.gemm` thinly wraps hipBLASLt for some shapes — beating it ≈ beating hipBLASLt, which is hard.
- Some AITER ops require `HIP_FORCE_DEV_KERNARG=1` env var to avoid host-side kernel-arg marshaling overhead.

## Versioning

AITER iterates fast; pin to a commit in your `baseline.json` and re-baseline on AITER bumps. Older AITER versions can be 2× slower on the same hardware due to autotune table improvements over time.
