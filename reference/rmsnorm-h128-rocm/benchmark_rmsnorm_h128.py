"""Standalone Phase 1 benchmark harness — rmsnorm h=128 on ROCm.

Times three rows per shape:
  1. ``torch_naive``  — torch reference (eager, correctness anchor)
  2. ``aiter_tuned``  — AITER's tuned rmsnorm op  (skipped if AITER not installed)
  3. ``hip_stock``    — naive HIP kernel (this fork's baseline-to-beat candidate)

Output: a CSV + a baseline.json next to it, in the AKO4X reference/<family>/
schema (so spawn.py + master can read it later without translation).

Usage::

    python benchmark_rmsnorm_h128.py [--n-rows 1,8,32,128,1024,8192]
                                     [--iters 200] [--warmup 5]
                                     [--dtype bfloat16]
                                     [--out ./baseline.json]

Run inside the rocm/pytorch container so torch can see the MI355X / MI350X GPUs.
"""
import argparse
import json
import time
from pathlib import Path

import torch


def rmsnorm_naive(x, w, eps=1e-6):
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    return (x.to(torch.float32) * inv_rms).to(x.dtype) * w


def try_aiter_rmsnorm():
    """Return a callable (x, w, eps) -> y if AITER is installed, else None."""
    try:
        import aiter
        # AITER's RMSNorm lives in aiter.ops.norm.rms_norm (signature may vary by version)
        from aiter.ops.norm import rms_norm_fwd  # noqa
        def call(x, w, eps):
            return rms_norm_fwd(x, w, eps)
        return call
    except Exception as e:
        print(f"  aiter not available: {e}")
        return None


def build_hip_kernel():
    """Compile hip_stock.hip into a torch extension via load_inline; return launcher."""
    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).parent / "baselines"
    module = load(
        name="rmsnorm_h128_hip_stock",
        sources=[str(src_dir / "hip_stock.hip")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        verbose=True,
    )

    def call(x, w, eps):
        y = torch.empty_like(x)
        # The .hip file exposes a C function; torch's extension auto-binds via
        # a generated stub for `extern "C"` symbols. Specifically for raw-symbol
        # exports we need a tiny wrapper — see Phase-1.1 followup.
        # Placeholder: if load_inline can't bind the raw C symbol, we'll wrap it
        # with a PYBIND11_MODULE block in hip_stock_wrapper.cpp (Phase 1.5b).
        module.rmsnorm_h128_launch(  # type: ignore[attr-defined]
            x.data_ptr(), w.data_ptr(), y.data_ptr(),
            x.shape[0], eps,
            torch.cuda.current_stream().cuda_stream,
        )
        return y
    return call


def time_call(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", default="1,8,32,128,1024,8192",
                    help="comma-separated batch sizes")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out", default="baseline.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: torch.cuda.is_available() == False. "
            "Make sure you're inside the rocm/pytorch container with --device /dev/kfd "
            "and --device /dev/dri so the wheel can see the GPUs."
        )

    device = "cuda:0"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).name})")

    # Build / load the available impls
    impls = {"torch_naive": rmsnorm_naive}
    aiter_call = try_aiter_rmsnorm()
    if aiter_call is not None:
        impls["aiter_tuned"] = aiter_call
    try:
        impls["hip_stock"] = build_hip_kernel()
    except Exception as e:
        print(f"  hip_stock build failed: {e}")

    # Run sweep
    HIDDEN = 128
    n_rows_list = [int(x) for x in args.n_rows.split(",")]
    results = {"operator": "rmsnorm_h128",
               "environment": {
                   "torch_version": torch.__version__,
                   "hip_version": torch.version.hip,
                   "device_name": torch.cuda.get_device_name(0),
                   "dtype": args.dtype,
               },
               "benchmark_config": {
                   "warmup": args.warmup,
                   "iters": args.iters,
               },
               "rows": []}

    print(f"\n{'n_rows':>8} {'hidden':>7} | " +
          " | ".join(f"{n:>14}" for n in impls.keys()))
    for n_rows in n_rows_list:
        x = torch.randn(n_rows, HIDDEN, dtype=dtype, device=device)
        w = torch.randn(HIDDEN, dtype=dtype, device=device)
        row = {"n_rows": n_rows, "hidden": HIDDEN, "latency_ms": {}}
        for name, fn in impls.items():
            try:
                row["latency_ms"][name] = time_call(fn, (x, w, 1e-6), args.warmup, args.iters)
            except Exception as e:
                row["latency_ms"][name] = None
                row.setdefault("errors", {})[name] = repr(e)[:200]
        results["rows"].append(row)
        cells = []
        for name in impls.keys():
            v = row["latency_ms"].get(name)
            cells.append(f"{v:>12.4f} ms" if v is not None else f"{'ERROR':>14}")
        print(f"{n_rows:>8d} {HIDDEN:>7d} | " + " | ".join(cells))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
