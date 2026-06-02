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
    """Return a callable (x, w, eps) -> y if AITER is installed, else None.

    AITER's API has shifted across versions; we probe a few known entry-point names
    in order. Most recent: aiter.ops.norm.rms_norm  (older was rms_norm_fwd).
    """
    try:
        import aiter  # noqa
    except ImportError as e:
        print(f"  aiter not installed: {e}")
        return None
    # Try several known API paths
    candidates = [
        ("aiter.ops.norm", "rms_norm"),
        ("aiter.ops.norm", "rms_norm_fwd"),
        ("aiter", "rms_norm"),
        ("aiter.ops.rmsnorm", "rms_norm_fwd"),
    ]
    for mod_path, name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[name])
            fn = getattr(mod, name)
            # Probe signature: most accept (x, w, eps) and return y; some take y first
            print(f"  aiter resolved to {mod_path}.{name}")
            def call(x, w, eps, _fn=fn):
                return _fn(x, w, eps)
            return call
        except (ImportError, AttributeError):
            continue
    # Last resort: list what's actually in aiter.ops.norm
    try:
        import aiter.ops.norm as n
        attrs = [a for a in dir(n) if "norm" in a.lower() or "rms" in a.lower()]
        print(f"  aiter present but no known rmsnorm entry; aiter.ops.norm has: {attrs}")
    except Exception:
        pass
    return None


def build_hip_kernel():
    """Compile hip_stock.hip + binding into a torch extension; return launcher."""
    from torch.utils.cpp_extension import load
    src_dir = Path(__file__).parent / "baselines"
    module = load(
        name="rmsnorm_h128_hip_stock",
        sources=[
            str(src_dir / "hip_stock.hip"),
            str(src_dir / "hip_stock_binding.cpp"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        verbose=False,
    )

    def call(x, w, eps):
        y = torch.empty_like(x)
        module.launch(x, w, y, eps)  # type: ignore[attr-defined]
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
    ap.add_argument("--hidden", default="128,4096,7168",
                    help="comma-separated hidden dims to sweep "
                         "(128=MLA-head-dim, 4096=Llama-hidden, 7168=DSR1-hidden). "
                         "NOTE: hip_stock kernel is hard-coded for HIDDEN=128; it will be "
                         "skipped for any other hidden size in this Phase-1 demo.")
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

    # Correctness check FIRST — pointless to time wrong answers
    HIDDEN = 128
    print("\n=== correctness check (n_rows=32) ===")
    x_corr = torch.randn(32, HIDDEN, dtype=dtype, device=device)
    w_corr = torch.randn(HIDDEN, dtype=dtype, device=device)
    y_ref = rmsnorm_naive(x_corr, w_corr, 1e-6)
    for name, fn in list(impls.items()):
        if name == "torch_naive":
            continue
        try:
            y_cand = fn(x_corr, w_corr, 1e-6)
            atol = (y_ref - y_cand).abs().max().item()
            rtol = ((y_ref - y_cand).abs() / (y_ref.abs() + 1e-6)).max().item()
            ok = atol < 0.05 and rtol < 0.05  # bf16 has ~1e-2 representation error
            tag = "✅" if ok else "❌"
            print(f"  {tag} {name:>14}: atol={atol:.4f} rtol={rtol:.4f}")
            if not ok:
                print(f"      ref[0,:4] = {y_ref[0, :4].tolist()}")
                print(f"      cand[0,:4]= {y_cand[0, :4].tolist()}")
                impls.pop(name)
                print(f"      ⚠ {name} REMOVED from benchmark — incorrect output")
        except Exception as e:
            print(f"  ❌ {name:>14}: crashed: {repr(e)[:120]}")
            impls.pop(name)

    # Run sweep
    n_rows_list = [int(x) for x in args.n_rows.split(",")]
    hidden_list = [int(x) for x in args.hidden.split(",")]
    results = {"operator": "rmsnorm",
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
    for hidden in hidden_list:
        for n_rows in n_rows_list:
            x = torch.randn(n_rows, hidden, dtype=dtype, device=device)
            w = torch.randn(hidden, dtype=dtype, device=device)
            row = {"n_rows": n_rows, "hidden": hidden, "latency_ms": {}}
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
                cells.append(f"{v:>12.4f} ms" if v is not None else f"{'-':>14}")
            print(f"{n_rows:>8d} {hidden:>7d} | " + " | ".join(cells))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
