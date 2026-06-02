"""Standalone Phase 1.5 benchmark — LayerNorm on ROCm.

Times rows: torch_naive, aiter_tuned, hip_stock, plus any variants/<name>/{kernel.hip,binding.cpp}.
Writes baseline.json in the AKO4X reference/<family>/ schema.
"""
import argparse, json, time
from pathlib import Path
import torch


def layernorm_naive(x, w, b, eps=1e-5):
    x_f = x.to(torch.float32)
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    return (centered * inv_std).to(x.dtype) * w + b


def try_aiter_layernorm():
    """Probe AITER for a layer_norm op (API name shifts across versions)."""
    try:
        import aiter  # noqa
    except ImportError as e:
        print(f"  aiter not installed: {e}")
        return None
    candidates = [
        ("aiter", "layer_norm"),
        ("aiter.ops.norm", "layer_norm"),
        ("aiter.ops.norm", "layernorm"),
        ("aiter.ops.norm", "layer_norm_fwd"),
    ]
    for mod_path, name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[name])
            fn = getattr(mod, name)
            print(f"  aiter resolved to {mod_path}.{name}")
            # Probe signature — pick the right call form
            def call(x, w, b, eps, _fn=fn):
                return _fn(x, w, b, eps)
            return call
        except (ImportError, AttributeError):
            continue
    # List options
    try:
        import aiter.ops.norm as n
        attrs = [a for a in dir(n) if "norm" in a.lower() or "layer" in a.lower()]
        print(f"  aiter present but no known layernorm entry; aiter.ops.norm has: {attrs}")
    except Exception:
        pass
    return None


def _build_hip(module_name, hip_path, cpp_path):
    from torch.utils.cpp_extension import load
    module = load(
        name=module_name,
        sources=[str(hip_path), str(cpp_path)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        verbose=False,
    )
    def call(x, w, b, eps):
        y = torch.empty_like(x)
        module.launch(x, w, b, y, eps)
        return y
    return call


def build_hip_stock():
    src_dir = Path(__file__).parent / "baselines"
    return _build_hip("layernorm_hip_stock", src_dir / "hip_stock.hip", src_dir / "hip_stock_binding.cpp")


def build_variants():
    impls = {}
    variants_dir = Path(__file__).parent / "variants"
    if not variants_dir.is_dir():
        return impls
    for v in sorted(variants_dir.iterdir()):
        if not v.is_dir(): continue
        hip_f, cpp_f = v / "kernel.hip", v / "binding.cpp"
        if not (hip_f.exists() and cpp_f.exists()):
            print(f"  variant {v.name}: missing files, skipped")
            continue
        try:
            impls[v.name] = _build_hip(f"layernorm_variant_{v.name}", hip_f, cpp_f)
            print(f"  variant {v.name}: built")
        except Exception as e:
            print(f"  variant {v.name}: build failed: {repr(e)[:200]}")
    return impls


def time_call(fn, args, warmup, iters):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", default="1,128,1024,8192")
    ap.add_argument("--hidden", default="128,4096,7168")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--out", default="baseline.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() == False — need rocm/pytorch container with /dev/kfd, /dev/dri")

    device = "cuda:0"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0)}")

    impls = {"torch_naive": layernorm_naive}
    aiter_call = try_aiter_layernorm()
    if aiter_call is not None:
        impls["aiter_tuned"] = aiter_call
    try:
        impls["hip_stock"] = build_hip_stock()
    except Exception as e:
        print(f"  hip_stock build failed: {e}")
    impls.update(build_variants())

    HIDDEN_PROBE = 4096
    print(f"\n=== correctness check (n_rows=32, hidden={HIDDEN_PROBE}) ===")
    x_corr = torch.randn(32, HIDDEN_PROBE, dtype=dtype, device=device)
    w_corr = torch.randn(HIDDEN_PROBE, dtype=dtype, device=device)
    b_corr = torch.randn(HIDDEN_PROBE, dtype=dtype, device=device)
    y_ref = layernorm_naive(x_corr, w_corr, b_corr, 1e-5)
    for name, fn in list(impls.items()):
        if name == "torch_naive": continue
        try:
            y_cand = fn(x_corr, w_corr, b_corr, 1e-5)
            atol = (y_ref - y_cand).abs().max().item()
            rtol = ((y_ref - y_cand).abs() / (y_ref.abs() + 1e-6)).max().item()
            ok = atol < 0.1 and rtol < 0.1
            print(f"  {'✅' if ok else '❌'} {name:>14}: atol={atol:.4f} rtol={rtol:.4f}")
            if not ok:
                impls.pop(name)
                print(f"      ⚠ {name} REMOVED")
        except Exception as e:
            print(f"  ❌ {name:>14}: crashed: {repr(e)[:120]}")
            impls.pop(name)

    n_rows_list = [int(x) for x in args.n_rows.split(",")]
    hidden_list = [int(x) for x in args.hidden.split(",")]
    results = {"operator": "layernorm",
               "environment": {"torch_version": torch.__version__, "hip_version": torch.version.hip,
                               "device_name": torch.cuda.get_device_name(0), "dtype": args.dtype},
               "benchmark_config": {"warmup": args.warmup, "iters": args.iters},
               "rows": []}

    col_w = 10
    print(f"\n{'n_rows':>8} {'hidden':>7} | " + " | ".join(f"{n[:col_w]:>{col_w}}" for n in impls.keys()))
    for hidden in hidden_list:
        for n_rows in n_rows_list:
            x = torch.randn(n_rows, hidden, dtype=dtype, device=device)
            w = torch.randn(hidden, dtype=dtype, device=device)
            b = torch.randn(hidden, dtype=dtype, device=device)
            row = {"n_rows": n_rows, "hidden": hidden, "latency_ms": {}}
            for name, fn in impls.items():
                try:
                    row["latency_ms"][name] = time_call(fn, (x, w, b, 1e-5), args.warmup, args.iters)
                except Exception as e:
                    row["latency_ms"][name] = None
                    row.setdefault("errors", {})[name] = repr(e)[:200]
            results["rows"].append(row)
            cells = []
            for name in impls.keys():
                v = row["latency_ms"].get(name)
                cells.append(f"{v*1000:>{col_w-3}.1f}µs" if v is not None else f"{'-':>{col_w}}")
            print(f"{n_rows:>8d} {hidden:>7d} | " + " | ".join(cells))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
