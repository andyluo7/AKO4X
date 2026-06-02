"""Phase 2 benchmark — FP8 GEMM (DSR1 FFN shape).

Single shape: M=4096, N=7168, K=2048 (override with --m/--n/--k).
FP8 e4m3 (fnuz on ROCm) inputs, BF16 output, per-token scale_a [M] + per-channel scale_b [N].

Times: torch_naive (fp32 ref), hipblaslt (via torch._scaled_mm if present),
aiter_tuned (probed), hip_stock, and any variants/<name>/{kernel.hip,binding.cpp}.

Writes baseline.json in the AKO4X reference/<family>/ schema.
"""
import argparse, json, time
from pathlib import Path
import torch

from baselines.torch_naive import fp8_gemm_naive


FP8_DTYPE = torch.float8_e4m3fnuz  # ROCm-native FP8 (fnuz = finite, no -0/NaN)


def quantize_per_token(x_bf16):
    """Per-row absmax quant: returns (fp8 tensor, fp32 scale [rows])."""
    amax = x_bf16.abs().to(torch.float32).amax(dim=-1).clamp(min=1e-4)
    fp8_max = 240.0  # e4m3fnuz max
    scale = amax / fp8_max
    q = (x_bf16.to(torch.float32) / scale.view(-1, 1)).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
    return q, scale


def try_aiter_fp8_gemm():
    try:
        import aiter
    except ImportError as e:
        print(f"  aiter not installed: {e}")
        return None
    # AITER 25.9 FP8 GEMM API names shift across versions; probe several.
    candidates = [
        ("aiter", "gemm_a8w8"),
        ("aiter", "gemm_a8w8_bpreshuffle"),
        ("aiter.ops.gemm_op_a8w8", "gemm_a8w8"),
        ("aiter.ops.gemm", "gemm_a8w8"),
    ]
    for mod_path, name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[name])
            fn = getattr(mod, name)
            print(f"  aiter resolved to {mod_path}.{name}")
            def call(a, b, scale_a, scale_b, _fn=fn):
                # Most AITER FP8 GEMMs expect (a, b, scale_a, scale_b, out_dtype)
                # — let's try the common signature and fall back to constructing out.
                try:
                    return _fn(a, b, scale_a, scale_b, torch.bfloat16)
                except TypeError:
                    out = torch.empty(a.size(0), b.size(0), dtype=torch.bfloat16, device=a.device)
                    _fn(a, b, scale_a, scale_b, out)
                    return out
            return call
        except (ImportError, AttributeError):
            continue
    try:
        import aiter as a
        attrs = [x for x in dir(a) if "gemm" in x.lower() or "a8w8" in x.lower()]
        print(f"  aiter present but no known FP8 GEMM; aiter has: {attrs[:20]}")
    except Exception:
        pass
    return None


def try_torch_scaled_mm():
    """torch._scaled_mm: HIPBLASLT-backed FP8 GEMM on recent torch+rocm.

    Signature varies by torch version. Returns None if unavailable / signature
    not matched.
    """
    if not hasattr(torch, "_scaled_mm"):
        print("  torch._scaled_mm not present")
        return None
    def call(a, b, scale_a, scale_b):
        # b is [N, K] (weight layout); _scaled_mm wants [K, N], so transpose-view.
        # _scaled_mm expects 2-D scales for per-token/per-channel on recent torch.
        try:
            return torch._scaled_mm(
                a, b.t(),
                scale_a=scale_a.view(-1, 1),
                scale_b=scale_b.view(1, -1),
                out_dtype=torch.bfloat16,
            )
        except Exception as e:
            raise RuntimeError(f"_scaled_mm call failed: {e!r}")
    return call


def _build_hip(module_name, hip_path, cpp_path):
    from torch.utils.cpp_extension import load
    module = load(
        name=module_name,
        sources=[str(hip_path), str(cpp_path)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        verbose=False,
    )
    def call(a, b, scale_a, scale_b):
        c = torch.empty(a.size(0), b.size(0), dtype=torch.bfloat16, device=a.device)
        module.launch(a, b, scale_a, scale_b, c)
        return c
    return call


def build_hip_stock():
    src = Path(__file__).parent / "baselines"
    return _build_hip("fp8_gemm_hip_stock", src / "hip_stock.hip", src / "hip_stock_binding.cpp")


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
            impls[v.name] = _build_hip(f"fp8_gemm_variant_{v.name}", hip_f, cpp_f)
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
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=7168)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="baseline.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() == False — need rocm/pytorch container.")

    device = "cuda:0"
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shape: M={args.m} N={args.n} K={args.k}")

    impls = {"torch_naive": fp8_gemm_naive}
    scaled = try_torch_scaled_mm()
    if scaled is not None:
        impls["hipblaslt_scaled_mm"] = scaled
    aiter_call = try_aiter_fp8_gemm()
    if aiter_call is not None:
        impls["aiter_tuned"] = aiter_call
    try:
        impls["hip_stock"] = build_hip_stock()
    except Exception as e:
        print(f"  hip_stock build failed: {e}")
    impls.update(build_variants())

    # Correctness: small shape so torch_naive is fast.
    print(f"\n=== correctness check (M=64, N=128, K={args.k}) ===")
    a_bf = torch.randn(64, args.k, dtype=torch.bfloat16, device=device)
    b_bf = torch.randn(128, args.k, dtype=torch.bfloat16, device=device)
    a_q, sa = quantize_per_token(a_bf)
    b_q, sb = quantize_per_token(b_bf)
    y_ref = fp8_gemm_naive(a_q, b_q, sa, sb)
    out_scale = y_ref.abs().to(torch.float32).max().item()
    print(f"  ref out abs-max: {out_scale:.2f}")
    for name, fn in list(impls.items()):
        if name == "torch_naive": continue
        try:
            y_cand = fn(a_q, b_q, sa, sb)
            diff = (y_ref - y_cand).abs().to(torch.float32)
            atol = diff.max().item()
            med = diff.median().item()
            # FP8 e4m3 has ~3 bits mantissa; expected BF16-output absolute error
            # scales roughly with output magnitude. Use 5% of out-magnitude as
            # the atol gate, with a floor of 0.5 to handle small outputs.
            atol_gate = max(0.5, 0.05 * out_scale)
            ok = atol < atol_gate
            print(f"  {'OK' if ok else 'BAD'} {name:>22}: atol={atol:.3f} median={med:.4f} (gate {atol_gate:.2f})")
            if not ok:
                impls.pop(name)
                print(f"      {name} REMOVED")
        except Exception as e:
            print(f"  BAD {name:>22}: crashed: {repr(e)[:160]}")
            impls.pop(name)

    a_bf = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
    b_bf = torch.randn(args.n, args.k, dtype=torch.bfloat16, device=device)
    a_q, sa = quantize_per_token(a_bf)
    b_q, sb = quantize_per_token(b_bf)

    results = {"operator": "fp8_gemm_dsr1",
               "shape": {"M": args.m, "N": args.n, "K": args.k},
               "environment": {"torch_version": torch.__version__, "hip_version": torch.version.hip,
                               "device_name": torch.cuda.get_device_name(0),
                               "input_dtype": "float8_e4m3fnuz", "output_dtype": "bfloat16",
                               "scale_scheme": "per-token-A + per-channel-B"},
               "benchmark_config": {"warmup": args.warmup, "iters": args.iters},
               "rows": []}

    # Compute throughput peak for context.
    flops = 2.0 * args.m * args.n * args.k
    print(f"\n=== timing (M={args.m} N={args.n} K={args.k}, {flops/1e9:.1f} GFLOP) ===")
    row = {"M": args.m, "N": args.n, "K": args.k, "latency_us": {}, "tflops": {}}
    for name, fn in impls.items():
        # torch_naive is hilariously slow at full shape — skip if M*N*K too big.
        if name == "torch_naive" and args.m * args.n * args.k > 1e8:
            print(f"  {name:>22}: SKIPPED (too slow at this shape)")
            continue
        try:
            ms = time_call(fn, (a_q, b_q, sa, sb), args.warmup, args.iters)
            us = ms * 1000.0
            tflops = flops / (ms / 1000.0) / 1e12
            row["latency_us"][name] = us
            row["tflops"][name] = tflops
            print(f"  {name:>22}: {us:8.1f} µs   {tflops:6.1f} TFLOP/s")
        except Exception as e:
            row.setdefault("errors", {})[name] = repr(e)[:200]
            print(f"  {name:>22}: FAILED {repr(e)[:120]}")
    results["rows"].append(row)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
