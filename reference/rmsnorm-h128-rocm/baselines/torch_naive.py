"""Naive torch RMSNorm reference — first baseline row for ako4x-rocm Phase 1.

This is the simplest correct reference, used to:
  (a) sanity-check torch+ROCm on the host (verify GPU access, dtype support);
  (b) serve as the ``ref_impl`` against which kernel candidates are compared
      for correctness in cheat_check().

Performance is NOT the goal here — naive torch is intentionally slow so we have
clear headroom for the agent's HIP / AITER / FlyDSL candidates to beat.

Operator
--------
RMSNorm with hidden dim h=128 (small / per-head shape, common in MLA / GQA).

  y = x * w / sqrt(mean(x^2) + eps)

Inputs
------
  x : tensor of shape (n_rows, 128), dtype bfloat16
  w : tensor of shape (128,),        dtype bfloat16
  eps : float

Output
------
  y : tensor of shape (n_rows, 128), dtype bfloat16
"""
import torch


def rmsnorm_naive(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference impl: explicit mean(x^2)+eps -> rsqrt -> scale by w."""
    # Cast to fp32 for the reduction (avoids bf16 accuracy loss on the var)
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    return (x.to(torch.float32) * inv_rms).to(x.dtype) * w


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    device = "cuda:0"  # PyTorch+ROCm exposes hip devices via the "cuda:" namespace
    print(f"torch={torch.__version__} hip={torch.version.hip} avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0: {torch.cuda.get_device_name(0)}")

    # Sweep batch sizes typical for decode-time RMSNorm (h=128, n_rows = batch * heads)
    SHAPES = [(1, 128), (8, 128), (32, 128), (128, 128), (1024, 128), (8192, 128)]
    DTYPE = torch.bfloat16
    EPS = 1e-6
    WARMUP, ITERS = 5, 200

    for n_rows, hidden in SHAPES:
        x = torch.randn(n_rows, hidden, dtype=DTYPE, device=device)
        w = torch.randn(hidden, dtype=DTYPE, device=device)
        # Warmup
        for _ in range(WARMUP):
            y = rmsnorm_naive(x, w, EPS)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            y = rmsnorm_naive(x, w, EPS)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0 / ITERS
        print(f"  n_rows={n_rows:>6d} hidden={hidden:>4d}  latency = {dt_ms:>8.4f} ms")
