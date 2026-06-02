"""Naive torch LayerNorm reference — correctness anchor for ako4x-rocm.

Formula: y = (x - mean(x)) / sqrt(var(x) + eps) * w + b
        where mean and var are over the last dim.
"""
import torch


def layernorm_naive(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference impl in fp32 for numerical stability, output cast back to x.dtype."""
    x_f = x.to(torch.float32)
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    return (centered * inv_std).to(x.dtype) * w + b
