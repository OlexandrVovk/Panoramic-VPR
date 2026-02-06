"""Poincare ball operations for hyperbolic space."""

import torch
from torch import Tensor


def exp_map_zero(v: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Exponential map from the origin of the Poincare ball.

    Maps Euclidean vectors into the Poincare ball:
        exp_0(v) = tanh(||v|| / 2) * v / ||v||

    Args:
        v: (..., d) Euclidean vectors.
        eps: Numerical stability epsilon.

    Returns:
        (..., d) points on the Poincare ball with norm < 1.
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    result = torch.tanh(norm / 2) * (v / norm)
    # Clamp output to stay strictly inside the ball
    return poincare_norm_clamp(result, max_norm=1.0 - eps)


def log_map_zero(x: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Logarithmic map to the origin (inverse of exp_map_zero).

    Maps Poincare ball points back to Euclidean tangent space:
        log_0(x) = arctanh(||x||) * x / ||x||

    Args:
        x: (..., d) points on the Poincare ball.
        eps: Numerical stability epsilon.

    Returns:
        (..., d) Euclidean vectors.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.atanh(norm.clamp(max=1 - eps)) * (x / norm)


def hyperbolic_distance(x: Tensor, y: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Poincare ball distance.

    d(x, y) = arccosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))

    Args:
        x: (..., d) points on the Poincare ball.
        y: (..., d) points on the Poincare ball.
        eps: Numerical stability epsilon.

    Returns:
        (...,) hyperbolic distances.
    """
    diff_sq = (x - y).pow(2).sum(dim=-1)
    nx = (1.0 - x.pow(2).sum(dim=-1)).clamp(min=eps)
    ny = (1.0 - y.pow(2).sum(dim=-1)).clamp(min=eps)
    arg = 1.0 + 2.0 * diff_sq / (nx * ny)
    return torch.acosh(arg.clamp(min=1.0 + eps))


def poincare_norm_clamp(x: Tensor, max_norm: float = 0.99) -> Tensor:
    """Clamp points to stay strictly inside the Poincare ball."""
    norm = x.norm(dim=-1, keepdim=True)
    clamped = torch.where(norm > max_norm, x * (max_norm / norm), x)
    return clamped
