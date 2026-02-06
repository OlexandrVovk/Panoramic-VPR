"""Einstein midpoint aggregation for hierarchical descriptors."""

import torch
from torch import Tensor

from .poincare import hyperbolic_distance


def einstein_midpoint(points: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Standard Einstein midpoint (hyperbolic centroid).

    m = (sum gamma_i * x_i) / (sum gamma_i)
    where gamma_i = 1 / sqrt(1 - ||x_i||^2) is the Lorentz factor.

    The Lorentz factor gives higher weight to points closer to the boundary,
    pulling the midpoint toward more distinctive children.

    Args:
        points: (K, d) child descriptors on the Poincare ball.
        eps: Numerical stability epsilon.

    Returns:
        (d,) root descriptor, closer to the center than any child.
    """
    norms_sq = points.pow(2).sum(dim=-1, keepdim=True)  # (K, 1)
    gamma = 1.0 / torch.sqrt((1.0 - norms_sq).clamp(min=eps))  # (K, 1)

    weighted_sum = (gamma * points).sum(dim=0)  # (d,)
    gamma_sum = gamma.sum(dim=0)  # (1,)

    midpoint = weighted_sum / gamma_sum

    # Clamp inside ball for numerical safety
    norm = midpoint.norm()
    if norm >= 1.0:
        midpoint = midpoint / (norm + eps) * 0.95

    return midpoint


def weighted_einstein_midpoint(points: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Weighted Einstein midpoint using uniqueness scores.

    Children dissimilar from their siblings receive higher weight,
    pulling the root toward rare visual content.

    Uniqueness: w_i = mean hyperbolic distance to other children.

    Args:
        points: (K, d) child descriptors on the Poincare ball.
        eps: Numerical stability epsilon.

    Returns:
        (d,) root descriptor.
    """
    K = points.shape[0]
    norms_sq = points.pow(2).sum(dim=-1, keepdim=True)  # (K, 1)
    gamma = 1.0 / torch.sqrt((1.0 - norms_sq).clamp(min=eps))  # (K, 1)

    # Compute pairwise hyperbolic distances
    dists = torch.zeros(K, K, device=points.device, dtype=points.dtype)
    for i in range(K):
        for j in range(i + 1, K):
            d = hyperbolic_distance(points[i], points[j])
            dists[i, j] = d
            dists[j, i] = d

    # Uniqueness = mean distance to other children
    weights = dists.sum(dim=1) / (K - 1)  # (K,)
    weights = weights / weights.sum()  # normalize

    # Combine Lorentz factor with uniqueness weight
    combined = gamma * weights.unsqueeze(-1)  # (K, 1)
    weighted_sum = (combined * points).sum(dim=0)  # (d,)
    combined_sum = combined.sum(dim=0)  # (1,)

    midpoint = weighted_sum / combined_sum

    norm = midpoint.norm()
    if norm >= 1.0:
        midpoint = midpoint / (norm + eps) * 0.95

    return midpoint
