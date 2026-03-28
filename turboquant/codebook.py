"""Lloyd-Max optimal scalar quantizer for hypersphere coordinate distribution.

Implements the optimal 1D quantizer for coordinates of uniformly distributed
vectors on the unit sphere S^(d-1). In high dimensions, coordinates follow
a Beta distribution converging to N(0, 1/d).

Reference: TurboQuant (arXiv:2504.19874), Section 3.1
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def hypersphere_coordinate_pdf(x: float, d: int) -> float:
    """PDF of a single coordinate of a uniform vector on S^(d-1).

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    Uses log-gamma to avoid overflow for large d.

    Args:
        x: Coordinate value in [-1, 1].
        d: Ambient dimension.

    Returns:
        Probability density at x.
    """
    if abs(x) >= 1.0:
        return 0.0
    log_coeff = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    log_body = ((d - 3) / 2) * np.log(max(1 - x**2, 1e-300))
    return np.exp(log_coeff + log_body)


class LloydMaxQuantizer:
    """Optimal scalar quantizer for hypersphere coordinate distribution.

    Computes centroids that minimize MSE distortion for the Beta-distributed
    coordinates arising from random rotation of unit vectors.

    Args:
        d: Embedding dimension.
        num_bits: Number of quantization bits (1-8).
        grid_points: Number of integration grid points (higher = more precise).
        max_iter: Maximum Lloyd-Max iterations.

    Example:
        >>> lmq = LloydMaxQuantizer(d=384, num_bits=4)
        >>> lmq.centroids  # array of 16 optimal centroids
        >>> idx = lmq.quantize(0.05)  # nearest centroid index
        >>> val = lmq.dequantize(idx)  # centroid value
    """

    def __init__(
        self,
        d: int,
        num_bits: int = 4,
        grid_points: int = 50000,
        max_iter: int = 200,
    ) -> None:
        self.d = d
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.centroids = self._compute_centroids(grid_points, max_iter)

    def _compute_centroids(self, grid_points: int, max_iter: int) -> np.ndarray:
        """Compute optimal centroids via Lloyd-Max algorithm."""
        from scipy.stats import norm

        sigma = 1.0 / np.sqrt(self.d)
        num_levels = self.num_levels

        # Initialize with Gaussian quantiles (good approximation for large d)
        quantiles = np.linspace(1 / (2 * num_levels), 1 - 1 / (2 * num_levels), num_levels)
        centroids = norm.ppf(quantiles, scale=sigma)

        # Fine integration grid covering ±5σ
        x_grid = np.linspace(-5 * sigma, 5 * sigma, grid_points)
        pdf_vals = np.array([hypersphere_coordinate_pdf(x, self.d) for x in x_grid])
        dx = x_grid[1] - x_grid[0]

        for _ in range(max_iter):
            # Boundaries = midpoints between centroids
            boundaries = np.concatenate(
                [[-np.inf], (centroids[:-1] + centroids[1:]) / 2, [np.inf]]
            )

            # Update centroids as conditional expectations
            new_centroids = np.zeros_like(centroids)
            for i in range(num_levels):
                lo, hi = boundaries[i], boundaries[i + 1]
                mask = (x_grid >= lo) & (x_grid < hi)
                if mask.sum() == 0:
                    new_centroids[i] = centroids[i]
                    continue
                weighted_sum = np.sum(x_grid[mask] * pdf_vals[mask]) * dx
                weight = np.sum(pdf_vals[mask]) * dx
                new_centroids[i] = weighted_sum / weight if weight > 1e-12 else centroids[i]

            if np.allclose(centroids, new_centroids, atol=1e-10):
                break
            centroids = new_centroids

        return np.sort(centroids)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize scalar values to nearest centroid indices.

        Args:
            values: Array of scalar values.

        Returns:
            Array of centroid indices (uint8).
        """
        dists = np.abs(values[..., np.newaxis] - self.centroids)
        return np.argmin(dists, axis=-1).astype(np.uint8)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Dequantize centroid indices back to scalar values.

        Args:
            indices: Array of centroid indices.

        Returns:
            Array of centroid values.
        """
        return self.centroids[indices]

    @property
    def theoretical_mse(self) -> float:
        """Upper bound on MSE from Theorem 1: (sqrt(3)*pi/2) * (1/4^b)."""
        return (np.sqrt(3) * np.pi / 2) * (1.0 / 4 ** self.num_bits)

    @property
    def shannon_lower_bound(self) -> float:
        """Information-theoretic lower bound on MSE: 1/4^b."""
        return 1.0 / 4 ** self.num_bits
