"""TurboQuant vector quantizers — MSE-optimal and inner-product-optimal.

Implements Algorithms 1 and 2 from the TurboQuant paper (arXiv:2504.19874).

Algorithm 1 (TurboQuantMSE):
    Random rotation → coordinate-wise Lloyd-Max quantization.
    Achieves near-optimal MSE distortion.

Algorithm 2 (TurboQuantProd):
    TurboQuantMSE with (b-1) bits + QJL 1-bit residual correction.
    Achieves unbiased inner product estimation with near-optimal distortion.

Reference: TurboQuant (arXiv:2504.19874), Section 3
"""

from __future__ import annotations

import numpy as np

from turboquant.codebook import LloydMaxQuantizer


class TurboQuantMSE:
    """MSE-optimal vector quantizer (Algorithm 1).

    Quantizes vectors on S^(d-1) by:
    1. Rotating with a random orthogonal matrix (QR decomposition)
    2. Quantizing each coordinate independently with Lloyd-Max

    Args:
        d: Vector dimension.
        num_bits: Bits per coordinate (1-8). Default 4.
        seed: Random seed for reproducible rotation matrix.

    Example:
        >>> tq = TurboQuantMSE(d=384, num_bits=4)
        >>> codes = tq.quantize(vectors)     # (N, d) → (N, d) uint8
        >>> recon = tq.dequantize(codes)     # (N, d) uint8 → (N, d) float32
        >>> mse = np.mean(np.sum((vectors - recon)**2, axis=1))
    """

    def __init__(self, d: int, num_bits: int = 4, seed: int = 42) -> None:
        self.d = d
        self.num_bits = num_bits
        self._rng = np.random.RandomState(seed)

        # Pre-compute random orthogonal rotation matrix via QR
        gaussian = self._rng.randn(d, d).astype(np.float32)
        self.rotation, _ = np.linalg.qr(gaussian)
        self.rotation = self.rotation.astype(np.float32)

        # Pre-compute optimal scalar quantizer
        self.codebook = LloydMaxQuantizer(d=d, num_bits=num_bits)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize vectors to centroid indices.

        Args:
            x: Input vectors, shape (N, d). Must be L2-normalized.

        Returns:
            Centroid indices, shape (N, d), dtype uint8.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # Step 1: Rotate
        y = x @ self.rotation.T

        # Step 2: Coordinate-wise quantization
        return self.codebook.quantize(y)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Dequantize centroid indices to reconstructed vectors.

        Args:
            indices: Centroid indices, shape (N, d), dtype uint8.

        Returns:
            Reconstructed vectors, shape (N, d), dtype float32.
        """
        # Step 1: Lookup centroids
        y_hat = self.codebook.dequantize(indices)

        # Step 2: Inverse rotation
        return y_hat @ self.rotation

    @property
    def bytes_per_vector(self) -> int:
        """Storage bytes per quantized vector."""
        return int(np.ceil(self.num_bits * self.d / 8))

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32 storage."""
        return (self.d * 4) / self.bytes_per_vector


class TurboQuantProd:
    """Inner-product-optimal vector quantizer (Algorithm 2).

    Two-stage quantization:
    1. TurboQuantMSE with (b-1) bits for the main signal
    2. QJL (Quantized Johnson-Lindenstrauss) with 1 bit for residual correction

    This eliminates the bias in inner product estimation that MSE-only
    quantization introduces, while maintaining near-optimal distortion.

    Args:
        d: Vector dimension.
        num_bits: Total bits per coordinate (≥2). Uses (b-1) for MSE + 1 for QJL.
        seed: Random seed for reproducible matrices.

    Example:
        >>> tq = TurboQuantProd(d=384, num_bits=4)
        >>> codes = tq.quantize(vectors)
        >>> # Inner product estimation is unbiased:
        >>> # E[<y, dequant(quant(x))>] = <y, x>
    """

    def __init__(self, d: int, num_bits: int = 4, seed: int = 42) -> None:
        if num_bits < 2:
            raise ValueError("TurboQuantProd requires num_bits >= 2 (1 bit reserved for QJL)")
        self.d = d
        self.num_bits = num_bits
        self._rng = np.random.RandomState(seed)

        # Stage 1: MSE quantizer with (b-1) bits
        self.mse_quantizer = TurboQuantMSE(d=d, num_bits=num_bits - 1, seed=seed)

        # Stage 2: QJL projection matrix S ~ N(0, 1), shape (d, d)
        self.qjl_matrix = self._rng.randn(d, d).astype(np.float32) / np.sqrt(d)

    def quantize(self, x: np.ndarray) -> dict:
        """Quantize vectors using MSE + QJL two-stage method.

        Args:
            x: Input vectors, shape (N, d). Must be L2-normalized.

        Returns:
            Dict with keys:
                'mse_codes': uint8 (N, d) — MSE stage indices
                'qjl_signs': int8 (N, d) — QJL sign bits (+1/-1)
                'residual_norms': float32 (N,) — L2 norm of residuals
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        # Stage 1: MSE quantization with (b-1) bits
        mse_codes = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(mse_codes)

        # Residual
        r = x - x_mse
        residual_norms = np.linalg.norm(r, axis=1).astype(np.float32)

        # Stage 2: QJL — sign of S·r
        projected = r @ self.qjl_matrix.T  # (N, d)
        qjl_signs = np.sign(projected).astype(np.int8)
        # Replace zeros with +1 (convention)
        qjl_signs[qjl_signs == 0] = 1

        return {
            "mse_codes": mse_codes,
            "qjl_signs": qjl_signs,
            "residual_norms": residual_norms,
        }

    def dequantize(self, codes: dict) -> np.ndarray:
        """Dequantize to reconstructed vectors with unbiased inner products.

        The reconstructed vector is:
            x̃ = x̃_mse + (sqrt(pi/2) / d) * γ * S^T · qjl_signs

        Args:
            codes: Dict from quantize().

        Returns:
            Reconstructed vectors, shape (N, d), dtype float32.
        """
        # Stage 1: MSE reconstruction
        x_mse = self.mse_quantizer.dequantize(codes["mse_codes"])

        # Stage 2: QJL correction
        gamma = codes["residual_norms"][:, np.newaxis]  # (N, 1)
        qjl_correction = (
            np.sqrt(np.pi / 2) / self.d
            * gamma
            * (codes["qjl_signs"].astype(np.float32) @ self.qjl_matrix)
        )

        return x_mse + qjl_correction

    @property
    def bytes_per_vector(self) -> int:
        """Storage bytes per quantized vector (MSE codes + QJL signs + norm)."""
        mse_bytes = int(np.ceil((self.num_bits - 1) * self.d / 8))
        qjl_bytes = int(np.ceil(self.d / 8))  # 1 bit per dimension
        norm_bytes = 4  # float32
        return mse_bytes + qjl_bytes + norm_bytes

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32 storage."""
        return (self.d * 4) / self.bytes_per_vector
