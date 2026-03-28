"""Tests for Lloyd-Max quantizer."""
import numpy as np
import pytest
from turboquant.codebook import LloydMaxQuantizer, hypersphere_coordinate_pdf


class TestHyperspherePDF:
    def test_pdf_symmetric(self):
        assert abs(hypersphere_coordinate_pdf(0.01, 384) - hypersphere_coordinate_pdf(-0.01, 384)) < 1e-6

    def test_pdf_zero_at_boundary(self):
        assert hypersphere_coordinate_pdf(1.0, 384) == 0.0
        assert hypersphere_coordinate_pdf(-1.0, 384) == 0.0

    def test_pdf_peaks_at_zero(self):
        assert hypersphere_coordinate_pdf(0.0, 384) > hypersphere_coordinate_pdf(0.05, 384)

    def test_pdf_positive(self):
        for x in [-0.05, -0.01, 0.0, 0.01, 0.05]:
            assert hypersphere_coordinate_pdf(x, 384) > 0


class TestLloydMaxQuantizer:
    def test_centroids_symmetric(self):
        lmq = LloydMaxQuantizer(d=384, num_bits=4)
        assert np.allclose(lmq.centroids, -lmq.centroids[::-1], atol=1e-6)

    def test_centroids_sorted(self):
        lmq = LloydMaxQuantizer(d=384, num_bits=3)
        assert np.all(np.diff(lmq.centroids) > 0)

    def test_num_centroids(self):
        for bits in [1, 2, 3, 4]:
            lmq = LloydMaxQuantizer(d=384, num_bits=bits)
            assert len(lmq.centroids) == 2 ** bits

    def test_quantize_dequantize(self):
        lmq = LloydMaxQuantizer(d=384, num_bits=4)
        values = np.array([0.0, 0.01, -0.05, 0.1])
        indices = lmq.quantize(values)
        recon = lmq.dequantize(indices)
        assert recon.shape == values.shape
        assert np.all(np.abs(values - recon) < 0.02)

    def test_theoretical_bounds(self):
        lmq = LloydMaxQuantizer(d=384, num_bits=4)
        assert lmq.theoretical_mse > lmq.shannon_lower_bound

    def test_different_dimensions(self):
        for d in [128, 384, 768, 1536]:
            lmq = LloydMaxQuantizer(d=d, num_bits=3)
            assert len(lmq.centroids) == 8
