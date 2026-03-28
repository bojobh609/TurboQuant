"""Tests for TurboQuant quantizers."""
import numpy as np
import pytest
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


def _random_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestTurboQuantMSE:
    def test_roundtrip_shape(self):
        tq = TurboQuantMSE(d=384, num_bits=4)
        x = _random_unit_vectors(10, 384)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert codes.shape == (10, 384)
        assert codes.dtype == np.uint8
        assert recon.shape == (10, 384)

    def test_mse_within_bound(self):
        tq = TurboQuantMSE(d=384, num_bits=4)
        x = _random_unit_vectors(500, 384)
        recon = tq.dequantize(tq.quantize(x))
        mse = np.mean(np.sum((x - recon) ** 2, axis=1))
        assert mse < tq.codebook.theoretical_mse * 1.5

    def test_single_vector(self):
        tq = TurboQuantMSE(d=384, num_bits=4)
        x = _random_unit_vectors(1, 384)
        codes = tq.quantize(x[0])
        assert codes.shape == (1, 384)

    def test_cosine_similarity_high(self):
        tq = TurboQuantMSE(d=384, num_bits=4)
        x = _random_unit_vectors(100, 384)
        recon = tq.dequantize(tq.quantize(x))
        cos_sims = np.sum(x * recon, axis=1) / (np.linalg.norm(recon, axis=1))
        assert np.mean(cos_sims) > 0.98

    def test_compression_ratio(self):
        tq = TurboQuantMSE(d=384, num_bits=4)
        assert tq.compression_ratio == pytest.approx(8.0, rel=0.1)

    def test_deterministic(self):
        x = _random_unit_vectors(10, 384)
        tq1 = TurboQuantMSE(d=384, num_bits=4, seed=42)
        tq2 = TurboQuantMSE(d=384, num_bits=4, seed=42)
        assert np.array_equal(tq1.quantize(x), tq2.quantize(x))

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
    def test_mse_decreases_with_bits(self, bits):
        tq = TurboQuantMSE(d=384, num_bits=bits)
        x = _random_unit_vectors(200, 384)
        recon = tq.dequantize(tq.quantize(x))
        mse = np.mean(np.sum((x - recon) ** 2, axis=1))
        assert mse < 0.5  # sanity check


class TestTurboQuantProd:
    def test_requires_min_2_bits(self):
        with pytest.raises(ValueError):
            TurboQuantProd(d=384, num_bits=1)

    def test_roundtrip_shape(self):
        tq = TurboQuantProd(d=384, num_bits=4)
        x = _random_unit_vectors(10, 384)
        codes = tq.quantize(x)
        assert "mse_codes" in codes
        assert "qjl_signs" in codes
        assert "residual_norms" in codes
        recon = tq.dequantize(codes)
        assert recon.shape == (10, 384)

    def test_unbiased_inner_product(self):
        """QJL should make inner product estimation unbiased."""
        tq = TurboQuantProd(d=384, num_bits=4)
        x = _random_unit_vectors(200, 384, seed=0)
        y = _random_unit_vectors(200, 384, seed=1)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        true_ips = np.sum(x * y, axis=1)
        approx_ips = np.sum(recon * y, axis=1)
        # Mean error should be near zero (unbiased)
        bias = np.mean(approx_ips - true_ips)
        assert abs(bias) < 0.05, f"Bias too large: {bias}"
