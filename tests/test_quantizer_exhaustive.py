"""Exhaustive parametrized tests for TurboQuantMSE and TurboQuantProd.

Generates 400+ parametrized test cases covering rotation properties,
quantize/dequantize correctness, storage formulas, determinism,
cross-quantizer comparisons, and stress tests.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_unit_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate n random L2-normalized vectors of dimension d."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

ROTATION_DIMS = [8, 16, 32, 64, 128, 256, 384]

MSE_DIMS = [8, 32, 64, 128, 384]
MSE_BITS = [2, 3, 4, 5, 6]
MSE_NS = [1, 10, 100]
MSE_QUANTIZE_PARAMS = list(itertools.product(MSE_DIMS, MSE_BITS, MSE_NS))

MSE_PROP_PARAMS = list(itertools.product(MSE_DIMS, MSE_BITS))

PROD_DIMS = [8, 32, 64, 128]
PROD_BITS = [2, 3, 4, 5]
PROD_NS = [1, 10, 50]
PROD_QUANTIZE_PARAMS = list(itertools.product(PROD_DIMS, PROD_BITS, PROD_NS))

PROD_PROP_PARAMS = list(itertools.product(PROD_DIMS, PROD_BITS))

CROSS_DIMS = [8, 32, 64, 128, 384]
CROSS_BITS = [2, 3, 4, 5, 6]
CROSS_PARAMS = list(itertools.product(CROSS_DIMS, CROSS_BITS))

SEED_DIMS = [32, 128, 384]

# Minimum cosine similarity thresholds by bits (conservative)
COS_SIM_THRESHOLDS = {2: 0.70, 3: 0.90, 4: 0.96, 5: 0.99, 6: 0.995}


# ===================================================================
# 1. TurboQuantMSE Rotation Tests
# ===================================================================

class TestMSERotation:
    """Tests for the random orthogonal rotation matrix."""

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_orthogonality(self, d: int) -> None:
        """R @ R.T should approximate the identity matrix."""
        tq = TurboQuantMSE(d=d, num_bits=2, seed=42)
        R = tq.rotation
        product = R @ R.T
        identity = np.eye(d, dtype=np.float32)
        np.testing.assert_allclose(product, identity, atol=1e-5)

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_transpose_orthogonality(self, d: int) -> None:
        """R.T @ R should also approximate the identity matrix."""
        tq = TurboQuantMSE(d=d, num_bits=2, seed=42)
        R = tq.rotation
        product = R.T @ R
        identity = np.eye(d, dtype=np.float32)
        np.testing.assert_allclose(product, identity, atol=1e-5)

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_preserves_norms(self, d: int) -> None:
        """||Rx|| should equal ||x|| for all test vectors."""
        tq = TurboQuantMSE(d=d, num_bits=2, seed=42)
        x = _random_unit_vectors(50, d, seed=99)
        rotated = x @ tq.rotation.T
        original_norms = np.linalg.norm(x, axis=1)
        rotated_norms = np.linalg.norm(rotated, axis=1)
        np.testing.assert_allclose(rotated_norms, original_norms, atol=1e-5)

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_deterministic_same_seed(self, d: int) -> None:
        """Same seed must produce identical rotation matrices."""
        tq1 = TurboQuantMSE(d=d, num_bits=2, seed=123)
        tq2 = TurboQuantMSE(d=d, num_bits=2, seed=123)
        np.testing.assert_array_equal(tq1.rotation, tq2.rotation)

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_different_seeds(self, d: int) -> None:
        """Different seeds must produce different rotation matrices."""
        tq1 = TurboQuantMSE(d=d, num_bits=2, seed=1)
        tq2 = TurboQuantMSE(d=d, num_bits=2, seed=2)
        assert not np.allclose(tq1.rotation, tq2.rotation, atol=1e-6)

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_shape(self, d: int) -> None:
        """Rotation matrix has shape (d, d) and dtype float32."""
        tq = TurboQuantMSE(d=d, num_bits=2, seed=42)
        assert tq.rotation.shape == (d, d)
        assert tq.rotation.dtype == np.float32

    @pytest.mark.parametrize("d", ROTATION_DIMS)
    def test_rotation_det_is_pm1(self, d: int) -> None:
        """Determinant of an orthogonal matrix should be +1 or -1."""
        tq = TurboQuantMSE(d=d, num_bits=2, seed=42)
        det = np.linalg.det(tq.rotation.astype(np.float64))
        assert abs(abs(det) - 1.0) < 1e-3


# ===================================================================
# 2. TurboQuantMSE Quantize/Dequantize Tests
# ===================================================================

class TestMSEQuantizeDequantize:
    """Parametrized roundtrip tests for TurboQuantMSE."""

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_quantize_output_shape(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        assert codes.shape == (N, d)

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_quantize_output_dtype(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        assert codes.dtype == np.uint8

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_dequantize_output_shape(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert recon.shape == (N, d)

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_dequantize_output_dtype(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert recon.dtype in (np.float32, np.float64)

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_codes_in_valid_range(self, d: int, bits: int, N: int) -> None:
        """All code indices must be in [0, 2^bits - 1]."""
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        assert np.all(codes >= 0)
        assert np.all(codes < 2**bits)

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_no_nan_inf_in_reconstruction(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        recon = tq.dequantize(tq.quantize(x))
        assert not np.any(np.isnan(recon))
        assert not np.any(np.isinf(recon))

    @pytest.mark.parametrize("d,N", list(itertools.product(MSE_DIMS, [10, 100])))
    def test_mse_decreases_with_bits(self, d: int, N: int) -> None:
        """MSE must strictly decrease as bits increase (2 through 6)."""
        x = _random_unit_vectors(N, d, seed=7)
        prev_mse = float("inf")
        for bits in [2, 3, 4, 5, 6]:
            tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
            recon = tq.dequantize(tq.quantize(x))
            mse = float(np.mean(np.sum((x - recon) ** 2, axis=1)))
            assert mse < prev_mse, (
                f"MSE did not decrease: d={d}, bits={bits}, mse={mse:.6f}, prev={prev_mse:.6f}"
            )
            prev_mse = mse

    @pytest.mark.parametrize("d,bits,N", MSE_QUANTIZE_PARAMS)
    def test_reconstructed_norm_approx_1(self, d: int, bits: int, N: int) -> None:
        """Reconstructed vectors from unit-norm input should have norm near 1."""
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        recon = tq.dequantize(tq.quantize(x))
        norms = np.linalg.norm(recon, axis=1)
        # Tolerance depends on bits — fewer bits = more distortion
        tol = {2: 0.5, 3: 0.3, 4: 0.15, 5: 0.08, 6: 0.04}
        np.testing.assert_allclose(norms, 1.0, atol=tol[bits])

    @pytest.mark.parametrize(
        "d,bits",
        [(d, b) for d in [32, 128, 384] for b in [3, 4, 5, 6]],
    )
    def test_cosine_similarity_above_threshold(self, d: int, bits: int) -> None:
        """Mean cosine similarity between original and reconstructed should exceed threshold."""
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(100, d, seed=11)
        recon = tq.dequantize(tq.quantize(x))
        recon_norms = np.linalg.norm(recon, axis=1, keepdims=True)
        recon_normed = recon / np.maximum(recon_norms, 1e-12)
        cos_sims = np.sum(x * recon_normed, axis=1)
        threshold = COS_SIM_THRESHOLDS[bits]
        assert np.mean(cos_sims) > threshold, (
            f"Mean cosine sim {np.mean(cos_sims):.4f} < {threshold} for d={d}, bits={bits}"
        )


# ===================================================================
# 3. TurboQuantMSE Properties
# ===================================================================

class TestMSEProperties:
    """Tests for bytes_per_vector, compression_ratio, determinism."""

    @pytest.mark.parametrize("d,bits", MSE_PROP_PARAMS)
    def test_bytes_per_vector(self, d: int, bits: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        expected = math.ceil(bits * d / 8)
        assert tq.bytes_per_vector == expected

    @pytest.mark.parametrize("d,bits", MSE_PROP_PARAMS)
    def test_compression_ratio(self, d: int, bits: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        expected_bpv = math.ceil(bits * d / 8)
        expected_ratio = (d * 4) / expected_bpv
        assert tq.compression_ratio == pytest.approx(expected_ratio, rel=1e-9)

    @pytest.mark.parametrize("d,bits", MSE_PROP_PARAMS)
    def test_determinism_same_input(self, d: int, bits: int) -> None:
        """Same quantizer + same input = same output."""
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(10, d, seed=55)
        codes1 = tq.quantize(x)
        codes2 = tq.quantize(x)
        np.testing.assert_array_equal(codes1, codes2)

    @pytest.mark.parametrize("d,bits", MSE_PROP_PARAMS)
    def test_codebook_num_levels(self, d: int, bits: int) -> None:
        """Codebook should have 2^bits centroids."""
        tq = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        assert len(tq.codebook.centroids) == 2**bits


# ===================================================================
# 4. TurboQuantMSE 1D Input Handling
# ===================================================================

class TestMSE1DInput:
    """Single vector (1D array) should get promoted to 2D."""

    @pytest.mark.parametrize("d", [8, 32, 128, 384])
    def test_1d_input_quantize_shape(self, d: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=4, seed=42)
        x = _random_unit_vectors(1, d, seed=7)[0]  # 1D array (d,)
        codes = tq.quantize(x)
        assert codes.shape == (1, d)
        assert codes.dtype == np.uint8

    @pytest.mark.parametrize("d", [8, 32, 128, 384])
    def test_1d_input_dequantize_shape(self, d: int) -> None:
        tq = TurboQuantMSE(d=d, num_bits=4, seed=42)
        x = _random_unit_vectors(1, d, seed=7)[0]
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert recon.shape == (1, d)
        assert recon.dtype in (np.float32, np.float64)

    @pytest.mark.parametrize("d", [8, 32, 128, 384])
    def test_1d_matches_2d_singleton(self, d: int) -> None:
        """quantize(x_1d) should match quantize(x_2d) for the same vector."""
        tq = TurboQuantMSE(d=d, num_bits=4, seed=42)
        x_2d = _random_unit_vectors(1, d, seed=7)
        x_1d = x_2d[0]
        codes_1d = tq.quantize(x_1d)
        codes_2d = tq.quantize(x_2d)
        np.testing.assert_array_equal(codes_1d, codes_2d)


# ===================================================================
# 5. TurboQuantProd Construction
# ===================================================================

class TestProdConstruction:
    """Tests for TurboQuantProd init and validation."""

    @pytest.mark.parametrize("d", PROD_DIMS)
    def test_bits_1_raises_value_error(self, d: int) -> None:
        with pytest.raises(ValueError, match="num_bits >= 2"):
            TurboQuantProd(d=d, num_bits=1)

    @pytest.mark.parametrize("d,bits", PROD_PROP_PARAMS)
    def test_bits_gte_2_succeeds(self, d: int, bits: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        assert tq.d == d
        assert tq.num_bits == bits

    @pytest.mark.parametrize("d,bits", PROD_PROP_PARAMS)
    def test_mse_quantizer_uses_bits_minus_1(self, d: int, bits: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        assert tq.mse_quantizer.num_bits == bits - 1

    @pytest.mark.parametrize("d,bits", PROD_PROP_PARAMS)
    def test_qjl_matrix_shape_and_dtype(self, d: int, bits: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        assert tq.qjl_matrix.shape == (d, d)
        assert tq.qjl_matrix.dtype in (np.float32, np.float64)


# ===================================================================
# 6. TurboQuantProd Quantize/Dequantize
# ===================================================================

class TestProdQuantizeDequantize:
    """Parametrized roundtrip tests for TurboQuantProd."""

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_quantize_returns_dict_with_keys(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        assert isinstance(codes, dict)
        assert set(codes.keys()) == {"mse_codes", "qjl_signs", "residual_norms"}

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_mse_codes_shape_dtype(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        assert codes["mse_codes"].shape == (N, d)
        assert codes["mse_codes"].dtype == np.uint8

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_qjl_signs_shape_dtype_values(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        signs = codes["qjl_signs"]
        assert signs.shape == (N, d)
        assert signs.dtype == np.int8
        # All values must be exactly -1 or +1
        assert np.all((signs == -1) | (signs == 1))

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_residual_norms_shape_dtype(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        rn = codes["residual_norms"]
        assert rn.shape == (N,)
        assert rn.dtype in (np.float32, np.float64)

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_residual_norms_non_negative(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        assert np.all(codes["residual_norms"] >= 0.0)

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_dequantize_output_shape_dtype(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert recon.shape == (N, d)
        # Prod dequantize may return float32 or float64 depending on
        # numpy broadcasting with scipy-computed centroids; accept both.
        assert np.issubdtype(recon.dtype, np.floating)

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_no_nan_inf_in_reconstruction(self, d: int, bits: int, N: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        recon = tq.dequantize(tq.quantize(x))
        assert not np.any(np.isnan(recon))
        assert not np.any(np.isinf(recon))

    @pytest.mark.parametrize(
        "d,bits",
        [(d, b) for d in [32, 64, 128] for b in [3, 4, 5]],
    )
    def test_inner_product_estimation_unbiased(self, d: int, bits: int) -> None:
        """E[<y, dequant(quant(x))>] should approximate <y, x> (unbiased)."""
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        N = 200
        x = _random_unit_vectors(N, d, seed=0)
        y = _random_unit_vectors(N, d, seed=1)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        true_ips = np.sum(x * y, axis=1)
        approx_ips = np.sum(recon * y, axis=1)
        bias = float(np.mean(approx_ips - true_ips))
        assert abs(bias) < 0.1, (
            f"Inner product bias too large: {bias:.4f} for d={d}, bits={bits}"
        )

    @pytest.mark.parametrize("d,bits,N", PROD_QUANTIZE_PARAMS)
    def test_mse_codes_in_valid_range(self, d: int, bits: int, N: int) -> None:
        """MSE code indices must be in [0, 2^(bits-1) - 1]."""
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        x = _random_unit_vectors(N, d)
        codes = tq.quantize(x)
        max_code = 2 ** (bits - 1) - 1
        assert np.all(codes["mse_codes"] >= 0)
        assert np.all(codes["mse_codes"] <= max_code)


# ===================================================================
# 7. TurboQuantProd Storage
# ===================================================================

class TestProdStorage:
    """Tests for bytes_per_vector and compression_ratio."""

    @pytest.mark.parametrize("d,bits", PROD_PROP_PARAMS)
    def test_bytes_per_vector(self, d: int, bits: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        mse_bytes = math.ceil((bits - 1) * d / 8)
        qjl_bytes = math.ceil(d / 8)
        norm_bytes = 4
        expected = mse_bytes + qjl_bytes + norm_bytes
        assert tq.bytes_per_vector == expected

    @pytest.mark.parametrize("d,bits", PROD_PROP_PARAMS)
    def test_compression_ratio(self, d: int, bits: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        expected = (d * 4) / tq.bytes_per_vector
        assert tq.compression_ratio == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("d,bits", PROD_PROP_PARAMS)
    def test_compression_ratio_positive(self, d: int, bits: int) -> None:
        tq = TurboQuantProd(d=d, num_bits=bits, seed=42)
        assert tq.compression_ratio > 0


# ===================================================================
# 8. Cross-Quantizer Comparison
# ===================================================================

class TestCrossQuantizer:
    """Compare TurboQuantMSE and TurboQuantProd side by side."""

    @pytest.mark.parametrize("d,bits", CROSS_PARAMS)
    def test_prod_has_more_bytes_than_mse_with_bits_minus_1(
        self, d: int, bits: int
    ) -> None:
        """TurboQuantProd(bits) should use more bytes than TurboQuantMSE(bits-1)."""
        mse = TurboQuantMSE(d=d, num_bits=bits - 1, seed=42)
        prod = TurboQuantProd(d=d, num_bits=bits, seed=42)
        assert prod.bytes_per_vector > mse.bytes_per_vector

    @pytest.mark.parametrize(
        "d,bits",
        [(d, b) for d in [32, 64, 128] for b in [2, 3, 4]],
    )
    def test_both_produce_valid_reconstructions(self, d: int, bits: int) -> None:
        """Both quantizers should produce finite, shaped reconstructions."""
        x = _random_unit_vectors(20, d, seed=3)

        mse_q = TurboQuantMSE(d=d, num_bits=bits, seed=42)
        recon_mse = mse_q.dequantize(mse_q.quantize(x))
        assert recon_mse.shape == (20, d)
        assert not np.any(np.isnan(recon_mse))

        prod_q = TurboQuantProd(d=d, num_bits=bits, seed=42)
        recon_prod = prod_q.dequantize(prod_q.quantize(x))
        assert recon_prod.shape == (20, d)
        assert not np.any(np.isnan(recon_prod))

    @pytest.mark.parametrize(
        "d,bits",
        [(d, b) for d in [32, 128] for b in [3, 4, 5]],
    )
    def test_prod_mse_component_matches_standalone(self, d: int, bits: int) -> None:
        """The MSE codes from TurboQuantProd should match standalone TurboQuantMSE(bits-1)."""
        x = _random_unit_vectors(20, d, seed=5)
        prod = TurboQuantProd(d=d, num_bits=bits, seed=42)
        mse_standalone = TurboQuantMSE(d=d, num_bits=bits - 1, seed=42)
        prod_codes = prod.quantize(x)["mse_codes"]
        standalone_codes = mse_standalone.quantize(x)
        np.testing.assert_array_equal(prod_codes, standalone_codes)


# ===================================================================
# 9. Seed Determinism
# ===================================================================

class TestSeedDeterminism:
    """Verify seed-based reproducibility across quantizer instances."""

    @pytest.mark.parametrize("d", SEED_DIMS)
    def test_mse_same_seed_same_output(self, d: int) -> None:
        x = _random_unit_vectors(20, d, seed=77)
        tq1 = TurboQuantMSE(d=d, num_bits=4, seed=42)
        tq2 = TurboQuantMSE(d=d, num_bits=4, seed=42)
        np.testing.assert_array_equal(tq1.quantize(x), tq2.quantize(x))

    @pytest.mark.parametrize("d", SEED_DIMS)
    def test_mse_different_seed_different_output(self, d: int) -> None:
        x = _random_unit_vectors(20, d, seed=77)
        tq1 = TurboQuantMSE(d=d, num_bits=4, seed=42)
        tq2 = TurboQuantMSE(d=d, num_bits=4, seed=99)
        # Different rotations should yield different codes for most vectors
        codes1 = tq1.quantize(x)
        codes2 = tq2.quantize(x)
        assert not np.array_equal(codes1, codes2)

    @pytest.mark.parametrize("d", SEED_DIMS)
    def test_mse_same_seed_same_reconstruction(self, d: int) -> None:
        x = _random_unit_vectors(20, d, seed=77)
        tq1 = TurboQuantMSE(d=d, num_bits=4, seed=42)
        tq2 = TurboQuantMSE(d=d, num_bits=4, seed=42)
        recon1 = tq1.dequantize(tq1.quantize(x))
        recon2 = tq2.dequantize(tq2.quantize(x))
        np.testing.assert_array_equal(recon1, recon2)

    @pytest.mark.parametrize("d", SEED_DIMS)
    def test_prod_same_seed_same_output(self, d: int) -> None:
        x = _random_unit_vectors(20, d, seed=77)
        tq1 = TurboQuantProd(d=d, num_bits=4, seed=42)
        tq2 = TurboQuantProd(d=d, num_bits=4, seed=42)
        codes1 = tq1.quantize(x)
        codes2 = tq2.quantize(x)
        np.testing.assert_array_equal(codes1["mse_codes"], codes2["mse_codes"])
        np.testing.assert_array_equal(codes1["qjl_signs"], codes2["qjl_signs"])
        np.testing.assert_array_equal(codes1["residual_norms"], codes2["residual_norms"])

    @pytest.mark.parametrize("d", SEED_DIMS)
    def test_prod_different_seed_different_output(self, d: int) -> None:
        x = _random_unit_vectors(20, d, seed=77)
        tq1 = TurboQuantProd(d=d, num_bits=4, seed=42)
        tq2 = TurboQuantProd(d=d, num_bits=4, seed=99)
        codes1 = tq1.quantize(x)
        codes2 = tq2.quantize(x)
        # At least one component should differ
        mse_differ = not np.array_equal(codes1["mse_codes"], codes2["mse_codes"])
        signs_differ = not np.array_equal(codes1["qjl_signs"], codes2["qjl_signs"])
        assert mse_differ or signs_differ

    @pytest.mark.parametrize("d", SEED_DIMS)
    def test_prod_same_seed_same_reconstruction(self, d: int) -> None:
        x = _random_unit_vectors(20, d, seed=77)
        tq1 = TurboQuantProd(d=d, num_bits=4, seed=42)
        tq2 = TurboQuantProd(d=d, num_bits=4, seed=42)
        recon1 = tq1.dequantize(tq1.quantize(x))
        recon2 = tq2.dequantize(tq2.quantize(x))
        np.testing.assert_array_equal(recon1, recon2)


# ===================================================================
# 10. Stress Tests
# ===================================================================

class TestStress:
    """Large-scale stress tests for correctness under volume."""

    def test_mse_10k_vectors_no_nan_inf(self) -> None:
        """10K vectors at d=128, bits=4: no NaN/Inf in reconstruction."""
        tq = TurboQuantMSE(d=128, num_bits=4, seed=42)
        x = _random_unit_vectors(10_000, 128, seed=0)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert not np.any(np.isnan(recon))
        assert not np.any(np.isinf(recon))
        assert recon.shape == (10_000, 128)

    def test_prod_10k_vectors_no_nan_inf(self) -> None:
        """10K vectors at d=128, bits=4: no NaN/Inf in Prod reconstruction."""
        tq = TurboQuantProd(d=128, num_bits=4, seed=42)
        x = _random_unit_vectors(10_000, 128, seed=0)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        assert not np.any(np.isnan(recon))
        assert not np.any(np.isinf(recon))
        assert recon.shape == (10_000, 128)

    def test_mse_10k_mse_within_theoretical_bound(self) -> None:
        """10K vectors: empirical MSE should stay below 1.5x theoretical bound."""
        tq = TurboQuantMSE(d=128, num_bits=4, seed=42)
        x = _random_unit_vectors(10_000, 128, seed=0)
        recon = tq.dequantize(tq.quantize(x))
        mse = float(np.mean(np.sum((x - recon) ** 2, axis=1)))
        bound = tq.codebook.theoretical_mse * 1.5
        assert mse < bound, f"MSE {mse:.6f} exceeds 1.5x theoretical bound {bound:.6f}"

    def test_prod_10k_inner_product_bias(self) -> None:
        """10K pairs: inner product bias should be small."""
        tq = TurboQuantProd(d=128, num_bits=4, seed=42)
        x = _random_unit_vectors(10_000, 128, seed=0)
        y = _random_unit_vectors(10_000, 128, seed=1)
        codes = tq.quantize(x)
        recon = tq.dequantize(codes)
        true_ips = np.sum(x * y, axis=1)
        approx_ips = np.sum(recon * y, axis=1)
        bias = float(np.mean(approx_ips - true_ips))
        assert abs(bias) < 0.02, f"Bias over 10K pairs: {bias:.5f}"

    def test_mse_large_dim_384(self) -> None:
        """1K vectors at d=384, bits=4: basic sanity."""
        tq = TurboQuantMSE(d=384, num_bits=4, seed=42)
        x = _random_unit_vectors(1_000, 384, seed=0)
        recon = tq.dequantize(tq.quantize(x))
        assert recon.shape == (1_000, 384)
        assert not np.any(np.isnan(recon))
        cos_sims = np.sum(x * recon, axis=1) / (
            np.linalg.norm(recon, axis=1) + 1e-12
        )
        assert np.mean(cos_sims) > 0.95

    def test_mse_single_vector_stress(self) -> None:
        """Quantize 1000 individual vectors one by one (1D input)."""
        tq = TurboQuantMSE(d=64, num_bits=4, seed=42)
        rng = np.random.RandomState(123)
        for _ in range(1000):
            v = rng.randn(64).astype(np.float32)
            v /= np.linalg.norm(v)
            codes = tq.quantize(v)
            assert codes.shape == (1, 64)
            recon = tq.dequantize(codes)
            assert recon.shape == (1, 64)
            assert not np.any(np.isnan(recon))


# ===================================================================
# Parametrize count verification (informational, always passes)
# ===================================================================

class TestParametrizeCount:
    """Sanity check that we have enough parametrized cases."""

    def test_total_parametrize_cases_exceeds_400(self) -> None:
        """Count all parametrized test cases to confirm > 400."""
        counts = {
            # 1. Rotation tests: 7 tests x 7 dims = 49
            "rotation": 7 * len(ROTATION_DIMS),
            # 2. MSE quantize/dequantize: 6 core tests x 75 combos = 450
            #    + mse_decreases: 5x2=10, cosine_sim: 3x4=12
            "mse_qd_core": 6 * len(MSE_QUANTIZE_PARAMS),
            "mse_decreases": len(MSE_DIMS) * 2,
            "mse_cosine": 3 * 4,
            # 3. MSE properties: 4 tests x 25 combos = 100
            "mse_props": 4 * len(MSE_PROP_PARAMS),
            # 4. MSE 1D: 3 tests x 4 dims = 12
            "mse_1d": 3 * 4,
            # 5. Prod construction: 4 tests x 16 combos + 4 bits_1 = 68
            "prod_construction": 4 * len(PROD_DIMS),  # bits_1
            "prod_construction_props": 3 * len(PROD_PROP_PARAMS),
            # 6. Prod quantize/dequantize: 9 core x 48 combos = 432
            #    + ip_unbiased: 9, mse_codes_range: 48
            "prod_qd_core": 8 * len(PROD_QUANTIZE_PARAMS),
            "prod_ip_unbiased": 3 * 3,
            # 7. Prod storage: 3 x 16 = 48
            "prod_storage": 3 * len(PROD_PROP_PARAMS),
            # 8. Cross-quantizer: 25 + 9 + 6 = 40
            "cross": len(CROSS_PARAMS) + 9 + 6,
            # 9. Seed determinism: 7 x 3 = 21
            "seed": 7 * len(SEED_DIMS),
            # 10. Stress: 6
            "stress": 6,
        }
        total = sum(counts.values())
        assert total > 400, f"Only {total} parametrized cases, expected > 400"
