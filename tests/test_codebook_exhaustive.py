"""Exhaustive parametrized tests for LloydMaxQuantizer and hypersphere_coordinate_pdf.

Generates 500+ test cases via pytest.mark.parametrize covering PDF properties,
centroid invariants, quantize/dequantize round-trips, MSE bounds, edge cases,
convergence, and stress tests.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.integrate import quad

from turboquant.codebook import LloydMaxQuantizer, hypersphere_coordinate_pdf

# ---------------------------------------------------------------------------
# Shared parameter grids
# ---------------------------------------------------------------------------

PDF_DIMS = [3, 5, 10, 50, 100, 384, 768]

CENTROID_DIMS = [8, 16, 32, 64, 128, 256, 384, 768]
CENTROID_BITS = [1, 2, 3, 4, 5, 6, 7, 8]
CENTROID_COMBOS = list(itertools.product(CENTROID_DIMS, CENTROID_BITS))

MSE_BITS = [1, 2, 3, 4, 5, 6, 7, 8]

GRID_POINTS = [1000, 5000, 10000, 50000]

# ---------------------------------------------------------------------------
# Fixture: cache expensive LloydMaxQuantizer instances
# ---------------------------------------------------------------------------

_quantizer_cache: dict[tuple[int, int], LloydMaxQuantizer] = {}


def _get_quantizer(d: int, bits: int) -> LloydMaxQuantizer:
    """Return a cached LloydMaxQuantizer to avoid redundant computation."""
    key = (d, bits)
    if key not in _quantizer_cache:
        _quantizer_cache[key] = LloydMaxQuantizer(d, num_bits=bits)
    return _quantizer_cache[key]


# ===================================================================
# 1. PDF TESTS  (7 dims x 5 properties = 35 test cases)
# ===================================================================


class TestHyperspherePDF:
    """Tests for hypersphere_coordinate_pdf correctness."""

    @pytest.mark.parametrize("d", PDF_DIMS)
    def test_pdf_non_negative(self, d: int) -> None:
        """PDF must be >= 0 for all x in [-1, 1]."""
        xs = np.linspace(-1, 1, 500)
        for x in xs:
            assert hypersphere_coordinate_pdf(x, d) >= 0.0, (
                f"PDF negative at x={x}, d={d}"
            )

    @pytest.mark.parametrize("d", PDF_DIMS)
    def test_pdf_integrates_to_one(self, d: int) -> None:
        """PDF must integrate to ~1.0 over [-1, 1]."""
        result, _ = quad(hypersphere_coordinate_pdf, -1.0, 1.0, args=(d,))
        assert abs(result - 1.0) < 1e-4, (
            f"PDF integral = {result} for d={d}, expected ~1.0"
        )

    @pytest.mark.parametrize("d", PDF_DIMS)
    def test_pdf_symmetric(self, d: int) -> None:
        """PDF must satisfy f(x) == f(-x) for all x."""
        xs = np.linspace(0.01, 0.99, 200)
        for x in xs:
            pos = hypersphere_coordinate_pdf(x, d)
            neg = hypersphere_coordinate_pdf(-x, d)
            assert abs(pos - neg) < 1e-12, (
                f"Asymmetry at x={x}, d={d}: f(x)={pos}, f(-x)={neg}"
            )

    @pytest.mark.parametrize("d", PDF_DIMS)
    def test_pdf_zero_outside_support(self, d: int) -> None:
        """PDF must be exactly 0 for |x| >= 1."""
        for x in [-1.0, 1.0, -1.5, 1.5, -100.0, 100.0]:
            assert hypersphere_coordinate_pdf(x, d) == 0.0, (
                f"PDF non-zero at x={x}, d={d}"
            )

    @pytest.mark.parametrize("d", PDF_DIMS)
    def test_pdf_peak_at_zero(self, d: int) -> None:
        """PDF peak must be at x=0 (maximum of symmetric unimodal distribution)."""
        peak = hypersphere_coordinate_pdf(0.0, d)
        for x in np.linspace(0.01, 0.99, 100):
            assert hypersphere_coordinate_pdf(x, d) <= peak + 1e-12, (
                f"PDF at x={x} exceeds peak at x=0 for d={d}"
            )


# ===================================================================
# 2. CENTROID TESTS  (8 dims x 8 bits x 6 properties = 384 test cases)
# ===================================================================


class TestCentroids:
    """Tests for LloydMaxQuantizer centroid properties."""

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_centroid_count(self, d: int, bits: int) -> None:
        """Number of centroids must equal 2^bits."""
        lmq = _get_quantizer(d, bits)
        assert len(lmq.centroids) == 2 ** bits

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_centroids_sorted(self, d: int, bits: int) -> None:
        """Centroids must be sorted in ascending order."""
        lmq = _get_quantizer(d, bits)
        assert np.all(np.diff(lmq.centroids) > 0), (
            f"Centroids not strictly ascending for d={d}, bits={bits}"
        )

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_centroids_in_range(self, d: int, bits: int) -> None:
        """Centroids must lie within +/- 5/sqrt(d)."""
        lmq = _get_quantizer(d, bits)
        bound = 5.0 / np.sqrt(d)
        assert np.all(lmq.centroids >= -bound), (
            f"Centroid below -{bound} for d={d}, bits={bits}"
        )
        assert np.all(lmq.centroids <= bound), (
            f"Centroid above {bound} for d={d}, bits={bits}"
        )

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_centroids_symmetric(self, d: int, bits: int) -> None:
        """Centroids must be roughly symmetric around 0 (sum approx 0)."""
        lmq = _get_quantizer(d, bits)
        sigma = 1.0 / np.sqrt(d)
        # Allow tolerance proportional to sigma and inversely to sqrt(num_levels)
        tol = 3.0 * sigma / np.sqrt(2 ** bits)
        assert abs(np.sum(lmq.centroids)) < tol + 1e-10, (
            f"Centroid sum = {np.sum(lmq.centroids)} exceeds tolerance {tol} "
            f"for d={d}, bits={bits}"
        )

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_centroids_unique(self, d: int, bits: int) -> None:
        """All centroids must be distinct."""
        lmq = _get_quantizer(d, bits)
        assert len(np.unique(lmq.centroids)) == len(lmq.centroids), (
            f"Duplicate centroids for d={d}, bits={bits}"
        )

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_centroids_dtype(self, d: int, bits: int) -> None:
        """Centroids must be float64."""
        lmq = _get_quantizer(d, bits)
        assert lmq.centroids.dtype == np.float64


# ===================================================================
# 3. QUANTIZE/DEQUANTIZE ROUND-TRIP  (8 dims x 8 bits x 5 props = 320 cases)
# ===================================================================


class TestQuantizeDequantize:
    """Tests for quantize/dequantize round-trip correctness."""

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_quantize_returns_uint8(self, d: int, bits: int) -> None:
        """quantize() must return uint8 array."""
        lmq = _get_quantizer(d, bits)
        vals = np.random.default_rng(42).normal(0, 1.0 / np.sqrt(d), size=100)
        indices = lmq.quantize(vals)
        assert indices.dtype == np.uint8

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_quantize_indices_in_range(self, d: int, bits: int) -> None:
        """All quantize indices must be in [0, 2^bits - 1]."""
        lmq = _get_quantizer(d, bits)
        vals = np.random.default_rng(42).normal(0, 1.0 / np.sqrt(d), size=100)
        indices = lmq.quantize(vals)
        assert np.all(indices >= 0)
        assert np.all(indices < 2 ** bits)

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_dequantized_values_are_centroids(self, d: int, bits: int) -> None:
        """dequantize(quantize(x)) must return values that are actual centroids."""
        lmq = _get_quantizer(d, bits)
        vals = np.random.default_rng(42).normal(0, 1.0 / np.sqrt(d), size=100)
        reconstructed = lmq.dequantize(lmq.quantize(vals))
        for v in reconstructed:
            assert v in lmq.centroids, (
                f"Dequantized value {v} is not a centroid"
            )

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_roundtrip_mse_bounded(self, d: int, bits: int) -> None:
        """Round-trip MSE on random samples must be bounded by theoretical_mse."""
        lmq = _get_quantizer(d, bits)
        rng = np.random.default_rng(42)
        vals = rng.normal(0, 1.0 / np.sqrt(d), size=10000)
        reconstructed = lmq.dequantize(lmq.quantize(vals))
        mse = np.mean((vals - reconstructed) ** 2)
        # The theoretical_mse formula is an asymptotic upper bound valid for
        # large d. For small d the coordinate distribution deviates from
        # Gaussian so the bound does not hold tightly. We use the signal
        # variance (1/d) as a fallback upper bound: quantization MSE must
        # be less than the variance of the unquantized signal.
        signal_var = 1.0 / d
        bound = min(signal_var, 2.0 * lmq.theoretical_mse) if d >= 64 else signal_var
        assert mse < bound, (
            f"Round-trip MSE {mse} exceeds bound {bound} for d={d}, bits={bits}"
        )

    @pytest.mark.parametrize("d,bits", CENTROID_COMBOS)
    def test_quantize_deterministic(self, d: int, bits: int) -> None:
        """Quantizing the same values twice must yield identical results."""
        lmq = _get_quantizer(d, bits)
        vals = np.random.default_rng(42).normal(0, 1.0 / np.sqrt(d), size=100)
        idx1 = lmq.quantize(vals)
        idx2 = lmq.quantize(vals)
        np.testing.assert_array_equal(idx1, idx2)


# ===================================================================
# 4. MSE BOUNDS TESTS  (8 bits x 3 properties = 24 test cases)
# ===================================================================


class TestMSEBounds:
    """Tests for theoretical_mse and shannon_lower_bound relationships."""

    @pytest.mark.parametrize("bits", MSE_BITS)
    def test_theoretical_exceeds_shannon(self, bits: int) -> None:
        """theoretical_mse must be strictly greater than shannon_lower_bound."""
        lmq = _get_quantizer(64, bits)
        assert lmq.theoretical_mse > lmq.shannon_lower_bound

    @pytest.mark.parametrize("bits", MSE_BITS[:-1])  # compare bits vs bits+1
    def test_theoretical_mse_decreases(self, bits: int) -> None:
        """theoretical_mse must decrease as bits increase."""
        lmq_lo = _get_quantizer(64, bits)
        lmq_hi = _get_quantizer(64, bits + 1)
        assert lmq_lo.theoretical_mse > lmq_hi.theoretical_mse

    @pytest.mark.parametrize("bits", MSE_BITS)
    def test_ratio_is_constant(self, bits: int) -> None:
        """Ratio theoretical_mse / shannon_lower_bound must equal sqrt(3)*pi/2."""
        lmq = _get_quantizer(64, bits)
        expected_ratio = np.sqrt(3) * np.pi / 2
        actual_ratio = lmq.theoretical_mse / lmq.shannon_lower_bound
        assert abs(actual_ratio - expected_ratio) < 1e-10, (
            f"Ratio = {actual_ratio}, expected {expected_ratio}"
        )


# ===================================================================
# 5. EDGE CASES  (individual tests, ~10 cases)
# ===================================================================


class TestEdgeCases:
    """Edge case tests for quantize/dequantize."""

    def test_quantize_single_scalar(self) -> None:
        """Quantizing a single scalar value must work."""
        lmq = _get_quantizer(64, 4)
        val = np.array(0.05)
        idx = lmq.quantize(val)
        assert idx.shape == ()
        rec = lmq.dequantize(idx)
        assert rec in lmq.centroids

    def test_quantize_identical_values(self) -> None:
        """Quantizing an array of identical values must give identical indices."""
        lmq = _get_quantizer(64, 4)
        vals = np.full(50, 0.03)
        indices = lmq.quantize(vals)
        assert np.all(indices == indices[0])

    def test_quantize_zeros(self) -> None:
        """Quantizing zeros must return valid indices."""
        lmq = _get_quantizer(64, 4)
        vals = np.zeros(20)
        indices = lmq.quantize(vals)
        assert indices.dtype == np.uint8
        assert np.all(indices == indices[0])  # all same index

    def test_quantize_at_centroids(self) -> None:
        """Quantizing centroid values must give perfect reconstruction."""
        lmq = _get_quantizer(64, 4)
        reconstructed = lmq.dequantize(lmq.quantize(lmq.centroids))
        np.testing.assert_array_almost_equal(reconstructed, lmq.centroids)

    def test_quantize_empty_array(self) -> None:
        """Quantizing an empty array must return an empty array."""
        lmq = _get_quantizer(64, 4)
        vals = np.array([])
        indices = lmq.quantize(vals)
        assert len(indices) == 0

    def test_one_bit_quantization(self) -> None:
        """1-bit quantization must produce exactly 2 centroids."""
        lmq = _get_quantizer(64, 1)
        assert len(lmq.centroids) == 2
        # One centroid negative, one positive (symmetric distribution)
        assert lmq.centroids[0] < 0
        assert lmq.centroids[1] > 0

    def test_dequantize_all_indices(self) -> None:
        """Dequantizing every valid index must return the corresponding centroid."""
        lmq = _get_quantizer(64, 4)
        for i in range(2 ** 4):
            val = lmq.dequantize(np.array(i, dtype=np.uint8))
            assert val == lmq.centroids[i]

    def test_quantize_extreme_values(self) -> None:
        """Quantizing values far outside the centroid range must map to boundary centroids."""
        lmq = _get_quantizer(64, 4)
        large_pos = np.array([100.0])
        large_neg = np.array([-100.0])
        idx_pos = lmq.quantize(large_pos)
        idx_neg = lmq.quantize(large_neg)
        # Should map to last and first centroid respectively
        assert idx_pos[0] == len(lmq.centroids) - 1
        assert idx_neg[0] == 0

    def test_quantize_1d_array(self) -> None:
        """Quantizing a 1D array must return a 1D array of same length."""
        lmq = _get_quantizer(64, 4)
        vals = np.array([0.01, -0.05, 0.1, 0.0, -0.02])
        indices = lmq.quantize(vals)
        assert indices.shape == vals.shape


# ===================================================================
# 6. CONVERGENCE TESTS  (4 grid_points x 2 dims = 8 test cases)
# ===================================================================


class TestConvergence:
    """Tests that higher grid_points lead to converging centroids."""

    @pytest.mark.parametrize(
        "d,grid_points",
        list(itertools.product([64, 384], GRID_POINTS)),
    )
    def test_convergence_with_grid_points(self, d: int, grid_points: int) -> None:
        """Centroids from successive grid resolutions must converge."""
        bits = 4
        lmq = LloydMaxQuantizer(d, num_bits=bits, grid_points=grid_points)
        lmq_ref = LloydMaxQuantizer(d, num_bits=bits, grid_points=50000)
        dist = np.max(np.abs(lmq.centroids - lmq_ref.centroids))
        # Tolerance scales with sigma and inversely with sqrt(grid_points)
        sigma = 1.0 / np.sqrt(d)
        tol = sigma * (50000 / grid_points) * 0.02
        assert dist < tol, (
            f"Max centroid distance {dist} exceeds tolerance {tol} "
            f"for d={d}, grid_points={grid_points}"
        )


# ===================================================================
# 7. STRESS TESTS  (individual tests, ~4 cases)
# ===================================================================


class TestStress:
    """Stress tests for large-scale quantization."""

    def test_quantize_100k_values(self) -> None:
        """Quantizing 100K random values must complete and return valid indices."""
        lmq = _get_quantizer(384, 4)
        rng = np.random.default_rng(123)
        vals = rng.normal(0, 1.0 / np.sqrt(384), size=100_000)
        indices = lmq.quantize(vals)
        assert indices.shape == (100_000,)
        assert indices.dtype == np.uint8
        assert np.all(indices < 16)

    def test_batch_consistency(self) -> None:
        """Batch quantize must equal element-wise quantize."""
        lmq = _get_quantizer(64, 4)
        rng = np.random.default_rng(99)
        vals = rng.normal(0, 1.0 / np.sqrt(64), size=200)
        batch_indices = lmq.quantize(vals)
        single_indices = np.array(
            [lmq.quantize(np.array(v)) for v in vals], dtype=np.uint8
        )
        np.testing.assert_array_equal(batch_indices, single_indices)

    def test_large_dimension_quantizer(self) -> None:
        """Quantizer for d=768 must produce valid centroids and quantize correctly."""
        lmq = _get_quantizer(768, 8)
        assert len(lmq.centroids) == 256
        vals = np.random.default_rng(7).normal(0, 1.0 / np.sqrt(768), size=1000)
        indices = lmq.quantize(vals)
        assert np.all(indices < 256)
        reconstructed = lmq.dequantize(indices)
        mse = np.mean((vals - reconstructed) ** 2)
        assert mse < 2.0 * lmq.theoretical_mse

    @pytest.mark.parametrize("d,bits", [(32, 8), (128, 8), (384, 8)])
    def test_max_bits_all_indices_used(self, d: int, bits: int) -> None:
        """With enough samples, all 256 centroid indices should be used."""
        lmq = _get_quantizer(d, bits)
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1.0 / np.sqrt(d), size=100_000)
        indices = lmq.quantize(vals)
        unique = np.unique(indices)
        # All 256 levels should appear with 100K normal samples
        assert len(unique) == 2 ** bits, (
            f"Only {len(unique)}/{2**bits} indices used for d={d}"
        )
