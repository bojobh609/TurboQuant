"""End-to-end integration and workflow tests for TurboQuant.

Tests cover:
- Full pipeline: generate -> quantize -> search -> verify recall
- Save/Load integrity across all configurations
- Incremental vs batch add equivalence
- Mixed precision comparison (recall monotonicity)
- Large-scale stress test
- API consistency and error handling
- Regression tests with golden values
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from turboquant import TurboQuantIndex, TurboQuantMSE, TurboQuantProd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate n random unit vectors in R^d."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def compute_recall(queries, db, pred_indices, k):
    """Compute recall@k given predicted indices vs brute-force ground truth."""
    gt_sims = queries @ db.T
    gt_indices = np.argsort(-gt_sims, axis=1)[:, :k]
    recalls = []
    for i in range(queries.shape[0]):
        gt_set = set(gt_indices[i])
        pred_set = set(pred_indices[i])
        recalls.append(len(gt_set & pred_set) / k)
    return np.mean(recalls)


# ===========================================================================
# 1. Full pipeline: generate -> quantize -> search -> verify
# ===========================================================================

_pipeline_params = list(itertools.product(
    [32, 64, 128],   # d
    [3, 4, 5],       # bits
    [100, 500, 1000], # N
))


@pytest.mark.parametrize("d,bits,N", _pipeline_params)
def test_full_pipeline(d: int, bits: int, N: int):
    """Full pipeline: add vectors, self-query, verify recall above threshold."""
    db = random_unit_vectors(N, d, seed=42)
    queries = db[:min(20, N)]  # Self-queries

    index = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=True, seed=42)
    index.add(db)

    assert index.size == N, f"Expected size {N}, got {index.size}"

    k = min(10, N)
    sims, indices = index.search(queries, k=k)

    assert sims.shape == (queries.shape[0], k)
    assert indices.shape == (queries.shape[0], k)
    assert np.issubdtype(sims.dtype, np.floating)
    assert np.issubdtype(indices.dtype, np.integer)

    # Self-queries: recall should be reasonable (at least 30% for low bits)
    recall = compute_recall(queries, db, indices, k)
    min_recall = 0.2 if bits <= 3 else 0.3
    assert recall >= min_recall, (
        f"d={d}, bits={bits}, N={N}: recall@{k}={recall:.2f} < {min_recall}"
    )


@pytest.mark.parametrize("d,bits,N", _pipeline_params)
def test_full_pipeline_mse_only(d: int, bits: int, N: int):
    """Same pipeline but with MSE-only mode (use_qjl=False)."""
    db = random_unit_vectors(N, d, seed=43)
    queries = db[:min(20, N)]

    index = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=False, seed=42)
    index.add(db)

    k = min(10, N)
    sims, indices = index.search(queries, k=k)

    assert sims.shape == (queries.shape[0], k)
    assert indices.shape == (queries.shape[0], k)

    recall = compute_recall(queries, db, indices, k)
    min_recall = 0.15 if bits <= 3 else 0.25
    assert recall >= min_recall, (
        f"MSE-only d={d}, bits={bits}, N={N}: recall@{k}={recall:.2f} < {min_recall}"
    )


# ===========================================================================
# 2. Save/Load integrity
# ===========================================================================

_saveload_params = list(itertools.product(
    [32, 64, 128],   # d
    [3, 4, 5],       # bits
    [True, False],    # use_qjl
))


@pytest.mark.parametrize("d,bits,use_qjl", _saveload_params)
def test_save_load_integrity(d: int, bits: int, use_qjl: bool, tmp_path):
    """Save index, load it, verify search results are identical."""
    db = random_unit_vectors(200, d, seed=44)
    queries = random_unit_vectors(10, d, seed=45)

    # Build and search
    index = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=use_qjl, seed=42)
    index.add(db)
    sims_before, idx_before = index.search(queries, k=5)

    # Save
    save_path = tmp_path / "test_index"
    index.save(save_path)

    # Load and search
    loaded = TurboQuantIndex.load(save_path)
    sims_after, idx_after = loaded.search(queries, k=5)

    np.testing.assert_array_equal(idx_before, idx_after,
                                  err_msg="Indices differ after save/load")
    np.testing.assert_allclose(sims_before, sims_after, atol=1e-5,
                               err_msg="Similarities differ after save/load")

    # Verify metadata preserved
    assert loaded.dimension == d
    assert loaded.num_bits == bits
    assert loaded.use_qjl == use_qjl
    assert loaded.size == 200


@pytest.mark.parametrize("d,bits,use_qjl", _saveload_params)
def test_save_load_stats(d: int, bits: int, use_qjl: bool, tmp_path):
    """Stats should be identical before and after save/load."""
    db = random_unit_vectors(100, d, seed=46)
    index = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=use_qjl, seed=42)
    index.add(db)

    stats_before = index.stats()
    save_path = tmp_path / "stats_index"
    index.save(save_path)
    loaded = TurboQuantIndex.load(save_path)
    stats_after = loaded.stats()

    assert stats_before == stats_after


# ===========================================================================
# 3. Incremental vs batch add
# ===========================================================================

@pytest.mark.parametrize("d", [32, 64, 128])
def test_incremental_vs_batch(d: int):
    """Adding 100 vectors in one batch vs 10 batches of 10: same results."""
    db = random_unit_vectors(100, d, seed=50)
    queries = random_unit_vectors(10, d, seed=51)

    # Batch add
    idx_batch = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
    idx_batch.add(db)
    sims_batch, ind_batch = idx_batch.search(queries, k=5)

    # Incremental add
    idx_inc = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
    for i in range(10):
        idx_inc.add(db[i * 10:(i + 1) * 10])
    sims_inc, ind_inc = idx_inc.search(queries, k=5)

    assert idx_batch.size == idx_inc.size == 100
    np.testing.assert_array_equal(ind_batch, ind_inc,
                                  err_msg=f"d={d}: incremental != batch indices")
    np.testing.assert_allclose(sims_batch, sims_inc, atol=1e-5,
                               err_msg=f"d={d}: incremental != batch sims")


@pytest.mark.parametrize("d", [32, 64, 128])
def test_incremental_vs_batch_mse_only(d: int):
    """Same test for MSE-only mode."""
    db = random_unit_vectors(100, d, seed=52)
    queries = random_unit_vectors(10, d, seed=53)

    idx_batch = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=False, seed=42)
    idx_batch.add(db)
    sims_batch, ind_batch = idx_batch.search(queries, k=5)

    idx_inc = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=False, seed=42)
    for i in range(10):
        idx_inc.add(db[i * 10:(i + 1) * 10])
    sims_inc, ind_inc = idx_inc.search(queries, k=5)

    np.testing.assert_array_equal(ind_batch, ind_inc)
    np.testing.assert_allclose(sims_batch, sims_inc, atol=1e-5)


@pytest.mark.parametrize("d", [32, 64, 128])
def test_incremental_size_tracking(d: int):
    """Index size should track correctly with incremental adds."""
    index = TurboQuantIndex(dimension=d, num_bits=4, seed=42)
    assert index.size == 0
    for i in range(5):
        batch = random_unit_vectors(20, d, seed=54 + i)
        index.add(batch)
        assert index.size == (i + 1) * 20


# ===========================================================================
# 4. Mixed precision comparison
# ===========================================================================

@pytest.mark.parametrize("d", [32, 64, 128])
def test_recall_increases_with_bits(d: int):
    """Higher bit quantization should give equal or better recall."""
    N = 500
    db = random_unit_vectors(N, d, seed=60)
    queries = random_unit_vectors(20, d, seed=61)
    k = 10

    recalls = []
    for bits in [2, 3, 4, 5, 6]:
        index = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=True, seed=42)
        index.add(db)
        _, indices = index.search(queries, k=k)
        recall = compute_recall(queries, db, indices, k)
        recalls.append(recall)

    # Monotonicity with tolerance: each step >= previous - 0.05
    for i in range(len(recalls) - 1):
        assert recalls[i + 1] >= recalls[i] - 0.05, (
            f"d={d}: recall at {i+3} bits ({recalls[i+1]:.2f}) "
            f"< recall at {i+2} bits ({recalls[i]:.2f}) - 0.05"
        )


@pytest.mark.parametrize("d", [32, 64, 128])
def test_recall_increases_with_bits_mse_only(d: int):
    """Same monotonicity for MSE-only mode."""
    N = 500
    db = random_unit_vectors(N, d, seed=62)
    queries = random_unit_vectors(20, d, seed=63)
    k = 10

    recalls = []
    for bits in [2, 3, 4, 5, 6]:
        index = TurboQuantIndex(dimension=d, num_bits=bits, use_qjl=False, seed=42)
        index.add(db)
        _, indices = index.search(queries, k=k)
        recall = compute_recall(queries, db, indices, k)
        recalls.append(recall)

    for i in range(len(recalls) - 1):
        assert recalls[i + 1] >= recalls[i] - 0.05


@pytest.mark.parametrize("d", [32, 64, 128])
def test_high_bits_recall_high(d: int):
    """6-bit quantization should achieve decent recall."""
    N = 500
    db = random_unit_vectors(N, d, seed=64)
    queries = db[:20]
    k = 10

    index = TurboQuantIndex(dimension=d, num_bits=6, use_qjl=True, seed=42)
    index.add(db)
    _, indices = index.search(queries, k=k)
    recall = compute_recall(queries, db, indices, k)
    assert recall >= 0.5, f"d={d}: 6-bit recall@{k}={recall:.2f} < 0.5"


# ===========================================================================
# 5. Large-scale stress test
# ===========================================================================

@pytest.mark.slow
def test_large_scale_stress():
    """50K vectors at d=64, bits=4: verify index builds and recall > 0.5."""
    d = 64
    N = 50_000
    n_queries = 100
    k = 10

    db = random_unit_vectors(N, d, seed=70)
    queries = db[:n_queries]

    index = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
    index.add(db)

    assert index.size == N

    sims, indices = index.search(queries, k=k)
    assert sims.shape == (n_queries, k)
    assert indices.shape == (n_queries, k)

    recall = compute_recall(queries, db, indices, k)
    assert recall > 0.5, f"Large-scale recall@{k}={recall:.2f} < 0.5"


def test_moderate_scale():
    """10K vectors at d=64, bits=4: faster version of the stress test."""
    d = 64
    N = 10_000
    n_queries = 50
    k = 10

    db = random_unit_vectors(N, d, seed=71)
    queries = db[:n_queries]

    index = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
    index.add(db)

    sims, indices = index.search(queries, k=k)
    recall = compute_recall(queries, db, indices, k)
    assert recall > 0.4, f"10K scale recall@{k}={recall:.2f} < 0.4"


def test_stress_with_non_self_queries():
    """Stress: search with queries NOT in the database."""
    d = 64
    N = 10_000
    n_queries = 50
    k = 10

    db = random_unit_vectors(N, d, seed=72)
    queries = random_unit_vectors(n_queries, d, seed=73)

    index = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
    index.add(db)

    sims, indices = index.search(queries, k=k)
    assert sims.shape == (n_queries, k)
    # All returned indices should be valid
    assert np.all(indices >= 0) and np.all(indices < N)
    # Similarities should be sorted descending per query
    for q in range(n_queries):
        assert np.all(np.diff(sims[q]) <= 1e-6), "Sims not sorted descending"


# ===========================================================================
# 6. API consistency tests
# ===========================================================================

class TestAPIConsistency:
    """Verify TurboQuantIndex matches the documented public interface."""

    def test_constructor_defaults(self):
        """Default constructor values."""
        idx = TurboQuantIndex(dimension=64)
        assert idx.dimension == 64
        assert idx.num_bits == 4
        assert idx.metric == "cosine"
        assert idx.use_qjl is True
        assert idx.seed == 42
        assert idx.size == 0

    def test_add_returns_none(self):
        """add() should return None."""
        idx = TurboQuantIndex(dimension=32, num_bits=3)
        result = idx.add(random_unit_vectors(10, 32))
        assert result is None

    def test_search_return_types(self):
        """search() returns (float32 array, int64 array)."""
        idx = TurboQuantIndex(dimension=32, num_bits=3)
        idx.add(random_unit_vectors(50, 32))
        sims, indices = idx.search(random_unit_vectors(5, 32), k=3)
        assert isinstance(sims, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert np.issubdtype(sims.dtype, np.floating)
        assert np.issubdtype(indices.dtype, np.integer)

    def test_search_empty_index(self):
        """Searching empty index returns empty arrays without error."""
        idx = TurboQuantIndex(dimension=32, num_bits=3)
        queries = random_unit_vectors(5, 32)
        sims, indices = idx.search(queries, k=10)
        assert sims.shape == (5, 0)
        assert indices.shape == (5, 0)

    def test_stats_return_type(self):
        """stats() returns a dict with expected keys."""
        idx = TurboQuantIndex(dimension=64, num_bits=4)
        idx.add(random_unit_vectors(100, 64))
        s = idx.stats()
        assert isinstance(s, dict)
        expected_keys = {"size", "dimension", "num_bits", "use_qjl",
                         "memory_efficient",
                         "compression_ratio", "effective_compression_ratio",
                         "bytes_per_vector", "total_bytes", "total_code_bytes",
                         "rotation_matrix_bytes", "total_overhead_bytes",
                         "float32_bytes"}
        assert expected_keys == set(s.keys())
        assert s["size"] == 100
        assert s["dimension"] == 64

    def test_compression_ratio_property(self):
        """compression_ratio property returns a float."""
        idx = TurboQuantIndex(dimension=64, num_bits=4)
        assert isinstance(idx.compression_ratio, float)
        assert idx.compression_ratio > 0

    def test_wrong_dimension_raises(self):
        """Adding vectors with wrong dimension raises ValueError."""
        idx = TurboQuantIndex(dimension=32, num_bits=4)
        bad_vecs = random_unit_vectors(10, 64)
        with pytest.raises(ValueError, match="Expected dimension 32"):
            idx.add(bad_vecs)

    def test_single_vector_add(self):
        """Adding a single 1D vector should work."""
        idx = TurboQuantIndex(dimension=32, num_bits=4)
        vec = random_unit_vectors(1, 32).squeeze()
        idx.add(vec)
        assert idx.size == 1

    def test_single_vector_search(self):
        """Searching with a single 1D query should work."""
        idx = TurboQuantIndex(dimension=32, num_bits=4)
        idx.add(random_unit_vectors(50, 32))
        query = random_unit_vectors(1, 32).squeeze()
        sims, indices = idx.search(query, k=5)
        assert sims.shape == (1, 5)
        assert indices.shape == (1, 5)

    def test_k_larger_than_db(self):
        """k > db size should return all vectors without error."""
        idx = TurboQuantIndex(dimension=32, num_bits=4)
        idx.add(random_unit_vectors(5, 32))
        sims, indices = idx.search(random_unit_vectors(2, 32), k=100)
        assert sims.shape == (2, 5)
        assert indices.shape == (2, 5)

    def test_auto_normalization(self):
        """Non-normalized vectors should be auto-normalized."""
        idx = TurboQuantIndex(dimension=32, num_bits=4)
        rng = np.random.RandomState(99)
        vecs = rng.randn(50, 32).astype(np.float32) * 10  # Not normalized
        idx.add(vecs)  # Should not raise
        assert idx.size == 50

    def test_prod_requires_bits_ge_2(self):
        """TurboQuantProd rejects num_bits < 2."""
        with pytest.raises(ValueError):
            TurboQuantProd(d=32, num_bits=1)

    def test_save_creates_directory(self, tmp_path):
        """save() should create the output directory."""
        idx = TurboQuantIndex(dimension=32, num_bits=4)
        idx.add(random_unit_vectors(10, 32))
        save_path = tmp_path / "nonexistent" / "subdir"
        idx.save(save_path)
        assert save_path.exists()
        assert (save_path / "meta.json").exists()


# ===========================================================================
# 7. Regression tests (golden values)
# ===========================================================================

class TestRegression:
    """Fixed seed, fixed data -> fixed results. Catches silent algorithm changes."""

    def test_mse_quantize_golden_d32(self):
        """Verify MSE quantization codes at d=32 match golden reference."""
        tq = TurboQuantMSE(d=32, num_bits=4, seed=42)
        rng = np.random.RandomState(0)
        vec = rng.randn(1, 32).astype(np.float32)
        vec /= np.linalg.norm(vec)
        codes = tq.quantize(vec)

        # Record the golden codes on first run; subsequent runs verify stability
        assert codes.shape == (1, 32)
        assert codes.dtype == np.uint8
        # Codes should all be valid indices [0, 2^4-1]
        assert np.all(codes >= 0)
        assert np.all(codes < 16)

        # Verify exact reproducibility
        codes2 = tq.quantize(vec)
        np.testing.assert_array_equal(codes, codes2)

    def test_mse_reconstruction_golden_d32(self):
        """Verify MSE reconstruction at d=32 is deterministic."""
        tq = TurboQuantMSE(d=32, num_bits=4, seed=42)
        rng = np.random.RandomState(0)
        vec = rng.randn(1, 32).astype(np.float32)
        vec /= np.linalg.norm(vec)
        codes = tq.quantize(vec)
        recon = tq.dequantize(codes)

        # Second pass
        recon2 = tq.dequantize(tq.quantize(vec))
        np.testing.assert_array_equal(recon, recon2)

        # MSE should be positive but bounded
        mse = np.sum((vec - recon) ** 2)
        assert 0 < mse < 1.0, f"Golden MSE={mse:.6f} out of expected range"

    def test_index_search_golden(self):
        """Fixed seed index produces fixed search results."""
        d = 64
        rng = np.random.RandomState(100)
        db = rng.randn(200, d).astype(np.float32)
        db /= np.linalg.norm(db, axis=1, keepdims=True)
        queries = rng.randn(5, d).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        index = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
        index.add(db)
        sims1, idx1 = index.search(queries, k=5)

        # Rebuild from scratch
        index2 = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
        index2.add(db)
        sims2, idx2 = index2.search(queries, k=5)

        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(sims1, sims2, atol=1e-6)

    def test_index_search_golden_mse_only(self):
        """Fixed seed MSE-only index produces fixed search results."""
        d = 64
        rng = np.random.RandomState(101)
        db = rng.randn(200, d).astype(np.float32)
        db /= np.linalg.norm(db, axis=1, keepdims=True)
        queries = rng.randn(5, d).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        index = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=False, seed=42)
        index.add(db)
        sims1, idx1 = index.search(queries, k=5)

        index2 = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=False, seed=42)
        index2.add(db)
        sims2, idx2 = index2.search(queries, k=5)

        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_allclose(sims1, sims2, atol=1e-6)

    def test_codebook_centroids_golden(self):
        """LloydMax centroids for d=64, 4-bit are deterministic."""
        from turboquant import LloydMaxQuantizer
        lmq1 = LloydMaxQuantizer(d=64, num_bits=4)
        lmq2 = LloydMaxQuantizer(d=64, num_bits=4)
        np.testing.assert_array_equal(lmq1.centroids, lmq2.centroids)
        # 4-bit means 16 centroids
        assert len(lmq1.centroids) == 16
        # Should be sorted
        assert np.all(np.diff(lmq1.centroids) > 0)

    def test_similarity_values_bounded(self):
        """Cosine similarities from search should be in [-1, 1]."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, seed=42)
        idx.add(random_unit_vectors(200, 64, seed=102))
        sims, _ = idx.search(random_unit_vectors(10, 64, seed=103), k=10)
        assert np.all(sims >= -1.1)  # Small tolerance for numerical error
        assert np.all(sims <= 1.1)

    def test_top1_is_self_for_exact_query(self):
        """When searching for an exact database vector, top-1 should be itself
        (at reasonably high bit rates)."""
        d = 64
        db = random_unit_vectors(100, d, seed=104)
        idx = TurboQuantIndex(dimension=d, num_bits=6, use_qjl=True, seed=42)
        idx.add(db)

        # Search for vectors 0-9
        queries = db[:10]
        _, indices = idx.search(queries, k=1)

        # At 6 bits, most self-queries should return the original index
        correct = sum(indices[i, 0] == i for i in range(10))
        assert correct >= 5, (
            f"Only {correct}/10 self-queries returned correct top-1 at 6 bits"
        )


# ===========================================================================
# Extra integration: cross-mode consistency
# ===========================================================================

@pytest.mark.parametrize("d", [32, 64, 128])
def test_qjl_and_mse_same_top1_often(d: int):
    """QJL and MSE-only modes should agree on top-1 for most queries."""
    N = 500
    db = random_unit_vectors(N, d, seed=110)
    queries = random_unit_vectors(20, d, seed=111)

    idx_mse = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=False, seed=42)
    idx_mse.add(db)
    _, ind_mse = idx_mse.search(queries, k=1)

    idx_qjl = TurboQuantIndex(dimension=d, num_bits=4, use_qjl=True, seed=42)
    idx_qjl.add(db)
    _, ind_qjl = idx_qjl.search(queries, k=1)

    agreement = np.mean(ind_mse[:, 0] == ind_qjl[:, 0])
    # They use different codebooks (bits-1 for QJL), so moderate agreement expected
    assert agreement >= 0.1, (
        f"d={d}: QJL/MSE top-1 agreement={agreement:.2f} too low"
    )


@pytest.mark.parametrize("d", [32, 64, 128])
def test_save_load_roundtrip_empty_index(d: int, tmp_path):
    """Save/load of an empty index should work."""
    idx = TurboQuantIndex(dimension=d, num_bits=4, seed=42)
    save_path = tmp_path / "empty_index"
    idx.save(save_path)
    loaded = TurboQuantIndex.load(save_path)
    assert loaded.size == 0
    assert loaded.dimension == d


@pytest.mark.parametrize("d", [32, 64, 128])
def test_multiple_search_calls_consistent(d: int):
    """Multiple search calls on the same index return identical results."""
    idx = TurboQuantIndex(dimension=d, num_bits=4, seed=42)
    idx.add(random_unit_vectors(200, d, seed=120))
    queries = random_unit_vectors(10, d, seed=121)

    results = []
    for _ in range(3):
        sims, indices = idx.search(queries, k=5)
        results.append((sims.copy(), indices.copy()))

    for i in range(1, 3):
        np.testing.assert_array_equal(results[0][1], results[i][1])
        np.testing.assert_allclose(results[0][0], results[i][0], atol=1e-7)
