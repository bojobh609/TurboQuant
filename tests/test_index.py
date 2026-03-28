"""Tests for TurboQuantIndex."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from turboquant.index import TurboQuantIndex


def _random_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestTurboQuantIndex:
    def test_add_and_search(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=False)
        db = _random_unit_vectors(100, 128)
        idx.add(db)
        assert idx.size == 100

        queries = _random_unit_vectors(5, 128, seed=99)
        sims, indices = idx.search(queries, k=10)
        assert sims.shape == (5, 10)
        assert indices.shape == (5, 10)
        assert np.all(sims[:, 0] >= sims[:, -1])  # sorted descending

    def test_recall_at_10(self):
        d = 384
        idx = TurboQuantIndex(dimension=d, num_bits=6, use_qjl=False)
        db = _random_unit_vectors(2000, d)
        queries = _random_unit_vectors(50, d, seed=99)

        idx.add(db)
        sims, indices = idx.search(queries, k=10)

        gt = np.argsort(-(queries @ db.T), axis=1)[:, :10]
        recalls = [len(set(gt[i]) & set(indices[i])) / 10 for i in range(50)]
        assert np.mean(recalls) > 0.90

    def test_empty_index(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4)
        queries = _random_unit_vectors(3, 128)
        sims, indices = idx.search(queries, k=10)
        assert sims.shape[1] == 0

    def test_auto_normalize(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=False)
        db = np.random.randn(50, 128).astype(np.float32) * 5  # not normalized
        idx.add(db)
        assert idx.size == 50

    def test_dimension_mismatch(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4)
        with pytest.raises(ValueError):
            idx.add(np.random.randn(10, 256).astype(np.float32))

    def test_save_load_roundtrip(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=False)
        db = _random_unit_vectors(100, 128)
        idx.add(db)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = TurboQuantIndex.load(tmpdir)

            assert loaded.size == 100
            assert loaded.dimension == 128
            assert loaded.num_bits == 4

            queries = _random_unit_vectors(5, 128, seed=99)
            s1, i1 = idx.search(queries, k=5)
            s2, i2 = loaded.search(queries, k=5)
            assert np.array_equal(i1, i2)

    def test_save_load_qjl(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=True)
        db = _random_unit_vectors(100, 128)
        idx.add(db)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = TurboQuantIndex.load(tmpdir)
            assert loaded.size == 100
            assert loaded.use_qjl is True

    def test_stats(self):
        idx = TurboQuantIndex(dimension=384, num_bits=6, use_qjl=False)
        db = _random_unit_vectors(1000, 384)
        idx.add(db)
        stats = idx.stats()
        assert stats["size"] == 1000
        assert stats["dimension"] == 384
        assert float(stats["compression_ratio"].replace("x", "")) > 4

    def test_incremental_add(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(50, 128, seed=1))
        idx.add(_random_unit_vectors(50, 128, seed=2))
        assert idx.size == 100

    def test_k_larger_than_db(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(5, 128))
        sims, indices = idx.search(_random_unit_vectors(2, 128, seed=99), k=100)
        assert indices.shape == (2, 5)


class TestCodebookMemoization:
    def test_same_params_reuse_centroids(self):
        """Two LloydMaxQuantizer instances with same (d, num_bits) share centroids."""
        from turboquant.codebook import LloydMaxQuantizer, _CENTROID_CACHE
        _CENTROID_CACHE.clear()
        lmq1 = LloydMaxQuantizer(d=64, num_bits=4)
        lmq2 = LloydMaxQuantizer(d=64, num_bits=4)
        assert lmq1.centroids is lmq2.centroids
        assert (64, 4) in _CENTROID_CACHE

    def test_different_params_different_centroids(self):
        """Different (d, num_bits) produce different centroids."""
        from turboquant.codebook import LloydMaxQuantizer, _CENTROID_CACHE
        _CENTROID_CACHE.clear()
        lmq1 = LloydMaxQuantizer(d=64, num_bits=4)
        lmq2 = LloydMaxQuantizer(d=64, num_bits=3)
        assert not np.array_equal(lmq1.centroids, lmq2.centroids)

    def test_cache_speedup(self):
        """Second instantiation is near-instant due to cache hit."""
        import time
        from turboquant.codebook import LloydMaxQuantizer, _CENTROID_CACHE
        _CENTROID_CACHE.clear()
        t0 = time.perf_counter()
        LloydMaxQuantizer(d=128, num_bits=4)
        first = time.perf_counter() - t0
        t0 = time.perf_counter()
        LloydMaxQuantizer(d=128, num_bits=4)
        second = time.perf_counter() - t0
        assert second < first * 0.1  # at least 10x faster


class TestLazyRebuild:
    def test_add_does_not_rebuild(self):
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        db = _random_unit_vectors(50, 64)
        idx.add(db)
        assert idx._dirty is True
        assert idx._reconstructed is None

    def test_search_triggers_rebuild(self):
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(50, 64))
        assert idx._dirty is True
        queries = _random_unit_vectors(2, 64, seed=99)
        idx.search(queries, k=5)
        assert idx._dirty is False
        assert idx._reconstructed is not None

    def test_multiple_adds_single_rebuild(self):
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(25, 64, seed=1))
        idx.add(_random_unit_vectors(25, 64, seed=2))
        idx.add(_random_unit_vectors(25, 64, seed=3))
        assert idx.size == 75
        assert idx._dirty is True
        assert idx._reconstructed is None
        queries = _random_unit_vectors(2, 64, seed=99)
        sims, indices = idx.search(queries, k=5)
        assert idx._dirty is False
        assert sims.shape == (2, 5)

    def test_save_triggers_rebuild_if_dirty(self):
        import tempfile
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(50, 64))
        assert idx._dirty is True
        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = TurboQuantIndex.load(tmpdir)
            assert loaded.size == 50


class TestAlwaysNormalize:
    def test_near_unit_vectors_get_normalized(self):
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        v = _random_unit_vectors(50, 64)
        v_scaled = v * 0.999
        idx.add(v_scaled)

        idx2 = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx2.add(v)

        q = _random_unit_vectors(5, 64, seed=99)
        s1, i1 = idx.search(q, k=5)
        s2, i2 = idx2.search(q, k=5)
        np.testing.assert_array_equal(i1, i2)

    def test_query_normalization(self):
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(100, 64))

        q_unit = _random_unit_vectors(3, 64, seed=99)
        q_scaled = q_unit * 2.5
        s1, i1 = idx.search(q_unit, k=5)
        s2, i2 = idx.search(q_scaled, k=5)
        np.testing.assert_array_equal(i1, i2)


class TestCodesConsolidation:
    def test_codes_consolidated_after_rebuild(self):
        """After search triggers rebuild, _codes should be a single-element list."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(25, 64, seed=1))
        idx.add(_random_unit_vectors(25, 64, seed=2))
        idx.add(_random_unit_vectors(25, 64, seed=3))
        assert len(idx._codes) == 3
        idx.search(_random_unit_vectors(2, 64, seed=99), k=5)
        assert len(idx._codes) == 1

    def test_codes_consolidated_qjl(self):
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=True)
        idx.add(_random_unit_vectors(25, 64, seed=1))
        idx.add(_random_unit_vectors(25, 64, seed=2))
        assert len(idx._codes) == 2
        idx.search(_random_unit_vectors(2, 64, seed=99), k=5)
        assert len(idx._codes) == 1
        assert idx._codes[0]["mse_codes"].shape[0] == 50


class TestStatsOverhead:
    def test_stats_reports_rotation_overhead(self):
        idx = TurboQuantIndex(dimension=384, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(100, 384))
        stats = idx.stats()
        assert "rotation_matrix_bytes" in stats
        assert stats["rotation_matrix_bytes"] == 384 * 384 * 4

    def test_stats_reports_effective_compression(self):
        idx = TurboQuantIndex(dimension=384, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(1000, 384))
        stats = idx.stats()
        assert "effective_compression_ratio" in stats
        raw = float(stats["compression_ratio"].replace("x", ""))
        effective = stats["effective_compression_ratio"]
        assert effective < raw
        assert effective > 1.0

    def test_stats_reports_qjl_overhead(self):
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=True)
        idx.add(_random_unit_vectors(100, 128))
        stats = idx.stats()
        assert "total_overhead_bytes" in stats
        assert stats["total_overhead_bytes"] > stats["rotation_matrix_bytes"]
