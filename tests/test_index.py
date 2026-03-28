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
