"""Tests for IVFTurboQuantIndex."""
import numpy as np
import pytest
import tempfile
from turboquant.ivf_index import IVFTurboQuantIndex


def _random_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class TestIVFTraining:
    def test_train_sets_centroids(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10, nprobe=3)
        data = _random_unit_vectors(500, 64)
        idx.train(data)
        assert idx._trained is True
        assert idx._centroids.shape == (10, 64)

    def test_add_before_train_raises(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10)
        with pytest.raises(RuntimeError, match="train"):
            idx.add(_random_unit_vectors(100, 64))

    def test_centroids_are_normalized(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10)
        idx.train(_random_unit_vectors(500, 64))
        norms = np.linalg.norm(idx._centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_kmeans_convergence(self):
        idx1 = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5, seed=42)
        idx2 = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5, seed=42)
        data = _random_unit_vectors(500, 64)
        idx1.train(data)
        idx2.train(data)
        np.testing.assert_array_equal(idx1._centroids, idx2._centroids)


class TestIVFAddSearch:
    @pytest.fixture
    def trained_index(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10, nprobe=5, use_qjl=False)
        data = _random_unit_vectors(1000, 64)
        idx.train(data)
        idx.add(data)
        return idx, data

    def test_add_distributes_vectors(self, trained_index):
        idx, _ = trained_index
        assert idx.size == 1000
        total_in_partitions = sum(p.size for p in idx._partitions)
        assert total_in_partitions == 1000

    def test_search_returns_correct_shape(self, trained_index):
        idx, _ = trained_index
        queries = _random_unit_vectors(5, 64, seed=99)
        sims, indices = idx.search(queries, k=10)
        assert sims.shape == (5, 10)
        assert indices.shape == (5, 10)

    def test_search_sorted_descending(self, trained_index):
        idx, _ = trained_index
        queries = _random_unit_vectors(5, 64, seed=99)
        sims, _ = idx.search(queries, k=10)
        for i in range(5):
            assert np.all(sims[i, :-1] >= sims[i, 1:])

    def test_recall_vs_brute_force(self):
        d = 128
        n = 2000
        db = _random_unit_vectors(n, d)
        queries = _random_unit_vectors(50, d, seed=99)

        idx = IVFTurboQuantIndex(dimension=d, num_bits=6, nlist=20, nprobe=10, use_qjl=False)
        idx.train(db)
        idx.add(db)
        _, ivf_indices = idx.search(queries, k=10)

        gt = np.argsort(-(queries @ db.T), axis=1)[:, :10]
        recalls = [len(set(gt[i]) & set(ivf_indices[i])) / 10 for i in range(50)]
        assert np.mean(recalls) > 0.70

    def test_nprobe_sweep(self):
        d = 64
        db = _random_unit_vectors(500, d)
        queries = _random_unit_vectors(20, d, seed=99)
        gt = np.argsort(-(queries @ db.T), axis=1)[:, :10]

        recalls = []
        for nprobe in [1, 3, 5, 10]:
            idx = IVFTurboQuantIndex(dimension=d, num_bits=4, nlist=10, nprobe=nprobe, use_qjl=False)
            idx.train(db)
            idx.add(db)
            _, indices = idx.search(queries, k=10)
            r = np.mean([len(set(gt[i]) & set(indices[i])) / 10 for i in range(20)])
            recalls.append(r)
        for i in range(len(recalls) - 1):
            assert recalls[i + 1] >= recalls[i] - 0.05


class TestIVFEdgeCases:
    def test_nprobe_greater_than_nlist(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5, nprobe=20, use_qjl=False)
        db = _random_unit_vectors(200, 64)
        idx.train(db)
        idx.add(db)
        sims, indices = idx.search(_random_unit_vectors(3, 64, seed=99), k=5)
        assert sims.shape == (3, 5)

    def test_empty_partitions(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=50, nprobe=5, use_qjl=False)
        db = _random_unit_vectors(100, 64)
        idx.train(db)
        idx.add(db)
        sims, indices = idx.search(_random_unit_vectors(3, 64, seed=99), k=5)
        assert sims.shape == (3, 5)

    def test_k_larger_than_probed_vectors(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10, nprobe=1, use_qjl=False)
        db = _random_unit_vectors(100, 64)
        idx.train(db)
        idx.add(db)
        sims, indices = idx.search(_random_unit_vectors(2, 64, seed=99), k=500)
        assert indices.shape[1] <= 100

    def test_single_vector(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5, nprobe=5, use_qjl=False)
        db = _random_unit_vectors(50, 64)
        idx.train(db)
        idx.add(_random_unit_vectors(1, 64, seed=77))
        assert idx.size == 1
        sims, indices = idx.search(_random_unit_vectors(1, 64, seed=99), k=1)
        assert sims.shape == (1, 1)

    def test_dimension_mismatch(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5)
        idx.train(_random_unit_vectors(100, 64))
        with pytest.raises(ValueError):
            idx.add(np.random.randn(10, 128).astype(np.float32))


class TestIVFSaveLoad:
    def test_save_load_roundtrip(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10, nprobe=5, use_qjl=False)
        db = _random_unit_vectors(500, 64)
        idx.train(db)
        idx.add(db)

        queries = _random_unit_vectors(5, 64, seed=99)
        s1, i1 = idx.search(queries, k=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = IVFTurboQuantIndex.load(tmpdir)
            assert loaded.size == 500
            assert loaded._trained is True
            s2, i2 = loaded.search(queries, k=10)
            np.testing.assert_array_equal(i1, i2)

    def test_save_load_with_qjl(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5, nprobe=3, use_qjl=True)
        db = _random_unit_vectors(200, 64)
        idx.train(db)
        idx.add(db)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = IVFTurboQuantIndex.load(tmpdir)
            assert loaded.size == 200
            assert loaded.use_qjl is True


class TestIVFStats:
    def test_stats_includes_partition_info(self):
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10, nprobe=5, use_qjl=False)
        db = _random_unit_vectors(500, 64)
        idx.train(db)
        idx.add(db)
        stats = idx.stats()
        assert stats["nlist"] == 10
        assert stats["nprobe"] == 5
        assert stats["size"] == 500
        assert "partition_sizes" in stats
