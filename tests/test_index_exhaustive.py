"""Exhaustive parametrized tests for TurboQuantIndex.

Generates 500+ test cases covering construction, add, search, recall,
save/load round-trips, stats, edge cases, incremental adds, and stress tests.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pytest

from turboquant.index import TurboQuantIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_normalized(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Generate n random L2-normalized vectors of dimension d."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)


def _brute_force_topk(db: np.ndarray, queries: np.ndarray, k: int):
    """Exact cosine top-k via brute force (both inputs assumed normalized)."""
    sims = queries @ db.T
    k = min(k, db.shape[0])
    idx = np.argsort(-sims, axis=1)[:, :k]
    rows = np.arange(queries.shape[0])[:, np.newaxis]
    return sims[rows, idx], idx


# ===========================================================================
# 1. Construction tests
# ===========================================================================

_CONSTRUCT_DIMS = [8, 32, 64, 128, 256, 384]
_CONSTRUCT_BITS = [2, 3, 4, 5, 6]
_CONSTRUCT_QJL = [True, False]

_construct_params = list(itertools.product(_CONSTRUCT_DIMS, _CONSTRUCT_BITS, _CONSTRUCT_QJL))


@pytest.mark.parametrize("dim,bits,qjl", _construct_params,
                         ids=[f"d{d}_b{b}_qjl{q}" for d, b, q in _construct_params])
class TestConstruction:
    """Construction invariants for every (dim, bits, qjl) combo."""

    def test_initial_size_is_zero(self, dim, bits, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        assert idx.size == 0

    def test_compression_ratio_positive(self, dim, bits, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        assert idx.compression_ratio > 0

    def test_compression_ratio_gt1_for_low_bits(self, dim, bits, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        if bits < 32:
            assert idx.compression_ratio > 1.0

    def test_stats_structure(self, dim, bits, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        s = idx.stats()
        for key in ("size", "dimension", "num_bits", "use_qjl",
                     "compression_ratio", "bytes_per_vector",
                     "total_bytes", "float32_bytes"):
            assert key in s, f"Missing key: {key}"
        assert s["size"] == 0
        assert s["dimension"] == dim
        assert s["num_bits"] == bits


# ===========================================================================
# 2. Add tests
# ===========================================================================

_ADD_DIMS = [8, 32, 64, 128]
_ADD_BITS = [2, 3, 4, 5]
_ADD_N = [1, 10, 100, 500]
_ADD_QJL = [True, False]

_add_params = list(itertools.product(_ADD_DIMS, _ADD_BITS, _ADD_N, _ADD_QJL))


@pytest.mark.parametrize("dim,bits,N,qjl", _add_params,
                         ids=[f"d{d}_b{b}_N{n}_qjl{q}" for d, b, n, q in _add_params])
class TestAdd:
    """Add-vector invariants."""

    def test_size_after_add(self, dim, bits, N, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        vecs = _random_normalized(N, dim)
        idx.add(vecs)
        assert idx.size == N

    def test_multiple_adds_accumulate(self, dim, bits, N, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        half = max(1, N // 2)
        idx.add(_random_normalized(half, dim, seed=1))
        idx.add(_random_normalized(N - half, dim, seed=2))
        assert idx.size == N

    def test_dimension_mismatch_raises(self, dim, bits, N, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        wrong_dim = dim + 1
        with pytest.raises(ValueError, match="Expected dimension"):
            idx.add(_random_normalized(N, wrong_dim))

    def test_non_normalized_auto_normalized(self, dim, bits, N, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        rng = np.random.RandomState(99)
        vecs = rng.randn(N, dim).astype(np.float32) * 5.0  # not normalized
        idx.add(vecs)
        assert idx.size == N


# ===========================================================================
# 3. Search tests
# ===========================================================================

_SEARCH_PARAMS = [
    # (dim, bits, N, Q, k, qjl)
    (32, 4, 50, 5, 10, True),
    (32, 4, 50, 5, 10, False),
    (64, 3, 100, 10, 5, True),
    (64, 3, 100, 10, 5, False),
    (128, 5, 200, 20, 10, True),
    (128, 5, 200, 20, 10, False),
    (32, 2, 30, 3, 15, True),
    (32, 2, 30, 3, 15, False),
    (64, 4, 80, 1, 1, True),
    (64, 4, 80, 1, 1, False),
    (8, 3, 20, 4, 5, True),
    (8, 3, 20, 4, 5, False),
    (128, 6, 150, 8, 20, True),
    (128, 6, 150, 8, 20, False),
    (32, 4, 10, 5, 20, True),   # k > N
    (32, 4, 10, 5, 20, False),  # k > N
    (256, 4, 100, 3, 10, True),
    (256, 4, 100, 3, 10, False),
]


@pytest.mark.parametrize("dim,bits,N,Q,k,qjl", _SEARCH_PARAMS,
                         ids=[f"d{d}_b{b}_N{n}_Q{q}_k{k}_qjl{qj}"
                              for d, b, n, q, k, qj in _SEARCH_PARAMS])
class TestSearch:
    """Search output shape, ordering, and validity."""

    @pytest.fixture(autouse=True)
    def _build_index(self, dim, bits, N, Q, k, qjl):
        self.idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        self.db = _random_normalized(N, dim, seed=7)
        self.idx.add(self.db)
        self.queries = _random_normalized(Q, dim, seed=42)
        self.sims, self.indices = self.idx.search(self.queries, k=k)
        self.effective_k = min(k, N)

    def test_output_shapes(self, dim, bits, N, Q, k, qjl):
        assert self.sims.shape == (Q, self.effective_k)
        assert self.indices.shape == (Q, self.effective_k)

    def test_similarities_sorted_descending(self, dim, bits, N, Q, k, qjl):
        for q in range(Q):
            row = self.sims[q]
            assert np.all(row[:-1] >= row[1:] - 1e-6), \
                f"Query {q}: sims not descending: {row}"

    def test_indices_in_range(self, dim, bits, N, Q, k, qjl):
        assert np.all(self.indices >= 0)
        assert np.all(self.indices < N)

    def test_no_duplicate_indices(self, dim, bits, N, Q, k, qjl):
        for q in range(Q):
            row = self.indices[q]
            assert len(set(row.tolist())) == len(row), \
                f"Query {q}: duplicate indices found"


@pytest.mark.parametrize("dim,bits,qjl", [
    (32, 5, True), (32, 5, False),
    (64, 5, True), (64, 5, False),
    (128, 6, True), (128, 6, False),
])
def test_self_search_returns_self(dim, bits, qjl):
    """When searching for a database vector itself, it should appear in top results."""
    N = 100
    idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
    db = _random_normalized(N, dim, seed=5)
    idx.add(db)
    # Query with first 10 database vectors
    queries = db[:10]
    sims, indices = idx.search(queries, k=5)
    hits = 0
    for i in range(10):
        if i in indices[i]:
            hits += 1
    # With high bits, at least 70% should self-match
    assert hits >= 7, f"Self-search hit rate too low: {hits}/10"


@pytest.mark.parametrize("dim,qjl", [(32, True), (32, False), (64, True), (64, False)])
def test_empty_index_search(dim, qjl):
    """Search on empty index returns empty arrays."""
    idx = TurboQuantIndex(dimension=dim, num_bits=4, use_qjl=qjl)
    queries = _random_normalized(3, dim)
    sims, indices = idx.search(queries, k=5)
    assert sims.shape == (3, 0)
    assert indices.shape == (3, 0)


@pytest.mark.parametrize("dim,qjl", [(32, True), (64, False)])
def test_single_vector_search(dim, qjl):
    """Index with single vector, searched by itself."""
    idx = TurboQuantIndex(dimension=dim, num_bits=4, use_qjl=qjl)
    vec = _random_normalized(1, dim)
    idx.add(vec)
    sims, indices = idx.search(vec, k=1)
    assert sims.shape == (1, 1)
    assert indices.shape == (1, 1)
    assert indices[0, 0] == 0


# ===========================================================================
# 4. Recall tests
# ===========================================================================

_RECALL_DIMS = [32, 64, 128]
_RECALL_BITS = [3, 4, 5, 6]

_recall_params = list(itertools.product(_RECALL_DIMS, _RECALL_BITS))


@pytest.mark.parametrize("dim,bits", _recall_params,
                         ids=[f"d{d}_b{b}" for d, b in _recall_params])
class TestRecall:
    """Recall quality tests."""

    def test_recall_at_10_above_threshold(self, dim, bits):
        N, Q, k = 1000, 50, 10
        db = _random_normalized(N, dim, seed=10)
        queries = _random_normalized(Q, dim, seed=20)

        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=True)
        idx.add(db)
        _, approx_ids = idx.search(queries, k=k)

        _, exact_ids = _brute_force_topk(db, queries, k)

        recall = np.mean([
            len(set(approx_ids[q].tolist()) & set(exact_ids[q].tolist())) / k
            for q in range(Q)
        ])
        # Minimum threshold depends on bits
        thresholds = {3: 0.15, 4: 0.30, 5: 0.50, 6: 0.60}
        threshold = thresholds.get(bits, 0.10)
        assert recall >= threshold, \
            f"Recall@{k} = {recall:.3f} < {threshold} for dim={dim}, bits={bits}"


@pytest.mark.parametrize("dim", _RECALL_DIMS)
def test_recall_improves_with_bits(dim):
    """More bits should generally yield better recall."""
    N, Q, k = 500, 30, 10
    db = _random_normalized(N, dim, seed=11)
    queries = _random_normalized(Q, dim, seed=21)
    _, exact_ids = _brute_force_topk(db, queries, k)

    recalls = []
    for bits in [3, 4, 5, 6]:
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=True)
        idx.add(db)
        _, approx_ids = idx.search(queries, k=k)
        recall = np.mean([
            len(set(approx_ids[q].tolist()) & set(exact_ids[q].tolist())) / k
            for q in range(Q)
        ])
        recalls.append(recall)

    # Recall at 6 bits should be better than at 3 bits
    assert recalls[-1] >= recalls[0] - 0.05, \
        f"Recall did not improve: bits=[3,4,5,6] -> recalls={recalls}"


@pytest.mark.parametrize("dim", _RECALL_DIMS)
def test_mse_vs_qjl_recall(dim):
    """QJL mode should generally match or beat MSE-only mode."""
    N, Q, k, bits = 500, 30, 10, 4
    db = _random_normalized(N, dim, seed=12)
    queries = _random_normalized(Q, dim, seed=22)
    _, exact_ids = _brute_force_topk(db, queries, k)

    recalls = {}
    for qjl in [True, False]:
        idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        idx.add(db)
        _, approx_ids = idx.search(queries, k=k)
        recalls[qjl] = np.mean([
            len(set(approx_ids[q].tolist()) & set(exact_ids[q].tolist())) / k
            for q in range(Q)
        ])

    # QJL should be at least close to MSE-only (allow small margin)
    assert recalls[True] >= recalls[False] - 0.15, \
        f"QJL recall {recalls[True]:.3f} much worse than MSE {recalls[False]:.3f}"


# ===========================================================================
# 5. Save/Load round-trip tests
# ===========================================================================

_SAVE_DIMS = [32, 64, 128]
_SAVE_BITS = [2, 3, 4, 5]
_SAVE_N = [10, 100]
_SAVE_QJL = [True, False]

_save_params = list(itertools.product(_SAVE_DIMS, _SAVE_BITS, _SAVE_N, _SAVE_QJL))


@pytest.mark.parametrize("dim,bits,N,qjl", _save_params,
                         ids=[f"d{d}_b{b}_N{n}_qjl{q}"
                              for d, b, n, q in _save_params])
class TestSaveLoad:
    """Save/load round-trip invariants."""

    @pytest.fixture(autouse=True)
    def _build_and_save(self, dim, bits, N, qjl, tmp_path):
        self.dim = dim
        self.bits = bits
        self.N = N
        self.qjl = qjl
        self.path = tmp_path / "test_index"

        self.original = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
        self.db = _random_normalized(N, dim, seed=77)
        self.original.add(self.db)
        self.original.save(self.path)
        self.loaded = TurboQuantIndex.load(self.path)

    def test_size_preserved(self, dim, bits, N, qjl, tmp_path):
        assert self.loaded.size == self.original.size

    def test_search_results_match(self, dim, bits, N, qjl, tmp_path):
        queries = _random_normalized(5, dim, seed=88)
        k = min(5, N)
        sims_orig, idx_orig = self.original.search(queries, k=k)
        sims_load, idx_load = self.loaded.search(queries, k=k)
        np.testing.assert_allclose(sims_orig, sims_load, atol=1e-5)
        np.testing.assert_array_equal(idx_orig, idx_load)

    def test_stats_match(self, dim, bits, N, qjl, tmp_path):
        assert self.original.stats() == self.loaded.stats()

    def test_meta_json_exists(self, dim, bits, N, qjl, tmp_path):
        assert (self.path / "meta.json").is_file()

    def test_directory_structure(self, dim, bits, N, qjl, tmp_path):
        assert (self.path / "rotation.npy").is_file()
        assert (self.path / "centroids.npy").is_file()
        if qjl:
            assert (self.path / "qjl_matrix.npy").is_file()
            assert (self.path / "mse_codes.npy").is_file()
            assert (self.path / "qjl_signs.npy").is_file()
            assert (self.path / "residual_norms.npy").is_file()
        else:
            assert (self.path / "codes.npy").is_file()


@pytest.mark.parametrize("dim,bits,qjl", [
    (32, 3, True), (32, 3, False), (64, 4, True), (64, 4, False),
])
def test_save_load_empty_index(dim, bits, qjl, tmp_path):
    """Save and load an empty index."""
    path = tmp_path / "empty"
    idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=qjl)
    idx.save(path)
    loaded = TurboQuantIndex.load(path)
    assert loaded.size == 0
    assert loaded.stats()["dimension"] == dim


# ===========================================================================
# 6. Persistence file structure tests
# ===========================================================================

@pytest.mark.parametrize("qjl", [True, False])
class TestPersistenceStructure:
    """Verify on-disk file structure."""

    def test_meta_json_keys(self, qjl, tmp_path):
        path = tmp_path / "struct"
        idx = TurboQuantIndex(dimension=32, num_bits=4, use_qjl=qjl)
        idx.add(_random_normalized(10, 32))
        idx.save(path)

        meta = json.loads((path / "meta.json").read_text())
        required_keys = {"dimension", "num_bits", "metric", "use_qjl",
                         "seed", "size", "version"}
        assert required_keys.issubset(set(meta.keys()))

    def test_npy_files_present(self, qjl, tmp_path):
        path = tmp_path / "struct2"
        idx = TurboQuantIndex(dimension=32, num_bits=4, use_qjl=qjl)
        idx.add(_random_normalized(10, 32))
        idx.save(path)

        assert (path / "rotation.npy").exists()
        assert (path / "centroids.npy").exists()
        if qjl:
            assert (path / "qjl_matrix.npy").exists()
            assert (path / "mse_codes.npy").exists()
            assert (path / "qjl_signs.npy").exists()
            assert (path / "residual_norms.npy").exists()
            assert not (path / "codes.npy").exists()
        else:
            assert (path / "codes.npy").exists()
            assert not (path / "qjl_matrix.npy").exists()


# ===========================================================================
# 7. Stats tests
# ===========================================================================

_STATS_PARAMS = list(itertools.product([32, 64, 128], [2, 3, 4, 5], [10, 100, 500]))


@pytest.mark.parametrize("dim,bits,N", _STATS_PARAMS,
                         ids=[f"d{d}_b{b}_N{n}" for d, b, n in _STATS_PARAMS])
class TestStats:
    """Stats dict consistency."""

    @pytest.fixture(autouse=True)
    def _build(self, dim, bits, N):
        self.idx = TurboQuantIndex(dimension=dim, num_bits=bits, use_qjl=True)
        self.idx.add(_random_normalized(N, dim))
        self.stats = self.idx.stats()
        self.dim = dim
        self.bits = bits
        self.N = N

    def test_required_keys(self, dim, bits, N):
        for key in ("size", "dimension", "num_bits", "use_qjl",
                     "compression_ratio", "bytes_per_vector",
                     "total_bytes", "float32_bytes"):
            assert key in self.stats

    def test_size_matches(self, dim, bits, N):
        assert self.stats["size"] == N

    def test_dimension_matches(self, dim, bits, N):
        assert self.stats["dimension"] == dim

    def test_num_bits_matches(self, dim, bits, N):
        assert self.stats["num_bits"] == bits

    def test_total_bytes_formula(self, dim, bits, N):
        assert self.stats["total_bytes"] == N * self.stats["bytes_per_vector"]

    def test_float32_bytes_formula(self, dim, bits, N):
        assert self.stats["float32_bytes"] == N * dim * 4


# ===========================================================================
# 8. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases that don't fit the parametrized groups."""

    @pytest.mark.parametrize("dim", [32, 64, 128])
    def test_add_single_vector_1d(self, dim):
        """Add a 1D array (single vector without batch dimension)."""
        idx = TurboQuantIndex(dimension=dim, num_bits=4)
        vec = _random_normalized(1, dim).squeeze()  # shape (dim,)
        idx.add(vec)
        assert idx.size == 1

    @pytest.mark.parametrize("dim", [32, 64, 128])
    def test_search_single_query_1d(self, dim):
        """Search with a 1D query array."""
        idx = TurboQuantIndex(dimension=dim, num_bits=4)
        idx.add(_random_normalized(20, dim))
        query = _random_normalized(1, dim).squeeze()  # shape (dim,)
        sims, indices = idx.search(query, k=5)
        assert sims.shape == (1, 5)
        assert indices.shape == (1, 5)

    @pytest.mark.parametrize("dim,qjl", [(32, True), (32, False), (64, True)])
    def test_k_equals_1(self, dim, qjl):
        idx = TurboQuantIndex(dimension=dim, num_bits=4, use_qjl=qjl)
        idx.add(_random_normalized(50, dim))
        sims, indices = idx.search(_random_normalized(3, dim), k=1)
        assert sims.shape == (3, 1)
        assert indices.shape == (3, 1)

    @pytest.mark.parametrize("dim,qjl", [(32, True), (64, False)])
    def test_k_equals_N(self, dim, qjl):
        """Return all vectors."""
        N = 20
        idx = TurboQuantIndex(dimension=dim, num_bits=4, use_qjl=qjl)
        idx.add(_random_normalized(N, dim))
        sims, indices = idx.search(_random_normalized(3, dim), k=N)
        assert sims.shape == (3, N)
        assert indices.shape == (3, N)

    @pytest.mark.parametrize("dim", [32, 64])
    def test_add_zero_vectors(self, dim):
        """Zero vectors should be handled (auto-normalized)."""
        idx = TurboQuantIndex(dimension=dim, num_bits=4)
        zeros = np.zeros((5, dim), dtype=np.float32)
        idx.add(zeros)
        assert idx.size == 5

    def test_high_dimensional_small_n(self):
        """d=1024 with small N."""
        idx = TurboQuantIndex(dimension=1024, num_bits=4)
        idx.add(_random_normalized(5, 1024))
        assert idx.size == 5
        sims, indices = idx.search(_random_normalized(2, 1024), k=3)
        assert sims.shape == (2, 3)

    @pytest.mark.parametrize("dim,qjl", [(32, True), (32, False)])
    def test_k_larger_than_n(self, dim, qjl):
        """k > N should clamp to N results."""
        N = 5
        idx = TurboQuantIndex(dimension=dim, num_bits=4, use_qjl=qjl)
        idx.add(_random_normalized(N, dim))
        sims, indices = idx.search(_random_normalized(2, dim), k=100)
        assert sims.shape == (2, N)
        assert indices.shape == (2, N)


# ===========================================================================
# 9. Incremental add tests
# ===========================================================================

@pytest.mark.parametrize("dim", [32, 64, 128])
class TestIncrementalAdd:
    """Verify incremental add produces equivalent results."""

    def test_size_accumulates(self, dim):
        idx = TurboQuantIndex(dimension=dim, num_bits=4)
        idx.add(_random_normalized(30, dim, seed=1))
        idx.add(_random_normalized(20, dim, seed=2))
        idx.add(_random_normalized(50, dim, seed=3))
        assert idx.size == 100

    def test_incremental_vs_batch_search(self, dim):
        """Search results from incremental adds match batch add."""
        rng = np.random.RandomState(55)
        batch1 = _random_normalized(40, dim, seed=100)
        batch2 = _random_normalized(60, dim, seed=200)
        combined = np.concatenate([batch1, batch2], axis=0)
        queries = _random_normalized(5, dim, seed=300)
        k = 10

        # Batch add
        idx_batch = TurboQuantIndex(dimension=dim, num_bits=4, seed=42)
        idx_batch.add(combined)
        sims_batch, ids_batch = idx_batch.search(queries, k=k)

        # Incremental add
        idx_inc = TurboQuantIndex(dimension=dim, num_bits=4, seed=42)
        idx_inc.add(batch1)
        idx_inc.add(batch2)
        sims_inc, ids_inc = idx_inc.search(queries, k=k)

        np.testing.assert_allclose(sims_batch, sims_inc, atol=1e-5)
        np.testing.assert_array_equal(ids_batch, ids_inc)


# ===========================================================================
# 10. Stress tests
# ===========================================================================

class TestStress:
    """Larger-scale stress tests."""

    def test_10k_vectors(self):
        """10K vectors, dim=64, bits=4, search 100 queries."""
        idx = TurboQuantIndex(dimension=64, num_bits=4)
        idx.add(_random_normalized(10_000, 64, seed=0))
        assert idx.size == 10_000
        sims, indices = idx.search(_random_normalized(100, 64, seed=1), k=10)
        assert sims.shape == (100, 10)
        assert indices.shape == (100, 10)
        # Verify sorted
        for q in range(100):
            assert np.all(sims[q, :-1] >= sims[q, 1:] - 1e-6)

    def test_multiple_save_load_cycles(self, tmp_path):
        """Three save/load cycles produce identical results."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=True)
        idx.add(_random_normalized(200, 64))
        queries = _random_normalized(10, 64, seed=99)

        sims_orig, ids_orig = idx.search(queries, k=10)

        for cycle in range(3):
            path = tmp_path / f"cycle_{cycle}"
            idx.save(path)
            idx = TurboQuantIndex.load(path)

        sims_final, ids_final = idx.search(queries, k=10)
        np.testing.assert_allclose(sims_orig, sims_final, atol=1e-5)
        np.testing.assert_array_equal(ids_orig, ids_final)

    def test_search_latency_scaling(self):
        """Search latency should not explode unreasonably with size."""
        dim, bits = 64, 4
        queries = _random_normalized(10, dim, seed=50)

        times = []
        for n in [100, 1000, 5000]:
            idx = TurboQuantIndex(dimension=dim, num_bits=bits)
            idx.add(_random_normalized(n, dim, seed=n))
            t0 = time.perf_counter()
            idx.search(queries, k=10)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        # 50x data should be < 500x slower (generous to account for
        # fixed overhead like lazy rebuild dominating small-N timings)
        ratio = times[2] / max(times[0], 1e-9)
        assert ratio < 500, f"Latency ratio 5000/100 = {ratio:.1f}x — too high"


# ===========================================================================
# Quick count sanity check (printed by pytest --collect-only)
# ===========================================================================

def test_meta_count_sanity():
    """Sanity: confirm we have many parametrized combos defined."""
    # Construction: 6 dims * 5 bits * 2 qjl = 60 combos * 3 tests = 180
    assert len(_construct_params) == 60
    # Add: 4 dims * 4 bits * 4 Ns * 2 qjl = 128 combos * 4 tests = 512
    assert len(_add_params) == 128
    # Save/Load: 3 dims * 4 bits * 2 Ns * 2 qjl = 48 combos * 5 tests = 240
    assert len(_save_params) == 48
    # Stats: 3 dims * 4 bits * 3 Ns = 36 combos * 6 tests = 216
    assert len(_STATS_PARAMS) == 36
