# TurboQuant Critical Fixes + IVF Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical issues identified by two external reviews and add IVF indexing for real ANN search.

**Architecture:** Phase 1 fixes existing code (lazy rebuild, always-normalize, codebook memoization, stats transparency). Phase 2 adds `IVFTurboQuantIndex` with K-means++ partitioning on top of existing `TurboQuantIndex`. Phase 3 adds CI/CD, benchmarks, and documentation updates.

**Tech Stack:** Python 3.10+, NumPy, SciPy, pytest, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-03-28-critical-fixes-and-ivf-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `turboquant/codebook.py` | Modify | Add centroid memoization cache |
| `turboquant/index.py` | Modify | Lazy rebuild, always-normalize, enhanced stats |
| `turboquant/ivf_index.py` | Create | IVFTurboQuantIndex with K-means++ partitioning |
| `turboquant/__init__.py` | Modify | Export IVFTurboQuantIndex |
| `tests/test_index.py` | Modify | Tests for lazy rebuild, normalization, stats |
| `tests/test_ivf_index.py` | Create | Full IVF test suite |
| `.github/workflows/ci.yml` | Create | CI pipeline |
| `examples/benchmark_query_time.py` | Create | Honest query-time benchmarks |
| `CHANGELOG.md` | Create | Release history |
| `pyproject.toml` | Modify | Version bump, status Alpha |
| `README.md` | Modify | Repositioning, IVF docs, query-time table |

---

## Task 1: Codebook Memoization

**Files:**
- Modify: `turboquant/codebook.py:56-66`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write failing test for memoization**

Add to `tests/test_index.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestCodebookMemoization -v`
Expected: FAIL -- `_CENTROID_CACHE` does not exist

- [ ] **Step 3: Implement memoization in codebook.py**

Add module-level cache and modify `__init__`:

```python
# Add after imports, before hypersphere_coordinate_pdf
_CENTROID_CACHE: dict[tuple[int, int], np.ndarray] = {}
```

Modify `LloydMaxQuantizer.__init__` (lines 56-66):

```python
def __init__(
    self,
    d: int,
    num_bits: int = 4,
    grid_points: int = 50000,
    max_iter: int = 200,
) -> None:
    self.d = d
    self.num_bits = num_bits
    self.num_levels = 2 ** num_bits
    cache_key = (d, num_bits)
    if cache_key in _CENTROID_CACHE:
        self.centroids = _CENTROID_CACHE[cache_key]
    else:
        self.centroids = self._compute_centroids(grid_points, max_iter)
        _CENTROID_CACHE[cache_key] = self.centroids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestCodebookMemoization -v`
Expected: 3 passed

- [ ] **Step 5: Run existing codebook tests to check no regressions**

Run: `cd /root/TurboQuant && python -m pytest tests/test_codebook.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd /root/TurboQuant
git add turboquant/codebook.py tests/test_index.py
git commit -m "perf: add centroid memoization cache to LloydMaxQuantizer

Same (d, num_bits) parameters now reuse pre-computed centroids,
avoiding redundant 0.44s+ Lloyd-Max iterations per instance.
Critical for IVFTurboQuantIndex which creates nlist sub-indexes."
```

---

## Task 2: Lazy Rebuild with `_dirty` Flag

**Files:**
- Modify: `turboquant/index.py:55-120`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write failing tests for lazy rebuild**

Add to `tests/test_index.py`:

```python
class TestLazyRebuild:
    def test_add_does_not_rebuild(self):
        """add() should not trigger _rebuild_reconstructed immediately."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        db = _random_unit_vectors(50, 64)
        idx.add(db)
        assert idx._dirty is True
        assert idx._reconstructed is None  # not rebuilt yet

    def test_search_triggers_rebuild(self):
        """search() should trigger rebuild when dirty."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(50, 64))
        assert idx._dirty is True
        queries = _random_unit_vectors(2, 64, seed=99)
        idx.search(queries, k=5)
        assert idx._dirty is False
        assert idx._reconstructed is not None

    def test_multiple_adds_single_rebuild(self):
        """Multiple add() calls should only rebuild once at search time."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(25, 64, seed=1))
        idx.add(_random_unit_vectors(25, 64, seed=2))
        idx.add(_random_unit_vectors(25, 64, seed=3))
        assert idx.size == 75
        assert idx._dirty is True
        assert idx._reconstructed is None
        # Now search triggers single rebuild
        queries = _random_unit_vectors(2, 64, seed=99)
        sims, indices = idx.search(queries, k=5)
        assert idx._dirty is False
        assert sims.shape == (2, 5)

    def test_save_triggers_rebuild_if_dirty(self):
        """save() should rebuild if dirty to serialize correct state."""
        import tempfile
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(50, 64))
        assert idx._dirty is True
        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = TurboQuantIndex.load(tmpdir)
            assert loaded.size == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestLazyRebuild -v`
Expected: FAIL -- `_dirty` attribute does not exist

- [ ] **Step 3: Implement lazy rebuild in index.py**

Modify `TurboQuantIndex.__init__` -- add `_dirty` flag:

```python
# Storage
self._codes: list = []
self._reconstructed: np.ndarray | None = None
self._size = 0
self._dirty = False
```

Modify `add()` -- remove `_rebuild_reconstructed()` call, set `_dirty`:

```python
def add(self, vectors: np.ndarray) -> None:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[np.newaxis, :]

    if vectors.shape[1] != self.dimension:
        raise ValueError(
            f"Expected dimension {self.dimension}, got {vectors.shape[1]}"
        )

    # Always normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-8, None)

    # Quantize
    codes = self._quantizer.quantize(vectors)

    # Store codes
    self._codes.append(codes)
    self._size += vectors.shape[0]
    self._dirty = True
```

Modify `_rebuild_reconstructed()` -- set `_dirty = False` at end:

```python
def _rebuild_reconstructed(self) -> None:
    """Rebuild the full reconstructed matrix from stored codes."""
    if not self._codes:
        self._reconstructed = None
        self._dirty = False
        return

    if self.use_qjl:
        all_mse = np.concatenate([c["mse_codes"] for c in self._codes], axis=0)
        all_signs = np.concatenate([c["qjl_signs"] for c in self._codes], axis=0)
        all_norms = np.concatenate([c["residual_norms"] for c in self._codes], axis=0)
        merged = {
            "mse_codes": all_mse,
            "qjl_signs": all_signs,
            "residual_norms": all_norms,
        }
        self._reconstructed = self._quantizer.dequantize(merged)
    else:
        all_codes = np.concatenate(self._codes, axis=0)
        self._reconstructed = self._quantizer.dequantize(all_codes)

    self._dirty = False
```

Modify `search()` -- add lazy rebuild check at top:

```python
def search(self, queries, k=10):
    if self._dirty:
        self._rebuild_reconstructed()

    if self._reconstructed is None or self._size == 0:
        # ... rest unchanged
```

Modify `save()` -- add lazy rebuild check:

```python
def save(self, path):
    if self._dirty:
        self._rebuild_reconstructed()
    # ... rest unchanged
```

- [ ] **Step 4: Run new tests**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestLazyRebuild -v`
Expected: 4 passed

- [ ] **Step 5: Run full test_index.py for regressions**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd /root/TurboQuant
git add turboquant/index.py tests/test_index.py
git commit -m "perf: lazy rebuild -- add() no longer triggers O(N) reconstruction

_rebuild_reconstructed() is now deferred until first search() or save()
after add(). Multiple add() calls accumulate codes in O(batch) each."
```

---

## Task 3: Always-Normalize

**Files:**
- Modify: `turboquant/index.py:107-111,168-171`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_index.py`:

```python
class TestAlwaysNormalize:
    def test_near_unit_vectors_get_normalized(self):
        """Vectors with norm 0.999 should be normalized, not skipped."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        # Create vectors with norm exactly 0.999 (within old atol=1e-3)
        v = _random_unit_vectors(50, 64)
        v_scaled = v * 0.999
        idx.add(v_scaled)

        idx2 = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx2.add(v)

        q = _random_unit_vectors(5, 64, seed=99)
        s1, i1 = idx.search(q, k=5)
        s2, i2 = idx2.search(q, k=5)
        # Results should be identical since both are normalized
        np.testing.assert_array_equal(i1, i2)

    def test_query_normalization(self):
        """Query vectors are always normalized regardless of input norm."""
        idx = TurboQuantIndex(dimension=64, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(100, 64))

        q_unit = _random_unit_vectors(3, 64, seed=99)
        q_scaled = q_unit * 2.5
        s1, i1 = idx.search(q_unit, k=5)
        s2, i2 = idx.search(q_scaled, k=5)
        np.testing.assert_array_equal(i1, i2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestAlwaysNormalize -v`
Expected: FAIL -- `test_near_unit_vectors_get_normalized` fails because old code skips normalization at atol=1e-3

- [ ] **Step 3: Implement always-normalize**

In `add()`, replace the conditional normalization (already done in Task 2 step 3 -- verify it reads):

```python
# Always normalize
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.clip(norms, 1e-8, None)
```

In `search()`, replace similarly:

```python
# Always normalize queries
norms = np.linalg.norm(queries, axis=1, keepdims=True)
queries = queries / np.clip(norms, 1e-8, None)
```

- [ ] **Step 4: Run tests**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestAlwaysNormalize -v`
Expected: 2 passed

- [ ] **Step 5: Run full test suite for regressions**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
cd /root/TurboQuant
git add turboquant/index.py tests/test_index.py
git commit -m "fix: always normalize vectors -- remove permissive atol=1e-3 threshold

Theoretical guarantees require vectors exactly on S^(d-1).
Normalization is O(N) and negligible vs quantization cost."
```

---

## Task 4: Enhanced Stats with Overhead Transparency

**Files:**
- Modify: `turboquant/index.py:286-297`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_index.py`:

```python
class TestStatsOverhead:
    def test_stats_reports_rotation_overhead(self):
        """stats() must report rotation matrix memory overhead."""
        idx = TurboQuantIndex(dimension=384, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(100, 384))
        stats = idx.stats()
        assert "rotation_matrix_bytes" in stats
        assert stats["rotation_matrix_bytes"] == 384 * 384 * 4  # d*d*float32

    def test_stats_reports_effective_compression(self):
        """stats() must report effective compression accounting for overhead."""
        idx = TurboQuantIndex(dimension=384, num_bits=4, use_qjl=False)
        idx.add(_random_unit_vectors(1000, 384))
        stats = idx.stats()
        assert "effective_compression_ratio" in stats
        # Effective ratio should be lower than raw ratio due to overhead
        raw = float(stats["compression_ratio"].replace("x", ""))
        effective = stats["effective_compression_ratio"]
        assert effective < raw
        assert effective > 1.0  # still compresses

    def test_stats_reports_qjl_overhead(self):
        """stats() with QJL must include QJL matrix overhead."""
        idx = TurboQuantIndex(dimension=128, num_bits=4, use_qjl=True)
        idx.add(_random_unit_vectors(100, 128))
        stats = idx.stats()
        assert "total_overhead_bytes" in stats
        # QJL overhead includes rotation + qjl_matrix + centroids
        assert stats["total_overhead_bytes"] > stats["rotation_matrix_bytes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestStatsOverhead -v`
Expected: FAIL -- keys not in stats dict

- [ ] **Step 3: Implement enhanced stats**

Replace `stats()` method in `index.py`:

```python
def stats(self) -> dict:
    """Return index statistics including memory overhead transparency."""
    d = self.dimension
    rotation_bytes = d * d * 4  # float32 rotation matrix
    if self.use_qjl:
        centroid_bytes = self._quantizer.mse_quantizer.codebook.num_levels * 4
    else:
        centroid_bytes = self._quantizer.codebook.num_levels * 4
    qjl_matrix_bytes = d * d * 4 if self.use_qjl else 0
    total_overhead = rotation_bytes + centroid_bytes + qjl_matrix_bytes

    code_bytes = self._size * self._quantizer.bytes_per_vector
    float32_bytes = self._size * d * 4
    total_with_overhead = code_bytes + total_overhead

    effective_ratio = float32_bytes / total_with_overhead if total_with_overhead > 0 else 0.0

    return {
        "size": self._size,
        "dimension": d,
        "num_bits": self.num_bits,
        "use_qjl": self.use_qjl,
        "compression_ratio": f"{self.compression_ratio:.1f}x",
        "effective_compression_ratio": round(effective_ratio, 2),
        "bytes_per_vector": self._quantizer.bytes_per_vector,
        "total_code_bytes": code_bytes,
        "rotation_matrix_bytes": rotation_bytes,
        "total_overhead_bytes": total_overhead,
        "float32_bytes": float32_bytes,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestStatsOverhead -v`
Expected: 3 passed

- [ ] **Step 5: Check existing stats test still passes**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py::TestTurboQuantIndex::test_stats -v`
Expected: PASS (the old keys are still present)

- [ ] **Step 6: Commit**

```bash
cd /root/TurboQuant
git add turboquant/index.py tests/test_index.py
git commit -m "feat: enhanced stats() with rotation matrix and overhead transparency

Reports rotation_matrix_bytes, total_overhead_bytes, and
effective_compression_ratio accounting for all structural overhead.
Addresses critique that d=1536 rotation matrix (9MB) was invisible."
```

---

## Task 5: IVFTurboQuantIndex

**Files:**
- Create: `turboquant/ivf_index.py`
- Modify: `turboquant/__init__.py`
- Create: `tests/test_ivf_index.py`

This is the largest task. We split into sub-steps: K-means++, then IVF class, then tests.

- [ ] **Step 1: Write test file for IVF**

Create `tests/test_ivf_index.py`:

```python
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
        """K-means should produce stable centroids."""
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
        """IVF recall should be reasonable vs brute-force ground truth."""
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
        assert np.mean(recalls) > 0.75  # IVF with quantization loses some recall

    def test_nprobe_sweep(self):
        """Higher nprobe should give equal or better recall."""
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
        # Recall should be monotonically non-decreasing
        for i in range(len(recalls) - 1):
            assert recalls[i + 1] >= recalls[i] - 0.05  # small tolerance for randomness


class TestIVFEdgeCases:
    def test_nprobe_greater_than_nlist(self):
        """nprobe > nlist should search all partitions without error."""
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=5, nprobe=20, use_qjl=False)
        db = _random_unit_vectors(200, 64)
        idx.train(db)
        idx.add(db)
        sims, indices = idx.search(_random_unit_vectors(3, 64, seed=99), k=5)
        assert sims.shape == (3, 5)

    def test_empty_partitions(self):
        """Some partitions may be empty -- search should still work."""
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=50, nprobe=5, use_qjl=False)
        # Only 100 vectors for 50 clusters -- some will be empty
        db = _random_unit_vectors(100, 64)
        idx.train(db)
        idx.add(db)
        sims, indices = idx.search(_random_unit_vectors(3, 64, seed=99), k=5)
        assert sims.shape == (3, 5)

    def test_k_larger_than_probed_vectors(self):
        """When k > total vectors in probed partitions, return all available."""
        idx = IVFTurboQuantIndex(dimension=64, num_bits=4, nlist=10, nprobe=1, use_qjl=False)
        db = _random_unit_vectors(100, 64)
        idx.train(db)
        idx.add(db)
        sims, indices = idx.search(_random_unit_vectors(2, 64, seed=99), k=500)
        assert indices.shape[1] <= 100  # can't return more than probed

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/TurboQuant && python -m pytest tests/test_ivf_index.py -v 2>&1 | head -20`
Expected: FAIL -- `turboquant.ivf_index` module does not exist

- [ ] **Step 3: Implement IVFTurboQuantIndex**

Create `turboquant/ivf_index.py`:

```python
"""IVFTurboQuantIndex -- Inverted File index with TurboQuant compression.

Combines K-means++ partitioning with TurboQuant quantization for
approximate nearest neighbor search with sub-linear query time.

Example:
    >>> from turboquant import IVFTurboQuantIndex
    >>>
    >>> index = IVFTurboQuantIndex(dimension=384, num_bits=6, nlist=100, nprobe=10)
    >>> index.train(training_vectors)
    >>> index.add(database_vectors)
    >>> similarities, indices = index.search(query_vectors, k=10)

Reference: TurboQuant (arXiv:2504.19874) + IVF partitioning
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from turboquant.index import TurboQuantIndex


def _kmeans_plus_plus(vectors: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """K-means++ initialization for cluster centroids.

    Args:
        vectors: Normalized vectors, shape (N, d).
        k: Number of clusters.
        rng: Random state for reproducibility.

    Returns:
        Initial centroids, shape (k, d).
    """
    n, d = vectors.shape
    centroids = np.empty((k, d), dtype=np.float32)
    idx = rng.randint(n)
    centroids[0] = vectors[idx]

    for i in range(1, k):
        # Squared distances to nearest existing centroid
        dists = 1.0 - vectors @ centroids[:i].T  # cosine distance
        min_dists = dists.min(axis=1)
        min_dists = np.clip(min_dists, 0.0, None)
        probs = min_dists / (min_dists.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centroids[i] = vectors[idx]

    return centroids


def _kmeans(vectors: np.ndarray, k: int, max_iter: int = 20,
            seed: int = 42) -> np.ndarray:
    """K-means clustering on normalized vectors using cosine similarity.

    Args:
        vectors: Normalized vectors, shape (N, d).
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed.

    Returns:
        Cluster centroids, shape (k, d), L2-normalized.
    """
    rng = np.random.RandomState(seed)
    centroids = _kmeans_plus_plus(vectors, k, rng)

    for _ in range(max_iter):
        # Assign to nearest centroid (cosine similarity)
        sims = vectors @ centroids.T  # (N, k)
        assignments = np.argmax(sims, axis=1)  # (N,)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = vectors[mask].mean(axis=0)
            else:
                # Empty cluster -- reinitialize randomly
                new_centroids[j] = vectors[rng.randint(vectors.shape[0])]

        # Normalize centroids
        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        new_centroids = new_centroids / np.clip(norms, 1e-8, None)

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


class IVFTurboQuantIndex:
    """Inverted File index with TurboQuant compression.

    Partitions vectors into nlist clusters via K-means, then stores each
    partition as a TurboQuantIndex. Search probes nprobe nearest partitions
    for sub-linear query time.

    Args:
        dimension: Vector dimension.
        num_bits: Bits per coordinate (2-8). Default 4.
        nlist: Number of IVF partitions. Default 100.
        nprobe: Number of partitions to search per query. Default 10.
        metric: Similarity metric -- 'cosine' or 'ip'.
        use_qjl: Use TurboQuantProd for unbiased inner products.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dimension: int,
        num_bits: int = 4,
        nlist: int = 100,
        nprobe: int = 10,
        metric: str = "cosine",
        use_qjl: bool = True,
        seed: int = 42,
    ) -> None:
        self.dimension = dimension
        self.num_bits = num_bits
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric
        self.use_qjl = use_qjl and num_bits >= 2
        self.seed = seed

        self._centroids: np.ndarray | None = None
        self._partitions: list[TurboQuantIndex] = []
        self._id_maps: list[list[int]] = []
        self._trained = False
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def train(self, vectors: np.ndarray) -> None:
        """Train IVF partitioning with K-means on provided vectors.

        Args:
            vectors: Training vectors, shape (N, d). Will be normalized.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-8, None)

        # Clamp nlist to number of vectors
        effective_nlist = min(self.nlist, vectors.shape[0])

        self._centroids = _kmeans(vectors, effective_nlist, seed=self.seed)

        # Initialize partitions
        self._partitions = [
            TurboQuantIndex(
                dimension=self.dimension,
                num_bits=self.num_bits,
                metric=self.metric,
                use_qjl=self.use_qjl,
                seed=self.seed,
            )
            for _ in range(effective_nlist)
        ]
        self._id_maps = [[] for _ in range(effective_nlist)]
        self._trained = True

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index, assigning each to its nearest partition.

        Args:
            vectors: Vectors to add, shape (N, d). Will be normalized.
        """
        if not self._trained:
            raise RuntimeError(
                "Index must be trained before adding vectors. Call train() first."
            )

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, got {vectors.shape[1]}"
            )

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-8, None)

        # Assign to nearest centroid
        sims = vectors @ self._centroids.T  # (N, nlist)
        assignments = np.argmax(sims, axis=1)  # (N,)

        # Add to partitions
        base_id = self._size
        for j in range(len(self._partitions)):
            mask = assignments == j
            if mask.sum() == 0:
                continue
            partition_vectors = vectors[mask]
            self._partitions[j].add(partition_vectors)
            global_ids = (np.where(mask)[0] + base_id).tolist()
            self._id_maps[j].extend(global_ids)

        self._size += vectors.shape[0]

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors across nprobe partitions.

        Args:
            queries: Query vectors, shape (Q, d).
            k: Number of nearest neighbors.

        Returns:
            Tuple of (similarities, indices) with global vector IDs.
        """
        if not self._trained or self._size == 0:
            empty_sim = np.zeros((queries.shape[0], 0), dtype=np.float32)
            empty_idx = np.zeros((queries.shape[0], 0), dtype=np.int64)
            return empty_sim, empty_idx

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]

        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.clip(norms, 1e-8, None)

        # Find nprobe nearest partitions per query
        effective_nprobe = min(self.nprobe, len(self._partitions))
        centroid_sims = queries @ self._centroids.T  # (Q, nlist)
        top_partitions = np.argsort(-centroid_sims, axis=1)[
            :, :effective_nprobe
        ]  # (Q, nprobe)

        num_queries = queries.shape[0]
        all_sims = []
        all_ids = []

        for qi in range(num_queries):
            query = queries[qi : qi + 1]
            q_sims = []
            q_ids = []

            for pi in top_partitions[qi]:
                partition = self._partitions[pi]
                if partition.size == 0:
                    continue
                local_k = min(k, partition.size)
                p_sims, p_local_idx = partition.search(query, k=local_k)
                # Map local indices to global IDs
                id_map = self._id_maps[pi]
                for j in range(p_sims.shape[1]):
                    local_idx = p_local_idx[0, j]
                    q_sims.append(p_sims[0, j])
                    q_ids.append(id_map[local_idx])

            # Select top-k from all probed partitions
            if len(q_sims) == 0:
                all_sims.append(np.zeros(0, dtype=np.float32))
                all_ids.append(np.zeros(0, dtype=np.int64))
            else:
                q_sims = np.array(q_sims, dtype=np.float32)
                q_ids = np.array(q_ids, dtype=np.int64)
                top_k = min(k, len(q_sims))
                if top_k < len(q_sims):
                    top_idx = np.argpartition(-q_sims, top_k)[:top_k]
                    top_idx = top_idx[np.argsort(-q_sims[top_idx])]
                else:
                    top_idx = np.argsort(-q_sims)
                all_sims.append(q_sims[top_idx])
                all_ids.append(q_ids[top_idx])

        # Pad to uniform shape
        max_k = max(len(s) for s in all_sims) if all_sims else 0
        result_sims = np.zeros((num_queries, max_k), dtype=np.float32)
        result_ids = np.full((num_queries, max_k), -1, dtype=np.int64)
        for qi in range(num_queries):
            n = len(all_sims[qi])
            result_sims[qi, :n] = all_sims[qi]
            result_ids[qi, :n] = all_ids[qi]

        return result_sims, result_ids

    def save(self, path: str | Path) -> None:
        """Save IVF index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "dimension": self.dimension,
            "num_bits": self.num_bits,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "metric": self.metric,
            "use_qjl": self.use_qjl,
            "seed": self.seed,
            "size": self._size,
            "trained": self._trained,
            "num_partitions": len(self._partitions),
            "version": "0.2.0",
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        if self._centroids is not None:
            np.save(path / "centroids.npy", self._centroids)

        # Save each non-empty partition
        for i, partition in enumerate(self._partitions):
            if partition.size > 0:
                pdir = path / f"partition_{i:04d}"
                partition.save(pdir)

        # Save ID maps as JSON (safe serialization)
        if self._id_maps:
            id_maps_data = {str(i): ids for i, ids in enumerate(self._id_maps)}
            (path / "id_maps.json").write_text(json.dumps(id_maps_data))

    @classmethod
    def load(cls, path: str | Path) -> IVFTurboQuantIndex:
        """Load IVF index from disk."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        index = cls(
            dimension=meta["dimension"],
            num_bits=meta["num_bits"],
            nlist=meta["nlist"],
            nprobe=meta["nprobe"],
            metric=meta.get("metric", "cosine"),
            use_qjl=meta.get("use_qjl", True),
            seed=meta["seed"],
        )

        index._size = meta["size"]
        index._trained = meta["trained"]

        if (path / "centroids.npy").exists():
            index._centroids = np.load(path / "centroids.npy")

        num_partitions = meta["num_partitions"]
        index._partitions = []
        for i in range(num_partitions):
            pdir = path / f"partition_{i:04d}"
            if pdir.exists():
                partition = TurboQuantIndex.load(pdir)
            else:
                partition = TurboQuantIndex(
                    dimension=meta["dimension"],
                    num_bits=meta["num_bits"],
                    metric=meta.get("metric", "cosine"),
                    use_qjl=meta.get("use_qjl", True),
                    seed=meta["seed"],
                )
            index._partitions.append(partition)

        if (path / "id_maps.json").exists():
            id_maps_data = json.loads((path / "id_maps.json").read_text())
            index._id_maps = [
                id_maps_data.get(str(i), []) for i in range(num_partitions)
            ]
        else:
            index._id_maps = [[] for _ in range(num_partitions)]

        return index

    def stats(self) -> dict:
        """Return index statistics."""
        partition_sizes = [p.size for p in self._partitions]
        return {
            "size": self._size,
            "dimension": self.dimension,
            "num_bits": self.num_bits,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "use_qjl": self.use_qjl,
            "trained": self._trained,
            "partition_sizes": partition_sizes,
            "avg_partition_size": np.mean(partition_sizes) if partition_sizes else 0,
            "empty_partitions": sum(1 for s in partition_sizes if s == 0),
        }
```

- [ ] **Step 4: Update `__init__.py` to export IVFTurboQuantIndex**

Add to `turboquant/__init__.py`:

```python
from turboquant.ivf_index import IVFTurboQuantIndex
```

And add `"IVFTurboQuantIndex"` to `__all__`.

- [ ] **Step 5: Run IVF tests**

Run: `cd /root/TurboQuant && python -m pytest tests/test_ivf_index.py -v`
Expected: all pass

- [ ] **Step 6: Run full existing test suite for regressions**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py tests/test_codebook.py tests/test_quantizer.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
cd /root/TurboQuant
git add turboquant/ivf_index.py turboquant/__init__.py tests/test_ivf_index.py
git commit -m "feat: add IVFTurboQuantIndex for sub-linear ANN search

K-means++ partitioning with TurboQuant compression per partition.
Reduces query complexity from O(N*d) to O(sqrt(N)*d) with nprobe control.
Includes train/add/search/save/load and comprehensive test suite."
```

---

## Task 6: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-fast:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run fast test suite
        run: |
          pytest tests/test_codebook.py tests/test_quantizer.py tests/test_index.py tests/test_ivf_index.py -v

  test-exhaustive:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run exhaustive test suite
        run: |
          pytest tests/ -v --timeout=900
```

- [ ] **Step 2: Commit**

```bash
cd /root/TurboQuant
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions pipeline -- fast suite on PRs, exhaustive on main"
```

---

## Task 7: CHANGELOG and Version Bump

**Files:**
- Create: `CHANGELOG.md`
- Modify: `pyproject.toml`
- Modify: `turboquant/__init__.py`

- [ ] **Step 1: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to TurboQuant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-03-28

### Added
- `IVFTurboQuantIndex` -- Inverted File index with K-means++ partitioning for sub-linear ANN search
- Codebook memoization cache -- same `(d, num_bits)` parameters reuse pre-computed centroids
- Enhanced `stats()` with rotation matrix overhead, effective compression ratio
- GitHub Actions CI pipeline (fast suite on PRs, exhaustive on main push)
- Query-time benchmark script (`examples/benchmark_query_time.py`)
- Comprehensive IVF test suite

### Fixed
- Lazy rebuild: `add()` no longer triggers O(N) reconstruction on every call
- Always-normalize: removed permissive `atol=1e-3` threshold that introduced bias
- README repositioned from "FAISS replacement" to "FAISS-compatible quantization library"

### Changed
- Development Status: Beta -> Alpha (honest maturity assessment)
- Version: 0.1.0 -> 0.2.0

## [0.1.0] - 2026-03-28

### Added
- Initial release
- `TurboQuantMSE` -- MSE-optimal quantizer (Algorithm 1 from paper)
- `TurboQuantProd` -- Inner-product-optimal quantizer with QJL correction (Algorithm 2)
- `TurboQuantIndex` -- FAISS-compatible vector search index
- `LloydMaxQuantizer` -- Optimal scalar quantizer for hypersphere coordinates
- 3,781 parametrized tests across 5 test suites
- Paper verification report (PAPER_VERIFICATION.md)
```

- [ ] **Step 2: Update pyproject.toml**

Change version to `0.2.0` and status to Alpha:

```
version = "0.2.0"
```

```
"Development Status :: 3 - Alpha",
```

- [ ] **Step 3: Update `__init__.py` version**

```python
__version__ = "0.2.0"
```

- [ ] **Step 4: Commit**

```bash
cd /root/TurboQuant
git add CHANGELOG.md pyproject.toml turboquant/__init__.py
git commit -m "chore: bump version to 0.2.0, add CHANGELOG, status Alpha"
```

---

## Task 8: Query-Time Benchmark Script

**Files:**
- Create: `examples/benchmark_query_time.py`

- [ ] **Step 1: Create benchmark script**

```python
"""Honest query-time benchmark: TurboQuantIndex vs IVFTurboQuantIndex.

Measures latency and recall at multiple dataset sizes.
Run: python examples/benchmark_query_time.py
"""

import time
import numpy as np
from turboquant import TurboQuantIndex, IVFTurboQuantIndex


def _random_unit_vectors(n, d, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def benchmark_brute_force(db, queries, k, num_bits):
    idx = TurboQuantIndex(dimension=db.shape[1], num_bits=num_bits, use_qjl=False)
    t0 = time.perf_counter()
    idx.add(db)
    add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sims, indices = idx.search(queries, k=k)
    query_time = time.perf_counter() - t0

    return add_time, query_time, indices


def benchmark_ivf(db, queries, k, num_bits, nlist, nprobe):
    idx = IVFTurboQuantIndex(
        dimension=db.shape[1],
        num_bits=num_bits,
        nlist=nlist,
        nprobe=nprobe,
        use_qjl=False,
    )
    t0 = time.perf_counter()
    idx.train(db)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx.add(db)
    add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sims, indices = idx.search(queries, k=k)
    query_time = time.perf_counter() - t0

    return train_time, add_time, query_time, indices


def compute_recall(pred_indices, gt_indices):
    recalls = []
    for i in range(len(pred_indices)):
        pred_set = set(int(x) for x in pred_indices[i] if x >= 0)
        gt_set = set(int(x) for x in gt_indices[i])
        if len(gt_set) > 0:
            recalls.append(len(pred_set & gt_set) / len(gt_set))
    return np.mean(recalls)


def main():
    d = 384
    k = 10
    num_bits = 6
    num_queries = 100

    print("=" * 80)
    print(f"TurboQuant Query-Time Benchmark (d={d}, bits={num_bits}, k={k})")
    print("=" * 80)
    print()

    for n in [10_000, 100_000]:
        print(f"--- Dataset: {n:,} vectors ---")
        db = _random_unit_vectors(n, d, seed=0)
        queries = _random_unit_vectors(num_queries, d, seed=99)

        # Ground truth (brute-force float32)
        gt = np.argsort(-(queries @ db.T), axis=1)[:, :k]

        # Brute-force TurboQuant
        add_t, query_t, bf_idx = benchmark_brute_force(db, queries, k, num_bits)
        bf_recall = compute_recall(bf_idx, gt)
        bf_qps = num_queries / query_t
        print(
            f"  TurboQuantIndex (brute):  add={add_t:.2f}s  "
            f"query={query_t * 1000:.1f}ms  "
            f"recall@{k}={bf_recall:.3f}  QPS={bf_qps:.0f}"
        )

        # IVF TurboQuant
        nlist = max(10, int(np.sqrt(n)))
        for nprobe in [1, 5, 10]:
            train_t, add_t, query_t, ivf_idx = benchmark_ivf(
                db, queries, k, num_bits, nlist, nprobe,
            )
            ivf_recall = compute_recall(ivf_idx, gt)
            ivf_qps = num_queries / query_t
            print(
                f"  IVFTurboQuant (nlist={nlist}, nprobe={nprobe:2d}): "
                f"train={train_t:.2f}s  add={add_t:.2f}s  "
                f"query={query_t * 1000:.1f}ms  "
                f"recall@{k}={ivf_recall:.3f}  QPS={ivf_qps:.0f}"
            )

        print()

    print("Note: Query time includes all queries. QPS = queries per second.")
    print("Note: Recall is measured against float32 brute-force ground truth.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /root/TurboQuant
git add examples/benchmark_query_time.py
git commit -m "bench: add honest query-time benchmark script

Measures latency and recall for TurboQuantIndex (brute-force) vs
IVFTurboQuantIndex at 10K and 100K vectors with nprobe sweep."
```

---

## Task 9: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with all repositioning changes**

Key changes:
1. Title line: replace "Pure Python FAISS replacement" with "FAISS-compatible vector quantization library"
2. Subtitle: replace "drop-in FAISS replacement" with "FAISS-compatible quantization backend"
3. Update "Why TurboQuant?" table -- add "Query complexity" row
4. Add IVF section with `IVFTurboQuantIndex` usage example
5. Add "Limitations and Transparency" section
6. Update comparison table with "Online/streaming" to "Yes (brute-force only)" and add query complexity
7. Add `IVFTurboQuantIndex` to API Reference
8. Update Architecture diagram with new files
9. Add query-time benchmark reference

- [ ] **Step 2: Run any README link checks or existing tests**

Run: `cd /root/TurboQuant && python -m pytest tests/test_index.py tests/test_ivf_index.py -v`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
cd /root/TurboQuant
git add README.md
git commit -m "docs: reposition README -- honest benchmarks, IVF docs, overhead transparency

- 'FAISS replacement' -> 'FAISS-compatible quantization library'
- Added IVFTurboQuantIndex usage and API docs
- Added Limitations section with rotation matrix overhead
- Noted recall benchmarks are on random vectors, not real embeddings
- Updated architecture diagram with new files"
```

---

## Task 10: Final Validation

- [ ] **Step 1: Run complete test suite**

Run: `cd /root/TurboQuant && python -m pytest tests/test_codebook.py tests/test_quantizer.py tests/test_index.py tests/test_ivf_index.py -v`
Expected: all pass

- [ ] **Step 2: Run benchmark to get real numbers for README**

Run: `cd /root/TurboQuant && python examples/benchmark_query_time.py`
Expected: prints benchmark table with real measurements

- [ ] **Step 3: Update README with real benchmark numbers if needed**

- [ ] **Step 4: Push to GitHub**

```bash
cd /root/TurboQuant && git push origin main
```
