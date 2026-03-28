# TurboQuant Critical Fixes + IVF Layer Design

**Date:** 2026-03-28
**Status:** Approved
**Scope:** Fix critical issues from external code review + add IVF indexing layer

---

## Context

An in-depth critical analysis of TurboQuant identified 8 issues, ranging from architectural scalability problems to misleading documentation. This spec addresses all of them through 6 prioritized interventions grouped into two phases.

### Issues Addressed (from two independent external reviews)

1. **`_rebuild_reconstructed()` called on every `add()`** — O(N) rebuild per batch insert
2. **Brute-force search marketed as ANN** — no indexing structure, O(N·d) per query
3. **Compression illusory at runtime** — `_reconstructed` float32 matrix negates storage savings
4. **Normalization threshold too permissive** — `atol=1e-3` introduces systematic bias
5. **No CI/CD** — 3,781 tests with no automated regression prevention
6. **README overpromises** — "FAISS replacement" without query-time benchmarks
7. **Rotation matrix overhead not reported** — 9MB for d=1536, invisible in stats()
8. **No codebook memoization** — same (d, num_bits) recomputes Lloyd-Max from scratch
9. **Development Status Beta → should be Alpha** — single-day project, no versioning
10. **No CHANGELOG** — version fixed at 0.1.0 with no release history
11. **Test quality vs quantity** — parametrized grid tests, missing real-world edge cases
12. **Benchmark only on 10K vectors** — insufficient for production claims

---

## Phase 1: Critical Fixes to Existing Code

### 1.1 Lazy Rebuild with `_dirty` Flag

**File:** `turboquant/index.py`

**Change:** Replace eager rebuild in `add()` with lazy pattern:

- `add()` sets `self._dirty = True` after appending codes
- `search()` checks `_dirty` and rebuilds only when needed
- `_rebuild_reconstructed()` sets `_dirty = False` after rebuild
- `save()` also triggers rebuild if dirty (to serialize correct state)
- `reset()` sets `_dirty = False`

**Impact:** `add()` becomes O(batch) instead of O(N_total). Multiple consecutive `add()` calls accumulate codes without redundant reconstruction.

### 1.2 Always-Normalize

**File:** `turboquant/index.py`

**Change:** Remove conditional normalization logic:

```python
# BEFORE
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
if not np.allclose(norms, 1.0, atol=1e-3):
    vectors = vectors / np.clip(norms, 1e-8, None)

# AFTER
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.clip(norms, 1e-8, None)
```

Apply to both `add()` and `search()`.

**Rationale:** Normalization is O(N) and negligible vs quantization cost. The theoretical guarantees of TurboQuant (Theorem 1) require vectors exactly on S^(d-1). Removing the tolerance eliminates accumulated bias.

### 1.3 Codebook Memoization

**File:** `turboquant/codebook.py`

**Change:** Add module-level cache for computed centroids keyed by `(d, num_bits)`:

```python
_CENTROID_CACHE: dict[tuple[int, int], np.ndarray] = {}
```

`LloydMaxQuantizer.__init__` checks the cache before running Lloyd-Max iterations. Cache hit avoids 0.44s+ initialization per instance. This matters when:
- Multiple `TurboQuantIndex` instances share the same (d, num_bits)
- Serverless cold-starts create fresh instances frequently
- `IVFTurboQuantIndex` creates nlist sub-indexes with identical codebooks

### 1.4 Stats Overhead Transparency

**File:** `turboquant/index.py`

**Change:** `stats()` now reports:
- `rotation_matrix_bytes`: d*d*4 bytes (the QR rotation matrix overhead)
- `total_overhead_bytes`: rotation + centroids + (QJL matrix if use_qjl)
- `effective_compression_ratio`: accounts for overhead, not just code bytes

This addresses the valid critique that d=1536 rotation matrices (~9MB) are invisible in current stats.

### 1.5 Development Status and CHANGELOG

**Files:** `pyproject.toml`, `CHANGELOG.md`

**Changes:**
- Development Status: `4 - Beta` → `3 - Alpha`
- Add `CHANGELOG.md` starting from v0.1.0 with honest description of current state
- Bump version to `0.2.0` to mark this release

### 1.6 README Repositioning

**File:** `README.md`

**Changes:**
- Title: "FAISS replacement" → "FAISS-compatible vector quantization library"
- Add explicit note that `TurboQuantIndex` uses brute-force search on compressed vectors
- Introduce `IVFTurboQuantIndex` as the ANN solution
- Add query-time benchmark table (measured, not theoretical)
- Clarify that runtime memory includes the reconstructed float32 matrix for `TurboQuantIndex`
- Document rotation matrix overhead for high-d embeddings
- Note that recall benchmarks are on random unit vectors; real embeddings may differ
- Remove "Online/streaming support: Yes" claim (contradicted by rebuild behavior, even with lazy fix the claim is misleading for IVF which requires training)

---

## Phase 2: IVF Indexing Layer

### 2.1 Architecture

**New file:** `turboquant/ivf_index.py`

`IVFTurboQuantIndex` implements Inverted File indexing on top of TurboQuant compression:

```
IVFTurboQuantIndex
  ├── centroids: np.ndarray (nlist, d)     # K-means cluster centers
  ├── partitions: list[TurboQuantIndex]    # One sub-index per cluster
  ├── id_map: list[np.ndarray]             # Original vector IDs per partition
  ├── nlist: int                           # Number of partitions
  ├── nprobe: int                          # Partitions to search per query
  └── _trained: bool                       # Whether K-means has been run
```

### 2.2 K-Means Implementation

Pure NumPy K-means (no new dependencies):

- Initialize with K-means++ (better convergence than random)
- Max 20 iterations, convergence check with `atol=1e-6`
- Operates on normalized vectors (consistent with TurboQuant assumptions)

### 2.3 API

```python
class IVFTurboQuantIndex:
    def __init__(self, dimension, num_bits=4, nlist=100, nprobe=10,
                 metric="cosine", use_qjl=True, seed=42): ...

    def train(self, vectors: np.ndarray) -> None:
        """Run K-means to learn cluster centroids."""

    def add(self, vectors: np.ndarray) -> None:
        """Assign vectors to nearest centroid and add to partition index."""

    def search(self, queries: np.ndarray, k=10) -> tuple[np.ndarray, np.ndarray]:
        """Search nprobe nearest partitions for top-k results."""

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> IVFTurboQuantIndex: ...

    def stats(self) -> dict: ...
```

### 2.4 Search Algorithm

```
search(queries, k, nprobe):
    1. Compute query-centroid similarities: queries @ centroids.T  → (Q, nlist)
    2. For each query, select top-nprobe partitions
    3. For each selected partition:
       - Search the partition's TurboQuantIndex for k results
       - Collect (similarity, global_id) pairs
    4. Merge all results per query, return top-k globally
```

### 2.5 Complexity

| Operation | TurboQuantIndex | IVFTurboQuantIndex |
|-----------|:-:|:-:|
| `add()` (per vector) | O(d) quantize | O(nlist·d) assign + O(d) quantize |
| `search()` (per query) | O(N·d) | O(nlist·d + nprobe·(N/nlist)·d) |
| Typical with nlist=sqrt(N) | O(N·d) | O(sqrt(N)·d) |

### 2.6 Save/Load Format

```
ivf_index/
  meta.json            # dimension, num_bits, nlist, nprobe, size, version
  centroids.npy        # (nlist, d) float32
  partition_000/       # TurboQuantIndex.save() format
  partition_001/
  ...
  id_map.npy           # flattened ID mapping
```

---

## Phase 3: CI/CD and Benchmarks

### 3.1 GitHub Actions

**New file:** `.github/workflows/ci.yml`

- **Trigger:** push to main, PRs
- **Matrix:** Python 3.10, 3.11, 3.12
- **Fast suite:** core tests (~35 tests, <20s) — runs on every trigger
- **Exhaustive suite:** full 3,781 tests (~13 min) — runs on push to main only

### 3.2 Query-Time Benchmark Script

**New file:** `examples/benchmark_query_time.py`

Measures and prints:
- Query latency (ms) for TurboQuantIndex and IVFTurboQuantIndex
- At dataset sizes 10K, 100K, 1M (random vectors)
- Reports recall@10 alongside latency for honest comparison

### 3.3 New Tests

**New file:** `tests/test_ivf_index.py`

Test categories:
- Training: K-means convergence, centroid properties, train-before-add guard
- Add: correct partition assignment, global ID mapping
- Search: recall vs brute-force baseline, nprobe sweep, merge correctness
- Save/Load: round-trip integrity, loaded index produces identical results
- Edge cases: nprobe > nlist, empty partitions, single vector, k > partition size
- Lazy rebuild: multiple add() calls don't trigger rebuild until search()

**Modified file:** `tests/test_index.py` — add tests for:
- Lazy rebuild behavior (no rebuild between consecutive add() calls)
- Always-normalize (vectors with norm 0.999 produce same results as 1.0)

**Quality-focused tests** (address "test theater" critique):
- Quasi-collinear vectors: pairs with cosine similarity > 0.999
- Clustered embeddings: vectors with anisotropic distribution (simulating real RAG embeddings)
- Large dimension edge cases: d=1536, d=3072 with realistic memory constraints
- Codebook memoization: verify cache hit avoids recomputation

---

## Non-Goals

- No HNSW or multi-level IVF — single level is sufficient for target scale (up to ~5M vectors)
- No ADC (Asymmetric Distance Computation) — search remains on reconstructed float32 within partitions
- No GPU support — pure NumPy/SciPy stays
- No breaking changes to `TurboQuantIndex` public API — lazy rebuild is internal
- `quantizer.py` remains untouched (codebook.py gets memoization only)

---

## File Change Summary

| File | Action |
|------|--------|
| `turboquant/index.py` | Modify: lazy rebuild, always-normalize, stats overhead |
| `turboquant/codebook.py` | Modify: add centroid memoization cache |
| `turboquant/ivf_index.py` | **New**: IVFTurboQuantIndex |
| `turboquant/__init__.py` | Modify: export IVFTurboQuantIndex |
| `README.md` | Modify: repositioning, query-time table, IVF docs, overhead transparency |
| `CHANGELOG.md` | **New**: release history starting v0.1.0 |
| `pyproject.toml` | Modify: version bump 0.2.0, status Alpha |
| `.github/workflows/ci.yml` | **New**: CI pipeline |
| `examples/benchmark_query_time.py` | **New**: query-time benchmark |
| `tests/test_ivf_index.py` | **New**: IVF test suite |
| `tests/test_index.py` | Modify: lazy rebuild + normalization + quality tests |
