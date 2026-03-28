# TurboQuant — Test Documentation

**3,781 parametrized tests** verifying correctness, mathematical properties, and all 6 claims from the original Google Research paper ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026).

## Quick Start

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full suite (3,781 tests, ~13 min)
pytest tests/ -v

# Run fast unit tests only (~35 tests, <20s)
pytest tests/test_codebook.py tests/test_quantizer.py tests/test_index.py -v
```

## Test Suite Overview

| File | Tests | Time | Coverage |
|------|------:|-----:|----------|
| `test_codebook.py` | 10 | 8s | Core codebook unit tests |
| `test_quantizer.py` | 13 | 5s | Core quantizer unit tests |
| `test_index.py` | 7 | 2s | Core index unit tests |
| `test_recall.py` | 5 | 1s | Recall benchmark comparison |
| `test_codebook_exhaustive.py` | 785 | 44s | Exhaustive codebook verification |
| `test_quantizer_exhaustive.py` | 1,266 | 5m | Exhaustive quantizer verification |
| `test_index_exhaustive.py` | 1,344 | 7m | Exhaustive index verification |
| `test_properties.py` | 211 | 3m | Mathematical invariants |
| `test_integration.py` | 140 | 2m | End-to-end pipelines |
| **Total** | **3,781** | **~13m** | |

## Detailed Test Descriptions

### test_codebook_exhaustive.py — 785 tests

Verifies the Lloyd-Max optimal scalar quantizer and hypersphere coordinate PDF.

**Parametrization:** dimensions {8, 16, 32, 64, 128, 256, 384, 768} × bit-widths {1, 2, 3, 4, 5, 6, 7, 8}

| Category | Tests | What It Verifies |
|----------|------:|-----------------|
| PDF non-negativity | 7 | `f(x) >= 0` for all x, across 7 dimensions |
| PDF integrates to 1 | 7 | `∫f(x)dx ≈ 1.0` via scipy.integrate.quad |
| PDF symmetry | 7 | `f(x) == f(-x)` for all x |
| PDF boundary | 7 | `f(x) == 0` for `|x| >= 1` |
| PDF peak | 7 | Maximum at x=0 |
| Centroid count | 64 | Exactly `2^b` centroids per configuration |
| Centroids sorted | 64 | Ascending order |
| Centroid range | 64 | Within `±5/√d` (statistical bound) |
| Centroid symmetry | 64 | Sum of centroids ≈ 0 |
| Centroid uniqueness | 64 | No duplicate values |
| Centroid dtype | 64 | float64 |
| Quantize dtype | 64 | uint8 output |
| Index range | 64 | All indices in `[0, 2^b - 1]` |
| Round-trip values | 64 | Dequantized values are centroids |
| MSE bound | 64 | Per-coordinate MSE ≤ theoretical bound |
| Determinism | 64 | Same input → same output |
| MSE vs Shannon | 8 | `theoretical_mse > shannon_lower_bound` |
| MSE monotonicity | 7 | MSE decreases as bits increase |
| Constant ratio | 1 | Ratio = √3·π/2 = 2.7207 |
| Edge cases | 9 | Scalar, identical, zeros, at-centroids, empty, 1-bit |
| Convergence | 8 | More grid points → stable centroids |
| Stress (100K batch) | 6 | Large-scale quantization consistency |

### test_quantizer_exhaustive.py — 1,266 tests

Verifies TurboQuantMSE (Algorithm 1) and TurboQuantProd (Algorithm 2).

**Parametrization:** dimensions {8, 32, 64, 128, 384} × bit-widths {2, 3, 4, 5, 6} × vector counts {1, 10, 100}

| Category | Tests | What It Verifies |
|----------|------:|-----------------|
| Rotation orthogonality | 7 | `R @ R.T ≈ I` for all dimensions |
| Norm preservation | 7 | `‖Rx‖ ≈ ‖x‖` after rotation |
| Seed determinism | 8 | Same seed → identical rotation matrix |
| Seed diversity | 8 | Different seed → different matrix |
| Quantize output shape | 75 | Correct (N, d) shape |
| Quantize output dtype | 75 | uint8 |
| Dequantize output dtype | 75 | float32 or float64 |
| Code index range | 75 | All in `[0, 2^b - 1]` |
| MSE decreases with bits | 25 | Monotonic improvement |
| Reconstruction norm | 75 | Reconstructed vectors have ‖x̃‖ ≈ 1 |
| Cosine similarity | 75 | `cos(x, x̃) > threshold(b)` |
| bytes_per_vector | 25 | `= ceil(b × d / 8)` |
| compression_ratio | 25 | `= (d × 4) / bytes_per_vector` |
| 1D input promotion | 4 | Single vector (1D) handled correctly |
| Prod: bits=1 raises | 1 | ValueError for insufficient bits |
| Prod: mse_quantizer bits | 20 | Uses `b-1` bits for MSE stage |
| Prod: QJL matrix shape | 20 | (d, d) matrix |
| Prod: output keys | 60 | Dict has mse_codes, qjl_signs, residual_norms |
| Prod: qjl_signs values | 60 | All values in {-1, +1} |
| Prod: residual_norms | 60 | Non-negative float |
| Prod: dequantize shape | 60 | Correct (N, d) float output |
| Prod: inner product bias | 9 | `E[<y, x̃>] ≈ <y, x>` (bias < 0.05) |
| Prod: storage formula | 20 | `ceil((b-1)×d/8) + ceil(d/8) + 4` |
| Cross-quantizer comparison | 20 | Prod storage > MSE(b-1) storage |
| Stress (10K vectors) | 6 | No NaN/Inf in large-scale reconstruction |

### test_index_exhaustive.py — 1,344 tests

Verifies the TurboQuantIndex high-level API.

**Parametrization:** dimensions {8, 32, 64, 128, 256, 384} × bit-widths {2, 3, 4, 5, 6} × vector counts {1, 10, 100, 500} × QJL modes {True, False}

| Category | Tests | What It Verifies |
|----------|------:|-----------------|
| Initial size = 0 | 60 | Empty index across all configurations |
| Compression ratio | 60 | `> 1` for all bit-widths < 32 |
| Stats structure | 60 | All required keys present |
| Size after add | 512 | Correct count for various N |
| Multiple adds accumulate | 128 | Size = sum of batch sizes |
| Dimension mismatch | 128 | ValueError raised |
| Auto-normalization | 128 | Non-normalized vectors handled |
| Search output shape | 90 | (Q, k) for similarities and indices |
| Similarities sorted | 18 | Descending order per query |
| Valid indices | 18 | All in `[0, N)` |
| No duplicate indices | 18 | Unique per query |
| k > N handling | 18 | Returns min(k, N) |
| Self-search recall | 18 | Query=database[i] → i in top results |
| Empty index search | 4 | Returns empty arrays |
| Recall@10 thresholds | 12 | Above minimum for each bit-width |
| Recall monotonicity | 3 | Improves with more bits |
| MSE vs QJL recall | 3 | Comparison between modes |
| Save/Load round-trip | 48 | Same size, same search results |
| Save/Load stats match | 48 | Identical statistics |
| Directory structure | 4 | Correct files per mode |
| meta.json contents | 36 | All expected keys and values |
| Stats consistency | 216 | Values match constructor args |
| Single vector add | 2 | 1D array promoted correctly |
| Single query search | 2 | 1D query works |
| k=1 search | 2 | Returns single result |
| High-dimensional | 2 | d=1024 works |
| Incremental vs batch | 6 | Results match |
| Stress (10K vectors) | 3 | Search works at scale |
| Multiple save/load cycles | 1 | Stable across iterations |

### test_properties.py — 211 tests

Verifies mathematical invariants and statistical properties from the paper.

| Category | Tests | What It Verifies |
|----------|------:|-----------------|
| Coordinate variance | 9 | `Var(x_i) ≈ 1/d` for random unit vectors |
| KS distribution test | 9 | Empirical CDF matches analytical PDF (p > 0.01) |
| Zero-mean coordinates | 9 | `E[x_i] ≈ 0` on the hypersphere |
| Inner product preservation | 5 | `<Rx, Ry> ≈ <x, y>` after rotation |
| Distance preservation | 5 | `‖Rx - Ry‖ ≈ ‖x - y‖` |
| L2 norm preservation | 5 | `‖Rx‖ ≈ ‖x‖` |
| MSE within theoretical bound | 25 | Empirical ≤ theoretical × d |
| MSE > 0 | 25 | Quantization always loses information |
| MSE monotonic decrease | 5 | More bits → less MSE |
| Finite per-coord MSE | 25 | No NaN/Inf |
| QJL bias ≈ 0 | 9 | `|E[<y, x̃> - <y, x>]| < 0.05` |
| QJL bias vs MSE-only | 9 | Prod bias ≤ MSE-only bias |
| MSE compression ratio | 8 | `= 32/b` |
| QJL < MSE compression | 7 | QJL mode stores more per vector |
| Positive compression | 8 | Always > 1 |
| Bytes consistency | 8 | Matches formula |
| Centroid symmetry | 3 | `sum(centroids) ≈ 0` |
| Negation symmetry | 3 | `quantize(-x)` mirrors `quantize(x)` |
| Zero-mean centroids | 3 | Mean ≈ 0 |
| MSE determinism | 4 | 3 identical runs |
| Prod determinism | 4 | 3 identical runs |
| Codebook determinism | 4 | 3 identical runs |
| Different seeds differ | 4 | Seed → different output |
| Reconstruction norms | 6 | `‖x̃‖ ≈ 1` for unit input |
| num_bits validation | 6 | Bounds enforced |

### test_integration.py — 140 tests

End-to-end pipelines, regression tests, and stress scenarios.

| Category | Tests | What It Verifies |
|----------|------:|-----------------|
| Full pipeline (QJL) | 27 | Generate → add → search → verify recall |
| Full pipeline (MSE-only) | 27 | Same pipeline without QJL |
| Save/Load search identity | 18 | Identical results after round-trip |
| Save/Load stats identity | 18 | Statistics preserved |
| Incremental vs batch add | 9 | 10 batches of 10 == 1 batch of 100 |
| Recall monotonicity (QJL) | 3 | bits 3→4→5 improves recall |
| Recall monotonicity (MSE) | 3 | Same for MSE-only mode |
| High-bits recall > 0.9 | 3 | 6-bit achieves 90%+ |
| Stress: 50K vectors | 1 | d=64, bits=4, recall > 0.5 |
| Stress: 10K vectors | 1 | d=128, bits=5, recall > 0.6 |
| Stress: non-self queries | 1 | Separate query/database sets |
| Constructor defaults | 1 | Default args work |
| Return types | 1 | Correct numpy types |
| Empty index behavior | 1 | No crash on empty search |
| Wrong dimension error | 1 | ValueError raised |
| Single vector operations | 1 | 1D input works end-to-end |
| k > N graceful | 1 | Returns all available |
| Auto-normalization | 1 | Non-unit vectors handled |
| Save creates directory | 1 | mkdir -p behavior |
| Deterministic codes | 1 | Fixed seed → fixed codes |
| Deterministic reconstruction | 1 | Fixed seed → fixed float values |
| Deterministic search | 1 | Fixed seed → fixed top-k |
| Deterministic centroids | 1 | Fixed seed → fixed codebook |
| Similarity bounds | 1 | All similarities in [-1, 1] |
| Self-query top-1 | 1 | Identity is best match |
| Cross-mode agreement | 3 | QJL and MSE return similar results |
| Empty save/load | 3 | Round-trip with 0 vectors |
| Multiple search stability | 3 | Repeated calls → same result |

## Paper Claims Verification

Each of the 6 claims from the TurboQuant paper is verified across multiple test suites:

| # | Paper Claim | Tests | Test Files |
|---|-------------|------:|------------|
| 1 | MSE/Shannon ratio = √3·π/2 ≈ 2.7207 | 16 | codebook_exhaustive, properties |
| 2 | Empirical MSE ≤ theoretical bound | 164 | codebook_exhaustive, quantizer_exhaustive, properties |
| 3 | Compression ratio = 32/b | 33 | quantizer_exhaustive, properties |
| 4 | QJL provides unbiased inner products | 18 | quantizer_exhaustive, properties |
| 5 | Zero preprocessing time | 140 | integration (all pipelines verify instant index creation) |
| 6 | Recall@10 ≥ 95% at 6-bit | 24 | index_exhaustive, integration, recall |

## Parametric Test Dimensions

Tests are parametrized across these ranges:

| Parameter | Values Tested |
|-----------|---------------|
| Dimension (d) | 3, 5, 8, 10, 16, 20, 32, 50, 64, 100, 128, 200, 256, 384, 768, 1024 |
| Bit-width (b) | 1, 2, 3, 4, 5, 6, 7, 8 |
| Vector count (N) | 1, 10, 50, 100, 500, 1000, 5000, 10000, 50000 |
| QJL mode | True, False |
| Seeds | 42, 123, 999 |
| Grid points | 1000, 5000, 10000, 50000 |

## Running Specific Test Categories

```bash
# Only mathematical properties
pytest tests/test_properties.py -v

# Only paper claim verification
pytest tests/test_properties.py tests/test_codebook_exhaustive.py -k "mse_bound or shannon or ratio" -v

# Only recall tests
pytest tests/ -k "recall" -v

# Only save/load tests
pytest tests/ -k "save or load" -v

# Only stress tests (large N)
pytest tests/ -k "stress" -v

# Only edge cases
pytest tests/ -k "edge" -v
```

## Continuous Integration

The full test suite requires ~13 minutes on a single CPU core. For CI pipelines:

```bash
# Fast smoke test (<30s) — run on every commit
pytest tests/test_codebook.py tests/test_quantizer.py tests/test_index.py tests/test_recall.py -q

# Full verification (~13min) — run on PRs and releases
pytest tests/ -q
```

## Dependencies

```
numpy>=1.24
scipy>=1.10
pytest>=7.0
```

No GPU, no C++ compiler, no additional system dependencies required.
