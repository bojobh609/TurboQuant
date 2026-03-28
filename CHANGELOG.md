# Changelog

All notable changes to TurboQuant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-03-28

### Added
- `memory_efficient=True` mode — ADC search without decompressed float32 matrix in RAM
- Rotation matrix caching by (d, seed) — eliminates redundant QR decompositions
- QJL matrix caching by (d, seed)
- Real-data benchmark script with clustered and anisotropic distributions
- Query performance table in README

### Fixed
- Codes list consolidation after rebuild prevents O(k²) concatenation
- README: clarified storage vs runtime compression, IVF training requirements
- README: added query time benchmarks and multi-distribution recall data

### Changed
- Version: 0.2.0 -> 0.3.0

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
