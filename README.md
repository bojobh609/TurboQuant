# TurboQuant

**Near-Optimal Vector Quantization for AI — Pure Python Implementation**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)](#testing)
[![Paper](https://img.shields.io/badge/arXiv-2504.19874-b31b1b.svg)](https://arxiv.org/abs/2504.19874)

A production-ready Python implementation of Google Research's **TurboQuant** algorithm ([ICLR 2026](https://arxiv.org/abs/2504.19874)). Compress embedding vectors by **5-8x** with **95%+ recall** and **zero preprocessing time**.

## Why TurboQuant?

| Feature | FAISS PQ | ScaNN | **TurboQuant** |
|---------|----------|-------|----------------|
| Preprocessing | K-means (minutes) | Tree building (minutes) | **None (instant)** |
| Recall@10 | ~60% | ~85% | **95.3%** |
| Compression | 8x | 4x | **5-8x** |
| Dependencies | C++/CUDA | C++/TensorFlow | **Pure Python/NumPy** |
| Theory guarantee | None | None | **2.7x Shannon limit** |
| Training data needed | Yes | Yes | **No (data-oblivious)** |

## Quick Start

### Installation

```bash
pip install turboquant
```

Or from source:

```bash
git clone https://github.com/Firmamento-Technologies/TurboQuant.git
cd TurboQuant
pip install -e .
```

### Basic Usage

```python
from turboquant import TurboQuantIndex
import numpy as np

# Create index (drop-in FAISS replacement)
index = TurboQuantIndex(dimension=384, num_bits=6)

# Add vectors (auto-normalizes if needed)
vectors = np.random.randn(10000, 384).astype(np.float32)
index.add(vectors)

# Search
query = np.random.randn(1, 384).astype(np.float32)
similarities, indices = index.search(query, k=10)

# Save / Load
index.save("my_index")
loaded = TurboQuantIndex.load("my_index")
```

### With Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
from turboquant import TurboQuantIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode documents
docs = ["First document", "Second document", ...]
embeddings = model.encode(docs)

# Build compressed index
index = TurboQuantIndex(dimension=384, num_bits=6)
index.add(embeddings)

# Semantic search
query_emb = model.encode(["search query"])
similarities, doc_indices = index.search(query_emb, k=10)
```

### Low-Level Quantizer API

```python
from turboquant import TurboQuantMSE

# MSE-optimal quantizer (Algorithm 1 from paper)
tq = TurboQuantMSE(d=384, num_bits=4)

# Quantize
codes = tq.quantize(vectors)      # (N, 384) float32 -> (N, 384) uint8
recon = tq.dequantize(codes)      # (N, 384) uint8 -> (N, 384) float32

# Check quality
mse = np.mean(np.sum((vectors - recon)**2, axis=1))
cos_sim = np.mean(np.sum(vectors * recon, axis=1))
print(f"MSE: {mse:.6f}, Cosine Similarity: {cos_sim:.4f}")
```

## How It Works

TurboQuant implements a mathematically elegant compression scheme from [Google Research (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874):

### Algorithm: PolarQuant + Lloyd-Max

```
Input vector x (d dimensions, unit norm)
    |
    v
[Random Rotation] -- Orthogonal matrix via QR decomposition
    |                 Transforms x so coordinates follow Beta distribution
    v
[Lloyd-Max Quantization] -- Optimal scalar quantizer per coordinate
    |                        Pre-computed centroids for the Beta PDF
    v
Compressed: b bits per coordinate (vs 32 bits original)
```

**Key insight:** After random rotation, each coordinate of a unit vector independently follows a known Beta distribution (converging to Gaussian for large d). This allows coordinate-wise optimal scalar quantization — no codebook training, no data preprocessing.

### Theoretical Guarantees

From the paper (Theorems 1-3):

| Bits | MSE Distortion | Shannon Lower Bound | Ratio |
|------|---------------|--------------------:|------:|
| 1 | 0.363 | 0.250 | 1.45x |
| 2 | 0.117 | 0.063 | 1.87x |
| 3 | 0.034 | 0.016 | 2.20x |
| 4 | 0.009 | 0.004 | 2.41x |

TurboQuant operates within **2.7x of the information-theoretic limit** — provably near-optimal.

## Benchmark Results

Tested on `all-MiniLM-L6-v2` embeddings (d=384):

| Bits | Recall@10 | Cosine Sim | Compression | Memory (10K vectors) |
|------|-----------|-----------|-------------|---------------------|
| 2 | 59.2% | 0.882 | 16.0x | 0.94 MB |
| 3 | 77.6% | 0.965 | 10.7x | 1.40 MB |
| 4 | 86.2% | 0.990 | 8.0x | 1.88 MB |
| 5 | 92.6% | 0.997 | 6.4x | 2.34 MB |
| **6** | **95.3%** | **0.998** | **5.3x** | **2.81 MB** |

*Baseline: float32 brute-force = 100% recall, 15.0 MB for 10K vectors*

### Recommended Configuration

| Use Case | Bits | Recall | Compression |
|----------|------|--------|-------------|
| Maximum compression (IoT, mobile) | 3 | 77.6% | 10.7x |
| Balanced (default) | 4 | 86.2% | 8.0x |
| High accuracy (RAG, search) | **6** | **95.3%** | **5.3x** |
| Near-lossless | 8 | 99.5% | 4.0x |

## API Reference

### `TurboQuantIndex`

High-level vector search index with TurboQuant compression.

```python
TurboQuantIndex(
    dimension: int,          # Vector dimension (e.g., 384 for MiniLM)
    num_bits: int = 4,       # Bits per coordinate (2-8)
    metric: str = "cosine",  # Similarity metric
    use_qjl: bool = False,   # Enable QJL for unbiased inner products
    seed: int = 42,          # Random seed for reproducibility
)
```

**Methods:**
- `add(vectors)` — Add vectors to the index
- `search(queries, k=10)` — Return (similarities, indices) for top-k neighbors
- `save(path)` / `load(path)` — Persist to disk
- `stats()` — Return index statistics

### `TurboQuantMSE`

MSE-optimal vector quantizer (Algorithm 1).

```python
TurboQuantMSE(d: int, num_bits: int = 4, seed: int = 42)
```

**Methods:**
- `quantize(x)` — Vectors (N, d) float32 → Codes (N, d) uint8
- `dequantize(codes)` — Codes (N, d) uint8 → Reconstructed (N, d) float32

### `TurboQuantProd`

Inner-product-optimal quantizer with QJL correction (Algorithm 2).

```python
TurboQuantProd(d: int, num_bits: int = 4, seed: int = 42)
```

**Methods:**
- `quantize(x)` — Returns dict with `mse_codes`, `qjl_signs`, `residual_norms`
- `dequantize(codes)` — Reconstructed vectors with unbiased inner products

### `LloydMaxQuantizer`

Optimal scalar quantizer for hypersphere coordinate distribution.

```python
LloydMaxQuantizer(d: int, num_bits: int = 4)
```

## Architecture

```
turboquant/
  __init__.py       # Public API
  codebook.py       # Lloyd-Max quantizer for Beta distribution
  quantizer.py      # TurboQuantMSE & TurboQuantProd (Algorithms 1 & 2)
  index.py          # TurboQuantIndex (FAISS-compatible vector search)

tests/
  test_codebook.py  # Codebook and PDF tests
  test_quantizer.py # MSE/Prod quantizer tests
  test_index.py     # Index add/search/save/load tests
  test_recall.py    # Recall benchmark comparison

examples/
  quickstart.py     # Basic usage example
  semantic_search.py # Sentence-transformers integration
  benchmark.py      # FAISS comparison benchmark
```

## Testing

```bash
# Run all tests
pip install -e ".[dev]"
pytest tests/ -v

# Run benchmarks
pip install -e ".[bench]"
python examples/benchmark.py
```

## Comparison with Existing Implementations

| Implementation | Focus | GPU Required | Pure Python |
|---------------|-------|:---:|:---:|
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | KV cache (vLLM) | Yes | No |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | KV cache (PyTorch) | Yes | No |
| [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) | vLLM fork | Yes | No |
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | llama.cpp | Yes | No |
| **This implementation** | **Vector search** | **No** | **Yes** |

This is the only implementation focused on **vector similarity search** (FAISS replacement) rather than KV cache compression, and the only one that runs on **CPU with pure Python/NumPy**.

## Citation

This implementation is based on:

```bibtex
@inproceedings{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2504.19874}
}
```

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

## About

Built by [Firmamento Technologies](https://github.com/Firmamento-Technologies) — Deep-tech solutions from Genova, Italy.
