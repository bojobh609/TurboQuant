"""TurboQuant — Near-Optimal Vector Quantization for AI.

A pure Python/NumPy implementation of Google Research's TurboQuant algorithm
(ICLR 2026, arXiv:2504.19874). Achieves near-Shannon-limit compression with
zero preprocessing time.

Features:
    - TurboQuantMSE: MSE-optimal scalar quantization via random rotation (Algorithm 1)
    - TurboQuantProd: Unbiased inner product estimation via QJL residual correction (Algorithm 2)
    - TurboQuantIndex: Drop-in FAISS replacement for vector search
    - 2-8 bit compression with configurable quality/size trade-off

Quick Start:
    >>> from turboquant import TurboQuantIndex
    >>> index = TurboQuantIndex(dimension=384, num_bits=4)
    >>> index.add(vectors)
    >>> distances, indices = index.search(query, k=10)

Paper: https://arxiv.org/abs/2504.19874
"""

__version__ = "0.1.0"

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from turboquant.index import TurboQuantIndex
from turboquant.codebook import LloydMaxQuantizer

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantIndex",
    "LloydMaxQuantizer",
]
