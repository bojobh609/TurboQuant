"""TurboQuantIndex — Drop-in replacement for FAISS vector search.

Provides a high-level vector index that uses TurboQuant compression for
memory-efficient approximate nearest neighbor search.

Example:
    >>> from turboquant import TurboQuantIndex
    >>>
    >>> # Build index
    >>> index = TurboQuantIndex(dimension=384, num_bits=4)
    >>> index.add(database_vectors)  # (N, 384) normalized vectors
    >>>
    >>> # Search
    >>> distances, indices = index.search(query_vectors, k=10)
    >>>
    >>> # Save / Load
    >>> index.save("my_index")
    >>> loaded = TurboQuantIndex.load("my_index")

Reference: TurboQuant (arXiv:2504.19874)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


class TurboQuantIndex:
    """Vector search index with TurboQuant compression.

    Supports both MSE-only and MSE+QJL (inner-product-optimal) modes.
    The QJL mode provides unbiased similarity estimation and higher recall
    at the cost of slightly more storage (1 extra bit per dimension).

    Args:
        dimension: Vector dimension.
        num_bits: Bits per coordinate (2-8). Default 4.
        metric: Similarity metric — 'cosine' or 'ip' (inner product).
        use_qjl: If True, use TurboQuantProd (MSE+QJL) for unbiased inner
            product estimation. Improves recall at cost of ~12% more storage.
            Default True.
        seed: Random seed for reproducible quantization.

    Attributes:
        size: Number of vectors in the index.
        compression_ratio: Compression factor vs float32.
    """

    def __init__(
        self,
        dimension: int,
        num_bits: int = 4,
        metric: str = "cosine",
        use_qjl: bool = True,
        seed: int = 42,
    ) -> None:
        self.dimension = dimension
        self.num_bits = num_bits
        self.metric = metric
        self.use_qjl = use_qjl and num_bits >= 2
        self.seed = seed

        # Build quantizer
        if self.use_qjl:
            self._quantizer = TurboQuantProd(d=dimension, num_bits=num_bits, seed=seed)
        else:
            self._quantizer = TurboQuantMSE(d=dimension, num_bits=num_bits, seed=seed)

        # Storage
        self._codes: list = []
        self._reconstructed: np.ndarray | None = None
        self._size = 0
        self._dirty = False

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._size

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        return self._quantizer.compression_ratio

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape (N, dimension). Vectors should be
                L2-normalized for best results. Non-normalized vectors
                are automatically normalized.
        """
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

        # Quantize
        codes = self._quantizer.quantize(vectors)

        # Store codes
        self._codes.append(codes)
        self._size += vectors.shape[0]
        self._dirty = True

    def _rebuild_reconstructed(self) -> None:
        """Rebuild the full reconstructed matrix from stored codes."""
        if not self._codes:
            self._reconstructed = None
            return

        if self.use_qjl:
            # Merge all code dicts
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

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            queries: Query vectors, shape (Q, dimension).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (similarities, indices):
                similarities: shape (Q, k), cosine similarities (descending)
                indices: shape (Q, k), database vector indices
        """
        if self._dirty:
            self._rebuild_reconstructed()

        if self._reconstructed is None or self._size == 0:
            empty_sim = np.zeros((queries.shape[0], 0), dtype=np.float32)
            empty_idx = np.zeros((queries.shape[0], 0), dtype=np.int64)
            return empty_sim, empty_idx

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]

        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.clip(norms, 1e-8, None)

        # Compute similarities
        similarities = queries @ self._reconstructed.T  # (Q, N)

        # Get top-k
        k = min(k, self._size)
        if k >= self._size:
            sorted_idx = np.argsort(-similarities, axis=1)
        else:
            # Use argpartition for efficiency when k << N
            top_k_idx = np.argpartition(-similarities, k, axis=1)[:, :k]
            # Sort the top-k by similarity
            rows = np.arange(queries.shape[0])[:, np.newaxis]
            top_k_sims = similarities[rows, top_k_idx]
            sort_order = np.argsort(-top_k_sims, axis=1)
            sorted_idx = top_k_idx[rows, sort_order]

        top_sims = similarities[np.arange(queries.shape[0])[:, np.newaxis], sorted_idx[:, :k]]

        return top_sims, sorted_idx[:, :k]

    def save(self, path: str | Path) -> None:
        """Save index to disk.

        Creates a directory with metadata, quantizer state, and compressed codes.

        Args:
            path: Directory path to save to.
        """
        if self._dirty:
            self._rebuild_reconstructed()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "dimension": self.dimension,
            "num_bits": self.num_bits,
            "metric": self.metric,
            "use_qjl": self.use_qjl,
            "seed": self.seed,
            "size": self._size,
            "version": "0.1.0",
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

        # Save rotation matrix
        if self.use_qjl:
            np.save(path / "rotation.npy", self._quantizer.mse_quantizer.rotation)
            np.save(path / "qjl_matrix.npy", self._quantizer.qjl_matrix)
            np.save(path / "centroids.npy", self._quantizer.mse_quantizer.codebook.centroids)
        else:
            np.save(path / "rotation.npy", self._quantizer.rotation)
            np.save(path / "centroids.npy", self._quantizer.codebook.centroids)

        # Save codes
        if self._codes:
            if self.use_qjl:
                all_mse = np.concatenate([c["mse_codes"] for c in self._codes], axis=0)
                all_signs = np.concatenate([c["qjl_signs"] for c in self._codes], axis=0)
                all_norms = np.concatenate([c["residual_norms"] for c in self._codes], axis=0)
                np.save(path / "mse_codes.npy", all_mse)
                np.save(path / "qjl_signs.npy", all_signs)
                np.save(path / "residual_norms.npy", all_norms)
            else:
                all_codes = np.concatenate(self._codes, axis=0)
                np.save(path / "codes.npy", all_codes)

    @classmethod
    def load(cls, path: str | Path) -> TurboQuantIndex:
        """Load index from disk.

        Args:
            path: Directory path to load from.

        Returns:
            Loaded TurboQuantIndex.
        """
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        index = cls(
            dimension=meta["dimension"],
            num_bits=meta["num_bits"],
            metric=meta.get("metric", "cosine"),
            use_qjl=meta.get("use_qjl", True),
            seed=meta["seed"],
        )

        # Restore rotation matrix and centroids
        if index.use_qjl:
            index._quantizer.mse_quantizer.rotation = np.load(path / "rotation.npy")
            index._quantizer.qjl_matrix = np.load(path / "qjl_matrix.npy")
            index._quantizer.mse_quantizer.codebook.centroids = np.load(path / "centroids.npy")

            if (path / "mse_codes.npy").exists():
                codes = {
                    "mse_codes": np.load(path / "mse_codes.npy"),
                    "qjl_signs": np.load(path / "qjl_signs.npy"),
                    "residual_norms": np.load(path / "residual_norms.npy"),
                }
                index._codes = [codes]
                index._size = codes["mse_codes"].shape[0]
                index._rebuild_reconstructed()
        else:
            index._quantizer.rotation = np.load(path / "rotation.npy")
            index._quantizer.codebook.centroids = np.load(path / "centroids.npy")

            if (path / "codes.npy").exists():
                codes = np.load(path / "codes.npy")
                index._codes = [codes]
                index._size = codes.shape[0]
                index._rebuild_reconstructed()

        return index

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "size": self._size,
            "dimension": self.dimension,
            "num_bits": self.num_bits,
            "use_qjl": self.use_qjl,
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "bytes_per_vector": self._quantizer.bytes_per_vector,
            "total_bytes": self._size * self._quantizer.bytes_per_vector,
            "float32_bytes": self._size * self.dimension * 4,
        }
