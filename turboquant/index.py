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
        memory_efficient: bool = False,
    ) -> None:
        self.dimension = dimension
        self.num_bits = num_bits
        self.metric = metric
        self.use_qjl = use_qjl and num_bits >= 2
        self.seed = seed
        self.memory_efficient = memory_efficient

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

        # ADC consolidated storage (used when memory_efficient=True)
        self._consolidated_codes: np.ndarray | None = None
        self._consolidated_signs: np.ndarray | None = None
        self._consolidated_norms: np.ndarray | None = None

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
        if self.memory_efficient:
            self._consolidate_codes()
            return

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
            self._codes = [merged]  # consolidate
        else:
            all_codes = np.concatenate(self._codes, axis=0)
            self._reconstructed = self._quantizer.dequantize(all_codes)
            self._codes = [all_codes]  # consolidate

        self._dirty = False

    def _consolidate_codes(self) -> None:
        """Consolidate codes into single arrays for ADC search (no float32 matrix)."""
        if not self._codes:
            self._consolidated_codes = None
            self._consolidated_signs = None
            self._consolidated_norms = None
            self._dirty = False
            return

        if self.use_qjl:
            all_mse = np.concatenate([c["mse_codes"] for c in self._codes], axis=0)
            all_signs = np.concatenate([c["qjl_signs"] for c in self._codes], axis=0)
            all_norms = np.concatenate([c["residual_norms"] for c in self._codes], axis=0)
            self._consolidated_codes = all_mse
            self._consolidated_signs = all_signs
            self._consolidated_norms = all_norms
            self._codes = [{
                "mse_codes": all_mse,
                "qjl_signs": all_signs,
                "residual_norms": all_norms,
            }]
        else:
            all_codes = np.concatenate(self._codes, axis=0)
            self._consolidated_codes = all_codes
            self._codes = [all_codes]

        self._dirty = False

    def _search_adc(
        self,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search using Asymmetric Distance Computation (no float32 matrix)."""
        if self._dirty:
            self._consolidate_codes()

        if self._consolidated_codes is None or self._size == 0:
            empty_sim = np.zeros((queries.shape[0], 0), dtype=np.float32)
            empty_idx = np.zeros((queries.shape[0], 0), dtype=np.int64)
            return empty_sim, empty_idx

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]

        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.clip(norms, 1e-8, None)

        # Get the MSE quantizer
        if self.use_qjl:
            quantizer = self._quantizer.mse_quantizer
        else:
            quantizer = self._quantizer

        # Step 1: Rotate queries
        q_rot = queries @ quantizer.rotation.T  # (Q, d)

        # Step 2: Build lookup table
        centroids = quantizer.codebook.centroids  # (num_levels,)
        # lut[q, j, k] = q_rot[q, j] * centroids[k]
        lut = q_rot[:, :, np.newaxis] * centroids[np.newaxis, np.newaxis, :]  # (Q, d, num_levels)

        # Step 3: Compute similarities via lookup (vectorized)
        codes = self._consolidated_codes  # (N, d) uint8
        Q = queries.shape[0]
        N = codes.shape[0]
        similarities = np.zeros((Q, N), dtype=np.float32)
        for j in range(self.dimension):
            similarities += lut[:, j, codes[:, j]]  # (Q, N) += (Q, N)

        # Step 4: QJL correction if applicable
        if self.use_qjl:
            S = self._quantizer.qjl_matrix  # (d, d)
            q_proj = queries @ S.T  # (Q, d)
            gamma = self._consolidated_norms  # (N,)
            signs = self._consolidated_signs.astype(np.float32)  # (N, d)
            scale = np.sqrt(np.pi / 2) / self.dimension
            # correction = scale * (q_proj @ signs.T) * gamma
            qjl_sim = scale * (q_proj @ signs.T) * gamma[np.newaxis, :]  # (Q, N)
            similarities += qjl_sim

        # Top-k selection
        k = min(k, self._size)
        if k >= self._size:
            sorted_idx = np.argsort(-similarities, axis=1)
        else:
            top_k_idx = np.argpartition(-similarities, k, axis=1)[:, :k]
            rows = np.arange(Q)[:, np.newaxis]
            top_k_sims = similarities[rows, top_k_idx]
            sort_order = np.argsort(-top_k_sims, axis=1)
            sorted_idx = top_k_idx[rows, sort_order]

        top_sims = similarities[np.arange(Q)[:, np.newaxis], sorted_idx[:, :k]]
        return top_sims, sorted_idx[:, :k]

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
        if self.memory_efficient:
            return self._search_adc(queries, k)

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
            if self.memory_efficient:
                self._consolidate_codes()
            else:
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
            "memory_efficient": self.memory_efficient,
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
            memory_efficient=meta.get("memory_efficient", False),
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
        """Return index statistics including memory overhead transparency."""
        d = self.dimension
        rotation_bytes = d * d * 4
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
            "memory_efficient": self.memory_efficient,
            "compression_ratio": f"{self.compression_ratio:.1f}x",
            "effective_compression_ratio": round(effective_ratio, 2),
            "bytes_per_vector": self._quantizer.bytes_per_vector,
            "total_bytes": code_bytes,
            "total_code_bytes": code_bytes,
            "rotation_matrix_bytes": rotation_bytes,
            "total_overhead_bytes": total_overhead,
            "float32_bytes": float32_bytes,
        }
