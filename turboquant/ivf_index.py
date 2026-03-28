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
    n, d = vectors.shape
    centroids = np.empty((k, d), dtype=np.float32)
    idx = rng.randint(n)
    centroids[0] = vectors[idx]

    for i in range(1, k):
        dists = 1.0 - vectors @ centroids[:i].T
        min_dists = dists.min(axis=1)
        min_dists = np.clip(min_dists, 0.0, None)
        probs = min_dists / (min_dists.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centroids[i] = vectors[idx]

    return centroids


def _kmeans(vectors: np.ndarray, k: int, max_iter: int = 20,
            seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    centroids = _kmeans_plus_plus(vectors, k, rng)

    for _ in range(max_iter):
        sims = vectors @ centroids.T
        assignments = np.argmax(sims, axis=1)

        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = vectors[mask].mean(axis=0)
            else:
                new_centroids[j] = vectors[rng.randint(vectors.shape[0])]

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
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-8, None)

        effective_nlist = min(self.nlist, vectors.shape[0])

        self._centroids = _kmeans(vectors, effective_nlist, seed=self.seed)

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

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-8, None)

        sims = vectors @ self._centroids.T
        assignments = np.argmax(sims, axis=1)

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
        if not self._trained or self._size == 0:
            empty_sim = np.zeros((queries.shape[0], 0), dtype=np.float32)
            empty_idx = np.zeros((queries.shape[0], 0), dtype=np.int64)
            return empty_sim, empty_idx

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]

        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = queries / np.clip(norms, 1e-8, None)

        effective_nprobe = min(self.nprobe, len(self._partitions))
        centroid_sims = queries @ self._centroids.T
        top_partitions = np.argsort(-centroid_sims, axis=1)[:, :effective_nprobe]

        num_queries = queries.shape[0]
        all_sims = []
        all_ids = []

        for qi in range(num_queries):
            query = queries[qi:qi+1]
            q_sims = []
            q_ids = []

            for pi in top_partitions[qi]:
                partition = self._partitions[pi]
                if partition.size == 0:
                    continue
                local_k = min(k, partition.size)
                p_sims, p_local_idx = partition.search(query, k=local_k)
                id_map = self._id_maps[pi]
                for j in range(p_sims.shape[1]):
                    local_idx = p_local_idx[0, j]
                    q_sims.append(p_sims[0, j])
                    q_ids.append(id_map[local_idx])

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

        max_k = max(len(s) for s in all_sims) if all_sims else 0
        result_sims = np.zeros((num_queries, max_k), dtype=np.float32)
        result_ids = np.full((num_queries, max_k), -1, dtype=np.int64)
        for qi in range(num_queries):
            n = len(all_sims[qi])
            result_sims[qi, :n] = all_sims[qi]
            result_ids[qi, :n] = all_ids[qi]

        return result_sims, result_ids

    def save(self, path: str | Path) -> None:
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

        for i, partition in enumerate(self._partitions):
            if partition.size > 0:
                pdir = path / f"partition_{i:04d}"
                partition.save(pdir)

        if self._id_maps:
            id_maps_data = {str(i): ids for i, ids in enumerate(self._id_maps)}
            (path / "id_maps.json").write_text(json.dumps(id_maps_data))

    @classmethod
    def load(cls, path: str | Path) -> IVFTurboQuantIndex:
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
